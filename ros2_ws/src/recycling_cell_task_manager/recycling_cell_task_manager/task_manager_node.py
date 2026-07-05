import time
import uuid

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

from recycling_cell_msgs.action import PickObject, PlaceObject
from recycling_cell_msgs.msg import DetectedObjectArray, RobotState, SortResult

CONFIDENCE_THRESHOLD = 0.5
GRASPABILITY_THRESHOLD = 0.3

CLASS_TO_BIN = {
    'plastic_bottle': 'plastic_bin',
    'can': 'metal_bin',
    'paper_cup': 'paper_bin',
    'glass_bottle': 'glass_bin',
}
DEFAULT_BIN_ID = 'reject_bin'

# Test coordinates chosen to stay well within the Panda arm's reachable
# workspace (see recycling_cell_moveit_manipulation for the pre/release/
# retreat z-offsets applied on top of these).
BIN_POSES = {
    'plastic_bin': (0.35, 0.20, 0.25),
    'metal_bin': (0.45, 0.20, 0.25),
    'paper_bin': (0.55, 0.15, 0.25),
    'glass_bin': (0.60, 0.10, 0.25),
    'reject_bin': (0.25, 0.20, 0.25),
}

STATE_IDLE = 'IDLE'
STATE_PICKING = 'PICKING'
STATE_PLACING = 'PLACING'


class TaskManagerNode(Node):

    def __init__(self):
        super().__init__('task_manager_node')

        self.declare_parameter('max_pick_retries', 2)
        self.declare_parameter('max_place_retries', 1)
        self.declare_parameter('enable_object_memory', True)

        self.callback_group_ = ReentrantCallbackGroup()

        self.detected_objects_sub_ = self.create_subscription(
            DetectedObjectArray,
            '/perception/detected_objects',
            self.detected_objects_callback,
            10,
            callback_group=self.callback_group_,
        )

        self.sort_result_pub_ = self.create_publisher(
            SortResult, '/task_manager/sort_results', 10)

        self.robot_state_pub_ = self.create_publisher(
            RobotState, '/robot/state', 10)

        self.pick_action_client_ = ActionClient(
            self, PickObject, '/manipulation/pick_object',
            callback_group=self.callback_group_)

        self.place_action_client_ = ActionClient(
            self, PlaceObject, '/manipulation/place_object',
            callback_group=self.callback_group_)

        self.robot_id_ = 'robot_1'
        self.state_ = STATE_IDLE
        self.is_error_ = False
        self.last_error_code_ = ''
        self.progress_ = 0.0
        self.cycle_start_time_ = 0.0

        self.current_task_id_ = ''
        self.current_object_id_ = ''
        self.current_class_name_ = ''
        self.current_confidence_ = 0.0
        self.current_target_bin_id_ = ''
        self.current_target_pose_ = Pose()
        self.current_graspability_score_ = 0.0

        self.pick_retry_count_ = 0
        self.place_retry_count_ = 0

        self.processed_object_ids_ = set()
        self.failed_object_ids_ = set()

        self.state_timer_ = self.create_timer(1.0, self.publish_robot_state)

        self.get_logger().info('task_manager_node started')

    # ---------- perception callback ----------

    def detected_objects_callback(self, msg):
        if self.state_ != STATE_IDLE:
            return

        target = self.select_best_target(msg.objects)
        if target is None:
            self.get_logger().info(
                'No valid unprocessed target object found.',
                throttle_duration_sec=5.0)
            return

        self.start_task(target)

    def select_best_target(self, objects):
        enable_object_memory = self.get_parameter(
            'enable_object_memory').value

        candidates = []
        for obj in objects:
            if obj.is_unknown or obj.class_name == 'unknown':
                continue
            if obj.confidence < CONFIDENCE_THRESHOLD:
                continue
            if obj.graspability_score < GRASPABILITY_THRESHOLD:
                continue
            if enable_object_memory \
                    and obj.object_id in self.processed_object_ids_:
                continue
            if obj.object_id in self.failed_object_ids_:
                continue
            candidates.append(obj)

        if not candidates:
            return None

        candidates.sort(
            key=lambda o: o.confidence * o.graspability_score, reverse=True)
        return candidates[0]

    @staticmethod
    def class_to_bin(class_name):
        return CLASS_TO_BIN.get(class_name, DEFAULT_BIN_ID)

    @staticmethod
    def bin_pose(bin_id):
        x, y, z = BIN_POSES.get(bin_id, BIN_POSES[DEFAULT_BIN_ID])
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = 1.0
        return pose

    # ---------- task flow: pick ----------

    def start_task(self, target):
        self.state_ = STATE_PICKING
        self.is_error_ = False
        self.last_error_code_ = ''
        self.progress_ = 0.0
        self.cycle_start_time_ = time.time()

        self.current_task_id_ = str(uuid.uuid4())
        self.current_object_id_ = target.object_id
        self.current_class_name_ = target.class_name
        self.current_confidence_ = target.confidence
        self.current_target_bin_id_ = self.class_to_bin(target.class_name)
        self.current_target_pose_ = target.pose_base
        self.current_graspability_score_ = target.graspability_score

        self.pick_retry_count_ = 0
        self.place_retry_count_ = 0

        self.publish_robot_state()

        self.get_logger().info(
            f'starting task {self.current_task_id_}: '
            f'object_id={target.object_id} class_name={target.class_name} '
            f'-> target_bin_id={self.current_target_bin_id_}')

        self.send_pick_goal()

    def send_pick_goal(self):
        # If the pick action server hasn't been discovered yet (e.g. right
        # after bringup, before manipulation node's action server has
        # finished registering), send_goal_async() would fire a goal request
        # that never reaches anyone and never resolves. Waiting here blocks
        # only this callback's worker thread, not the whole executor.
        self.pick_action_client_.wait_for_server()

        goal_msg = PickObject.Goal()
        goal_msg.object_id = self.current_object_id_
        goal_msg.class_name = self.current_class_name_
        goal_msg.target_pose = self.current_target_pose_
        goal_msg.graspability_score = self.current_graspability_score_

        send_goal_future = self.pick_action_client_.send_goal_async(
            goal_msg, feedback_callback=self.pick_feedback_callback)
        send_goal_future.add_done_callback(self.pick_goal_response_callback)

    def pick_feedback_callback(self, feedback_msg):
        if not rclpy.ok():
            return
        feedback = feedback_msg.feedback
        self.progress_ = feedback.progress
        self.get_logger().info(
            f'[pick] stage={feedback.current_stage} '
            f'progress={feedback.progress:.2f}')
        self.publish_robot_state()

    def pick_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.retry_pick_or_finish(
                'PICK_GOAL_REJECTED',
                'pick goal rejected by manipulation server')
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.pick_result_callback)

    def pick_result_callback(self, future):
        result_response = future.result()
        result = result_response.result

        if result_response.status != GoalStatus.STATUS_SUCCEEDED \
                or not result.success:
            self.retry_pick_or_finish(
                result.error_code or 'PICK_FAILED',
                result.message or 'pick failed')
            return

        self.get_logger().info('pick succeeded, sending place goal')
        self.send_place_goal()

    def retry_pick_or_finish(self, error_code, message):
        max_retries = self.get_parameter('max_pick_retries').value

        if self.pick_retry_count_ < max_retries:
            self.pick_retry_count_ += 1
            self.get_logger().warn(
                f'Pick failed with {error_code}. Retrying pick '
                f'{self.pick_retry_count_}/{max_retries}')
            self.progress_ = 0.0
            self.publish_robot_state()
            self.send_pick_goal()
            return

        self.get_logger().error(
            f'Pick failed with {error_code} after {max_retries} retries, '
            f'giving up')
        self.finish_task(
            pick_success=False, place_success=False,
            error_code='MAX_PICK_RETRY_EXCEEDED',
            message=(
                f'pick of {self.current_object_id_} failed after '
                f'{max_retries} retries; last error: {error_code} - '
                f'{message}'))

    # ---------- task flow: place ----------

    def send_place_goal(self):
        self.state_ = STATE_PLACING
        self.progress_ = 0.0
        self.publish_robot_state()

        # See send_pick_goal() for why this wait is necessary.
        self.place_action_client_.wait_for_server()

        place_pose = self.bin_pose(self.current_target_bin_id_)
        self.get_logger().info(
            f'Selected target bin pose: '
            f'bin_id={self.current_target_bin_id_} '
            f'x={place_pose.position.x:.2f} '
            f'y={place_pose.position.y:.2f} '
            f'z={place_pose.position.z:.2f}')

        goal_msg = PlaceObject.Goal()
        goal_msg.object_id = self.current_object_id_
        goal_msg.class_name = self.current_class_name_
        goal_msg.target_bin_id = self.current_target_bin_id_
        goal_msg.place_pose = place_pose

        send_goal_future = self.place_action_client_.send_goal_async(
            goal_msg, feedback_callback=self.place_feedback_callback)
        send_goal_future.add_done_callback(self.place_goal_response_callback)

    def place_feedback_callback(self, feedback_msg):
        if not rclpy.ok():
            return
        feedback = feedback_msg.feedback
        self.progress_ = feedback.progress
        self.get_logger().info(
            f'[place] stage={feedback.current_stage} '
            f'progress={feedback.progress:.2f}')
        self.publish_robot_state()

    def place_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.retry_place_or_finish(
                'PLACE_GOAL_REJECTED',
                'place goal rejected by manipulation server')
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.place_result_callback)

    def place_result_callback(self, future):
        result_response = future.result()
        result = result_response.result

        if result_response.status != GoalStatus.STATUS_SUCCEEDED \
                or not result.success:
            self.retry_place_or_finish(
                result.error_code or 'PLACE_FAILED',
                result.message or 'place failed')
            return

        self.get_logger().info('place succeeded, task complete')
        self.finish_task(
            pick_success=True, place_success=True,
            error_code='', message='sort task completed successfully')

    def retry_place_or_finish(self, error_code, message):
        max_retries = self.get_parameter('max_place_retries').value

        if self.place_retry_count_ < max_retries:
            self.place_retry_count_ += 1
            self.get_logger().warn(
                f'Place failed with {error_code}. Retrying place '
                f'{self.place_retry_count_}/{max_retries}')
            self.progress_ = 0.0
            self.publish_robot_state()
            self.send_place_goal()
            return

        self.get_logger().error(
            f'Place failed with {error_code} after {max_retries} retries, '
            f'giving up')
        self.finish_task(
            pick_success=True, place_success=False,
            error_code='MAX_PLACE_RETRY_EXCEEDED',
            message=(
                f'place of {self.current_object_id_} into '
                f'{self.current_target_bin_id_} failed after '
                f'{max_retries} retries; last error: {error_code} - '
                f'{message}'))

    # ---------- finishing ----------

    def finish_task(self, pick_success, place_success, error_code, message):
        overall_success = pick_success and place_success
        cycle_time_sec = time.time() - self.cycle_start_time_

        sort_result = SortResult()
        sort_result.header.stamp = self.get_clock().now().to_msg()
        sort_result.header.frame_id = 'base_link'
        sort_result.task_id = self.current_task_id_
        sort_result.object_id = self.current_object_id_
        sort_result.class_name = self.current_class_name_
        sort_result.target_bin_id = self.current_target_bin_id_
        sort_result.pick_success = pick_success
        sort_result.place_success = place_success
        sort_result.overall_success = overall_success
        sort_result.confidence = self.current_confidence_
        sort_result.cycle_time_sec = cycle_time_sec
        sort_result.error_code = error_code
        sort_result.message = message
        self.sort_result_pub_.publish(sort_result)

        self.get_logger().info(
            f'task {self.current_task_id_} finished: '
            f'overall_success={overall_success} '
            f'cycle_time_sec={cycle_time_sec:.2f}')

        if overall_success:
            self.processed_object_ids_.add(self.current_object_id_)
            self.get_logger().info(
                f'Object marked as processed: '
                f'object_id={self.current_object_id_}')
        else:
            self.failed_object_ids_.add(self.current_object_id_)
            self.get_logger().info(
                f'Object marked as failed: '
                f'object_id={self.current_object_id_}')

        self.is_error_ = not overall_success
        self.last_error_code_ = error_code
        self.progress_ = 1.0
        self.state_ = STATE_IDLE
        self.publish_robot_state()

        self.current_task_id_ = ''

    # ---------- robot state ----------

    def publish_robot_state(self):
        if not rclpy.ok():
            return

        msg = RobotState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.robot_id = self.robot_id_
        msg.state = self.state_
        msg.current_task_id = self.current_task_id_
        msg.last_error_code = self.last_error_code_
        msg.is_busy = self.state_ != STATE_IDLE
        msg.is_error = self.is_error_
        msg.progress = self.progress_
        self.robot_state_pub_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TaskManagerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
