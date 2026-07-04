import time

import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

from recycling_cell_msgs.action import PickObject, PlaceObject

PICK_STAGES = [
    'MOVING_TO_PRE_GRASP',
    'MOVING_TO_GRASP',
    'CLOSING_GRIPPER',
    'LIFTING_OBJECT',
    'PICK_DONE',
]

PLACE_STAGES = [
    'MOVING_TO_BIN_PRE_PLACE',
    'MOVING_TO_PLACE_POSE',
    'OPENING_GRIPPER',
    'PLACE_DONE',
]

STAGE_DURATION_SEC = 0.5


class FakeManipulationNode(Node):

    def __init__(self):
        super().__init__('fake_manipulation_node')

        self.declare_parameter('force_pick_failure', False)
        self.declare_parameter('force_place_failure', False)
        self.declare_parameter('min_graspability', 0.5)

        self.callback_group_ = ReentrantCallbackGroup()

        self.pick_action_server_ = ActionServer(
            self,
            PickObject,
            '/manipulation/pick_object',
            execute_callback=self.execute_pick_callback,
            callback_group=self.callback_group_,
        )

        self.place_action_server_ = ActionServer(
            self,
            PlaceObject,
            '/manipulation/place_object',
            execute_callback=self.execute_place_callback,
            callback_group=self.callback_group_,
        )

        self.get_logger().info('fake_manipulation_node started')

    @staticmethod
    def _shutdown_result(goal_handle, result_type):
        # Explicitly mark the goal as aborted (best-effort) so the action
        # server does not have to fall back to its own "goal state not
        # set, assuming aborted" auto-abort path once we return.
        try:
            goal_handle.abort()
        except Exception:
            pass
        result = result_type.Result()
        result.success = False
        result.error_code = 'NODE_SHUTTING_DOWN'
        result.message = 'fake_manipulation_node is shutting down'
        return result

    def execute_pick_callback(self, goal_handle):
        request = goal_handle.request
        self.get_logger().info(
            f'PickObject goal received: object_id={request.object_id}, '
            f'class_name={request.class_name}')

        min_graspability = self.get_parameter('min_graspability').value
        force_pick_failure = self.get_parameter('force_pick_failure').value

        # --- failure simulation: graspability check ---
        if request.graspability_score < min_graspability:
            self.get_logger().warn(
                f'[pick] graspability_score={request.graspability_score:.2f} '
                f'below min_graspability={min_graspability:.2f}, aborting')
            goal_handle.abort()
            result = PickObject.Result()
            result.success = False
            result.error_code = 'GRASPABILITY_TOO_LOW'
            result.message = (
                f'object_id={request.object_id} has '
                f'graspability_score={request.graspability_score:.2f}, '
                f'below required min_graspability={min_graspability:.2f}')
            return result

        # --- failure simulation: forced pick failure ---
        if force_pick_failure:
            self.get_logger().warn(
                '[pick] force_pick_failure is true, aborting')
            goal_handle.abort()
            result = PickObject.Result()
            result.success = False
            result.error_code = 'PICK_EXECUTION_FAILED'
            result.message = (
                f'pick of {request.object_id} failed '
                f'(force_pick_failure=true)')
            return result

        feedback_msg = PickObject.Feedback()
        num_stages = len(PICK_STAGES)

        for i, stage in enumerate(PICK_STAGES):
            if not rclpy.ok():
                return self._shutdown_result(goal_handle, PickObject)

            feedback_msg.current_stage = stage
            feedback_msg.progress = (i + 1) / num_stages
            try:
                goal_handle.publish_feedback(feedback_msg)
                self.get_logger().info(
                    f'[pick] stage={stage} '
                    f'progress={feedback_msg.progress:.2f}')
            except Exception:
                # node is shutting down mid-goal; the publisher/context
                # backing this goal handle may already be torn down
                return self._shutdown_result(goal_handle, PickObject)
            time.sleep(STAGE_DURATION_SEC)

        if not rclpy.ok():
            return self._shutdown_result(goal_handle, PickObject)

        result = PickObject.Result()
        try:
            goal_handle.succeed()
            result.success = True
            result.error_code = ''
            result.message = f'pick of {request.object_id} completed'
        except Exception:
            return self._shutdown_result(goal_handle, PickObject)
        return result

    def execute_place_callback(self, goal_handle):
        request = goal_handle.request
        self.get_logger().info(
            f'PlaceObject goal received: object_id={request.object_id}, '
            f'target_bin_id={request.target_bin_id}')

        force_place_failure = self.get_parameter('force_place_failure').value

        # --- failure simulation: forced place failure ---
        if force_place_failure:
            self.get_logger().warn(
                '[place] force_place_failure is true, aborting')
            goal_handle.abort()
            result = PlaceObject.Result()
            result.success = False
            result.error_code = 'PLACE_EXECUTION_FAILED'
            result.message = (
                f'place of {request.object_id} into {request.target_bin_id} '
                f'failed (force_place_failure=true)')
            return result

        feedback_msg = PlaceObject.Feedback()
        num_stages = len(PLACE_STAGES)

        for i, stage in enumerate(PLACE_STAGES):
            if not rclpy.ok():
                return self._shutdown_result(goal_handle, PlaceObject)

            feedback_msg.current_stage = stage
            feedback_msg.progress = (i + 1) / num_stages
            try:
                goal_handle.publish_feedback(feedback_msg)
                self.get_logger().info(
                    f'[place] stage={stage} '
                    f'progress={feedback_msg.progress:.2f}')
            except Exception:
                # node is shutting down mid-goal; the publisher/context
                # backing this goal handle may already be torn down
                return self._shutdown_result(goal_handle, PlaceObject)
            time.sleep(STAGE_DURATION_SEC)

        if not rclpy.ok():
            return self._shutdown_result(goal_handle, PlaceObject)

        result = PlaceObject.Result()
        try:
            goal_handle.succeed()
            result.success = True
            result.error_code = ''
            result.message = (
                f'place of {request.object_id} into '
                f'{request.target_bin_id} completed')
        except Exception:
            return self._shutdown_result(goal_handle, PlaceObject)
        return result


def main(args=None):
    rclpy.init(args=args)
    node = FakeManipulationNode()
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
