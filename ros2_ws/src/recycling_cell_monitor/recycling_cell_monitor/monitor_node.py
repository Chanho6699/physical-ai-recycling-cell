import shutil

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from recycling_cell_msgs.msg import DetectedObjectArray, RobotState, SortResult


class MonitorNode(Node):

    def __init__(self):
        super().__init__('monitor_node')

        self.robot_state_ = None
        self.detected_objects_ = None
        self.last_sort_result_ = None

        self.robot_state_sub_ = self.create_subscription(
            RobotState,
            '/robot/state',
            self.robot_state_callback,
            10,
        )

        self.sort_result_sub_ = self.create_subscription(
            SortResult,
            '/task_manager/sort_results',
            self.sort_result_callback,
            10,
        )

        self.detected_objects_sub_ = self.create_subscription(
            DetectedObjectArray,
            '/perception/detected_objects',
            self.detected_objects_callback,
            10,
        )

        self.render_timer_ = self.create_timer(1.0, self.render)

        self.get_logger().info('monitor_node started')

    def robot_state_callback(self, msg):
        self.robot_state_ = msg

    def sort_result_callback(self, msg):
        self.last_sort_result_ = msg

    def detected_objects_callback(self, msg):
        self.detected_objects_ = msg

    def render(self):
        width = shutil.get_terminal_size(fallback=(80, 24)).columns
        lines = []

        lines.append('=' * width)
        lines.append('Recycling Cell Monitor'.center(width))
        lines.append('=' * width)

        lines.extend(self.render_robot_state())
        lines.append('')
        lines.extend(self.render_perception())
        lines.append('')
        lines.extend(self.render_last_sort_result())

        print('\033[2J\033[H' + '\n'.join(lines), flush=True)

    def render_robot_state(self):
        lines = ['[Robot State]']
        state = self.robot_state_

        if state is None:
            lines.append('  (no data received yet)')
            return lines

        lines.append(f'  robot_id         : {state.robot_id}')
        lines.append(f'  state            : {state.state}')
        lines.append(f'  is_busy          : {state.is_busy}')
        lines.append(f'  is_error         : {state.is_error}')
        lines.append(f'  progress         : {state.progress:.2f}')
        lines.append(f'  current_task_id  : {state.current_task_id}')
        lines.append(f'  last_error_code  : {state.last_error_code}')
        return lines

    def render_perception(self):
        lines = ['[Perception]']
        objects_msg = self.detected_objects_

        if objects_msg is None:
            lines.append('  (no data received yet)')
            return lines

        lines.append(f'  detected object count: {len(objects_msg.objects)}')
        for obj in objects_msg.objects:
            lines.append(
                f'    - object_id={obj.object_id} '
                f'class_name={obj.class_name} '
                f'confidence={obj.confidence:.2f} '
                f'graspability_score={obj.graspability_score:.2f} '
                f'is_unknown={obj.is_unknown}')
        return lines

    def render_last_sort_result(self):
        lines = ['[Last Sort Result]']
        result = self.last_sort_result_

        if result is None:
            lines.append('  (no data received yet)')
            return lines

        lines.append(f'  task_id          : {result.task_id}')
        lines.append(f'  object_id        : {result.object_id}')
        lines.append(f'  class_name       : {result.class_name}')
        lines.append(f'  target_bin_id    : {result.target_bin_id}')
        lines.append(f'  pick_success     : {result.pick_success}')
        lines.append(f'  place_success    : {result.place_success}')
        lines.append(f'  overall_success  : {result.overall_success}')
        lines.append(f'  confidence       : {result.confidence:.2f}')
        lines.append(f'  cycle_time_sec   : {result.cycle_time_sec:.2f}')
        lines.append(f'  error_code       : {result.error_code}')
        lines.append(f'  message          : {result.message}')
        return lines


def main(args=None):
    rclpy.init(args=args)
    node = MonitorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
