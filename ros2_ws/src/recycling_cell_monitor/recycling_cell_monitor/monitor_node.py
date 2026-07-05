import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from recycling_cell_msgs.msg import RobotState, SortResult


class MonitorNode(Node):

    def __init__(self):
        super().__init__('monitor_node')

        self.declare_parameter('summary_period_sec', 10.0)
        self.declare_parameter('enable_detailed_result_log', True)

        self.start_time_ = time.time()
        self.total_processed_ = 0
        self.success_count_ = 0
        self.failure_count_ = 0
        self.cycle_time_sum_sec_ = 0.0
        self.class_counts_ = {}
        self.bin_counts_ = {}
        self.error_counts_ = {}
        self.last_robot_state_ = None

        self.sort_result_sub_ = self.create_subscription(
            SortResult,
            '/task_manager/sort_results',
            self.sort_result_callback,
            10,
        )

        self.robot_state_sub_ = self.create_subscription(
            RobotState,
            '/robot/state',
            self.robot_state_callback,
            10,
        )

        summary_period_sec = self.get_parameter('summary_period_sec').value
        self.summary_timer_ = self.create_timer(
            summary_period_sec, self.log_summary)

        self.get_logger().info('monitor_node started')

    # ---------- subscriptions ----------

    def robot_state_callback(self, msg):
        self.last_robot_state_ = msg

    def sort_result_callback(self, msg):
        self.total_processed_ += 1
        self.cycle_time_sum_sec_ += msg.cycle_time_sec

        self.class_counts_[msg.class_name] = (
            self.class_counts_.get(msg.class_name, 0) + 1)
        self.bin_counts_[msg.target_bin_id] = (
            self.bin_counts_.get(msg.target_bin_id, 0) + 1)

        if msg.overall_success:
            self.success_count_ += 1
        else:
            self.failure_count_ += 1
            if msg.error_code:
                self.error_counts_[msg.error_code] = (
                    self.error_counts_.get(msg.error_code, 0) + 1)

        if self.get_parameter('enable_detailed_result_log').value:
            self.get_logger().info(
                f'Sort result received: object_id={msg.object_id} '
                f'class={msg.class_name} success={msg.overall_success} '
                f'cycle_time={msg.cycle_time_sec:.1f}s')

    # ---------- periodic summary ----------

    def log_summary(self):
        if self.total_processed_ == 0:
            self.get_logger().info(
                '[CellMetrics] no sort results received yet')
            return

        success_rate = 100.0 * self.success_count_ / self.total_processed_
        avg_cycle_time_sec = (
            self.cycle_time_sum_sec_ / self.total_processed_)

        elapsed_min = (time.time() - self.start_time_) / 60.0
        throughput_per_min = (
            self.total_processed_ / elapsed_min if elapsed_min > 0 else 0.0)

        class_counts_str = ', '.join(
            f'{name}:{count}' for name, count in self.class_counts_.items())
        bin_counts_str = ', '.join(
            f'{bin_id}:{count}' for bin_id, count in self.bin_counts_.items())
        error_counts_str = ', '.join(
            f'{code}:{count}' for code, count in self.error_counts_.items())

        lines = [
            '[CellMetrics]',
            f'total_processed={self.total_processed_}',
            f'success={self.success_count_}',
            f'failure={self.failure_count_}',
            f'success_rate={success_rate:.1f}%',
            f'avg_cycle_time_sec={avg_cycle_time_sec:.1f}',
            f'throughput_per_min={throughput_per_min:.1f}',
            f'class_counts={{{class_counts_str}}}',
            f'bin_counts={{{bin_counts_str}}}',
            f'error_counts={{{error_counts_str}}}',
        ]

        if self.last_robot_state_ is not None:
            lines.append(
                f'robot_state={self.last_robot_state_.state} '
                f'is_busy={self.last_robot_state_.is_busy} '
                f'is_error={self.last_robot_state_.is_error}')

        self.get_logger().info('\n'.join(lines))


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
