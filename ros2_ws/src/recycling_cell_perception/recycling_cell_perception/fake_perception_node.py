import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from geometry_msgs.msg import Pose
from recycling_cell_msgs.msg import DetectedObject, DetectedObjectArray


class FakePerceptionNode(Node):

    def __init__(self):
        super().__init__('fake_perception_node')

        self.publisher_ = self.create_publisher(
            DetectedObjectArray, '/perception/detected_objects', 10)

        self.timer_period_sec = 2.0
        self.timer_ = self.create_timer(
            self.timer_period_sec, self.publish_fake_detections)

        self.get_logger().info('fake_perception_node started')

    def publish_fake_detections(self):
        if not rclpy.ok():
            return

        msg = DetectedObjectArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.objects = [
            self._make_object(
                object_id='obj_1',
                class_name='plastic_bottle',
                confidence=0.92,
                position=(0.35, 0.10, 0.05),
                size=(0.06, 0.06, 0.18),
                graspability_score=0.85,
                is_unknown=False,
            ),
            self._make_object(
                object_id='obj_2',
                class_name='can',
                confidence=0.88,
                position=(0.30, -0.15, 0.05),
                size=(0.06, 0.06, 0.11),
                graspability_score=0.80,
                is_unknown=False,
            ),
            self._make_object(
                object_id='obj_3',
                class_name='unknown',
                confidence=0.40,
                position=(0.45, 0.02, 0.05),
                size=(0.08, 0.08, 0.08),
                graspability_score=0.20,
                is_unknown=True,
            ),
        ]

        self.publisher_.publish(msg)
        self.get_logger().info(
            f'published {len(msg.objects)} fake detected objects')

    def _make_object(self, object_id, class_name, confidence, position,
                      size, graspability_score, is_unknown):
        obj = DetectedObject()
        obj.object_id = object_id
        obj.class_name = class_name
        obj.confidence = confidence

        pose = Pose()
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]
        pose.orientation.w = 1.0

        obj.pose_base = pose
        obj.pose_camera = pose

        obj.width = size[0]
        obj.height = size[1]
        obj.depth = size[2]

        obj.graspability_score = graspability_score
        obj.is_unknown = is_unknown

        return obj


def main(args=None):
    rclpy.init(args=args)
    node = FakePerceptionNode()
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
