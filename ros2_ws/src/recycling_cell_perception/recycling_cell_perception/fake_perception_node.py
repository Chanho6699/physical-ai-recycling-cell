import random

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from geometry_msgs.msg import Pose
from recycling_cell_msgs.msg import DetectedObject, DetectedObjectArray

CONVEYOR_CLASSES = ['plastic_bottle', 'can', 'paper_cup', 'glass_bottle']

CLASS_SIZES = {
    'plastic_bottle': (0.06, 0.06, 0.18),
    'can': (0.06, 0.06, 0.11),
    'paper_cup': (0.05, 0.05, 0.10),
    'glass_bottle': (0.06, 0.06, 0.20),
    'unknown': (0.08, 0.08, 0.08),
}

# Kept within the Panda arm's reachable workspace (matches the poses
# recycling_cell_moveit_manipulation already plans/executes against).
CONVEYOR_POSE_X_RANGE = (0.35, 0.55)
CONVEYOR_POSE_Y_RANGE = (-0.10, 0.15)
CONVEYOR_POSE_Z = 0.05


class FakePerceptionNode(Node):

    def __init__(self):
        super().__init__('fake_perception_node')

        self.declare_parameter('enable_conveyor_mode', True)
        self.declare_parameter('spawn_period_sec', 30.0)
        self.declare_parameter('objects_per_batch', 2)
        self.declare_parameter('max_batches', 3)
        self.declare_parameter('unknown_every_n', 0)

        self.publisher_ = self.create_publisher(
            DetectedObjectArray, '/perception/detected_objects', 10)

        # conveyor mode state
        self.active_objects_ = []
        self.next_object_index_ = 1
        self.next_class_cycle_index_ = 0
        self.batch_count_ = 0
        self.conveyor_finished_logged_ = False

        self.timer_period_sec = 2.0
        self.timer_ = self.create_timer(
            self.timer_period_sec, self.publish_detections)

        if self.get_parameter('enable_conveyor_mode').value:
            spawn_period_sec = self.get_parameter('spawn_period_sec').value
            self.spawn_timer_ = self.create_timer(
                spawn_period_sec, self.spawn_next_batch)
            # Spawn the first batch immediately so the belt isn't empty for
            # the first spawn_period_sec after startup.
            self.spawn_next_batch()

        self.get_logger().info('fake_perception_node started')

    # ---------- conveyor mode ----------

    def spawn_next_batch(self):
        if not rclpy.ok():
            return

        max_batches = self.get_parameter('max_batches').value
        if self.batch_count_ >= max_batches:
            if not self.conveyor_finished_logged_:
                self.get_logger().info(
                    'Conveyor simulation finished. No more new objects.')
                self.conveyor_finished_logged_ = True
            return

        objects_per_batch = self.get_parameter('objects_per_batch').value
        unknown_every_n = self.get_parameter('unknown_every_n').value

        self.batch_count_ += 1
        new_objects = []
        summary_parts = []

        for _ in range(objects_per_batch):
            object_id = f'obj_{self.next_object_index_}'

            is_unknown = (
                unknown_every_n > 0
                and self.next_object_index_ % unknown_every_n == 0)

            if is_unknown:
                class_name = 'unknown'
                confidence = random.uniform(0.30, 0.45)
                graspability_score = random.uniform(0.15, 0.25)
            else:
                class_name = CONVEYOR_CLASSES[
                    self.next_class_cycle_index_ % len(CONVEYOR_CLASSES)]
                self.next_class_cycle_index_ += 1
                confidence = random.uniform(0.85, 0.95)
                graspability_score = random.uniform(0.75, 0.90)

            position = (
                random.uniform(*CONVEYOR_POSE_X_RANGE),
                random.uniform(*CONVEYOR_POSE_Y_RANGE),
                CONVEYOR_POSE_Z,
            )

            obj = self._make_object(
                object_id=object_id,
                class_name=class_name,
                confidence=confidence,
                position=position,
                size=CLASS_SIZES[class_name],
                graspability_score=graspability_score,
                is_unknown=is_unknown,
            )
            new_objects.append(obj)
            summary_parts.append(f'{object_id} {class_name}')
            self.next_object_index_ += 1

        self.active_objects_.extend(new_objects)

        self.get_logger().info(
            f'Conveyor batch {self.batch_count_} published: '
            + ', '.join(summary_parts))

    # ---------- publishing ----------

    def publish_detections(self):
        if not rclpy.ok():
            return

        if self.get_parameter('enable_conveyor_mode').value:
            self.publish_objects(self.active_objects_)
        else:
            self.publish_fixed_detections()

    def publish_objects(self, objects):
        if not objects:
            return

        msg = DetectedObjectArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.objects = list(objects)

        self.publisher_.publish(msg)
        self.get_logger().info(
            f'published {len(msg.objects)} detected objects')

    def publish_fixed_detections(self):
        fixed_objects = [
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
        self.publish_objects(fixed_objects)

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
