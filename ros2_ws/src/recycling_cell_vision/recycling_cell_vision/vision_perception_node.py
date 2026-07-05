import random

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from geometry_msgs.msg import Pose
from recycling_cell_msgs.msg import DetectedObject, DetectedObjectArray

VALID_IMAGE_SOURCES = ('synthetic', 'image_file', 'camera')

MOCK_CLASSES = ['plastic_bottle', 'can', 'paper_cup', 'glass_bottle']

CLASS_SIZES = {
    'plastic_bottle': (0.06, 0.06, 0.18),
    'can': (0.06, 0.06, 0.11),
    'paper_cup': (0.05, 0.05, 0.10),
    'glass_bottle': (0.06, 0.06, 0.20),
}

# Kept within the Panda arm's reachable workspace (same convention as
# fake_perception_node / recycling_cell_moveit_manipulation).
MOCK_POSE_X_RANGE = (0.40, 0.55)
MOCK_POSE_Y_RANGE = (-0.10, 0.10)
MOCK_POSE_Z = 0.05


class VisionPerceptionNode(Node):
    """v1 skeleton: image input pipeline + DetectedObjectArray publishing.

    No YOLO/ONNX inference yet. _acquire_frame() is the seam where a real
    model will run in a later version -- it already reads a frame from the
    selected source and returns it, but the frame is currently discarded in
    favor of mock detections.
    """

    def __init__(self):
        super().__init__('vision_perception_node')

        self.declare_parameter('image_source', 'synthetic')
        self.declare_parameter('image_path', '')
        self.declare_parameter('publish_period_sec', 5.0)
        self.declare_parameter('enable_mock_detection', True)
        self.declare_parameter('camera_frame_id', 'camera_link')
        self.declare_parameter('base_frame_id', 'base_link')

        image_source = self.get_parameter('image_source').value
        if image_source not in VALID_IMAGE_SOURCES:
            self.get_logger().warn(
                f"Unknown image_source '{image_source}', falling back to "
                f"'synthetic'. Valid options: {VALID_IMAGE_SOURCES}")
            image_source = 'synthetic'
        self.image_source_ = image_source

        self.publisher_ = self.create_publisher(
            DetectedObjectArray, '/perception/detected_objects', 10)

        self.active_objects_ = []
        self.next_object_index_ = 1
        self.next_class_cycle_index_ = 0

        self.camera_ = None
        if self.image_source_ == 'camera':
            self._open_camera()

        publish_period_sec = self.get_parameter('publish_period_sec').value
        self.timer_ = self.create_timer(publish_period_sec, self.tick)

        self.get_logger().info(
            f"vision_perception_node started (image_source="
            f"'{self.image_source_}')")

    # ---------- frame acquisition (no YOLO inference yet) ----------

    def _open_camera(self):
        if not CV2_AVAILABLE:
            self.get_logger().error(
                'image_source=camera requires OpenCV (cv2), which is not '
                'installed. Camera frames will not be read.')
            return

        self.camera_ = cv2.VideoCapture(0)
        if not self.camera_.isOpened():
            self.get_logger().error(
                'Failed to open camera device 0 (cv2.VideoCapture(0))')
            self.camera_ = None
        else:
            self.get_logger().info('Camera device 0 opened successfully')

    def _acquire_frame(self):
        if self.image_source_ == 'synthetic':
            return None

        if self.image_source_ == 'image_file':
            if not CV2_AVAILABLE:
                self.get_logger().error(
                    'image_source=image_file requires OpenCV (cv2), which '
                    'is not installed. Cannot load image_path.')
                return None

            image_path = self.get_parameter('image_path').value
            frame = cv2.imread(image_path)
            if frame is None:
                self.get_logger().error(
                    f"Failed to load image from image_path='{image_path}'")
            else:
                self.get_logger().info(
                    f"Loaded image from image_path='{image_path}' "
                    f'shape={frame.shape}')
            return frame

        if self.image_source_ == 'camera':
            if self.camera_ is None:
                self.get_logger().error(
                    'Camera is not open; skipping frame read')
                return None

            ret, frame = self.camera_.read()
            if not ret:
                self.get_logger().error('Failed to read frame from camera')
                return None

            self.get_logger().info(f'Read camera frame shape={frame.shape}')
            return frame

        return None

    # ---------- main loop ----------

    def tick(self):
        if not rclpy.ok():
            return

        # `frame` is where a future YOLO/ONNX model would run inference.
        # For v1 it is only read (to exercise/validate the input pipeline
        # and its logging) and then discarded in favor of mock detections.
        self._acquire_frame()

        if self.get_parameter('enable_mock_detection').value:
            self._spawn_mock_object()
            self._publish_active_objects()

    def _spawn_mock_object(self):
        object_id = f'vision_obj_{self.next_object_index_}'
        self.next_object_index_ += 1

        class_name = MOCK_CLASSES[
            self.next_class_cycle_index_ % len(MOCK_CLASSES)]
        self.next_class_cycle_index_ += 1

        position = (
            random.uniform(*MOCK_POSE_X_RANGE),
            random.uniform(*MOCK_POSE_Y_RANGE),
            MOCK_POSE_Z,
        )

        obj = self._make_object(
            object_id=object_id,
            class_name=class_name,
            confidence=random.uniform(0.90, 0.97),
            position=position,
            size=CLASS_SIZES[class_name],
            graspability_score=random.uniform(0.80, 0.92),
            is_unknown=False,
        )
        self.active_objects_.append(obj)

        self.get_logger().info(
            f'mock detection spawned: object_id={object_id} '
            f'class_name={class_name}')

    def _publish_active_objects(self):
        if not self.active_objects_:
            return

        msg = DetectedObjectArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.get_parameter('base_frame_id').value
        msg.objects = list(self.active_objects_)

        self.publisher_.publish(msg)
        self.get_logger().info(
            f'published {len(msg.objects)} detected objects')

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

    def destroy_node(self):
        if self.camera_ is not None:
            self.camera_.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisionPerceptionNode()
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
