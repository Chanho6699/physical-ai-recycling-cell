import os
import random
import time

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ort = None
    ONNXRUNTIME_AVAILABLE = False

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from geometry_msgs.msg import Pose
from recycling_cell_msgs.msg import (
    DetectedObject, DetectedObjectArray, SortResult)

VALID_IMAGE_SOURCES = ('synthetic', 'image_file', 'camera', 'image_folder')
VALID_FOLDER_ADVANCE_MODES = ('time', 'result')
VALID_FOLDER_RESULT_POLICIES = ('single_best_object', 'all_objects')

# Mirrors task_manager_node's own unknown-object score penalty, so ranking
# here is consistent with how task_manager would rank the same objects.
UNKNOWN_SCORE_PENALTY = 0.8

MOCK_CLASSES = ['plastic_bottle', 'can', 'paper_cup', 'glass_bottle']

CLASS_SIZES = {
    'plastic_bottle': (0.06, 0.06, 0.18),
    'can': (0.06, 0.06, 0.11),
    'paper_cup': (0.05, 0.05, 0.10),
    'glass_bottle': (0.06, 0.06, 0.20),
    'unknown': (0.08, 0.08, 0.08),
}

# Kept within the Panda arm's reachable workspace (same convention as
# fake_perception_node / recycling_cell_moveit_manipulation).
MOCK_POSE_X_RANGE = (0.40, 0.55)
MOCK_POSE_Y_RANGE = (-0.10, 0.10)
MOCK_POSE_Z = 0.05

# Minimal COCO class-id -> project class-name mapping for the stock
# Ultralytics COCO-pretrained export. can/glass_bottle have no confident
# COCO equivalent, so they stay unmapped until a custom-trained model
# exists; unmapped detections are reported as 'unknown' rather than
# guessed, since task_manager already knows how to route unknown objects
# (reject_bin) instead of silently mislabeling something.
COCO_CLASS_ID_TO_PROJECT_CLASS = {
    39: 'plastic_bottle',  # COCO 'bottle'
    41: 'paper_cup',       # COCO 'cup'
}


class VisionPerceptionNode(Node):
    """Image input pipeline + optional ONNX YOLO inference.

    v1 behavior (enable_onnx_inference=false, the default) is untouched:
    frames are read for pipeline validation only and mock detections are
    published. v2 adds an ONNX Runtime YOLO path that runs on image_file/
    camera frames and converts detections to DetectedObjectArray; pose_base
    is still a simple bbox-to-workspace mapping, not real depth estimation.
    """

    def __init__(self):
        super().__init__('vision_perception_node')

        self.declare_parameter('image_source', 'synthetic')
        self.declare_parameter('image_path', '')
        self.declare_parameter('publish_period_sec', 5.0)
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('enable_mock_detection', True)
        self.declare_parameter('camera_frame_id', 'camera_link')
        self.declare_parameter('base_frame_id', 'base_link')

        self.declare_parameter('enable_onnx_inference', False)
        self.declare_parameter('onnx_model_path', '')
        self.declare_parameter('onnx_input_size', 640)
        self.declare_parameter('confidence_threshold', 0.50)
        self.declare_parameter('nms_threshold', 0.45)
        self.declare_parameter('use_mock_pose_for_onnx', True)

        self.declare_parameter('image_folder_path', '')
        self.declare_parameter('image_extensions', '.jpg,.jpeg,.png')
        self.declare_parameter('loop_folder', False)
        self.declare_parameter('publish_once_per_image', True)
        self.declare_parameter('folder_advance_mode', 'time')
        self.declare_parameter('result_wait_timeout_sec', 20.0)
        self.declare_parameter('folder_result_policy', 'single_best_object')

        image_source = self.get_parameter('image_source').value
        if image_source not in VALID_IMAGE_SOURCES:
            self.get_logger().warn(
                f"Unknown image_source '{image_source}', falling back to "
                f"'synthetic'. Valid options: {VALID_IMAGE_SOURCES}")
            image_source = 'synthetic'
        self.image_source_ = image_source

        folder_advance_mode = self.get_parameter('folder_advance_mode').value
        if folder_advance_mode not in VALID_FOLDER_ADVANCE_MODES:
            self.get_logger().warn(
                f"Unknown folder_advance_mode '{folder_advance_mode}', "
                f"falling back to 'time'. Valid options: "
                f"{VALID_FOLDER_ADVANCE_MODES}")

        folder_result_policy = self.get_parameter(
            'folder_result_policy').value
        if folder_result_policy not in VALID_FOLDER_RESULT_POLICIES:
            self.get_logger().warn(
                f"Unknown folder_result_policy '{folder_result_policy}', "
                f"falling back to 'single_best_object'. Valid options: "
                f"{VALID_FOLDER_RESULT_POLICIES}")

        self.publisher_ = self.create_publisher(
            DetectedObjectArray, '/perception/detected_objects', 10)

        self.sort_result_sub_ = self.create_subscription(
            SortResult, '/task_manager/sort_results',
            self.sort_result_callback, 10)

        self.active_objects_ = []
        self.next_object_index_ = 1
        self.next_class_cycle_index_ = 0

        self.camera_ = None
        if self.image_source_ == 'camera':
            self._open_camera()

        self.image_folder_files_ = []
        self.image_folder_index_ = 0
        self.image_folder_finished_logged_ = False

        # folder_advance_mode="result" state
        self.waiting_for_result_ = False
        self.pending_object_ids_ = set()
        self.current_image_path_ = ''
        self.result_wait_start_time_ = 0.0
        self.last_publish_had_objects_ = False

        if self.image_source_ == 'image_folder':
            self._load_image_folder()

        self.onnx_session_ = None
        self.onnx_input_name_ = None
        self.onnx_output_names_ = None
        if self.get_parameter('enable_onnx_inference').value:
            self.init_onnx_model()

        publish_period_sec = self.get_parameter('publish_period_sec').value
        self.timer_ = self.create_timer(publish_period_sec, self.tick)

        self.get_logger().info(
            f"vision_perception_node started (image_source="
            f"'{self.image_source_}', enable_onnx_inference="
            f"{self.get_parameter('enable_onnx_inference').value})")

    # ---------- frame acquisition ----------

    def _open_camera(self):
        if not CV2_AVAILABLE:
            self.get_logger().error(
                'image_source=camera requires OpenCV (cv2), which is not '
                'installed. Camera frames will not be read.')
            return

        camera_index = self.get_parameter('camera_index').value
        self.camera_ = cv2.VideoCapture(camera_index)
        if not self.camera_.isOpened():
            self.get_logger().error(
                f'Failed to open camera device {camera_index} '
                f'(cv2.VideoCapture({camera_index}))')
            self.camera_ = None
        else:
            self.get_logger().info(
                f'Camera device {camera_index} opened successfully')

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

    # ---------- image_folder scanning ----------

    def _load_image_folder(self):
        folder_path = self.get_parameter('image_folder_path').value
        extensions = tuple(
            ext.strip().lower() for ext in
            self.get_parameter('image_extensions').value.split(',')
            if ext.strip())

        if not folder_path or not os.path.isdir(folder_path):
            self.get_logger().error(
                f"image_source=image_folder but image_folder_path="
                f"'{folder_path}' is not a valid directory")
            return

        files = sorted(
            os.path.join(folder_path, name)
            for name in os.listdir(folder_path)
            if name.lower().endswith(extensions)
        )

        if not files:
            self.get_logger().error(
                f'No images with extensions {extensions} found in '
                f"image_folder_path='{folder_path}'")
            return

        self.image_folder_files_ = files
        self.get_logger().info(
            f'image_folder scan found {len(files)} image(s) in '
            f"'{folder_path}'")

    def _detect_objects_in_image(self, image_path):
        frame = None
        if not CV2_AVAILABLE:
            self.get_logger().error(
                'image_source=image_folder requires OpenCV (cv2), which '
                'is not installed. Cannot load images.')
        else:
            frame = cv2.imread(image_path)
            if frame is None:
                self.get_logger().error(
                    f"Failed to load image from image_path='{image_path}'")

        objects = None
        if self.get_parameter('enable_onnx_inference').value \
                and frame is not None:
            objects = self._run_onnx_pipeline(frame)

        if objects is None:
            if self.get_parameter('enable_mock_detection').value:
                objects = [self._build_mock_object()]
            else:
                objects = []

        return objects

    def _advance_folder_index(self):
        self.image_folder_index_ += 1
        total = len(self.image_folder_files_)

        if self.image_folder_index_ >= total:
            if self.get_parameter('loop_folder').value:
                self.image_folder_index_ = 0
            elif not self.image_folder_finished_logged_:
                self.get_logger().info('Image folder scan completed')
                self.image_folder_finished_logged_ = True

    def _tick_image_folder_time_mode(self):
        if not self.image_folder_files_:
            # Already logged in _load_image_folder(); nothing to do.
            return

        total = len(self.image_folder_files_)

        if self.image_folder_index_ >= total:
            if self.get_parameter('loop_folder').value:
                self.image_folder_index_ = 0
            else:
                if not self.image_folder_finished_logged_:
                    self.get_logger().info('Image folder scan completed')
                    self.image_folder_finished_logged_ = True
                return

        image_path = self.image_folder_files_[self.image_folder_index_]
        image_name = os.path.basename(image_path)

        self.get_logger().info(
            f'Processing image [{self.image_folder_index_ + 1}/{total}]: '
            f'{image_name}')

        objects = self._detect_objects_in_image(image_path)
        self._publish_objects_for_current_image(objects, image_name)

        # publish_once_per_image=false intentionally keeps re-processing
        # the same image every tick instead of advancing -- useful for
        # dwelling on one image without switching to image_file mode.
        if self.get_parameter('publish_once_per_image').value:
            self.image_folder_index_ += 1

    def _tick_image_folder_result_mode(self):
        if not self.image_folder_files_:
            return

        if self.waiting_for_result_:
            self._check_result_wait_timeout()
            return

        total = len(self.image_folder_files_)
        if self.image_folder_index_ >= total:
            # Finished (non-looping) and nothing pending -- stay idle.
            return

        image_path = self.image_folder_files_[self.image_folder_index_]
        image_name = os.path.basename(image_path)
        self.current_image_path_ = image_path

        self.get_logger().info(
            f'Processing image [{self.image_folder_index_ + 1}/{total}]: '
            f'{image_name}')

        objects = self._detect_objects_in_image(image_path)
        objects = self._apply_folder_result_policy(objects, image_name)
        self._publish_objects_for_current_image(objects, image_name)

        if objects:
            self.last_publish_had_objects_ = True
            self.pending_object_ids_ = {obj.object_id for obj in objects}
            self.waiting_for_result_ = True
            self.result_wait_start_time_ = time.time()
        else:
            # Nothing was published for task_manager to act on, so there
            # is no SortResult to wait for -- move on right away.
            self.last_publish_had_objects_ = False
            self._advance_folder_index()

    def _apply_folder_result_policy(self, objects, image_name):
        if len(objects) <= 1:
            return objects

        folder_result_policy = self.get_parameter(
            'folder_result_policy').value

        if folder_result_policy == 'all_objects':
            self.get_logger().warn(
                f'folder_result_policy=all_objects with {len(objects)} '
                f'detections for {image_name}; task_manager only acts on '
                f'one candidate per message, so extra pending object_ids '
                f'may run into result_wait_timeout_sec')
            return objects

        # Default: single_best_object.
        best = self._select_single_best_object(objects)
        self.get_logger().info(
            f'folder_result_policy=single_best_object: selected '
            f'object_id={best.object_id} class_name={best.class_name} '
            f'out of {len(objects)} detections for {image_name}')
        return [best]

    @staticmethod
    def _select_single_best_object(objects):
        known_candidates = []
        unknown_candidates = []

        for obj in objects:
            is_unknown = obj.is_unknown or obj.class_name == 'unknown'
            score = obj.confidence * 0.6 + obj.graspability_score * 0.4
            if is_unknown:
                unknown_candidates.append(
                    (score * UNKNOWN_SCORE_PENALTY, obj))
            else:
                known_candidates.append((score, obj))

        # Same known-over-unknown priority tier task_manager_node uses, so
        # the object we forward is the one it would have picked anyway.
        candidates = known_candidates if known_candidates \
            else unknown_candidates
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        return candidates[0][1]

    def _check_result_wait_timeout(self):
        result_wait_timeout_sec = self.get_parameter(
            'result_wait_timeout_sec').value
        elapsed_sec = time.time() - self.result_wait_start_time_
        if elapsed_sec < result_wait_timeout_sec:
            return

        image_name = os.path.basename(self.current_image_path_)
        self.get_logger().warn(
            f'Result wait timeout for image {image_name}, '
            f'pending_object_ids={self.pending_object_ids_}; '
            f'advancing to next image')
        self.pending_object_ids_ = set()
        self.waiting_for_result_ = False
        self._advance_folder_index()

    def sort_result_callback(self, msg):
        if not self.waiting_for_result_:
            return
        if msg.object_id not in self.pending_object_ids_:
            return

        self.pending_object_ids_.discard(msg.object_id)
        self.get_logger().info(
            f'Received SortResult for pending object_id={msg.object_id} '
            f'success={msg.overall_success}')

        if self.pending_object_ids_:
            return

        image_name = os.path.basename(self.current_image_path_)
        self.get_logger().info(
            f'All pending objects for image {image_name} completed, '
            f'advancing to next image')
        self.waiting_for_result_ = False
        self._advance_folder_index()

    def _publish_objects_for_current_image(self, objects, image_name):
        msg = DetectedObjectArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.get_parameter('base_frame_id').value
        msg.objects = list(objects)

        self.publisher_.publish(msg)
        self.get_logger().info(
            f'Published {len(msg.objects)} detections from {image_name}')

    # ---------- ONNX model setup ----------

    def init_onnx_model(self):
        if not ONNXRUNTIME_AVAILABLE:
            self.get_logger().error(
                'enable_onnx_inference=true requires onnxruntime, which is '
                'not installed (pip install onnxruntime). Falling back to '
                'mock detections.')
            return

        if not NUMPY_AVAILABLE:
            self.get_logger().error(
                'enable_onnx_inference=true requires numpy, which is not '
                'installed. Falling back to mock detections.')
            return

        onnx_model_path = self.get_parameter('onnx_model_path').value
        if not onnx_model_path:
            self.get_logger().error(
                'enable_onnx_inference=true but onnx_model_path is empty. '
                'Falling back to mock detections.')
            return

        if not os.path.isfile(onnx_model_path):
            self.get_logger().error(
                f"onnx_model_path='{onnx_model_path}' does not exist. "
                'Falling back to mock detections.')
            return

        try:
            self.onnx_session_ = ort.InferenceSession(
                onnx_model_path, providers=['CPUExecutionProvider'])
        except Exception as exc:
            self.get_logger().error(
                f"Failed to load ONNX model from '{onnx_model_path}': {exc}")
            self.onnx_session_ = None
            return

        input_info = self.onnx_session_.get_inputs()[0]
        self.onnx_input_name_ = input_info.name
        self.onnx_output_names_ = [
            output.name for output in self.onnx_session_.get_outputs()]

        self.get_logger().info(
            f"ONNX model loaded from '{onnx_model_path}': "
            f'input_name={self.onnx_input_name_} '
            f'input_shape={input_info.shape} '
            f'output_names={self.onnx_output_names_}')

    # ---------- ONNX preprocess / inference / postprocess ----------

    def preprocess_yolo(self, frame):
        input_size = self.get_parameter('onnx_input_size').value

        resized = cv2.resize(frame, (input_size, input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(chw, axis=0)
        return np.ascontiguousarray(batched, dtype=np.float32)

    def run_onnx_inference(self, frame):
        input_tensor = self.preprocess_yolo(frame)
        outputs = self.onnx_session_.run(
            self.onnx_output_names_, {self.onnx_input_name_: input_tensor})

        self.get_logger().info(
            f'ONNX raw output shapes: {[out.shape for out in outputs]}')
        return outputs

    def postprocess_yolo(self, outputs, original_shape):
        """Decode Ultralytics-style post-NMS ONNX export output.

        Expects outputs[0] shaped (1, N, 6) with rows of
        [x1, y1, x2, y2, confidence, class_id] in onnx_input_size pixel
        space. Raw pre-NMS prediction tensors (e.g. (1, 84, 8400)) are not
        decoded here -- that needs its own NMS step and is left for a
        follow-up version. Returns None to signal "fall back to mock",
        or a (possibly empty) list of detections on success.
        """
        output = outputs[0]

        if NUMPY_AVAILABLE and output.ndim == 3 and output.shape[-1] == 6:
            confidence_threshold = self.get_parameter(
                'confidence_threshold').value
            input_size = self.get_parameter('onnx_input_size').value
            orig_h, orig_w = original_shape[0], original_shape[1]
            scale_x = orig_w / input_size
            scale_y = orig_h / input_size

            detections = []
            for x1, y1, x2, y2, confidence, class_id in output[0]:
                if confidence < confidence_threshold:
                    continue
                detections.append({
                    'bbox': (
                        float(x1) * scale_x, float(y1) * scale_y,
                        float(x2) * scale_x, float(y2) * scale_y),
                    'confidence': float(confidence),
                    'class_id': int(class_id),
                })
            return detections

        self.get_logger().warn(
            f'Unsupported ONNX output shape {output.shape}; expected a '
            f'post-NMS Ultralytics export shaped (1, N, 6). Raw pre-NMS '
            f'prediction tensors are not decoded in this version -- '
            f'falling back to mock detections.')
        return None

    # ---------- class mapping / pose mapping ----------

    @staticmethod
    def map_class_id_to_class_name(class_id):
        return COCO_CLASS_ID_TO_PROJECT_CLASS.get(class_id, 'unknown')

    @staticmethod
    def bbox_to_mock_pose(bbox, original_shape):
        # No depth estimation yet: map the bbox center's position within
        # the image onto the Panda arm's reachable workspace rectangle.
        x1, y1, x2, y2 = bbox
        orig_h, orig_w = original_shape[0], original_shape[1]

        norm_x = ((x1 + x2) / 2.0) / orig_w if orig_w > 0 else 0.5
        norm_y = ((y1 + y2) / 2.0) / orig_h if orig_h > 0 else 0.5

        x = MOCK_POSE_X_RANGE[0] + norm_x * (
            MOCK_POSE_X_RANGE[1] - MOCK_POSE_X_RANGE[0])
        y = MOCK_POSE_Y_RANGE[0] + norm_y * (
            MOCK_POSE_Y_RANGE[1] - MOCK_POSE_Y_RANGE[0])
        return (x, y, MOCK_POSE_Z)

    # ---------- main loop ----------

    def tick(self):
        if not rclpy.ok():
            return

        if self.image_source_ == 'image_folder':
            if self.get_parameter('folder_advance_mode').value == 'result':
                self._tick_image_folder_result_mode()
            else:
                self._tick_image_folder_time_mode()
            return

        frame = self._acquire_frame()

        if self.get_parameter('enable_onnx_inference').value \
                and self.image_source_ in ('image_file', 'camera') \
                and frame is not None:
            new_objects = self._run_onnx_pipeline(frame)
            if new_objects is not None:
                self.active_objects_.extend(new_objects)
                self._publish_active_objects()
                return

        if self.get_parameter('enable_mock_detection').value:
            self._spawn_mock_object()
            self._publish_active_objects()

    def _run_onnx_pipeline(self, frame):
        if self.onnx_session_ is None:
            self.get_logger().error(
                'enable_onnx_inference=true but no ONNX session is '
                'loaded; falling back to mock detection for this tick')
            return None

        outputs = self.run_onnx_inference(frame)
        detections = self.postprocess_yolo(outputs, frame.shape)

        if detections is None:
            return None

        if not detections:
            self.get_logger().info('No ONNX detections above threshold')
            return []

        use_mock_pose = self.get_parameter('use_mock_pose_for_onnx').value
        objects = []
        for detection in detections:
            class_name = self.map_class_id_to_class_name(
                detection['class_id'])

            if use_mock_pose:
                position = self.bbox_to_mock_pose(
                    detection['bbox'], frame.shape)
            else:
                position = (
                    random.uniform(*MOCK_POSE_X_RANGE),
                    random.uniform(*MOCK_POSE_Y_RANGE),
                    MOCK_POSE_Z,
                )

            is_unknown = class_name == 'unknown'
            graspability_score = (
                random.uniform(0.15, 0.25) if is_unknown
                else min(0.95, detection['confidence'] + 0.05))

            object_id = f'vision_obj_{self.next_object_index_}'
            self.next_object_index_ += 1

            obj = self._make_object(
                object_id=object_id,
                class_name=class_name,
                confidence=detection['confidence'],
                position=position,
                size=CLASS_SIZES[class_name],
                graspability_score=graspability_score,
                is_unknown=is_unknown,
            )
            objects.append(obj)

            self.get_logger().info(
                f'ONNX detection: object_id={object_id} '
                f'class_name={class_name} '
                f"confidence={detection['confidence']:.2f} "
                f"bbox={tuple(round(v, 1) for v in detection['bbox'])}")

        return objects

    # ---------- mock detection (v1, unchanged) ----------

    def _build_mock_object(self):
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

        return self._make_object(
            object_id=object_id,
            class_name=class_name,
            confidence=random.uniform(0.90, 0.97),
            position=position,
            size=CLASS_SIZES[class_name],
            graspability_score=random.uniform(0.80, 0.92),
            is_unknown=False,
        )

    def _spawn_mock_object(self):
        obj = self._build_mock_object()
        self.active_objects_.append(obj)

        self.get_logger().info(
            f'mock detection spawned: object_id={obj.object_id} '
            f'class_name={obj.class_name}')

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
