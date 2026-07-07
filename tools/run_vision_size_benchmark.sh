#!/usr/bin/env bash
# Runs the vision_sorting_cell pipeline once per ONNX input size (640/416/320)
# against an image folder (test_images/ by default, or any dataset via
# IMAGE_FOLDER_PATH), so the [VisionPerf] logs for each run can be compared
# side by side. This does NOT parse the logs into a CSV -- that's done by
# tools/parse_vision_benchmark_logs.py as a follow-up step.
#
# Usage:
#   tools/run_vision_size_benchmark.sh
#   tools/run_vision_size_benchmark.sh 640 320       # subset of sizes
#
#   IMAGE_FOLDER_PATH=/path/to/test_images_real \
#   RECURSIVE_IMAGE_FOLDER=true \
#   RUN_DURATION_SEC=180 \
#   tools/run_vision_size_benchmark.sh
#
#   # Fast ONNX-only sweep over a full dataset -- no task_manager/MoveIt,
#   # so it isn't gated on pick/place cycle time:
#   IMAGE_FOLDER_PATH=/path/to/test_images_real \
#   RECURSIVE_IMAGE_FOLDER=true \
#   BENCHMARK_MODE=vision_only \
#   RUN_DURATION_SEC=30 \
#   tools/run_vision_size_benchmark.sh
#
#   # custom_autolabel_v0 model instead of the pretrained COCO model --
#   # MODEL_STEM must match tools/export_recycling_yolo_autolabel_v0_onnx.sh's
#   # OUTPUT naming (models/<MODEL_STEM>_<size>.onnx), and MODEL_CLASS_MODE
#   # must be recycling_custom so vision_perception_node decodes class_id
#   # 0-3 as plastic/paper/can/glass_bottle instead of COCO ids:
#   MODEL_STEM=yolo11n_recycling_autolabel_v0 \
#   MODEL_CLASS_MODE=recycling_custom \
#   IMAGE_FOLDER_PATH=/path/to/test_images_real \
#   RECURSIVE_IMAGE_FOLDER=true \
#   BENCHMARK_MODE=vision_only \
#   tools/run_vision_size_benchmark.sh 640
#
# Requires: models/<MODEL_STEM>_<size>.onnx to already exist for each size
# (see tools/export_yolo_onnx_sizes.py for the pretrained COCO models, or
# tools/export_recycling_yolo_autolabel_v0_onnx.sh for the custom model)
# and the ROS2 workspace to already be built (colcon build in ros2_ws).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS2_WS="$PROJECT_ROOT/ros2_ws"
MODELS_DIR="$PROJECT_ROOT/models"
DEFAULT_IMAGE_FOLDER="$PROJECT_ROOT/test_images"
IMAGE_FOLDER="${IMAGE_FOLDER_PATH:-$DEFAULT_IMAGE_FOLDER}"
RECURSIVE_IMAGE_FOLDER="${RECURSIVE_IMAGE_FOLDER:-false}"
BENCHMARK_MODE="${BENCHMARK_MODE:-end_to_end}"
MODEL_STEM="${MODEL_STEM:-yolo11n}"
MODEL_CLASS_MODE="${MODEL_CLASS_MODE:-coco}"

if [ "$MODEL_CLASS_MODE" != "coco" ] && [ "$MODEL_CLASS_MODE" != "recycling_custom" ]; then
  echo "ERROR: MODEL_CLASS_MODE must be 'coco' or 'recycling_custom', got '${MODEL_CLASS_MODE}'" >&2
  exit 1
fi

if [ "$BENCHMARK_MODE" != "end_to_end" ] && [ "$BENCHMARK_MODE" != "vision_only" ]; then
  echo "ERROR: BENCHMARK_MODE must be 'end_to_end' or 'vision_only', got '${BENCHMARK_MODE}'" >&2
  exit 1
fi

if [ "$#" -gt 0 ]; then
  SIZES=("$@")
else
  SIZES=(640 416 320)
fi

RUN_DURATION_SEC="${RUN_DURATION_SEC:-60}"

# Keep the default test_images/ smoke-test logs at their original path
# (logs/vision_benchmark/yolo11n_<size>.log) so existing tooling/invocations
# stay unaffected; any other dataset gets its own subdirectory named after
# the image folder, so runs against different datasets never overwrite
# each other's logs. vision_only additionally gets its own subfolder so it
# never overwrites an end_to_end run's log for the same dataset/size.
DATASET_NAME="$(basename "$IMAGE_FOLDER")"
if [ "$IMAGE_FOLDER" = "$DEFAULT_IMAGE_FOLDER" ]; then
  BASE_LOG_DIR="$PROJECT_ROOT/logs/vision_benchmark"
else
  BASE_LOG_DIR="$PROJECT_ROOT/logs/vision_benchmark/$DATASET_NAME"
fi

if [ "$BENCHMARK_MODE" = "vision_only" ]; then
  LOG_DIR="$BASE_LOG_DIR/vision_only"
else
  LOG_DIR="$BASE_LOG_DIR"
fi

mkdir -p "$LOG_DIR"

# ROS2's setup.bash scripts reference some variables without checking if
# they're set, which trips `set -u`; disable it just for sourcing them.
set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source "$ROS2_WS/install/setup.bash"
set -u

for size in "${SIZES[@]}"; do
  model_path="$MODELS_DIR/${MODEL_STEM}_${size}.onnx"
  log_path="$LOG_DIR/${MODEL_STEM}_${size}.log"

  if [ ! -f "$model_path" ]; then
    echo "SKIP size=${size}: model not found at ${model_path}"
    echo "      (run tools/export_yolo_onnx_sizes.py first)"
    continue
  fi

  echo "== Running benchmark for input_size=${size} (benchmark_mode=${BENCHMARK_MODE}) =="
  echo "   model:     ${model_path} (model_class_mode=${MODEL_CLASS_MODE})"
  echo "   log:       ${log_path}"
  echo "   dataset:   ${IMAGE_FOLDER} (recursive=${RECURSIVE_IMAGE_FOLDER})"

  # setsid makes the launch process the leader of its own new session and
  # process group (pgid == its own pid), so `kill -- -PID` below reliably
  # kills the whole tree it spawns without touching this script's own shell.
  if [ "$BENCHMARK_MODE" = "vision_only" ]; then
    # vision_only never touches task_manager/MoveIt, so run the vision
    # node standalone instead of the full launch file -- this skips
    # move_group/RViz/ros2_control startup entirely, which is what makes a
    # full 50-image sweep finish in seconds instead of minutes.
    setsid ros2 run recycling_cell_vision vision_perception_node --ros-args \
      -p image_source:=image_folder \
      -p image_folder_path:="$IMAGE_FOLDER" \
      -p recursive_image_folder:="$RECURSIVE_IMAGE_FOLDER" \
      -p enable_onnx_inference:=true \
      -p onnx_model_path:="$model_path" \
      -p model_class_mode:="$MODEL_CLASS_MODE" \
      -p onnx_input_size:="$size" \
      -p confidence_threshold:=0.5 \
      -p benchmark_mode:=vision_only \
      -p publish_detections_in_vision_only:=false \
      -p enable_vision_perf_logging:=true \
      -p vision_perf_log_period:=1 \
      -p loop_folder:=false \
      > "$log_path" 2>&1 &
  else
    setsid ros2 launch recycling_cell_bringup vision_sorting_cell.launch.py \
      image_source:=image_folder \
      image_folder_path:="$IMAGE_FOLDER" \
      recursive_image_folder:="$RECURSIVE_IMAGE_FOLDER" \
      enable_onnx_inference:=true \
      onnx_model_path:="$model_path" \
      model_class_mode:="$MODEL_CLASS_MODE" \
      onnx_input_size:="$size" \
      confidence_threshold:=0.5 \
      benchmark_mode:=end_to_end \
      folder_advance_mode:=result \
      folder_result_policy:=single_best_object \
      result_wait_timeout_sec:=20.0 \
      enable_vision_perf_logging:=true \
      vision_perf_log_period:=1 \
      loop_folder:=false \
      route_unknown_to_reject_bin:=true \
      > "$log_path" 2>&1 &
  fi
  launch_pid=$!

  sleep "$RUN_DURATION_SEC"

  kill -- -"$launch_pid" 2>/dev/null || true
  wait "$launch_pid" 2>/dev/null || true
  sleep 2

  echo "== Summary for input_size=${size} =="
  grep -E \
    '\[VisionModel\]|\[VisionPerf\]|ONNX detection|overall_success|Image folder scan completed' \
    "$log_path" || echo "  (no matching lines found -- check ${log_path})"
  echo
done

echo "All requested sizes attempted. Logs saved under: $LOG_DIR"
