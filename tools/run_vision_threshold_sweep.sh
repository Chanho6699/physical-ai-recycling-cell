#!/usr/bin/env bash
# Confidence threshold sweep: runs vision_perception_node standalone
# (benchmark_mode=vision_only, no task_manager/MoveIt) once per
# (confidence_threshold, input_size) combination against a full image
# folder, so tools/analyze_threshold_sweep_results.py can compare
# [PerceptionPolicy] decisions (ACCEPT_SORT/ROUTE_TO_REJECT/
# SKIP_NO_DETECTION/RETRY_VIEW/MANUAL_REVIEW) across thresholds afterward.
#
# Usage:
#   bash tools/run_vision_threshold_sweep.sh
#   THRESHOLDS="0.3 0.5" SIZES="640 320" bash tools/run_vision_threshold_sweep.sh
#
# Requires: models/yolo11n_<size>.onnx to already exist for each size in
# SIZES (see tools/export_yolo_onnx_sizes.py) and the ROS2 workspace to
# already be built (colcon build in ros2_ws).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS2_WS="$PROJECT_ROOT/ros2_ws"
MODELS_DIR="$PROJECT_ROOT/models"

IMAGE_FOLDER_PATH="${IMAGE_FOLDER_PATH:-$PROJECT_ROOT/test_images_real}"
RECURSIVE_IMAGE_FOLDER="${RECURSIVE_IMAGE_FOLDER:-true}"
BENCHMARK_MODE="${BENCHMARK_MODE:-vision_only}"
RUN_DURATION_SEC="${RUN_DURATION_SEC:-30}"
THRESHOLDS="${THRESHOLDS:-0.3 0.5 0.7}"
SIZES="${SIZES:-640 416 320}"

if [ "$BENCHMARK_MODE" != "vision_only" ]; then
  echo "ERROR: this sweep only supports BENCHMARK_MODE=vision_only (it runs" >&2
  echo "       vision_perception_node standalone, with no task_manager/MoveIt" >&2
  echo "       to gate on); got BENCHMARK_MODE=${BENCHMARK_MODE}" >&2
  exit 1
fi

SWEEP_LOG_ROOT="$PROJECT_ROOT/logs/vision_benchmark/test_images_real/vision_only_threshold_sweep"
mkdir -p "$SWEEP_LOG_ROOT"

# ROS2's setup.bash scripts reference some variables without checking if
# they're set, which trips `set -u`; disable it just for sourcing them.
set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source "$ROS2_WS/install/setup.bash"
set -u

for threshold in $THRESHOLDS; do
  conf_dir="$SWEEP_LOG_ROOT/conf_${threshold}"
  mkdir -p "$conf_dir"

  for size in $SIZES; do
    model_path="$MODELS_DIR/yolo11n_${size}.onnx"
    log_path="$conf_dir/yolo11n_${size}.log"

    if [ ! -f "$model_path" ]; then
      echo "SKIP threshold=${threshold} size=${size}: model not found at ${model_path}"
      echo "      (run tools/export_yolo_onnx_sizes.py first)"
      continue
    fi

    echo "== Running threshold=${threshold} input_size=${size} =="
    echo "   model:   ${model_path}"
    echo "   log:     ${log_path}"
    echo "   dataset: ${IMAGE_FOLDER_PATH} (recursive=${RECURSIVE_IMAGE_FOLDER})"

    # policy_confidence_threshold is kept equal to confidence_threshold so
    # the [PerceptionPolicy] decision boundary matches the same threshold
    # being swept for detection itself, rather than sweeping one and
    # holding the other fixed.
    #
    # setsid makes the node the leader of its own new session/process
    # group (pgid == its own pid), so `kill -- -PID` below reliably kills
    # it without touching this script's own shell.
    setsid ros2 run recycling_cell_vision vision_perception_node --ros-args \
      -p image_source:=image_folder \
      -p image_folder_path:="$IMAGE_FOLDER_PATH" \
      -p recursive_image_folder:="$RECURSIVE_IMAGE_FOLDER" \
      -p enable_onnx_inference:=true \
      -p onnx_model_path:="$model_path" \
      -p onnx_input_size:="$size" \
      -p confidence_threshold:="$threshold" \
      -p benchmark_mode:=vision_only \
      -p publish_detections_in_vision_only:=false \
      -p enable_perception_policy:=true \
      -p policy_confidence_threshold:="$threshold" \
      -p enable_vision_perf_logging:=true \
      -p vision_perf_log_period:=1 \
      -p loop_folder:=false \
      > "$log_path" 2>&1 &
    launch_pid=$!

    sleep "$RUN_DURATION_SEC"

    kill -- -"$launch_pid" 2>/dev/null || true
    wait "$launch_pid" 2>/dev/null || true
    sleep 1

    echo "== Summary for threshold=${threshold} input_size=${size} =="
    grep -E \
      '\[VisionPerf\]|ONNX detection|PerceptionPolicy|Image folder scan completed' \
      "$log_path" || echo "  (no matching lines found -- check ${log_path})"
    echo
  done
done

echo "Sweep complete. Logs saved under: $SWEEP_LOG_ROOT"
