#!/usr/bin/env bash
# Runs the vision_sorting_cell pipeline once per ONNX input size (640/416/320)
# against test_images/, so the [VisionPerf] logs for each run can be
# compared side by side. This does NOT parse the logs into a CSV yet --
# that's a follow-up step. It only makes the 3 runs reproducible.
#
# Usage:
#   tools/run_vision_size_benchmark.sh
#   tools/run_vision_size_benchmark.sh 640 320       # subset of sizes
#
# Requires: models/yolo11n_<size>.onnx to already exist for each size
# (see tools/export_yolo_onnx_sizes.py) and the ROS2 workspace to already
# be built (colcon build in ros2_ws).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS2_WS="$PROJECT_ROOT/ros2_ws"
MODELS_DIR="$PROJECT_ROOT/models"
IMAGE_FOLDER="$PROJECT_ROOT/test_images"
LOG_DIR="$PROJECT_ROOT/logs/vision_benchmark"

if [ "$#" -gt 0 ]; then
  SIZES=("$@")
else
  SIZES=(640 416 320)
fi

RUN_DURATION_SEC="${RUN_DURATION_SEC:-60}"

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
  model_path="$MODELS_DIR/yolo11n_${size}.onnx"
  log_path="$LOG_DIR/yolo11n_${size}.log"

  if [ ! -f "$model_path" ]; then
    echo "SKIP size=${size}: model not found at ${model_path}"
    echo "      (run tools/export_yolo_onnx_sizes.py first)"
    continue
  fi

  echo "== Running benchmark for input_size=${size} =="
  echo "   model: ${model_path}"
  echo "   log:   ${log_path}"

  # setsid makes the launch process the leader of its own new session and
  # process group (pgid == its own pid), so `kill -- -PID` below reliably
  # kills the whole tree it spawns (move_group, RViz, ros2_control, every
  # ROS node) without touching this script's own shell.
  setsid ros2 launch recycling_cell_bringup vision_sorting_cell.launch.py \
    image_source:=image_folder \
    image_folder_path:="$IMAGE_FOLDER" \
    enable_onnx_inference:=true \
    onnx_model_path:="$model_path" \
    onnx_input_size:="$size" \
    confidence_threshold:=0.5 \
    folder_advance_mode:=result \
    folder_result_policy:=single_best_object \
    result_wait_timeout_sec:=20.0 \
    enable_vision_perf_logging:=true \
    vision_perf_log_period:=1 \
    loop_folder:=false \
    route_unknown_to_reject_bin:=true \
    > "$log_path" 2>&1 &
  launch_pid=$!

  sleep "$RUN_DURATION_SEC"

  kill -- -"$launch_pid" 2>/dev/null || true
  wait "$launch_pid" 2>/dev/null || true
  sleep 2

  echo "== Summary for input_size=${size} =="
  grep -E \
    '\[VisionPerf\]|ONNX detection|overall_success|Image folder scan completed' \
    "$log_path" || echo "  (no matching lines found -- check ${log_path})"
  echo
done

echo "All requested sizes attempted. Logs saved under: $LOG_DIR"
