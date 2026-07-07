#!/usr/bin/env bash
# Export the fine-tuned recycling_yolo_material_v1 checkpoint (see
# tools/train_recycling_yolo_material_v1.sh) to ONNX, matching the same
# nms=True post-NMS export convention used for the other yolo11n exports
# in this project, so vision_perception_node.py's postprocess_yolo() can
# decode it without any changes.
#
# Usage:
#   bash tools/export_recycling_yolo_material_v1_onnx.sh
#   TRAINED_MODEL=runs/recycling_yolo/some_other_run/weights/best.pt \
#     OUTPUT=models/custom_v2.onnx \
#     bash tools/export_recycling_yolo_material_v1_onnx.sh
#
# Requires the `yolo` CLI (ultralytics) to be installed and on PATH.

set -e

TRAINED_MODEL="${TRAINED_MODEL:-runs/recycling_yolo/yolo11n_material_v1_640/weights/best.pt}"
OUTPUT="${OUTPUT:-models/yolo11n_recycling_material_v1_640.onnx}"
IMGSZ="${IMGSZ:-640}"

if [ ! -f "$TRAINED_MODEL" ]; then
  echo "ERROR: trained model not found at ${TRAINED_MODEL}" >&2
  echo "       Run tools/train_recycling_yolo_material_v1.sh first, or" >&2
  echo "       set TRAINED_MODEL= to point at the right weights." >&2
  echo "       Ultralytics can sometimes write to a nested path (e.g." >&2
  echo "       runs/detect/runs/recycling_yolo/.../best.pt) instead of" >&2
  echo "       the plain runs/recycling_yolo/<name>/weights/best.pt this" >&2
  echo "       script expects by default -- here are all best.pt files" >&2
  echo "       found under runs/ and runs/detect/:" >&2
  find runs runs/detect -name best.pt 2>/dev/null >&2 || true
  exit 1
fi

echo "== Exporting recycling YOLO (material_v1) to ONNX =="
echo "   trained_model: ${TRAINED_MODEL}"
echo "   output:        ${OUTPUT}"
echo "   imgsz:         ${IMGSZ}"
echo

yolo export model="$TRAINED_MODEL" format=onnx imgsz="$IMGSZ" nms=True

EXPORTED_PATH="${TRAINED_MODEL%.pt}.onnx"

mkdir -p "$(dirname "$OUTPUT")"

if [ "$EXPORTED_PATH" != "$OUTPUT" ]; then
  mv "$EXPORTED_PATH" "$OUTPUT"
fi

echo
echo "ONNX model saved to: ${OUTPUT}"
