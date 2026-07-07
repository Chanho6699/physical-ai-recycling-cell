#!/usr/bin/env bash
# Export the strong-augmentation recycling_yolo_material_v1 checkpoint
# (see tools/train_recycling_yolo_material_v1_aug_strong.sh) to ONNX,
# matching the same nms=True post-NMS export convention used for every
# other yolo11n export in this project, so vision_perception_node.py's
# postprocess_yolo() can decode it without any changes.
#
# This is a separate script from
# tools/export_recycling_yolo_material_v1_onnx.sh (the baseline export)
# -- that script is NOT modified by this experiment.
#
# Usage:
#   bash tools/export_recycling_yolo_material_v1_aug_strong_onnx.sh
#   TRAINED_MODEL=runs/recycling_yolo/some_other_run/weights/best.pt \
#     OUTPUT=models/custom_v2.onnx \
#     bash tools/export_recycling_yolo_material_v1_aug_strong_onnx.sh
#
# Requires the `yolo` CLI (ultralytics) to be installed and on PATH.

set -e

TRAINED_MODEL="${TRAINED_MODEL:-runs/recycling_yolo/yolo11n_material_v1_aug_strong_640/weights/best.pt}"
OUTPUT="${OUTPUT:-models/yolo11n_recycling_material_v1_aug_strong_640.onnx}"
IMGSZ="${IMGSZ:-640}"

if [ ! -f "$TRAINED_MODEL" ]; then
  echo "ERROR: trained model not found at ${TRAINED_MODEL}" >&2
  echo "       Run tools/train_recycling_yolo_material_v1_aug_strong.sh" >&2
  echo "       first, or set TRAINED_MODEL= to point at the right" >&2
  echo "       weights. Ultralytics can sometimes write to a nested path" >&2
  echo "       (e.g. runs/detect/runs/recycling_yolo/.../best.pt) instead" >&2
  echo "       of the plain runs/recycling_yolo/<name>/weights/best.pt" >&2
  echo "       this script expects by default -- here are all best.pt" >&2
  echo "       files found under runs/ and runs/detect/:" >&2
  find runs runs/detect -name best.pt 2>/dev/null >&2 || true
  exit 1
fi

echo "== Exporting recycling YOLO (material_v1, aug=strong) to ONNX =="
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
