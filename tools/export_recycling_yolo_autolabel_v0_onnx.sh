#!/usr/bin/env bash
# Export the fine-tuned recycling_yolo_autolabel_v0 checkpoint (see
# tools/train_recycling_yolo_autolabel_v0.sh) to ONNX, matching the same
# nms=True post-NMS export convention used for the pretrained-COCO
# yolo11n models (tools/export_yolo_onnx_sizes.py), so
# vision_perception_node.py's postprocess_yolo() can decode it without
# any changes.
#
# Usage:
#   bash tools/export_recycling_yolo_autolabel_v0_onnx.sh
#   TRAINED_MODEL=runs/recycling_yolo/some_other_run/weights/best.pt \
#     OUTPUT=models/custom_v1.onnx \
#     bash tools/export_recycling_yolo_autolabel_v0_onnx.sh
#
# Requires the `yolo` CLI (ultralytics) to be installed and on PATH.

set -e

TRAINED_MODEL="${TRAINED_MODEL:-runs/recycling_yolo/yolo11n_autolabel_v0_640/weights/best.pt}"
OUTPUT="${OUTPUT:-models/yolo11n_recycling_autolabel_v0_640.onnx}"
IMGSZ="${IMGSZ:-640}"

if [ ! -f "$TRAINED_MODEL" ]; then
  echo "ERROR: trained model not found at ${TRAINED_MODEL}" >&2
  echo "       Run tools/train_recycling_yolo_autolabel_v0.sh first." >&2
  exit 1
fi

echo "== Exporting recycling YOLO (autolabel_v0) to ONNX =="
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
