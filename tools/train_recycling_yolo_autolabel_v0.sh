#!/usr/bin/env bash
# Fine-tune YOLO11n on the recycling_yolo_autolabel_v0 pseudo-label
# dataset (see tools/autolabel_recycling_groundingdino.py and
# docs/groundingdino_autolabeling_plan.md for how it was generated and
# why its labels are a v0 draft, not ground truth).
#
# Usage:
#   bash tools/train_recycling_yolo_autolabel_v0.sh
#   EPOCHS=100 BATCH=16 bash tools/train_recycling_yolo_autolabel_v0.sh
#
# Requires the `yolo` CLI (ultralytics) to be installed and on PATH.

set -e

DATA="${DATA:-datasets/recycling_yolo_autolabel_v0/recycling.yaml}"
MODEL="${MODEL:-models/yolo11n.pt}"
IMGSZ="${IMGSZ:-640}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-8}"
PROJECT="${PROJECT:-runs/recycling_yolo}"
NAME="${NAME:-yolo11n_autolabel_v0_640}"

if [ ! -f "$DATA" ]; then
  echo "ERROR: dataset config not found at ${DATA}" >&2
  echo "       Run tools/autolabel_recycling_groundingdino.py first." >&2
  exit 1
fi

if [ ! -f "$MODEL" ]; then
  echo "ERROR: base model weights not found at ${MODEL}" >&2
  exit 1
fi

echo "== Training recycling YOLO (autolabel_v0) =="
echo "   data:    ${DATA}"
echo "   model:   ${MODEL}"
echo "   imgsz:   ${IMGSZ}"
echo "   epochs:  ${EPOCHS}"
echo "   batch:   ${BATCH}"
echo "   project: ${PROJECT}"
echo "   name:    ${NAME}"
echo

yolo detect train model="$MODEL" data="$DATA" imgsz="$IMGSZ" \
  epochs="$EPOCHS" batch="$BATCH" project="$PROJECT" name="$NAME"

echo
echo "Training complete. Weights should be under:"
echo "  ${PROJECT}/${NAME}/weights/best.pt"
