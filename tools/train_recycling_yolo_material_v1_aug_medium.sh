#!/usr/bin/env bash
# Fine-tune YOLO11n on recycling_yolo_material_v1 with MEDIUM data
# augmentation, as a domain-gap-mitigation experiment: the baseline model
# (tools/train_recycling_yolo_material_v1.sh) had strong validation mAP
# (plastic 0.915 / metal 0.908 / glass 0.882) but detected almost nothing
# on test_images_real (0 detections at confidence_threshold=0.5, only 2
# at threshold=0.05) -- see docs/material_v1_augmentation_experiment_plan.md
# for the full rationale. This script does NOT rebuild the dataset --
# same images/labels as baseline, only the augmentation hyperparameters
# change.
#
# This is a separate script from tools/train_recycling_yolo_material_v1.sh
# (the baseline) -- that script is NOT modified by this experiment.
#
# Uses RUN_NAME (not NAME) for the run name: NAME is a common env var
# that some shells/desktop sessions already export (e.g. from
# /etc/os-release), which would silently override a `NAME=...` default
# in this script via `${NAME:-default}` and produce an unexpected run
# directory -- exactly what happened with an earlier training script.
#
# Usage:
#   bash tools/train_recycling_yolo_material_v1_aug_medium.sh
#   EPOCHS=100 bash tools/train_recycling_yolo_material_v1_aug_medium.sh
#
# Requires the `yolo` CLI (ultralytics) to be installed and on PATH.

set -e

DATA="${DATA:-datasets/recycling_yolo_material_v1/recycling_material_v1.yaml}"
MODEL="${MODEL:-models/yolo11n.pt}"
PROJECT="${PROJECT:-runs/recycling_yolo}"
RUN_NAME="${RUN_NAME:-yolo11n_material_v1_aug_medium_640}"
IMGSZ="${IMGSZ:-640}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-8}"

MOSAIC="${MOSAIC:-0.7}"
MIXUP="${MIXUP:-0.1}"
DEGREES="${DEGREES:-10.0}"
TRANSLATE="${TRANSLATE:-0.1}"
SCALE="${SCALE:-0.4}"
SHEAR="${SHEAR:-2.0}"
PERSPECTIVE="${PERSPECTIVE:-0.0005}"
FLIPLR="${FLIPLR:-0.5}"
HSV_H="${HSV_H:-0.015}"
HSV_S="${HSV_S:-0.5}"
HSV_V="${HSV_V:-0.4}"

if [ ! -f "$DATA" ]; then
  echo "ERROR: dataset config not found at ${DATA}" >&2
  echo "       Run tools/build_recycling_material_v1_dataset.py first." >&2
  exit 1
fi

if [ ! -f "$MODEL" ]; then
  echo "ERROR: base model weights not found at ${MODEL}" >&2
  exit 1
fi

echo "== Training recycling YOLO (material_v1, aug=medium) =="
echo "   data:     ${DATA}"
echo "   model:    ${MODEL}"
echo "   project:  ${PROJECT}"
echo "   run_name: ${RUN_NAME}"
echo "   imgsz:    ${IMGSZ}"
echo "   epochs:   ${EPOCHS}"
echo "   batch:    ${BATCH}"
echo "   -- augmentation --"
echo "   mosaic:      ${MOSAIC}"
echo "   mixup:       ${MIXUP}"
echo "   degrees:     ${DEGREES}"
echo "   translate:   ${TRANSLATE}"
echo "   scale:       ${SCALE}"
echo "   shear:       ${SHEAR}"
echo "   perspective: ${PERSPECTIVE}"
echo "   fliplr:      ${FLIPLR}"
echo "   hsv_h:       ${HSV_H}"
echo "   hsv_s:       ${HSV_S}"
echo "   hsv_v:       ${HSV_V}"
echo

# project=/name= are passed explicitly and quoted -- but observed in
# practice, Ultralytics can still resolve a relative project= path under
# its own runs/detect/ root (e.g. writing to
# runs/detect/runs/recycling_yolo/<name>/ instead of
# runs/recycling_yolo/<name>/), so the fallback search below checks both
# locations rather than assuming project=/name= pins the exact path.
yolo detect train \
  model="$MODEL" \
  data="$DATA" \
  imgsz="$IMGSZ" \
  epochs="$EPOCHS" \
  batch="$BATCH" \
  project="$PROJECT" \
  name="$RUN_NAME" \
  mosaic="$MOSAIC" \
  mixup="$MIXUP" \
  degrees="$DEGREES" \
  translate="$TRANSLATE" \
  scale="$SCALE" \
  shear="$SHEAR" \
  perspective="$PERSPECTIVE" \
  fliplr="$FLIPLR" \
  hsv_h="$HSV_H" \
  hsv_s="$HSV_S" \
  hsv_v="$HSV_V"

BEST_PT="${PROJECT}/${RUN_NAME}/weights/best.pt"
echo
if [ -f "$BEST_PT" ]; then
  echo "Training complete. Best weights: ${BEST_PT}"
else
  echo "Training finished, but ${BEST_PT} was not found where expected."
  echo "Searching for best.pt under ${PROJECT} and runs/detect/ (Ultralytics"
  echo "sometimes nests relative project= paths there instead)..."
  find "$PROJECT" runs/detect -name best.pt 2>/dev/null || true
fi
