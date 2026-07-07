#!/usr/bin/env bash
# Fine-tune YOLO11n on the recycling_yolo_material_v1 dataset (see
# tools/build_recycling_material_v1_dataset.py and
# docs/recycling_material_v1_dataset_plan.md for how it was built and
# why the taxonomy moved to material-level plastic/metal/glass/paper).
#
# Uses RUN_NAME (not NAME) for the run name: NAME is a common env var
# that some shells/desktop sessions already export (e.g. from
# /etc/os-release), which would silently override a `NAME=...` default
# in this script via `${NAME:-default}` and produce an unexpected run
# directory -- exactly what happened with an earlier training script.
#
# Usage:
#   bash tools/train_recycling_yolo_material_v1.sh
#   EPOCHS=100 BATCH=16 bash tools/train_recycling_yolo_material_v1.sh
#
# Requires the `yolo` CLI (ultralytics) to be installed and on PATH.

set -e

DATA="${DATA:-datasets/recycling_yolo_material_v1/recycling_material_v1.yaml}"
MODEL="${MODEL:-models/yolo11n.pt}"
PROJECT="${PROJECT:-runs/recycling_yolo}"
RUN_NAME="${RUN_NAME:-yolo11n_material_v1_640}"
IMGSZ="${IMGSZ:-640}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-8}"

if [ ! -f "$DATA" ]; then
  echo "ERROR: dataset config not found at ${DATA}" >&2
  echo "       Run tools/build_recycling_material_v1_dataset.py first." >&2
  exit 1
fi

if [ ! -f "$MODEL" ]; then
  echo "ERROR: base model weights not found at ${MODEL}" >&2
  exit 1
fi

echo "== Training recycling YOLO (material_v1) =="
echo "   data:     ${DATA}"
echo "   model:    ${MODEL}"
echo "   project:  ${PROJECT}"
echo "   run_name: ${RUN_NAME}"
echo "   imgsz:    ${IMGSZ}"
echo "   epochs:   ${EPOCHS}"
echo "   batch:    ${BATCH}"
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
  name="$RUN_NAME"

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
