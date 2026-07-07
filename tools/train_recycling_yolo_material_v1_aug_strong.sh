#!/usr/bin/env bash
# Fine-tune YOLO11n on recycling_yolo_material_v1 with STRONG data
# augmentation, as a domain-gap-mitigation experiment: the baseline model
# (tools/train_recycling_yolo_material_v1.sh) had strong validation mAP
# (plastic 0.915 / metal 0.908 / glass 0.882) but detected almost nothing
# on test_images_real (0 detections at confidence_threshold=0.5, only 2
# at threshold=0.05) -- see docs/material_v1_augmentation_experiment_plan.md
# for the full rationale. This script does NOT rebuild the dataset --
# same images/labels as baseline, only the augmentation hyperparameters
# change (heavier than tools/train_recycling_yolo_material_v1_aug_medium.sh).
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
#   bash tools/train_recycling_yolo_material_v1_aug_strong.sh
#   EPOCHS=100 bash tools/train_recycling_yolo_material_v1_aug_strong.sh
#
# Requires the `yolo` CLI (ultralytics) to be installed and on PATH.
#
# WSL/VS Code disconnects kill this process along with the terminal --
# a 50-epoch run took ~50min on GPU / much longer on CPU and got cut off
# mid-run this way once already (stopped at epoch 38/50, only discovered
# after the fact by comparing `results.csv` row count to EPOCHS). Two
# ways to avoid losing the run:
#
#   1. Run inside tmux so the training survives a dropped connection --
#      see docs/material_v1_augmentation_experiment_plan.md's "Running
#      long training under tmux" section for the full walkthrough:
#        tmux new -s strong_train
#        cd ~/Projects/physical-ai-recycling-cell
#        bash tools/train_recycling_yolo_material_v1_aug_strong.sh
#      then detach with Ctrl+B, D and `tmux attach -t strong_train` to
#      check back in.
#
#   2. If it does get cut off anyway: check whether `weights/last.pt`
#      exists and compare `results.csv`'s row count to EPOCHS (if rows <
#      EPOCHS, training didn't finish -- best.pt existing is NOT proof of
#      completion, it's just the best checkpoint seen so far). If
#      incomplete, resume from the last checkpoint instead of restarting
#      from scratch -- pass RESUME=1 to this script:
#        RESUME=1 bash tools/train_recycling_yolo_material_v1_aug_strong.sh
#      This continues from `last.pt`'s saved epoch/optimizer state up to
#      the original EPOCHS total (Ultralytics reads the run's own
#      args.yaml, so augmentation params below are ignored in this mode).
#      If `last.pt` looks corrupt or resume errors out, fall back to a
#      fresh run (drop RESUME, optionally remove the old run dir first).

set -e

DATA="${DATA:-datasets/recycling_yolo_material_v1/recycling_material_v1.yaml}"
MODEL="${MODEL:-models/yolo11n.pt}"
PROJECT="${PROJECT:-runs/recycling_yolo}"
RUN_NAME="${RUN_NAME:-yolo11n_material_v1_aug_strong_640}"
IMGSZ="${IMGSZ:-640}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-8}"

MOSAIC="${MOSAIC:-1.0}"
MIXUP="${MIXUP:-0.2}"
DEGREES="${DEGREES:-15.0}"
TRANSLATE="${TRANSLATE:-0.15}"
SCALE="${SCALE:-0.5}"
SHEAR="${SHEAR:-3.0}"
PERSPECTIVE="${PERSPECTIVE:-0.001}"
FLIPLR="${FLIPLR:-0.5}"
HSV_H="${HSV_H:-0.02}"
HSV_S="${HSV_S:-0.7}"
HSV_V="${HSV_V:-0.5}"

if [ ! -f "$DATA" ]; then
  echo "ERROR: dataset config not found at ${DATA}" >&2
  echo "       Run tools/build_recycling_material_v1_dataset.py first." >&2
  exit 1
fi

if [ ! -f "$MODEL" ]; then
  echo "ERROR: base model weights not found at ${MODEL}" >&2
  exit 1
fi

echo "== Training recycling YOLO (material_v1, aug=strong) =="
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

if [ "${RESUME:-0}" = "1" ]; then
  LAST_PT="${PROJECT}/${RUN_NAME}/weights/last.pt"
  if [ ! -f "$LAST_PT" ]; then
    LAST_PT="runs/detect/${PROJECT}/${RUN_NAME}/weights/last.pt"
  fi
  if [ ! -f "$LAST_PT" ]; then
    echo "ERROR: RESUME=1 but no last.pt found under ${PROJECT}/${RUN_NAME}/weights/" >&2
    echo "       or runs/detect/${PROJECT}/${RUN_NAME}/weights/ -- nothing to resume." >&2
    exit 1
  fi
  echo "== Resuming from ${LAST_PT} (augmentation/epoch settings above are ignored;"
  echo "   Ultralytics reuses the original run's args.yaml) =="
  yolo detect train resume="$LAST_PT"
else
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
fi

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
