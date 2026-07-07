#!/usr/bin/env bash
# Fine-tune YOLO11n on recycling_yolo_material_v1_real_aug (material_v1's
# candidates_v1/v0_remapped composition PLUS real-camera-style photos from
# datasets/recycling_material_real_selected/) using the same STRONG
# augmentation settings as tools/train_recycling_yolo_material_v1_aug_
# strong.sh -- this experiment isolates ONE variable (real photos added to
# the training set) against a matched augmentation strength, so results
# are comparable to aug_strong's own test_images_real numbers. See
# docs/material_v1_real_aug_experiment_plan.md for the full rationale.
#
# This is a separate script from tools/train_recycling_yolo_material_v1_
# aug_strong.sh -- that script and its dataset are NOT touched by this one.
#
# GPU: run this from the GPU-enabled .venv-autolabel (torch 2.12.1+cu126,
# NVIDIA driver-compatible), NOT the system-wide `yolo` on PATH -- the
# system-wide ultralytics install has torch+cu130, which is newer than
# this machine's NVIDIA driver supports and silently falls back to CPU
# (confirmed during the aug_strong resume: a 12-epoch resume that should
# take minutes on GPU took ~16min on CPU). Before running, always check:
#   source .venv-autolabel/bin/activate
#   which python3   # should be .../venv-autolabel/bin/python3
#   which yolo       # should be .../venv-autolabel/bin/yolo
#   python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# This script re-checks torch.cuda.is_available() itself before training
# (see below) and refuses to silently train on CPU with device=0 requested.
#
# Uses RUN_NAME (not NAME) for the run name -- NAME is a common env var
# some shells/desktop sessions already export, which would silently
# override a `NAME=...` default via `${NAME:-default}` (bit earlier
# scripts in this project once by accident).
#
# Long training + WSL/VS Code disconnects: run this inside tmux so the
# job survives a dropped connection (see docs/material_v1_augmentation_
# experiment_plan.md's "Running long training under tmux" section, same
# pattern applies here):
#   sudo apt install -y tmux   # first time only
#   tmux new -s real_aug_train
#   cd ~/Projects/physical-ai-recycling-cell
#   source .venv-autolabel/bin/activate
#   bash tools/train_recycling_yolo_material_v1_real_aug.sh
#   # detach: Ctrl+B, then D
#   # reattach later: tmux attach -t real_aug_train
# If a run gets cut off anyway, compare `results.csv`'s row count to
# EPOCHS before trusting best.pt as "done" (best.pt exists from epoch 1
# onward, complete or not), then pass RESUME=1 to continue from last.pt:
#   RESUME=1 bash tools/train_recycling_yolo_material_v1_real_aug.sh
#
# Usage:
#   source .venv-autolabel/bin/activate
#   bash tools/train_recycling_yolo_material_v1_real_aug.sh
#   EPOCHS=100 bash tools/train_recycling_yolo_material_v1_real_aug.sh
#
# Requires the `yolo` CLI (ultralytics) installed in .venv-autolabel --
# see the module docstring of tools/build_recycling_material_v1_real_aug_
# dataset.py for install steps.

set -e

DATA="${DATA:-datasets/recycling_yolo_material_v1_real_aug/recycling_material_v1_real_aug.yaml}"
MODEL="${MODEL:-models/yolo11n.pt}"
PROJECT="${PROJECT:-runs/recycling_yolo}"
RUN_NAME="${RUN_NAME:-yolo11n_material_v1_real_aug_640}"
IMGSZ="${IMGSZ:-640}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-8}"
DEVICE="${DEVICE:-0}"

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
  echo "       Run tools/build_recycling_material_v1_real_aug_dataset.py first." >&2
  exit 1
fi

if [ ! -f "$MODEL" ]; then
  echo "ERROR: base model weights not found at ${MODEL}" >&2
  exit 1
fi

if [ "$DEVICE" != "cpu" ]; then
  CUDA_OK="$(python3 -c "import torch; print('1' if torch.cuda.is_available() else '0')" 2>/dev/null || echo 0)"
  if [ "$CUDA_OK" != "1" ]; then
    echo "ERROR: DEVICE=${DEVICE} requested but torch.cuda.is_available() is False" >&2
    echo "       in the current python3 (\$(which python3) = $(which python3))." >&2
    echo "       Activate the GPU-enabled venv first:" >&2
    echo "         source .venv-autolabel/bin/activate" >&2
    echo "       or explicitly pass DEVICE=cpu to train on CPU anyway (slow)." >&2
    exit 1
  fi
fi

echo "== Training recycling YOLO (material_v1_real_aug, aug=strong) =="
echo "   data:     ${DATA}"
echo "   model:    ${MODEL}"
echo "   project:  ${PROJECT}"
echo "   run_name: ${RUN_NAME}"
echo "   imgsz:    ${IMGSZ}"
echo "   epochs:   ${EPOCHS}"
echo "   batch:    ${BATCH}"
echo "   device:   ${DEVICE}"
echo "   python3:  $(which python3)"
echo "   yolo:     $(which yolo)"
echo "   -- augmentation (matches aug_strong) --"
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
  echo "== Resuming from ${LAST_PT} (augmentation/epoch/device settings above are"
  echo "   ignored; Ultralytics reuses the original run's args.yaml) =="
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
    device="$DEVICE" \
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
