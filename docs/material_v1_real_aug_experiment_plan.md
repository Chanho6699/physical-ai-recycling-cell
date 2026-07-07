# material_v1_real_aug Experiment Plan

## Where this picks up

`docs/material_v1_augmentation_experiment_plan.md` tested whether heavier
training-time augmentation alone could close the domain gap between
`recycling_yolo_material_v1`'s training photos and `test_images_real/`.
Result (`results/material_v1_augmentation_comparison.md`): augmentation
helped and `strong` beat `medium`, but absolute real-world detection
stayed low --

| threshold | baseline | medium | strong |
|---|---|---|---|
| 0.5 | 0/50 | 2/50 | 3/50 |
| 0.05 | 2/50 | 4/50 | 4/50 |

no class collapse in any of the three, and `strong` even eliminated the
baseline's 1-3 unsafe `unknown`->`ACCEPT_SORT` cases. But single-digit
detections out of 50 real images means augmentation reduced, not closed,
the gap. This experiment tests a different lever: adding a small number
of real-camera-style photos directly into the training set, instead of
further augmenting the same studio-ish photos.

## What's new: real_selected

`datasets/recycling_material_real_selected/{plastic,metal,glass,paper}/`
-- photos taken specifically to resemble `test_images_real/`'s domain:
wood floor / desk, shadows, oblique angles, partial framing, dim
lighting, background clutter. Small (91 images total: plastic 35 / metal
24 / paper 22 / glass 10), auto-labeled fresh with GroundingDINO using
its own prompt map (tuned for cluttered/off-angle real photos -- e.g. a
generic "metal object" prompt alongside "aluminum can", since a real
photo isn't guaranteed to be a clean can shot the way a candidates_v1
photo is).

## Dataset design: one variable at a time

`recycling_yolo_material_v1_real_aug` = material_v1's own
candidates_v1 (100/class) + v0_remapped (paper excluded) composition,
UNCHANGED, plus real_selected added on top. Reusing the exact same
`--limit-per-class 100`/seed/prompt-map/`--exclude-too-large` settings
material_v1 was built with means the only variable that changed versus
`aug_strong` is "real photos in the training set, yes/no" -- both use
identical strong-augmentation training hyperparameters (see below), so a
real-image-benchmark delta between `aug_strong` and `real_aug` isolates
the effect of adding real_selected, not a confound from also changing
augmentation strength.

Build:
```bash
source .venv-autolabel/bin/activate
python3 tools/build_recycling_material_v1_real_aug_dataset.py \
  --limit-per-class 100 \
  --exclude-too-large
```

`--skip-real-selected` reproduces material_v1's own candidates_v1+
v0_remapped subset alone (useful as a sanity check that this builder's
independent GroundingDINO re-run matches material_v1's original labels
closely).

## Training: same strong augmentation as aug_strong, GPU (device=0)

```bash
source .venv-autolabel/bin/activate
bash tools/train_recycling_yolo_material_v1_real_aug.sh
```

Augmentation params are identical to `tools/train_recycling_yolo_
material_v1_aug_strong.sh`: `mosaic=1.0 mixup=0.2 degrees=15.0
translate=0.15 scale=0.5 shear=3.0 perspective=0.001 fliplr=0.5
hsv_h=0.02 hsv_s=0.7 hsv_v=0.5`, `epochs=50`, `imgsz=640`.

**GPU note:** the system-wide `yolo` on PATH (outside any venv) resolves
to a torch+cu130 install that is newer than this machine's NVIDIA driver
supports and silently falls back to CPU (confirmed during the aug_strong
resume -- a 12-epoch resume that should take minutes on GPU took ~16min
on CPU). `.venv-autolabel` has a matching torch+cu126 build
(`torch.cuda.is_available()` -> `True`, GPU: RTX 3050) and now also has
`ultralytics` installed directly into it (via `pip install ultralytics
--no-deps`, to avoid pip resolving a plain `pip install ultralytics` back
onto a torch+cu130 wheel and undoing the whole point of this venv) --
`--no-deps` means ultralytics' own soft dependencies aren't pulled in
automatically, so also install these once (confirmed missing/needed by
hitting a mid-training crash the first time this dataset was trained --
`ModuleNotFoundError: No module named 'polars'` at the end of epoch 1,
inside Ultralytics' own `save_model()`):
```bash
source .venv-autolabel/bin/activate
pip install ultralytics --no-deps   # first time only
pip install polars psutil nvidia-ml-py ultralytics-thop   # first time only
```
its own `yolo`/`python3` then resolve to the GPU-correct build. Always
activate it and verify before a training run:
```bash
source .venv-autolabel/bin/activate
which python3   # .../physical-ai-recycling-cell/.venv-autolabel/bin/python3
which yolo      # .../physical-ai-recycling-cell/.venv-autolabel/bin/yolo
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```
`tools/train_recycling_yolo_material_v1_real_aug.sh` also re-checks
`torch.cuda.is_available()` itself before launching and refuses to
silently train on CPU with `device=0` (the default) requested -- pass
`DEVICE=cpu` explicitly to override.

## Running long training under tmux

Same reasoning as `docs/material_v1_augmentation_experiment_plan.md`'s
own tmux section -- a WSL/VS Code disconnect kills the training process
along with the terminal. Don't trust `best.pt`'s existence alone as
"training finished" (compare `results.csv`'s row count to `EPOCHS`
instead -- this bit the `aug_strong` run once already, stopping at
epoch 38/50 with `best.pt`/`last.pt` both present and looking complete
at a glance).

```bash
sudo apt install -y tmux   # first time only

tmux new -s real_aug_train
cd ~/Projects/physical-ai-recycling-cell
source .venv-autolabel/bin/activate
bash tools/train_recycling_yolo_material_v1_real_aug.sh
```

Detach: `Ctrl+B`, then `D`. Reattach: `tmux attach -t real_aug_train`.
List sessions: `tmux list-sessions`.

If cut off mid-run, resume instead of restarting from scratch:
```bash
RESUME=1 bash tools/train_recycling_yolo_material_v1_real_aug.sh
```

## Export to ONNX

```bash
bash tools/export_recycling_yolo_material_v1_real_aug_onnx.sh
```
Writes `models/yolo11n_recycling_material_v1_real_aug_640.onnx`
(gitignored, like every other `models/*.onnx` in this project -- re-run
the export script locally to regenerate it, don't expect it to be in the
repo).

## ROS2 vision node compatibility

No code changes needed. `recycling_yolo_material_v1_real_aug` uses the
exact same 4-class taxonomy and class_id order as `material_v1`/
`aug_medium`/`aug_strong` (0=plastic, 1=metal, 2=glass, 3=paper), which
`vision_perception_node.py`'s `model_class_mode=recycling_material_v1`
already decodes via `RECYCLING_MATERIAL_V1_CLASS_ID_TO_PROJECT_CLASS`:

```python
RECYCLING_MATERIAL_V1_CLASS_ID_TO_PROJECT_CLASS = {
    0: 'plastic',
    1: 'metal',
    2: 'glass',
    3: 'paper',
}
```

and `task_manager_node.py` already maps all four project classes to
their bins (`plastic`->`plastic_bin`, `metal`->`metal_bin`,
`glass`->`glass_bin`, `paper`->`paper_bin`), same as the other
material_v1 variants -- swapping in `real_aug`'s ONNX file is a drop-in
model change, nothing else.

Launch example (end-to-end, real image folder):
```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 launch recycling_cell_bringup vision_sorting_cell.launch.py \
  image_source:=image_folder \
  image_folder_path:=/home/rlack/Projects/physical-ai-recycling-cell/test_images_real \
  recursive_image_folder:=true \
  enable_onnx_inference:=true \
  onnx_model_path:=/home/rlack/Projects/physical-ai-recycling-cell/models/yolo11n_recycling_material_v1_real_aug_640.onnx \
  model_class_mode:=recycling_material_v1 \
  onnx_input_size:=640 \
  confidence_threshold:=0.5 \
  benchmark_mode:=end_to_end \
  folder_advance_mode:=result \
  folder_result_policy:=single_best_object \
  route_unknown_to_reject_bin:=true
```

vision_only smoke test (skips task_manager/MoveIt, matches how the
benchmark logs below were produced):
```bash
ros2 run recycling_cell_vision vision_perception_node --ros-args \
  -p image_source:=image_folder \
  -p image_folder_path:=/home/rlack/Projects/physical-ai-recycling-cell/test_images_real \
  -p recursive_image_folder:=true \
  -p enable_onnx_inference:=true \
  -p onnx_model_path:=/home/rlack/Projects/physical-ai-recycling-cell/models/yolo11n_recycling_material_v1_real_aug_640.onnx \
  -p model_class_mode:=recycling_material_v1 \
  -p onnx_input_size:=640 \
  -p confidence_threshold:=0.05 \
  -p benchmark_mode:=vision_only \
  -p publish_detections_in_vision_only:=false \
  -p enable_vision_perf_logging:=true \
  -p vision_perf_log_period:=1 \
  -p loop_folder:=false
```

Note: `tools/run_vision_size_benchmark.sh`'s own `MODEL_CLASS_MODE`
validation currently only accepts `coco`/`recycling_custom`, even though
`vision_perception_node.py` has supported `recycling_material_v1` since
the material_v1 baseline -- the benchmark logs referenced in this
project's material_v1 comparisons (baseline/medium/strong/real_aug) were
all produced with the direct `ros2 run` invocation above (also setting
`confidence_threshold:=0.05`, which the wrapper script hardcodes to
`0.5` and can't override), not through the wrapper script.

## Benchmark comparison

Four models compared against `test_images_real/` at
confidence_threshold in {0.5, 0.3, 0.1, 0.05}:

| model | ONNX |
|---|---|
| baseline | `models/yolo11n_recycling_material_v1_640.onnx` |
| aug_medium | `models/yolo11n_recycling_material_v1_aug_medium_640.onnx` |
| aug_strong | `models/yolo11n_recycling_material_v1_aug_strong_640.onnx` |
| real_aug | `models/yolo11n_recycling_material_v1_real_aug_640.onnx` |

See `results/material_v1_real_aug_comparison.md` for the full
per-threshold tables and interpretation.

## Limitations going in

- real_selected is small (91 images across 4 classes, glass at only 10)
  -- even a clear real-benchmark improvement from this experiment is a
  proof-of-concept signal ("more real photos help"), not proof that 91
  images is *enough* real data for production use.
- real_selected's GroundingDINO labels are unreviewed pseudo-labels, same
  caveat as candidates_v1/v0 -- if this experiment's real-image detection
  doesn't improve, a bad-bbox check on `previews/real_selected_*` is a
  cheaper next step than immediately concluding "real photos don't help"
  (see `results/material_v1_real_aug_comparison.md`'s "Next steps"
  section once benchmarked).
- Single training run per model (no repeated-seed variance estimate), same
  caveat as the augmentation experiment.
