# material_v1 Augmentation Experiment Plan

## Baseline problem: strong validation, near-zero real detection

`recycling_yolo_material_v1` baseline (`tools/train_recycling_yolo_
material_v1.sh`, 50 epochs, default Ultralytics augmentation) validated
well on its own held-out split:

| class | mAP50 | Recall |
|---|---|---|
| plastic | 0.915 | 0.909 |
| metal | 0.908 | 0.786 |
| glass | 0.882 | 0.879 |
| paper | 0.502 | 0.244 |

But run against `test_images_real/` (a real-world photo set the model
never saw during training, and visually quite different from
`datasets/recycling_yolo_candidates_v1/`'s photos), it detected almost
nothing: **0 detections across all 50 images at confidence_threshold=
0.5**, and only **2 detections total even at threshold=0.05**. High
validation mAP alongside near-zero real-world recall is a classic
symptom of **overfitting to the training distribution's specific
backgrounds/lighting/framing** rather than learning the object/material
itself -- i.e. a domain gap between `candidates_v1` and `test_images_
real/`, not a labeling or class-mapping problem (the earlier
`custom_autolabel_v0` "paper collapse" was a different failure mode --
that model over-detected everything as one class; this one barely
detects anything at all).

## Why augmentation, not a new dataset

This experiment deliberately does **not** rebuild
`datasets/recycling_yolo_material_v1/` -- same images, same labels, same
`recycling_material_v1.yaml`. Only the training-time augmentation
strength changes. The hypothesis: heavier geometric/color augmentation
during training forces the model to rely less on `candidates_v1`'s
specific backgrounds/lighting/exact framing and more on the object's
actual shape/texture/color, which should transfer better to `test_
images_real/`'s different conditions. This is a cheap experiment (no new
labeling work) to run before deciding whether the real fix is more/
different data (see `docs/recycling_material_v1_dataset_plan.md`'s
"Next steps"-style options).

## medium vs. strong

Both keep `DATA`/`MODEL`/`PROJECT`/`IMGSZ`/`EPOCHS`/`BATCH` identical to
baseline (`datasets/recycling_yolo_material_v1/recycling_material_v1.
yaml`, `models/yolo11n.pt`, `runs/recycling_yolo`, 640, 50, 8) -- only
the augmentation parameters passed to `yolo detect train` differ:

| parameter | baseline (Ultralytics default) | medium | strong |
|---|---|---|---|
| mosaic | 1.0 | 0.7 | 1.0 |
| mixup | 0.0 | 0.1 | 0.2 |
| degrees | 0.0 | 10.0 | 15.0 |
| translate | 0.1 | 0.1 | 0.15 |
| scale | 0.5 | 0.4 | 0.5 |
| shear | 0.0 | 2.0 | 3.0 |
| perspective | 0.0 | 0.0005 | 0.001 |
| fliplr | 0.5 | 0.5 | 0.5 |
| hsv_h | 0.015 | 0.015 | 0.02 |
| hsv_s | 0.7 | 0.5 | 0.7 |
| hsv_v | 0.4 | 0.4 | 0.5 |

`strong` pushes every geometric/color parameter further than `medium`
(more rotation/shear/perspective/translate/scale jitter, more mixup, a
wider hue/saturation/value range) -- the idea being: if `medium` helps
but doesn't fully close the gap, `strong` tests whether pushing further
helps more, or instead starts hurting validation performance too much to
be worth it (heavy augmentation on an already-small ~700-image dataset
risks the model failing to learn clean class boundaries at all).

## Evaluation criteria (not validation mAP alone)

Validation mAP on `recycling_yolo_material_v1`'s own held-out split is
**not** the deciding metric here -- the baseline already scored well on
it and still failed on real images. Compare baseline vs. medium vs.
strong on:

1. **Validation mAP50/recall per class** (sanity check that augmentation
   didn't make the model worse at the thing it can still measure).
2. **`test_images_real/` detection count** at a fixed confidence_
   threshold (e.g. 0.5) and at a low threshold (e.g. 0.05) -- does
   augmentation actually produce more real detections at all, not just
   different ones?
3. **Class collapse check** -- of whatever detections do appear, are
   they spread across plastic/metal/glass/paper in a way that tracks the
   real image's ground truth, or does the model still collapse onto one
   dominant class (the same failure shape as `custom_autolabel_v0`'s
   paper collapse, just potentially with a different dominant class)?
   Reuse `tools/analyze_custom_yolo_real_benchmark.py`'s ground_truth
   vs. predicted_class cross-tab against each model's `test_images_real/`
   log to check this directly.

## Example commands

Medium:
```bash
bash tools/train_recycling_yolo_material_v1_aug_medium.sh
bash tools/export_recycling_yolo_material_v1_aug_medium_onnx.sh
```

Strong:
```bash
bash tools/train_recycling_yolo_material_v1_aug_strong.sh
bash tools/export_recycling_yolo_material_v1_aug_strong_onnx.sh
```

Evaluate against real images (repeat per model, swapping `onnx_model_
path`):
```bash
MODEL_STEM=yolo11n_recycling_material_v1_aug_medium \
MODEL_CLASS_MODE=recycling_material_v1 \
IMAGE_FOLDER_PATH=/home/rlack/Projects/physical-ai-recycling-cell/test_images_real \
RECURSIVE_IMAGE_FOLDER=true \
BENCHMARK_MODE=vision_only \
bash tools/run_vision_size_benchmark.sh 640
```

## Outputs

| script | writes |
|---|---|
| `tools/train_recycling_yolo_material_v1_aug_medium.sh` | `runs/recycling_yolo/yolo11n_material_v1_aug_medium_640/` (or `runs/detect/runs/recycling_yolo/...` -- see below) |
| `tools/train_recycling_yolo_material_v1_aug_strong.sh` | `runs/recycling_yolo/yolo11n_material_v1_aug_strong_640/` (or nested, same caveat) |
| `tools/export_recycling_yolo_material_v1_aug_medium_onnx.sh` | `models/yolo11n_recycling_material_v1_aug_medium_640.onnx` |
| `tools/export_recycling_yolo_material_v1_aug_strong_onnx.sh` | `models/yolo11n_recycling_material_v1_aug_strong_640.onnx` |

As observed when training the baseline, Ultralytics can resolve a
relative `project=` path under its own `runs/detect/` root instead of
the plain `${PROJECT}/${RUN_NAME}/` path, even when `project=`/`name=`
are passed explicitly and quoted. Both train scripts' own fallback
message and both export scripts search `runs/` and `runs/detect/` for
`best.pt` if it isn't where expected -- pass `TRAINED_MODEL=` explicitly
to the export script if that happens.

## Running long training under tmux (avoid WSL/VS Code disconnects)

A 50-epoch run takes long enough that a dropped WSL/VS Code connection is
likely to kill it mid-run -- this happened once already to the `strong`
run (it silently stopped at epoch 38/50; the only tell was `results.csv`
having fewer rows than `EPOCHS`, since `best.pt`/`last.pt` both still
existed and looked like a normal completed run at a glance). Two rules
of thumb:

1. **Don't trust `best.pt`'s existence alone as "training finished."**
   Compare `results.csv`'s row count to the run's `args.yaml` `epochs:`
   value (or the `EPOCHS` the script was invoked with). `best.pt` is
   just the best checkpoint seen so far -- it exists from epoch 1
   onward, complete run or not.
2. **Run inside tmux (or nohup) so the job survives a dropped
   connection.** tmux is the simpler option since you can reattach and
   watch live progress:

```bash
sudo apt install -y tmux   # first time only

tmux new -s strong_train
cd ~/Projects/physical-ai-recycling-cell
bash tools/train_recycling_yolo_material_v1_aug_strong.sh
```

Detach (leaves it running in the background): `Ctrl+B`, then `D`.

Reattach later to check progress or see it finish:
```bash
tmux attach -t strong_train
```

List sessions if you forget the name: `tmux list-sessions`. Kill a
session once you're done with it: `tmux kill-session -t strong_train`.

Same pattern applies to `tools/train_recycling_yolo_material_v1_aug_
medium.sh` and the baseline script -- swap the session name and command.

If a run does get cut short, see the `RESUME=1` option documented at the
top of `tools/train_recycling_yolo_material_v1_aug_strong.sh` before
re-running from scratch -- resuming from `last.pt` preserves the epochs
already trained instead of wasting them.

## Baseline is unchanged

`tools/train_recycling_yolo_material_v1.sh` and `tools/export_
recycling_yolo_material_v1_onnx.sh` are not modified by this experiment
-- they remain the object-level baseline reference to compare medium/
strong against.
