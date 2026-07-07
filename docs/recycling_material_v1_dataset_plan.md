# Recycling Material v1 Dataset Plan

## Why this exists: the v0 paper collapse

`custom_autolabel_v0` was trained on a 4-class object-level taxonomy
(`plastic_bottle`/`paper_cup`/`can`/`glass_bottle`, later renamed to
`plastic`/`paper`/`can`/`glass_bottle`) and benchmarked against
`test_images_real/` with `tools/analyze_custom_yolo_real_benchmark.py`.
The result was a severe **paper collapse**: 44 of 50 real images were
predicted as `paper`, regardless of ground truth --

| ground_truth | predicted `paper` |
|---|---|
| can | 8/10 (80%) |
| glass_bottle | 9/10 (90%) |
| paper_cup | 10/10 (100%) |
| plastic_bottle | 9/10 (90%) |
| unknown | 8/10 (80%) |

(see `results/custom_yolo_v0_real_failure_analysis.md`). 88% of
non-paper images were still predicted as paper, and most of those were
auto-accepted by the failure-aware policy (`ACCEPT_SORT`) despite being
wrong. Several of v0's own "paper" pseudo-labels (visible in
`datasets/recycling_yolo_autolabel_v0/previews/`) were loose/oversized
boxes covering nearly the whole image -- a plausible training-data bias
toward the model learning "big box, ambiguous background -> paper"
rather than actual paper content.

## Why the taxonomy changed again

Two changes address this at once:

1. **A coarser, material-level taxonomy** (`plastic`/`metal`/`glass`/
   `paper` instead of `plastic_bottle`/`paper_cup`/`can`/`glass_bottle`)
   -- material categories are visually more distinct by texture/color/
   reflectivity than the finer object-level split, which should be
   easier for the detector to separate.
2. **More varied source photos.** `datasets/recycling_yolo_candidates_v1/`
   is a freshly collected set with varied backgrounds and framings,
   unlike the more uniform earlier photos -- reducing the risk of the
   model learning background/framing artifacts as a shortcut to a
   particular class.

`--exclude-v0-paper` additionally lets the "paper" class from v0 be
dropped entirely from the reused v0 data, since it's the lead suspect
for the collapse; the documented example commands in this project use
this flag on.

## Previous taxonomy vs. material_v1 taxonomy

| | custom_autolabel_v0 | recycling_yolo_material_v1 |
|---|---|---|
| 0 | plastic | plastic |
| 1 | paper | metal |
| 2 | can | glass |
| 3 | glass_bottle | paper |

v0 -> material_v1 remap rule (used by `--include-v0-remap`):

| v0 class_id | v0 name | material_v1 class_id | material_v1 name |
|---|---|---|---|
| 0 | plastic | 0 | plastic |
| 1 | paper | 3 | paper (excludable via `--exclude-v0-paper`) |
| 2 | can | 1 | metal |
| 3 | glass_bottle | 2 | glass |

## Data sources

1. **`candidates_v1`** -- `datasets/recycling_yolo_candidates_v1/
   {plastic,metal,glass,paper}/`, classification-style images with no
   bbox annotation, auto-labeled fresh with GroundingDINO using the
   prompt map below. This is the primary, larger source for material_v1
   (thousands of images per class vs. v0's ~100/class).
2. **`v0_remapped`** -- `datasets/recycling_yolo_autolabel_v0/`'s
   existing YOLO labels (bboxes + `autolabel_report.csv` confidence
   values), remapped into the material taxonomy and reused as-is --
   GroundingDINO is NOT re-run on these images, since they're already
   labeled. Enabled with `--include-v0-remap`.

`tools/build_recycling_material_v1_dataset.py`'s `autolabel_report.csv`
records `source_type` per row so the two are always distinguishable
after the fact.

## GroundingDINO prompt map (candidates_v1 only)

| class | prompts |
|---|---|
| plastic | "plastic bottle", "plastic container", "plastic cup", "plastic bag", "plastic packaging" |
| metal | "aluminum can", "metal can", "tin can", "metal container" |
| glass | "glass bottle", "glass jar", "glass container" |
| paper | "paper", "cardboard box", "paper box", "magazine", "book" |

Only a class's own prompts are ever used for its own candidate images,
same policy as v0's builder.

## bbox quality warnings

Computed per labeled detection (both source types) from the normalized
bbox and (where available) confidence:

- `too_large_bbox`: normalized area ratio (width * height) > 0.75
- `too_small_bbox`: normalized area ratio < 0.02
- `low_confidence`: confidence < 0.35
- `boundary_clamped`: the raw box extended past the image edge and was
  clamped before normalizing (candidates_v1 only -- v0's boxes were
  already clamped/normalized when v0 was built, so this never applies
  to v0_remapped rows)

By default, warned boxes are still written to the label files (a
warning is a flag for review, not an automatic rejection). Two CLI
options narrow that:
- `--exclude-warned`: drop any labeled box with *any* warning flag
- `--exclude-too-large`: drop only boxes flagged `too_large_bbox`
  (targeted at the specific oversized-box pattern implicated in the v0
  paper collapse), regardless of other warnings

Excluded rows keep `status=excluded` in `autolabel_report.csv` (the
image is still copied into `images/`, just without a paired label
file) rather than being silently removed from the dataset entirely.

## Human validation plan

Same shape as v0's: check `previews/` first (bbox + class_name +
confidence + warning_flags are all drawn on), spot-check
`autolabel_report.csv` rows flagged with warnings or low confidence
across both source types, and treat any class with a high `no_box` or
warning rate as unreliable until the prompts/thresholds are adjusted or
the labels are hand-corrected -- before trusting this dataset beyond
pipeline validation.

## Training plan

`tools/train_recycling_yolo_material_v1.sh` fine-tunes `models/
yolo11n.pt` on `recycling_yolo_material_v1/recycling_material_v1.yaml`,
using `RUN_NAME` (not `NAME`) for the run name -- `NAME` is a common
pre-set environment variable on some systems (e.g. from
`/etc/os-release`), which would silently override a naive
`NAME=${NAME:-default}` default and was a likely contributor to an
earlier unexpected `runs/detect/runs/recycling_yolo/...` nesting.
`project=`/`name=` are passed explicitly and quoted, but in practice
Ultralytics still resolved this run to
`runs/detect/runs/recycling_yolo/yolo11n_material_v1_640/` rather than
the plain `${PROJECT}/${RUN_NAME}/` -- i.e. relative `project=` paths
get nested under Ultralytics' own `runs/detect/` root regardless of the
`NAME` env var issue. Both `tools/train_recycling_yolo_material_v1.sh`'s
own fallback message and `tools/export_recycling_yolo_material_v1_onnx.sh`
search both `${PROJECT}` and `runs/detect/` for `best.pt`, so this is a
recoverable inconvenience, not a correctness problem -- just pass
`TRAINED_MODEL=` explicitly to the export script if it happens.

## ONNX export plan

`tools/export_recycling_yolo_material_v1_onnx.sh` exports with the same
`nms=True` convention as every other yolo11n export in this project. If
`best.pt` isn't found at the expected path (Ultralytics can sometimes
write to a nested location), the script runs `find` under `runs/` and
`runs/detect/` and prints every `best.pt` it finds so the right path can
be passed via `TRAINED_MODEL=`.

## Evaluation plan

Same pipeline already validated on the pretrained-COCO and
custom_autolabel_v0 models:
1. `tools/run_vision_size_benchmark.sh` with `MODEL_STEM=yolo11n_
   recycling_material_v1 MODEL_CLASS_MODE=recycling_material_v1` against
   `test_images_real/`.
2. `tools/analyze_custom_yolo_real_benchmark.py` (or a similar ground_
   truth/predicted_class cross-tab) against the resulting log, to check
   directly whether the paper-collapse pattern is gone or has moved to
   a different dominant class.
3. `results/policy_effectiveness_summary.md`-style unsafe_accept_rate
   metrics once material_v1 is included in a full threshold sweep.

## ROS2 integration

`vision_perception_node.py` gained a third `model_class_mode`:
`recycling_material_v1` (class_id 0/1/2/3 -> plastic/metal/glass/paper
directly, same pattern as `recycling_custom`'s class-id-is-already-the
-project-class mapping). `task_manager_node.py`'s `CLASS_TO_BIN` and
`perception_policy.py`'s `KNOWN_CLASSES`/`SUPPORTED_EMITTED_CLASSES`
were extended with `metal`/`glass` (`plastic`/`paper` already existed
from `recycling_custom`) so the policy layer and bin routing work
correctly under this taxonomy too, exactly as was done for
`recycling_custom`.

## Example commands

Build (recommended default, matching this project's documented
examples):
```bash
python3 tools/build_recycling_material_v1_dataset.py \
  --include-v0-remap \
  --exclude-v0-paper \
  --exclude-too-large
```

Train:
```bash
bash tools/train_recycling_yolo_material_v1.sh
```

Export to ONNX:
```bash
bash tools/export_recycling_yolo_material_v1_onnx.sh
```

Run the vision node against the material_v1 model:
```bash
MODEL_STEM=yolo11n_recycling_material_v1 \
MODEL_CLASS_MODE=recycling_material_v1 \
IMAGE_FOLDER_PATH=/path/to/test_images_real \
RECURSIVE_IMAGE_FOLDER=true \
BENCHMARK_MODE=vision_only \
bash tools/run_vision_size_benchmark.sh 640
```
