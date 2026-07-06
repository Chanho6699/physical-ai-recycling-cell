# GroundingDINO Auto-labeling Plan

## Why auto-labeling is needed

The sorting cell's vision pipeline has so far used a pretrained,
COCO-trained YOLO11n ONNX model (`models/yolo11n_{640,416,320}.onnx`).
Real-image benchmarking (`results/real_image_detection_analysis.md`,
`results/threshold_sweep_summary.md`, `results/policy_effectiveness_
summary.md`) showed this model has a structural gap: COCO has no `can`
or `glass_bottle` category, so `COCO_CLASS_ID_TO_PROJECT_CLASS` in
`vision_perception_node.py` can only ever emit `plastic_bottle`,
`paper_cup`, or `unknown` -- `can`/`glass_bottle` images always land on
`no_detection` or, worse, get **misclassified as `paper_cup`/
`plastic_bottle` with high confidence**. The policy-effectiveness
analysis quantified this directly: ACCEPT_SORT on a can/glass_bottle
image was unsafe **100% of the time** across every threshold tested
(0.3/0.5/0.7), because the model has no way to say "this is a can" in
the first place. Confidence threshold sweeping alone cannot fix a
missing class -- there is no threshold at which a model emits a label it
was never trained to produce. The only fix is a model trained on the
project's own class taxonomy.

## Kaggle dataset limitation

The external dataset being used as a source, the Kaggle "Garbage
Classification" dataset, is an **image classification** dataset: each
image has one folder-name label and no bounding-box annotation at all.
YOLO detection training needs `class_id x_center y_center width height`
labels per image. Hand-annotating bounding boxes for the ~2000 candidate
images now under `datasets/recycling_yolo_candidates/` was judged too
slow for a first validation pass, so this plan uses GroundingDINO
zero-shot detection to generate pseudo bounding boxes instead ("auto
-labeling"), accepting a known quality cost in exchange for speed (see
Expected risks below).

## Why class taxonomy changed

The Kaggle dataset's own class names (cardboard/glass/metal/paper/
plastic/trash) don't line up with what this project originally assumed.
Inspecting the actual image content:
- Kaggle `glass` is glass bottles, which matches `glass_bottle` fine.
- Kaggle `paper` is general paper/paper sheets -- **not** paper cups.
  Using it to represent `paper_cup` would mislabel plain paper as a
  paper cup.
- Kaggle `metal` is predominantly cans, matching `can`.
- Kaggle `plastic` is plastic waste broadly (bottles, cups, containers,
  bags) -- **not** just plastic bottles. Using it to represent
  `plastic_bottle` specifically would mislabel a lot of non-bottle
  plastic as a bottle.

Keeping the original `plastic_bottle`/`paper_cup` taxonomy while
sourcing labels from this dataset would silently corrupt the semantics
of two of the four classes. The taxonomy was redefined to match what the
source data actually contains.

## Previous class taxonomy vs new taxonomy

| | previous | new |
|---|---|---|
| 0 | plastic_bottle | plastic |
| 1 | paper_cup | paper |
| 2 | can | can |
| 3 | glass_bottle | glass_bottle |

Bin mapping under the new taxonomy:

| class | bin |
|---|---|
| plastic | plastic_bin |
| paper | paper_bin |
| can | metal_bin |
| glass_bottle | glass_bin |
| unknown | reject_bin |

`can` and `glass_bottle` keep their names since the Kaggle content
already matched those concepts; only the two classes whose Kaggle
content didn't match the original assumption (`plastic_bottle` ->
`plastic`, `paper_cup` -> `paper`) were renamed.

## Prompt mapping

Only a class's own prompts are ever used for its own candidate images
(a `can/` image is only queried with can-prompts, never with paper/
plastic/glass prompts) -- see
`tools/autolabel_recycling_groundingdino.py`'s `PROMPT_MAP`:

| class | prompts |
|---|---|
| plastic | "plastic bottle", "plastic cup", "plastic container", "plastic waste" |
| paper | "paper", "paper sheet", "crumpled paper" |
| can | "aluminum can", "soda can", "metal can" |
| glass_bottle | "glass bottle" |

When multiple prompts for the same class each return a box, the single
highest-confidence box across all of that class's prompts is kept (one
label per image).

## Pseudo-label dataset definition

Output: `datasets/recycling_yolo_autolabel_v0/`
- `images/train/`, `images/val/` -- candidate images, split by
  `--val-ratio` (default 0.2) per class
- `labels/train/`, `labels/val/` -- YOLO-format labels; images with no
  GroundingDINO detection above threshold get no label file at all
  (tracked instead in `autolabel_report.csv` as `no_box`)
- `recycling.yaml` -- Ultralytics dataset config (paths + the 4 class
  names)
- `autolabel_report.csv` -- per-image status/prompt/confidence/bbox
- `previews/` -- a sample (`--preview-limit-per-class`, default 10) of
  images with the selected box drawn on them, for a quick look before
  trusting the dataset with a training run
- `README.md` -- generated into the dataset folder itself, repeating the
  pseudo-label quality warning so it travels with the data

This is explicitly a **v0** dataset: good enough to validate that
training/export/benchmarking works end-to-end on the new taxonomy, not a
final production dataset.

## Expected risks

- **False positive**: GroundingDINO draws a box where the described
  object isn't actually present (or isn't the primary subject).
- **no_box**: no prompt for the image's class clears
  `box_threshold`/`text_threshold`; the image ships with no label.
- **Wrong object**: a box is drawn, but around something other than the
  intended recyclable item (e.g. a hand, a background object, packaging
  text).
- **Oversized bbox**: GroundingDINO boxes are open-vocabulary detections,
  not tuned for tight recycling-object boxes -- expect some boxes looser
  than a human annotator would draw.
- **Class ambiguity**: some Kaggle images plausibly straddle categories
  (e.g. a plastic-coated paper cup, a crushed can that barely resembles
  one) -- the human curation step that produced `datasets/recycling_yolo
  _candidates/` is the first filter against this, but it isn't perfect.

None of these are hypothetical -- they are the reason human validation
(next section) exists as an explicit step before this data is trusted
for anything beyond pipeline validation.

## Human validation plan

1. Review every image in `previews/` first (fast, bounded sample size)
   -- if a large fraction of previews look wrong, fix `PROMPT_MAP` or
   thresholds and re-run before going further.
2. Spot-check a random sample of `autolabel_report.csv` rows per class
   (not just previews), specifically the `no_box` and lowest-confidence
   `labeled` rows -- these are the likeliest false positives/negatives.
3. For any class with a high `no_box` rate or an obviously wrong prompt
   match rate, treat that class's labels as unreliable until re-labeled
   (with adjusted prompts/thresholds) or hand-corrected.
4. Only after this pass should the dataset be used for anything beyond
   the pipeline-validation training run described below.
5. Human corrections (deleting bad boxes, redrawing tight boxes, moving
   an image between `no_box` and `labeled`) should produce a
   `recycling_yolo_autolabel_v1` (or a hand-corrected variant), not
   silently overwrite v0 -- keep the auto-labeled and human-corrected
   datasets distinguishable.

## Training plan

`tools/train_recycling_yolo_autolabel_v0.sh` fine-tunes the existing
`models/yolo11n.pt` checkpoint (not from scratch) on
`recycling_yolo_autolabel_v0/recycling.yaml`:

```bash
bash tools/train_recycling_yolo_autolabel_v0.sh
# equivalent to:
yolo detect train model=models/yolo11n.pt \
  data=datasets/recycling_yolo_autolabel_v0/recycling.yaml \
  imgsz=640 epochs=50 batch=8 \
  project=runs/recycling_yolo name=yolo11n_autolabel_v0_640
```

Since the labels are pseudo-labels, training metrics (mAP, precision/
recall reported by `yolo detect train`) should be read as "how well the
model learned to reproduce GroundingDINO's pseudo-labels", not as a
proxy for real-world accuracy -- that's what the Evaluation plan below
is for.

## ONNX export plan

`tools/export_recycling_yolo_autolabel_v0_onnx.sh` exports the trained
checkpoint with the same `nms=True` convention as the existing
`yolo11n_{640,416,320}.onnx` models, so `vision_perception_node.py`'s
`postprocess_yolo()` (which expects a post-NMS `(1, N, 6)` output) can
load it without any code changes -- only `onnx_model_path` (and a new
`COCO_CLASS_ID_TO_PROJECT_CLASS`-equivalent mapping, since this model's
4 output classes are `plastic`/`paper`/`can`/`glass_bottle` directly,
not COCO ids) need to change in the launch config:

```bash
bash tools/export_recycling_yolo_autolabel_v0_onnx.sh
# writes models/yolo11n_recycling_autolabel_v0_640.onnx
```

## Evaluation plan

Re-run the same real-image analysis pipeline already built for the
pretrained model, pointed at the new custom model, so the comparison
uses tools already validated against real data:
1. `tools/run_vision_size_benchmark.sh` / `tools/run_vision_threshold_
   sweep.sh` with `onnx_model_path` pointed at the new custom ONNX file
   and the class-name mapping updated for the 4-class output.
2. `tools/analyze_real_image_detections.py` and `tools/analyze_policy_
   effectiveness.py` against the resulting logs.
3. A `benchmark_mode=end_to_end` run on a small representative subset to
   confirm real pick/place behavior, not just vision-only detection.

## Pretrained YOLO11n vs custom_autolabel_v0 comparison metrics

| metric | source |
|---|---|
| class-level detection success | `analyze_real_image_detections.py`'s Class-level Summary (`correct` count per class) |
| no_detection count | same report's `no_detection` column |
| misclassified count | same report's `misclassified` column |
| policy ACCEPT_SORT safety | `analyze_policy_effectiveness.py`'s `accept_correct`/`accept_total` |
| unsafe_accept_rate | `analyze_policy_effectiveness.py`'s Overall Policy Effectiveness Table |
| vision latency | `parse_vision_benchmark_logs.py`'s `avg_total_ms`/`avg_inference_ms`/`avg_fps` |
| end-to-end sorting success on selected subset | `monitor_node`'s `[CellMetrics]` success rate from a `benchmark_mode=end_to_end` run |

The headline question this comparison needs to answer: does
custom_autolabel_v0 actually fix the can/glass_bottle domain gap (real
`accept_correct` > 0, `unsafe_accept_rate` for those two classes no
longer stuck at 100%), and does it do so without regressing paper/
plastic performance or vision latency versus the pretrained COCO model.

## Example commands

Auto-label:
```bash
python3 tools/autolabel_recycling_groundingdino.py \
  --input-dir datasets/recycling_yolo_candidates \
  --output-dir datasets/recycling_yolo_autolabel_v0 \
  --val-ratio 0.2 \
  --box-threshold 0.35 \
  --text-threshold 0.25 \
  --preview-limit-per-class 10
```

Train:
```bash
bash tools/train_recycling_yolo_autolabel_v0.sh
```

Export to ONNX:
```bash
bash tools/export_recycling_yolo_autolabel_v0_onnx.sh
```
