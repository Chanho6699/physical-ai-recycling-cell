# Custom YOLO v0 Real-Image Failure Analysis

## Experiment Purpose

Investigate an observed failure mode of the custom_autolabel_v0 model (model_class_mode=recycling_custom) on test_images_real/: a large fraction of images being predicted as a single class ("paper") regardless of their actual content. This cross-tabulates ground_truth (from the folder name) against predicted_class (the highest-confidence ONNX detection) to quantify exactly how many can/glass_bottle/plastic_bottle/unknown images were misread as paper.

## Ground truth vs. taxonomy

**Note:** test_images_real/ still uses the older folder names (`can`, `glass_bottle`, `paper_cup`, `plastic_bottle`, `unknown`), predating the custom model's 4-class taxonomy (`plastic`, `paper`, `can`, `glass_bottle`). ground_truth below is the raw folder name, compared to predicted_class as plain strings -- so a `paper_cup` image correctly predicted as `paper`, or a `plastic_bottle` image correctly predicted as `plastic`, will NOT count as "correct" in the accuracy number below (exact string match only). Only `can` and `glass_bottle` are named identically in both. This does not affect the ground_truth=`unknown` rows or the paper-collapse analysis, which are unambiguous regardless of taxonomy.

## Dataset

- Log file: `logs/vision_benchmark/test_images_real/vision_only/yolo11n_recycling_autolabel_v0_640.log`
- Total images parsed: 50
- Ground truth classes: can, glass_bottle, paper_cup, plastic_bottle, unknown
- Predicted classes observed: can, no_detection, paper

## Overall Accuracy

- Exact-match accuracy (predicted_class == ground_truth): 1/50 = 2.0% (see taxonomy note above -- this undercounts paper_cup/plastic_bottle correctness)
- no_detection count: 5/50 (10.0%)

## Ground-truth-wise Predicted-class Distribution

| ground_truth | can | no_detection | paper | total |
|---|---|---|---|---|
| can | 1 | 1 | 8 | 10 |
| glass_bottle | 0 | 1 | 9 | 10 |
| paper_cup | 0 | 0 | 10 | 10 |
| plastic_bottle | 0 | 1 | 9 | 10 |
| unknown | 0 | 2 | 8 | 10 |

## Paper Collapse Check

- Most-predicted class overall: **paper** -- 44/50 of all predictions (88.0%)
- Of the 50 images whose ground_truth is NOT `paper`, 44 (88.0%) were still predicted as `paper`
- Collapse verdict (>= 50% cross-class rate): **CONFIRMED**
- Per-ground_truth breakdown of images predicted as `paper`: can: 8/10 (80%); glass_bottle: 9/10 (90%); paper_cup: 10/10 (100%); plastic_bottle: 9/10 (90%); unknown: 8/10 (80%)

## Unsafe ACCEPT_SORT Examples

44 image(s) were auto-accepted (policy_decision=ACCEPT_SORT) despite predicted_class not matching ground_truth (exact-string comparison -- see taxonomy note above for the paper_cup/plastic_bottle caveat):

| image | ground_truth | predicted_class | confidence |
|---|---|---|---|
| can/can_clutter_bg_005.jpg.jpg | can | paper | 0.88 |
| can/can_crushed_008.jpg.jpg | can | paper | 0.72 |
| can/can_edge_frame_006.jpg.jpg | can | paper | 0.84 |
| can/can_hard_case_010.jpg.jpg | can | paper | 0.79 |
| can/can_multi_object_009.jpg.jpg | can | paper | 0.71 |
| can/can_normal_front_001.jpg.jpg | can | paper | 0.63 |
| can/can_partial_cut_007.jpg.jpg | can | paper | 0.75 |
| can/can_small_far_004.jpg.jpg | can | paper | 0.59 |
| glass_bottle/glass_bottle_angle_view_002.jpg.jpg | glass_bottle | paper | 0.66 |
| glass_bottle/glass_bottle_clutter_bg_005.jpg.jpg | glass_bottle | paper | 0.61 |
| glass_bottle/glass_bottle_dark_light_003.jpg.jpg | glass_bottle | paper | 0.71 |
| glass_bottle/glass_bottle_edge_frame_006.jpg.jpg | glass_bottle | paper | 0.91 |
| glass_bottle/glass_bottle_hard_case_010.jpg.jpg | glass_bottle | paper | 0.79 |
| glass_bottle/glass_bottle_multi_object_009.jpg.jpg | glass_bottle | paper | 0.85 |
| glass_bottle/glass_bottle_partial_cut_007.jpg.jpg | glass_bottle | paper | 0.51 |

... and 29 more (see `results/custom_yolo_v0_real_image_predictions.csv`).

## Limitations

- Exact-string ground_truth/predicted_class comparison only -- see the taxonomy note above; a real per-class semantic mapping (paper_cup -> paper, plastic_bottle -> plastic) would give a more meaningful accuracy number.
- No ground-truth bounding boxes -- this is an image-level class comparison, not an IoU/mAP evaluation.
- Single run at one input_size/confidence_threshold/box_threshold combination; the paper-collapse behavior has not been checked across other sizes/thresholds in this report.
- Root cause is not diagnosed here (e.g., whether it is a training-data imbalance, a GroundingDINO pseudo-label bias toward loose/oversized "paper"-prompt boxes, or a training convergence issue) -- this tool only quantifies the symptom.

## Next Steps

- Inspect training data balance and pseudo-label bbox quality for the "paper" class specifically (see `datasets/recycling_yolo_autolabel_v0/previews/` and `autolabel_report.csv`) for a systematic bias (e.g., oversized boxes nearly covering the whole image, which several of the "paper" detections in this run showed).
- Re-run this analysis after retraining/rebalancing to confirm the collapse is resolved, not just shifted to a different dominant class.
- Cross-check against `results/policy_effectiveness_summary.md`-style metrics (unsafe_accept_rate) once the custom model is included in a full threshold sweep.
