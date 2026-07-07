# Policy Effectiveness Analysis

## Experiment Purpose

threshold_sweep_summary.md showed how detection stability and policy-decision *counts* shift with confidence_threshold. This analysis asks a sharper question of the same sweep data: when perception_policy.py said ACCEPT_SORT, was it actually right (vs. the folder-name ground truth), and when the ground truth says a detection was risky (misclassified/false_known), did the policy actually block it rather than sorting it anyway? This is a policy-effectiveness evaluation, not a raw accuracy evaluation.

## Dataset and Inputs

- Sweep log directory: `logs/vision_benchmark/test_images_real/vision_only_threshold_sweep`
- Dataset: test_images_real/ (recursive_image_folder=true), same 50 images (5 classes x 10 shooting conditions) used throughout the real-image analyses
- confidence_threshold values: 0.3, 0.5, 0.7 (policy_confidence_threshold matched to each)
- input_size values: 640, 416, 320
- benchmark_mode=vision_only for every run (no task_manager/MoveIt)
- **This is an offline evaluation of the runtime policy against a folder-name-derived ground truth label, not a bbox-annotation -based mAP evaluation, and not a live test of the policy actually gating the sort pipeline** (task_manager does not consume policy_decision yet).

## Analysis Definitions

- `accept_correct`: `policy_decision == ACCEPT_SORT` and `status == correct`
- `unsafe_accept_count`: `policy_decision == ACCEPT_SORT` and `status in {misclassified, false_known}`; `unsafe_accept_rate = unsafe_accept_count / accept_total`
- risky case: `status in {misclassified, false_known}`; `blocked_risky_count`: a risky case where `policy_decision != ACCEPT_SORT`; `blocked_risky_rate = blocked_risky_count / risky_total`
- `known_correct_total`: `expected_class != unknown` and `status == correct`; `missed_correct_count`: one of those where `policy_decision != ACCEPT_SORT` anyway (a real object the policy could safely have sorted, but didn't); `missed_correct_rate = missed_correct_count / known_correct_total`

## Overall Policy Effectiveness Table

| threshold | input_size | accept_total | accept_correct | unsafe_accept_count | unsafe_accept_rate | risky_total | blocked_risky_count | blocked_risky_rate | known_correct_total | missed_correct_count | missed_correct_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.3 | 640 | 17 | 9 | 8 | 47.1% | 18 | 10 | 55.6% | 14 | 5 | 35.7% |
| 0.3 | 416 | 22 | 10 | 12 | 54.5% | 18 | 6 | 33.3% | 12 | 2 | 16.7% |
| 0.3 | 320 | 20 | 9 | 11 | 55.0% | 17 | 6 | 35.3% | 12 | 3 | 25.0% |
| 0.5 | 640 | 14 | 8 | 6 | 42.9% | 12 | 6 | 50.0% | 11 | 3 | 27.3% |
| 0.5 | 416 | 18 | 8 | 10 | 55.6% | 16 | 6 | 37.5% | 9 | 1 | 11.1% |
| 0.5 | 320 | 15 | 7 | 8 | 53.3% | 14 | 6 | 42.9% | 8 | 1 | 12.5% |
| 0.7 | 640 | 11 | 6 | 5 | 45.5% | 11 | 6 | 54.5% | 8 | 2 | 25.0% |
| 0.7 | 416 | 12 | 5 | 7 | 58.3% | 12 | 5 | 41.7% | 6 | 1 | 16.7% |
| 0.7 | 320 | 13 | 5 | 8 | 61.5% | 12 | 4 | 33.3% | 5 | 0 | 0.0% |

## ACCEPT_SORT Safety Table

| threshold | input_size | accept_total | accept_correct | accept_misclassified | accept_false_known | accept_unknown_or_no_detection | unsafe_accept_rate |
|---|---|---|---|---|---|---|---|
| 0.3 | 640 | 17 | 9 | 8 | 0 | 0 | 47.1% |
| 0.3 | 416 | 22 | 10 | 12 | 0 | 0 | 54.5% |
| 0.3 | 320 | 20 | 9 | 11 | 0 | 0 | 55.0% |
| 0.5 | 640 | 14 | 8 | 6 | 0 | 0 | 42.9% |
| 0.5 | 416 | 18 | 8 | 10 | 0 | 0 | 55.6% |
| 0.5 | 320 | 15 | 7 | 8 | 0 | 0 | 53.3% |
| 0.7 | 640 | 11 | 6 | 5 | 0 | 0 | 45.5% |
| 0.7 | 416 | 12 | 5 | 7 | 0 | 0 | 58.3% |
| 0.7 | 320 | 13 | 5 | 8 | 0 | 0 | 61.5% |

## Risk Blocking Table

| threshold | input_size | risky_total | blocked_risky_count | blocked_risky_rate |
|---|---|---|---|---|
| 0.3 | 640 | 18 | 10 | 55.6% |
| 0.3 | 416 | 18 | 6 | 33.3% |
| 0.3 | 320 | 17 | 6 | 35.3% |
| 0.5 | 640 | 12 | 6 | 50.0% |
| 0.5 | 416 | 16 | 6 | 37.5% |
| 0.5 | 320 | 14 | 6 | 42.9% |
| 0.7 | 640 | 11 | 6 | 54.5% |
| 0.7 | 416 | 12 | 5 | 41.7% |
| 0.7 | 320 | 12 | 4 | 33.3% |

## Missed Opportunity Table

| threshold | input_size | known_correct_total | missed_correct_count | missed_correct_rate |
|---|---|---|---|---|
| 0.3 | 640 | 14 | 5 | 35.7% |
| 0.3 | 416 | 12 | 2 | 16.7% |
| 0.3 | 320 | 12 | 3 | 25.0% |
| 0.5 | 640 | 11 | 3 | 27.3% |
| 0.5 | 416 | 9 | 1 | 11.1% |
| 0.5 | 320 | 8 | 1 | 12.5% |
| 0.7 | 640 | 8 | 2 | 25.0% |
| 0.7 | 416 | 6 | 1 | 16.7% |
| 0.7 | 320 | 5 | 0 | 0.0% |

## Per-class ACCEPT_SORT Safety

| threshold | input_size | expected_class | accept_total | accept_correct | accept_unsafe | unsafe_accept_rate |
|---|---|---|---|---|---|---|
| 0.3 | 640 | can | 3 | 0 | 3 | 100.0% |
| 0.3 | 640 | glass_bottle | 5 | 0 | 5 | 100.0% |
| 0.3 | 640 | paper_cup | 6 | 6 | 0 | 0.0% |
| 0.3 | 640 | plastic_bottle | 3 | 3 | 0 | 0.0% |
| 0.3 | 640 | unknown | 0 | 0 | 0 | 0.0% |
| 0.3 | 416 | can | 3 | 0 | 3 | 100.0% |
| 0.3 | 416 | glass_bottle | 7 | 0 | 7 | 100.0% |
| 0.3 | 416 | paper_cup | 7 | 5 | 2 | 28.6% |
| 0.3 | 416 | plastic_bottle | 5 | 5 | 0 | 0.0% |
| 0.3 | 416 | unknown | 0 | 0 | 0 | 0.0% |
| 0.3 | 320 | can | 3 | 0 | 3 | 100.0% |
| 0.3 | 320 | glass_bottle | 6 | 0 | 6 | 100.0% |
| 0.3 | 320 | paper_cup | 5 | 3 | 2 | 40.0% |
| 0.3 | 320 | plastic_bottle | 6 | 6 | 0 | 0.0% |
| 0.3 | 320 | unknown | 0 | 0 | 0 | 0.0% |
| 0.5 | 640 | can | 2 | 0 | 2 | 100.0% |
| 0.5 | 640 | glass_bottle | 4 | 0 | 4 | 100.0% |
| 0.5 | 640 | paper_cup | 6 | 6 | 0 | 0.0% |
| 0.5 | 640 | plastic_bottle | 2 | 2 | 0 | 0.0% |
| 0.5 | 640 | unknown | 0 | 0 | 0 | 0.0% |
| 0.5 | 416 | can | 1 | 0 | 1 | 100.0% |
| 0.5 | 416 | glass_bottle | 7 | 0 | 7 | 100.0% |
| 0.5 | 416 | paper_cup | 6 | 4 | 2 | 33.3% |
| 0.5 | 416 | plastic_bottle | 4 | 4 | 0 | 0.0% |
| 0.5 | 416 | unknown | 0 | 0 | 0 | 0.0% |
| 0.5 | 320 | can | 0 | 0 | 0 | 0.0% |
| 0.5 | 320 | glass_bottle | 6 | 0 | 6 | 100.0% |
| 0.5 | 320 | paper_cup | 5 | 3 | 2 | 40.0% |
| 0.5 | 320 | plastic_bottle | 4 | 4 | 0 | 0.0% |
| 0.5 | 320 | unknown | 0 | 0 | 0 | 0.0% |
| 0.7 | 640 | can | 1 | 0 | 1 | 100.0% |
| 0.7 | 640 | glass_bottle | 4 | 0 | 4 | 100.0% |
| 0.7 | 640 | paper_cup | 5 | 5 | 0 | 0.0% |
| 0.7 | 640 | plastic_bottle | 1 | 1 | 0 | 0.0% |
| 0.7 | 640 | unknown | 0 | 0 | 0 | 0.0% |
| 0.7 | 416 | can | 1 | 0 | 1 | 100.0% |
| 0.7 | 416 | glass_bottle | 6 | 0 | 6 | 100.0% |
| 0.7 | 416 | paper_cup | 2 | 2 | 0 | 0.0% |
| 0.7 | 416 | plastic_bottle | 3 | 3 | 0 | 0.0% |
| 0.7 | 416 | unknown | 0 | 0 | 0 | 0.0% |
| 0.7 | 320 | can | 0 | 0 | 0 | 0.0% |
| 0.7 | 320 | glass_bottle | 7 | 0 | 7 | 100.0% |
| 0.7 | 320 | paper_cup | 3 | 2 | 1 | 33.3% |
| 0.7 | 320 | plastic_bottle | 3 | 3 | 0 | 0.0% |
| 0.7 | 320 | unknown | 0 | 0 | 0 | 0.0% |

## Key Findings

- Mapping-gap contribution to unsafe_accept_rate: can/glass_bottle ACCEPT_SORT cases (summed across all thresholds/sizes) are unsafe 100.0% of the time (66/66), vs. 11.8% (9/76) for paper_cup/plastic_bottle. The mapping gap, not the confidence-based policy logic itself, is the dominant driver of the overall unsafe_accept_rate above.
- Lowest unsafe_accept_rate: threshold=0.5, input_size=640 at 42.9%.
- Most accept_correct: threshold=0.3, input_size=416 with 10 correctly-accepted image(s).
- Highest blocked_risky_rate: threshold=0.3, input_size=640 at 55.6%.
- Highest missed_correct_rate: threshold=0.3, input_size=640 at 35.7%.
- threshold=0.3 vs threshold=0.7 (summed across input sizes): accept_total 59 vs 36 (more ACCEPT_SORT at 0.3), unsafe_accept_rate 52.5% vs 55.6% (not higher at 0.3). Does NOT confirm that lowering the threshold trades more ACCEPT_SORT volume for a higher unsafe-accept rate on this dataset.
- threshold=0.7 vs threshold=0.3 (summed across input sizes): blocked_risky_rate 42.9% vs 41.5% (better risk-blocking at 0.7), missed_correct_rate 15.8% vs 26.3% (not more missed opportunity at 0.7). Does NOT confirm that raising the threshold trades better risk-blocking for more missed sorting opportunities on this dataset.

## Limitations

- **can/glass_bottle can never be `accept_correct` today**: the current YOLO/COCO mapping never emits `can`/`glass_bottle` as class_name (only plastic_bottle/paper_cup/unknown), so any ACCEPT_SORT on a can/glass_bottle image is structurally impossible -- see Per-class ACCEPT_SORT Safety, where those two classes should show accept_total=0 or accept_correct=0 regardless of threshold. This is a class-mapping gap, not a policy-effectiveness result for those classes.
- This evaluates the runtime policy_decision against an offline, folder-name-derived expected_class -- not a bbox-annotation-based mAP/precision-recall evaluation, and there is no ground-truth bounding box data for test_images_real/.
- task_manager does not consume policy_decision yet, so "unsafe_accept"/"blocked_risky" describe what the policy *would* have decided, not an outcome actually observed on the real sort pipeline.
- Same 50-image dataset as the other real-image analyses -- 10 images per class is a small sample, so a single image flipping status can swing a per-class rate by 10-100 percentage points.
- Only 3 threshold values (0.3/0.5/0.7) were tested; the true safety/opportunity trade-off curve between them is not known.

## Next Steps

- Extend the class mapping so can/glass_bottle can actually be accept_correct, then re-run this analysis -- right now their rows are not informative about policy effectiveness at all.
- Once a threshold is chosen from this trade-off, wire policy_decision into task_manager so ROUTE_TO_REJECT/SKIP_NO_DETECTION/RETRY_VIEW/MANUAL_REVIEW actually change sort pipeline behavior, then re-measure unsafe_accept_rate/blocked_risky_rate against real outcomes instead of this offline estimate.
- Sweep confidence_threshold and policy_confidence_threshold independently to see whether tightening just the policy gate (while leaving the ONNX-level filter looser) can lower unsafe_accept_rate without also raising missed_correct_rate as much as tightening both together.
- Grow the dataset (more images per class/condition) before treating any single per-class or per-condition rate in this report as final.
