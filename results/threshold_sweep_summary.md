# Confidence Threshold Sweep Summary

## Experiment Purpose

The real-image vision_only benchmark and detection-stability analysis so far used a single confidence_threshold=0.5. This sweep runs the same test_images_real/ dataset at multiple confidence_threshold values (with policy_confidence_threshold matched to the same value) across all three input sizes, to see how no_detection, misclassified, and the failure-aware policy decisions (ACCEPT_SORT/ROUTE_TO_REJECT/SKIP_NO_DETECTION/RETRY_VIEW/MANUAL_REVIEW) actually shift as the threshold moves, instead of assuming the trend from first principles.

## Dataset

- Sweep log directory: `logs/vision_benchmark/test_images_real/vision_only_threshold_sweep`
- Dataset: test_images_real/ (recursive_image_folder=true), same 50 images (5 classes x 10 shooting conditions) used in the single-threshold real-image analysis
- benchmark_mode=vision_only for every run (no task_manager/MoveIt)

## Thresholds and Input Sizes

- confidence_threshold values swept: 0.3, 0.5, 0.7 (policy_confidence_threshold set equal to each)
- input_size values swept: 640, 416, 320
- Total combinations: 9 (3 thresholds x 3 sizes)

## Overall Summary Table

| threshold | input_size | total_images | correct_or_reject | no_detection | misclassified_or_false_known | image_level_success_rate |
|---|---|---|---|---|---|---|
| 0.3 | 640 | 50 | 24 | 8 | 18 | 48.0% |
| 0.3 | 416 | 50 | 22 | 10 | 18 | 44.0% |
| 0.3 | 320 | 50 | 22 | 11 | 17 | 44.0% |
| 0.5 | 640 | 50 | 21 | 17 | 12 | 42.0% |
| 0.5 | 416 | 50 | 19 | 15 | 16 | 38.0% |
| 0.5 | 320 | 50 | 18 | 18 | 14 | 36.0% |
| 0.7 | 640 | 50 | 18 | 21 | 11 | 36.0% |
| 0.7 | 416 | 50 | 16 | 22 | 12 | 32.0% |
| 0.7 | 320 | 50 | 15 | 23 | 12 | 30.0% |

## Policy-level Summary Table

| threshold | input_size | ACCEPT_SORT | ROUTE_TO_REJECT | SKIP_NO_DETECTION | RETRY_VIEW | MANUAL_REVIEW |
|---|---|---|---|---|---|---|
| 0.3 | 640 | 17 | 18 | 13 | 0 | 2 |
| 0.3 | 416 | 22 | 12 | 16 | 0 | 0 |
| 0.3 | 320 | 20 | 12 | 17 | 0 | 1 |
| 0.5 | 640 | 14 | 9 | 25 | 0 | 2 |
| 0.5 | 416 | 18 | 9 | 23 | 0 | 0 |
| 0.5 | 320 | 15 | 11 | 24 | 0 | 0 |
| 0.7 | 640 | 11 | 9 | 30 | 0 | 0 |
| 0.7 | 416 | 12 | 8 | 30 | 0 | 0 |
| 0.7 | 320 | 13 | 7 | 30 | 0 | 0 |

## Key Findings

- Highest image_level_success_rate: threshold=0.3, input_size=640 at 48.0%.
- Lowest no_detection count: threshold=0.3, input_size=640 with 8 no_detection image(s).
- Most ACCEPT_SORT decisions: threshold=0.3, input_size=416 with 22 ACCEPT_SORT image(s).
- Most SKIP_NO_DETECTION decisions: threshold=0.7, input_size=320, threshold=0.7, input_size=416, threshold=0.7, input_size=640 with 30 SKIP_NO_DETECTION image(s).
- Lowering the threshold from 0.7 to 0.3 (summed across all input sizes): no_detection went from 66 to 29 (lower at the lower threshold), misclassified_or_false_known went from 35 to 53 (higher at the lower threshold). This confirms the expected pattern of "lower threshold -> fewer no_detection, more misclassified" on this dataset.
- Raising the threshold from 0.3 to 0.7 (summed across all input sizes): ACCEPT_SORT went from 59 to 36 (down), SKIP_NO_DETECTION+ROUTE_TO_REJECT went from 88 to 114 (up). This confirms the expected pattern of "higher threshold -> fewer ACCEPT_SORT, more SKIP/REJECT" on this dataset.

## Limitations

- Same 50-image dataset as the single-threshold analysis -- still no ground-truth bounding boxes, and 10 images per class is a small sample per (threshold, input_size, class) cell.
- Only 3 threshold values (0.3/0.5/0.7) were tested; the real decision boundary between "too strict" and "too loose" could sit anywhere between them and wasn't swept continuously.
- policy_confidence_threshold was always kept equal to confidence_threshold in this sweep, so this cannot separate the effect of the ONNX postprocess filter from the effect of the perception policy's own confidence gate -- a follow-up sweep would need to vary them independently.
- can/glass_bottle still cannot score ACCEPT_SORT at any threshold (see the single-threshold analysis's mapping-gap finding) -- this sweep does not change that structural gap.
- Each (threshold, input_size) combination was only run once; no repeated trials to quantify run-to-run variance.

## Next Steps

- Narrow the sweep around whichever threshold in this run looked best (see Key Findings) with finer steps (e.g. 0.4, 0.45, 0.55, 0.6) to find the actual local optimum instead of just comparing three coarse points.
- Sweep confidence_threshold and policy_confidence_threshold independently to see whether the ONNX-level filter or the policy-level gate is the bigger lever on ACCEPT_SORT/RETRY_VIEW rates.
- Once a preferred threshold is chosen, validate it with a benchmark_mode=end_to_end run on a representative subset to confirm the sorting pipeline still behaves as expected end to end, not just at the vision-only/policy level.
- Extend the sweep to a larger/more varied image set before treating any single threshold as final, per the sample-size limitations already noted in the single-threshold analysis.
