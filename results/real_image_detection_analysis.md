# Real-Image Detection Stability Analysis

## Experiment Purpose

Break down the benchmark_mode=vision_only real-image run (test_images_real/, 5 classes x 10 shooting conditions) by input_size to see not just latency/FPS but whether detection *stability* changes with input_size -- per class and per shooting condition, not just in aggregate.
**This is an image-level detection stability analysis, not a bbox-annotation-based mAP/precision-recall evaluation** -- test_images_real/ has no ground-truth bounding boxes, only a folder-name convention (`<class>/<class>_<condition>_<index>.jpg.jpg`) used as a per-image expected label.

## Dataset

- Log directory: `logs/vision_benchmark/test_images_real/vision_only`
- Classes (5): can, glass_bottle, paper_cup, plastic_bottle, unknown
- Shooting conditions observed (21): angle_view, book, bottle_cap, charger, clutter_bg, crushed, dark_light, edge_frame, hand, hard_case, mixed_objects, mouse, multi_object, normal_front, paper_piece, partial_cut, plastic_bag, reflection, small_far, snack_bag, tissue
- Images per input_size: 640=50, 416=50, 320=50
- expected_class = first path segment of the image (e.g. `can/can_dark_light_003.jpg.jpg` -> `can`)
- condition = filename with the expected_class prefix, trailing `_<index>`, and `.jpg`/`.jpg.jpg` extension stripped (e.g. `can_dark_light_003.jpg.jpg` -> `dark_light`)

## Analysis Rule

For `expected_class != unknown`:
- `correct`: at least one detection has `class_name == expected_class`
- `no_detection`: `detections_count == 0`
- `misclassified`: detections exist, but none match `expected_class`

For `expected_class == unknown`:
- `correct_reject`: no detections, or every detection is `class_name == unknown`
- `false_known`: at least one detection is a known class (can, glass_bottle, paper_cup, plastic_bottle)
- Note: the current YOLO/COCO mapping only ever emits `plastic_bottle`/`paper_cup`/`unknown` as class_name -- `can` and `glass_bottle` are still included in the known-class set for false_known checks so the rule stays correct if that mapping is extended later.

## Input-size Summary

| input_size | total_images | correct_or_reject | no_detection | misclassified_or_false_known | image_level_success_rate |
|---|---|---|---|---|---|
| 640 | 50 | 21 | 17 | 12 | 42.0% |
| 416 | 50 | 19 | 15 | 16 | 38.0% |
| 320 | 50 | 18 | 18 | 14 | 36.0% |

## Class-level Summary

| input_size | expected_class | total | correct | no_detection | misclassified | correct_reject | false_known | main_detected_classes |
|---|---|---|---|---|---|---|---|---|
| 640 | can | 10 | 0 | 7 | 3 | 0 | 0 | paper_cup:1,plastic_bottle:1,unknown:1 |
| 640 | glass_bottle | 10 | 0 | 2 | 8 | 0 | 0 | unknown:6,paper_cup:6,plastic_bottle:2 |
| 640 | paper_cup | 10 | 8 | 2 | 0 | 0 | 0 | paper_cup:8,unknown:2,plastic_bottle:2 |
| 640 | plastic_bottle | 10 | 3 | 6 | 1 | 0 | 0 | plastic_bottle:3,unknown:3 |
| 640 | unknown | 10 | 0 | 0 | 0 | 10 | 0 | unknown:2 |
| 416 | can | 10 | 0 | 8 | 2 | 0 | 0 | paper_cup:1,unknown:1 |
| 416 | glass_bottle | 10 | 0 | 1 | 9 | 0 | 0 | paper_cup:7,unknown:3,plastic_bottle:2 |
| 416 | paper_cup | 10 | 4 | 2 | 4 | 0 | 0 | paper_cup:4,plastic_bottle:3,unknown:2 |
| 416 | plastic_bottle | 10 | 5 | 4 | 1 | 0 | 0 | plastic_bottle:5,unknown:4 |
| 416 | unknown | 10 | 0 | 0 | 0 | 10 | 0 | unknown:2 |
| 320 | can | 10 | 0 | 9 | 1 | 0 | 0 | unknown:1 |
| 320 | glass_bottle | 10 | 0 | 2 | 8 | 0 | 0 | paper_cup:7,unknown:3,plastic_bottle:2 |
| 320 | paper_cup | 10 | 3 | 3 | 4 | 0 | 0 | paper_cup:3,unknown:2,plastic_bottle:2 |
| 320 | plastic_bottle | 10 | 5 | 4 | 1 | 0 | 0 | plastic_bottle:5,unknown:3 |
| 320 | unknown | 10 | 0 | 0 | 0 | 10 | 0 | unknown:4 |

## Condition-level Summary

| input_size | condition | total | correct_or_reject | failures | failure_rate | common_failure_modes |
|---|---|---|---|---|---|---|
| 640 | angle_view | 4 | 2 | 2 | 50.0% | no_detection:2 |
| 640 | book | 1 | 1 | 0 | 0.0% | none |
| 640 | bottle_cap | 1 | 1 | 0 | 0.0% | none |
| 640 | charger | 1 | 1 | 0 | 0.0% | none |
| 640 | clutter_bg | 4 | 1 | 3 | 75.0% | no_detection:2,misclassified:1 |
| 640 | crushed | 3 | 0 | 3 | 100.0% | no_detection:3 |
| 640 | dark_light | 4 | 1 | 3 | 75.0% | misclassified:2,no_detection:1 |
| 640 | edge_frame | 4 | 0 | 4 | 100.0% | no_detection:3,misclassified:1 |
| 640 | hand | 1 | 1 | 0 | 0.0% | none |
| 640 | hard_case | 4 | 2 | 2 | 50.0% | misclassified:2 |
| 640 | mixed_objects | 1 | 1 | 0 | 0.0% | none |
| 640 | mouse | 1 | 1 | 0 | 0.0% | none |
| 640 | multi_object | 4 | 1 | 3 | 75.0% | misclassified:3 |
| 640 | normal_front | 4 | 1 | 3 | 75.0% | no_detection:2,misclassified:1 |
| 640 | paper_piece | 1 | 1 | 0 | 0.0% | none |
| 640 | partial_cut | 4 | 1 | 3 | 75.0% | no_detection:2,misclassified:1 |
| 640 | plastic_bag | 1 | 1 | 0 | 0.0% | none |
| 640 | reflection | 1 | 0 | 1 | 100.0% | no_detection:1 |
| 640 | small_far | 4 | 2 | 2 | 50.0% | no_detection:1,misclassified:1 |
| 640 | snack_bag | 1 | 1 | 0 | 0.0% | none |
| 640 | tissue | 1 | 1 | 0 | 0.0% | none |
| 416 | angle_view | 4 | 1 | 3 | 75.0% | misclassified:2,no_detection:1 |
| 416 | book | 1 | 1 | 0 | 0.0% | none |
| 416 | bottle_cap | 1 | 1 | 0 | 0.0% | none |
| 416 | charger | 1 | 1 | 0 | 0.0% | none |
| 416 | clutter_bg | 4 | 1 | 3 | 75.0% | misclassified:2,no_detection:1 |
| 416 | crushed | 3 | 0 | 3 | 100.0% | no_detection:2,misclassified:1 |
| 416 | dark_light | 4 | 1 | 3 | 75.0% | misclassified:2,no_detection:1 |
| 416 | edge_frame | 4 | 0 | 4 | 100.0% | no_detection:3,misclassified:1 |
| 416 | hand | 1 | 1 | 0 | 0.0% | none |
| 416 | hard_case | 4 | 1 | 3 | 75.0% | misclassified:2,no_detection:1 |
| 416 | mixed_objects | 1 | 1 | 0 | 0.0% | none |
| 416 | mouse | 1 | 1 | 0 | 0.0% | none |
| 416 | multi_object | 4 | 0 | 4 | 100.0% | misclassified:3,no_detection:1 |
| 416 | normal_front | 4 | 1 | 3 | 75.0% | no_detection:2,misclassified:1 |
| 416 | paper_piece | 1 | 1 | 0 | 0.0% | none |
| 416 | partial_cut | 4 | 2 | 2 | 50.0% | no_detection:1,misclassified:1 |
| 416 | plastic_bag | 1 | 1 | 0 | 0.0% | none |
| 416 | reflection | 1 | 0 | 1 | 100.0% | no_detection:1 |
| 416 | small_far | 4 | 2 | 2 | 50.0% | no_detection:1,misclassified:1 |
| 416 | snack_bag | 1 | 1 | 0 | 0.0% | none |
| 416 | tissue | 1 | 1 | 0 | 0.0% | none |
| 320 | angle_view | 4 | 1 | 3 | 75.0% | misclassified:2,no_detection:1 |
| 320 | book | 1 | 1 | 0 | 0.0% | none |
| 320 | bottle_cap | 1 | 1 | 0 | 0.0% | none |
| 320 | charger | 1 | 1 | 0 | 0.0% | none |
| 320 | clutter_bg | 4 | 0 | 4 | 100.0% | no_detection:2,misclassified:2 |
| 320 | crushed | 3 | 0 | 3 | 100.0% | no_detection:3 |
| 320 | dark_light | 4 | 1 | 3 | 75.0% | no_detection:2,misclassified:1 |
| 320 | edge_frame | 4 | 0 | 4 | 100.0% | no_detection:3,misclassified:1 |
| 320 | hand | 1 | 1 | 0 | 0.0% | none |
| 320 | hard_case | 4 | 1 | 3 | 75.0% | misclassified:2,no_detection:1 |
| 320 | mixed_objects | 1 | 1 | 0 | 0.0% | none |
| 320 | mouse | 1 | 1 | 0 | 0.0% | none |
| 320 | multi_object | 4 | 0 | 4 | 100.0% | misclassified:3,no_detection:1 |
| 320 | normal_front | 4 | 2 | 2 | 50.0% | no_detection:1,misclassified:1 |
| 320 | paper_piece | 1 | 1 | 0 | 0.0% | none |
| 320 | partial_cut | 4 | 1 | 3 | 75.0% | no_detection:2,misclassified:1 |
| 320 | plastic_bag | 1 | 1 | 0 | 0.0% | none |
| 320 | reflection | 1 | 0 | 1 | 100.0% | no_detection:1 |
| 320 | small_far | 4 | 2 | 2 | 50.0% | no_detection:1,misclassified:1 |
| 320 | snack_bag | 1 | 1 | 0 | 0.0% | none |
| 320 | tissue | 1 | 1 | 0 | 0.0% | none |

## Policy-level Summary

| input_size | ACCEPT_SORT | ROUTE_TO_REJECT | SKIP_NO_DETECTION | RETRY_VIEW | MANUAL_REVIEW |
|---|---|---|---|---|---|
| 640 | 14 | 9 | 25 | 0 | 2 |
| 416 | 18 | 9 | 23 | 0 | 0 |
| 320 | 15 | 11 | 24 | 0 | 0 |

Counts come from the [PerceptionPolicy] log line (recycling_cell_vision/perception_policy.py), which runs independently of what actually gets published -- it reflects what a failure-aware policy layer would decide for each image's detections, not the current publish behavior.

## Key Findings

- Least stable classes: **can and glass_bottle** (tied) with an overall 0% correct/correct_reject rate across all input sizes, the lowest of all 5 classes (can=0%, glass_bottle=0%, paper_cup=50%, plastic_bottle=43%, unknown=100%).
- Domain gap confirmed for can and glass_bottle: can 0/30 correct (0%), glass_bottle 0/30 correct (0%). The current YOLO/COCO class mapping (COCO_CLASS_ID_TO_PROJECT_CLASS in vision_perception_node.py) never emits a "can" or "glass_bottle" class_name at all (only plastic_bottle/paper_cup/unknown), so these classes can only ever land on no_detection or misclassified, independent of input_size -- this is a labeling/model coverage gap, not a resolution problem.
- Conditions with the most failures across all input sizes: crushed (9/9 failed, no_detection:8,misclassified:1); edge_frame (12/12 failed, no_detection:9,misclassified:3); reflection (3/3 failed, no_detection:3).
- input_size=320 vs. 640: image-level success rate *dropped* by 6.0 points (36% vs. 42%). Combined with the throughput results in results/vision_benchmark_real_vision_only_summary.md (320 is fastest), this indicates 320 is faster but measurably less stable, so the speed gain has a real detection-quality cost on this dataset.

## Limitations

- No ground-truth bounding boxes -- this measures whether the expected class name shows up anywhere in an image's detections, not localization accuracy (IoU/mAP).
- 10 images per class per condition-family is still a small sample; single misclassified/no_detection images swing a per-condition rate by 100 percentage points at this size.
- can/glass_bottle can structurally never score `correct` today (see Key Findings) -- their no_detection/misclassified counts reflect a class-mapping gap, not necessarily a harder visual case, and should not be read as "the model is bad at bottles".
- confidence_threshold=0.5 and CPUExecutionProvider only; no sweep over threshold or provider in this analysis.
- Each image was only run once per input_size (from the vision_only benchmark), so there is no repeat-run variance data for borderline-confidence detections.

## Next Steps

- Extend the ONNX/COCO class mapping (or fine-tune on the project's own classes) so can/glass_bottle can actually be emitted as class_name, then re-run this analysis to see their real detection stability instead of a guaranteed miss.
- Collect more images per condition (10 is thin) to make the condition-level failure_rate numbers less sensitive to a single image flipping status.
- Add bounding-box ground truth for a subset of images to move from this image-level stability check to a real IoU/mAP evaluation.
- Re-run at additional confidence_threshold values to see whether misclassified/no_detection images are near-miss (just under threshold) or genuinely undetected.
