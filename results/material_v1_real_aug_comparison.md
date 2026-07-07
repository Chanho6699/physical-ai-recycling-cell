# material_v1_real_aug Comparison: baseline vs. medium vs. strong vs. real_aug

## Experiment Purpose

material_v1/aug_medium/aug_strong (see `docs/material_v1_augmentation_experiment_plan.md`) tested whether heavier training-time augmentation alone could close the domain gap between material_v1's training photos and test_images_real/. It helped (strong beat medium beat baseline) but stayed far from reliable (3/50 detections at confidence_threshold=0.5 for strong). This adds a 4th model, `real_aug`: the SAME strong-augmentation recipe, but trained on strong's dataset PLUS a small set of real-camera-style photos (`datasets/recycling_material_real_selected/`, 91 images) added directly to the training set -- see `docs/material_v1_real_aug_experiment_plan.md`.

## Source logs

- baseline: `logs/vision_benchmark/test_images_real/vision_only/yolo11n_recycling_material_v1_640_conf005.log`
- medium: `logs/vision_benchmark/test_images_real/vision_only/yolo11n_recycling_material_v1_aug_medium_640_conf005.log`
- strong: `logs/vision_benchmark/test_images_real/vision_only/yolo11n_recycling_material_v1_aug_strong_640_conf005.log`
- real_aug: `logs/vision_benchmark/test_images_real/vision_only/yolo11n_recycling_material_v1_real_aug_640_conf005.log`
- All four logged at confidence_threshold=0.05 against the same 50-image test_images_real/ set; every other threshold's results below are re-derived by filtering those same raw per-image detections and re-running the actual failure-aware policy (perception_policy.evaluate_detections) at each virtual threshold, not read from a separately re-run log.

## Inference Speed (threshold-independent)

| model | avg_inference_ms | avg_total_ms | avg_fps |
|---|---|---|---|
| baseline | 56.18 | 226.01 | 4.42 |
| medium | 51.43 | 174.98 | 5.72 |
| strong | 26.97 | 117.58 | 8.5 |
| real_aug | 26.36 | 117.09 | 8.54 |

## Validation mAP (final epoch, own held-out split)

| model | val_mAP50 | val_mAP50-95 |
|---|---|---|
| baseline | 0.7686 | 0.7287 |
| medium | 0.7629 | 0.6198 |
| strong | 0.7622 | 0.6135 |
| real_aug | 0.7196 | 0.5448 |

## Images With Detection, by Threshold

| threshold | baseline | medium | strong | real_aug |
|---|---|---|---|---|
| 0.5 | 0/50 (0.0%) | 2/50 (4.0%) | 3/50 (6.0%) | 0/50 (0.0%) |
| 0.3 | 2/50 (4.0%) | 4/50 (8.0%) | 4/50 (8.0%) | 5/50 (10.0%) |
| 0.1 | 2/50 (4.0%) | 4/50 (8.0%) | 4/50 (8.0%) | 8/50 (16.0%) |
| 0.05 | 2/50 (4.0%) | 4/50 (8.0%) | 4/50 (8.0%) | 8/50 (16.0%) |

## No-detection Count, by Threshold

| threshold | baseline | medium | strong | real_aug |
|---|---|---|---|---|
| 0.5 | 50 | 48 | 47 | 50 |
| 0.3 | 48 | 46 | 46 | 45 |
| 0.1 | 48 | 46 | 46 | 42 |
| 0.05 | 48 | 46 | 46 | 42 |

## Predicted-class Distribution, by Threshold

**threshold=0.5**

| model | distribution |
|---|---|
| baseline | no_detection:50 |
| medium | no_detection:48;metal:1;glass:1 |
| strong | no_detection:47;metal:2;plastic:1 |
| real_aug | no_detection:50 |

**threshold=0.3**

| model | distribution |
|---|---|
| baseline | no_detection:48;metal:1;plastic:1 |
| medium | no_detection:46;glass:2;metal:1;plastic:1 |
| strong | no_detection:46;metal:2;plastic:1;paper:1 |
| real_aug | no_detection:45;paper:3;plastic:2 |

**threshold=0.1**

| model | distribution |
|---|---|
| baseline | no_detection:48;metal:1;plastic:1 |
| medium | no_detection:46;glass:2;metal:1;plastic:1 |
| strong | no_detection:46;metal:2;plastic:1;paper:1 |
| real_aug | no_detection:42;paper:5;plastic:3 |

**threshold=0.05**

| model | distribution |
|---|---|
| baseline | no_detection:48;metal:1;plastic:1 |
| medium | no_detection:46;glass:2;metal:1;plastic:1 |
| strong | no_detection:46;metal:2;plastic:1;paper:1 |
| real_aug | no_detection:42;paper:5;plastic:3 |

## Correct-case Counts (v0-remap targets), by Threshold

| threshold | model | can->metal | glass_bottle->glass | plastic_bottle->plastic | paper_cup->paper |
|---|---|---|---|---|---|
| 0.5 | baseline | 0/10 | 0/10 | 0/10 | 0/10 |
| 0.5 | medium | 1/10 | 0/10 | 0/10 | 0/10 |
| 0.5 | strong | 2/10 | 0/10 | 0/10 | 0/10 |
| 0.5 | real_aug | 0/10 | 0/10 | 0/10 | 0/10 |
| 0.3 | baseline | 1/10 | 0/10 | 0/10 | 0/10 |
| 0.3 | medium | 1/10 | 0/10 | 0/10 | 0/10 |
| 0.3 | strong | 2/10 | 0/10 | 0/10 | 1/10 |
| 0.3 | real_aug | 0/10 | 0/10 | 1/10 | 0/10 |
| 0.1 | baseline | 1/10 | 0/10 | 0/10 | 0/10 |
| 0.1 | medium | 1/10 | 0/10 | 0/10 | 0/10 |
| 0.1 | strong | 2/10 | 0/10 | 0/10 | 1/10 |
| 0.1 | real_aug | 0/10 | 0/10 | 1/10 | 0/10 |
| 0.05 | baseline | 1/10 | 0/10 | 0/10 | 0/10 |
| 0.05 | medium | 1/10 | 0/10 | 0/10 | 0/10 |
| 0.05 | strong | 2/10 | 0/10 | 0/10 | 1/10 |
| 0.05 | real_aug | 0/10 | 0/10 | 1/10 | 0/10 |

## Unsafe ACCEPT_SORT on unknown Ground Truth, by Threshold

| threshold | baseline | medium | strong | real_aug |
|---|---|---|---|---|
| 0.5 | 0/10 | 0/10 | 0/10 | 0/10 |
| 0.3 | 1/10 | 0/10 | 0/10 | 2/10 |
| 0.1 | 1/10 | 0/10 | 0/10 | 4/10 |
| 0.05 | 1/10 | 0/10 | 0/10 | 4/10 |

## Class Collapse Check, by Threshold

| threshold | model | dominant_class | dominant_share | collapse_flag |
|---|---|---|---|---|
| 0.5 | baseline | no_detection | 0% | False |
| 0.5 | medium | metal | 50% | False |
| 0.5 | strong | metal | 67% | False |
| 0.5 | real_aug | no_detection | 0% | False |
| 0.3 | baseline | metal | 50% | False |
| 0.3 | medium | glass | 50% | False |
| 0.3 | strong | metal | 50% | False |
| 0.3 | real_aug | paper | 60% | False |
| 0.1 | baseline | metal | 50% | False |
| 0.1 | medium | glass | 50% | False |
| 0.1 | strong | metal | 50% | False |
| 0.1 | real_aug | paper | 62% | False |
| 0.05 | baseline | metal | 50% | False |
| 0.05 | medium | glass | 50% | False |
| 0.05 | strong | metal | 50% | False |
| 0.05 | real_aug | paper | 62% | False |

`collapse_flag` only triggers with >=5 non-empty detections and a dominant share >=70% -- with this few real detections overall, most cells here are too small a sample for the flag to be meaningful and are left `False` by design rather than reporting a spurious 100%-of-1 "collapse".

## Interpretation

### 1. Did adding real_selected increase detection count?

real_aug vs. baseline (summed across all 4 thresholds): images_with_detection 6 -> 21 (+15). real_aug vs. strong (the more relevant comparison, since both share the same augmentation strength and base dataset -- the only difference is real_selected): images_with_detection 15 -> 21 (+6).

### 2. Did real_aug beat strong (augmentation alone) on real photos?

correct v0-remap-target predictions (can->metal + glass_bottle->glass + plastic_bottle->plastic), summed across all 4 thresholds: strong=8, real_aug=3. detection count: strong=15, real_aug=21. real_aug does NOT clearly beat strong on both counts -- see the per-threshold tables above for where it does/doesn't.

### 3. Did unsafe unknown->ACCEPT_SORT increase?

unsafe unknown->ACCEPT_SORT, summed across all 4 thresholds: baseline=3, strong=0, real_aug=10. real_aug INCREASED unsafe accepts versus strong -- worth checking which unknown images flipped to ACCEPT_SORT before shipping this model.

### 4. Did a new class collapse appear?

class_collapse observed anywhere across thresholds: baseline=False, strong=False, real_aug=False. No collapse in real_aug.

### 5. Validation mAP vs. real-image detection: same direction or not?

val_mAP50: baseline=0.7686, medium=0.7629, strong=0.7622, real_aug=0.7196. real-image detection_sum (summed across thresholds): baseline=6, medium=14, strong=15, real_aug=21. The two metrics rank the four models in a DIFFERENT order -- validation mAP on material_v1's own held-out split is NOT a reliable proxy for real-image detection here, consistent with the original material_v1 failure mode (strong validation mAP, near-zero real-world recall). Use the real-image numbers above, not validation mAP, to judge which model is actually better for test_images_real/-like conditions.

### 6. Next steps

- real_aug detects MORE than strong (images_with_detection 15 -> 21) but this is NOT an unambiguous win: correct v0-remap-target predictions went 8 -> 3 and unsafe unknown->ACCEPT_SORT went 0 -> 10. The extra detections are disproportionately WRONG-class or false-positive-on-unknown, not more correct sorts -- do NOT treat higher detection_sum alone as "real_aug is better." Before collecting more real_selected data, first check `datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_*` for bad GroundingDINO boxes (unreviewed pseudo-labels) and check which specific `unknown/` images in test_images_real/ flipped to ACCEPT_SORT (see the per-threshold unsafe-accept table above) -- a few noisy real_selected labels teaching the model to fire more readily on background clutter is a plausible cause given real_selected images were shot with more background clutter than candidates_v1.
- Unsafe accepts increased and/or class collapse appeared -- do NOT deploy real_aug past benchmarking as-is. Consider tightening `policy_confidence_threshold` (see `ros2_ws/src/recycling_cell_vision/recycling_cell_vision/perception_policy.py`) and re-running this comparison, or reverting to `strong` (0 unsafe accepts) until real_aug's false-positive-on-unknown behavior is understood.

## Limitations

- Real detection counts are very low in absolute terms (single digits out of 50 images even at threshold=0.05) -- every metric above is working with a small sample, so differences of 1-2 detections should not be over-interpreted as a robust trend.
- Each model was trained once (no repeated runs), so there is no variance estimate to know whether a different random seed would shift these results as much as adding real_selected did.
- ground_truth is test_images_real/'s folder name compared to predicted_class as a plain string via the v0-remap mapping -- no bounding-box IoU is checked, only whether the right class name appears at all.
- real_selected's own labels are unreviewed GroundingDINO pseudo-labels (same caveat as candidates_v1/v0) -- a real_aug result that looks worse than expected could be a labeling-quality artifact, not evidence that real photos don't help in general.
