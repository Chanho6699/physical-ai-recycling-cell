# material_v1_real_aug_manual_label Experiment Plan

**Status: PLANNED, NOT STARTED.** No dataset builder for this experiment
exists yet, no dataset has been built, and no training has happened.
This document fixes the design ahead of time, to be executed once
`results/real_selected_manual_relabeling_tracker.csv` shows real_selected
images with `labeling_status=LABELED` (see `docs/real_selected_manual_
relabeling_plan.md` and `docs/label_studio_manual_labeling_guide.md` for
that workflow).

## Where this picks up

`docs/real_selected_global_label_quality_correction.md` found real
bbox-placement defects (not just in `paper`, also `plastic`/`glass`) in
real_selected's GroundingDINO pseudo-labels. Rather than filtering the
pseudo-labels down to a `KEEP` subset (`docs/material_v1_real_aug_label_
clean_experiment_plan.md`, now a fallback -- see `docs/real_selected_
manual_relabeling_plan.md` section 1 for why), this experiment replaces
real_selected's labels entirely with human-drawn YOLO boxes, preserving
the full 91-image sample instead of shrinking it to whatever fraction
happens to already have a correct pseudo-label.

## Design

- **candidates_v1: unchanged from `material_v1` baseline.** Same
  `--limit-per-class 100`, seed, and prompts as every material_v1
  variant so far -- this experiment isolates "real_selected label
  source" as the one variable, not candidates_v1 composition.
- **v0_remapped: unchanged.** Same reuse-as-is approach as
  `material_v1`/`aug_medium`/`aug_strong`/`real_aug`. v0 `paper` stays
  excluded by default, same as every variant since `aug_strong`.
- **real_selected: human-drawn YOLO labels, not GroundingDINO
  pseudo-labels.** Source: `results/real_selected_manual_relabeling_
  tracker.csv` rows with `labeling_status=LABELED`, using each row's
  `manual_label_path` for the box instead of running GroundingDINO on
  that image at all.
- **Images with no manual label are excluded.** Any real_selected image
  still `TODO`, `DROP`, or `REVIEW` in the tracker at build time is left
  out of this dataset entirely -- no pseudo-label fallback for excluded
  images (the whole point is to stop depending on pseudo-labels for this
  source).
- **Taxonomy unchanged:** `0=plastic, 1=metal, 2=glass, 3=paper` -- no
  ROS2 vision-node changes needed (same compatibility note as every
  other material_v1 variant, see `docs/material_v1_real_aug_experiment_
  plan.md`'s ROS2 section).
- **Training hyperparameters:** same strong-augmentation recipe as
  `real_aug`/`aug_strong`, so any benchmark delta is attributable to the
  real_selected label-source change, not augmentation strength.

## Planned outputs (none created yet)

| artifact | path |
|---|---|
| dataset | `datasets/recycling_yolo_material_v1_real_aug_manual_label/` |
| ONNX model | `models/yolo11n_recycling_material_v1_real_aug_manual_label_640.onnx` |
| comparison CSV | `results/material_v1_real_aug_manual_label_comparison.csv` |
| comparison MD | `results/material_v1_real_aug_manual_label_comparison.md` |

Dataset/model paths follow this project's existing naming convention
(distinct from `material_v1_real_aug` and `material_v1_real_aug_label_
clean` so none of the three overwrite each other) -- same pattern as how
`aug_medium`/`aug_strong`/`real_aug` each got their own `runs/recycling_
yolo/<name>/` and `models/*.onnx` names.

## What still needs to be built (not done yet)

1. A new dataset builder (or an option added to `tools/build_recycling_
   material_v1_real_aug_dataset.py`) that:
   - Reuses that script's candidates_v1/v0_remapped logic unchanged.
   - For real_selected, reads `results/real_selected_manual_relabeling_
     tracker.csv`, filters to `labeling_status=LABELED`, and copies each
     image + its `manual_label_path` `.txt` directly into `images/
     <split>/` and `labels/<split>/` -- no GroundingDINO call for this
     source.
   - Applies the same train/val split convention (seed, ratio) to the
     `LABELED` pool.
2. `tools/train_recycling_yolo_material_v1_real_aug_manual_label.sh` --
   copy of `tools/train_recycling_yolo_material_v1_real_aug.sh` pointed
   at the new dataset yaml and run name.
3. `tools/export_recycling_yolo_material_v1_real_aug_manual_label_onnx
   .sh` -- copy of the `real_aug` export script pointed at the new run.
4. Extend `tools/analyze_material_v1_real_aug_comparison.py` (or a new
   sibling script, matching how the 3-way script was extended to 4-way
   for `real_aug`) to a 5-way comparison including this model.

None of the above exists yet -- this section is the checklist for when
the labeling tracker shows enough `LABELED` rows to make the experiment
worth running.

## Success criteria (same benchmark methodology as `results/material_v1_
real_aug_comparison.md`: confidence_threshold in {0.5, 0.3, 0.1, 0.05}
against `test_images_real/`)

`material_v1_real_aug_manual_label` is a **success** if:

1. **threshold=0.5: unsafe `unknown`->`ACCEPT_SORT` = 0** (hard
   requirement, matching `strong`'s and the production default's
   existing safety bar).
2. **threshold=0.3: unsafe `unknown`->`ACCEPT_SORT` is LOWER than real_
   aug's** (real_aug had 2 at this threshold).
3. **Correct v0-remap-target predictions (can->metal + glass_bottle->
   glass + plastic_bottle->plastic) are HIGHER than real_aug's** (3,
   summed across all 4 thresholds).
4. **Stretch goal:** correct-case count approaches or exceeds `strong`'s
   8 (summed across all 4 thresholds).

**Failure condition, explicit:** if `images_with_detection` increases
relative to `real_aug`/`strong` but unsafe `unknown`->`ACCEPT_SORT` does
NOT improve (flat or worse), that's a failure -- it would mean even
correctly-drawn human labels didn't fix the regression, which would
point the causal analysis back toward the visual-confusion hypothesis
(`docs/real_selected_manual_review_findings.md` Findings 2-3) or toward
needing negative/`unknown`-class training data (Hypothesis B in the same
doc) as the actual primary lever.

## Relationship to `material_v1_real_aug_label_clean`

These are two different experiments testing overlapping but distinct
hypotheses, both currently unstarted:

- `real_aug_label_clean` (`docs/material_v1_real_aug_label_clean_
  experiment_plan.md`): filters existing pseudo-labels down to a
  human-verified `KEEP` subset. Cheaper (no manual bbox drawing), but
  shrinks the real-photo sample size to whatever fraction already has a
  correct pseudo-label.
- `real_aug_manual_label` (this document): replaces pseudo-labels with
  human-drawn boxes for every usable image. More expensive (91 images
  to hand-label), but preserves sample size and removes labeling-tool
  error as a variable entirely.

Whichever finishes review/labeling first, or whichever result is more
informative once run, can inform whether the other is worth pursuing
too -- they are not mutually exclusive, but there's no need to run both
simultaneously before seeing either result.

## Sources

- `docs/real_selected_manual_relabeling_plan.md`
- `docs/label_studio_manual_labeling_guide.md`
- `results/real_selected_manual_relabeling_tracker.csv` / `.md`
- `docs/real_selected_global_label_quality_correction.md`
- `docs/material_v1_real_aug_label_clean_experiment_plan.md`
- `results/material_v1_real_aug_comparison.md`
- `results/real_aug_unsafe_unknown_cases.md`
