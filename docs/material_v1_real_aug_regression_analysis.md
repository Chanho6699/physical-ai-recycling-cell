# real_aug Regression Root-cause Analysis

## Context

`results/material_v1_real_aug_comparison.md` found that `real_aug`
(strong-augmentation training + 91 real_selected photos) detects more on
`test_images_real/` than `strong` (augmentation alone) but is **not** a
clean win: correct v0-remap-target predictions dropped (strong=8,
real_aug=3, summed across 4 thresholds) and unsafe
`unknown`->`ACCEPT_SORT` cases jumped from 0 to 10. **Read this as:
unreviewed real pseudo-labels increased raw detection count but degraded
correctness and safety -- not as "real_aug is an improvement."** This doc
synthesizes the three follow-up analyses that investigate why, and
recommends the next experiment.

## 1. Where the unsafe accepts come from

Full detail: `results/real_aug_unsafe_unknown_cases.csv` /
`.md` (built by `tools/analyze_real_aug_unsafe_unknown_cases.py`).

- All 10 unsafe cases come from just **4 unique `unknown/` images**:
  `unknown_bottle_cap_003`, `unknown_charger_007`, `unknown_snack_bag_002`,
  `unknown_tissue_001` -- none of which contain a plastic/metal/glass/
  paper object.
- Predicted-class skew: **paper=7 (70%), plastic=3 (30%)**. Paper is the
  class most responsible for the regression.
- **Every single unsafe case is below confidence_threshold=0.5** (the
  project's own default) -- the highest confidence observed among all 10
  cases is 0.44 (`unknown_tissue_001` -> paper). At threshold=0.5,
  real_aug produces the same 0 unsafe accepts as `strong` does.
- Practical implication: this specific regression, as benchmarked, is
  **not a threshold-independent failure** -- it only shows up when
  `confidence_threshold`/`policy_confidence_threshold` is lowered to
  0.3/0.1/0.05 for benchmarking sensitivity. It would not have been
  visible at the production default at all.

## 2. Is it a labeling-quality problem?

Full detail: `results/real_selected_pseudolabel_quality.csv` / `.md`
(built by `tools/analyze_real_selected_pseudolabel_quality.py`).

- **0 of 91** real_selected images triggered `too_large_bbox`,
  `too_small_bbox`, or `low_confidence` -- the three hard quality-warning
  flags this project uses everywhere else (candidates_v1, v0). By that
  measure, real_selected's pseudo-labels do **not** look obviously bad.
- 18/91 (20%) are `boundary_clamped` (box touched the image edge) --
  expected and mostly benign for close-up real photos, not a sign of a
  wrong box.
- Softer signal worth noting: **paper has the lowest avg_confidence
  (0.7478) and the largest avg_bbox_area_ratio (0.3243) of the four
  classes** (plastic/metal are both ~0.93 confidence, glass 0.78). This
  doesn't cross any hard warning threshold, but it's directionally
  consistent with section 1's finding that `paper` is the class driving
  the false positives -- real_selected's paper boxes are the loosest and
  least-confident of the four classes, even though none are flagged.
- **Conclusion: this does not look like a "bad bounding boxes" problem**
  in the way `too_large_bbox`/`low_confidence` would normally catch. The
  more likely mechanism is a **generalization effect**: real_selected's
  photos have more background clutter than candidates_v1 by design (that
  was the whole point -- they were shot to resemble test_images_real/'s
  domain), and training on them likely taught the model to fire more
  readily on textures/clutter that resemble a real background, not just
  cleaner GroundingDINO boxes elsewhere in the same photos.

## 3. What to manually check first

Full detail: `results/real_selected_manual_review_list.csv` / `.md`
(built by `tools/build_real_selected_manual_review_list.py`).

Since tiers 1-2 (too_large_bbox / low_confidence) are empty, the two
tiers actually worth a human look are:
- **Tier 4 (10 images):** bbox_area_ratio > 0.35 but below the
  too_large_bbox cutoff -- large-but-not-flagged boxes, the closest thing
  to a "soft too_large_bbox" in this dataset.
- **Tier 5 (30 images):** `paper`/`plastic` class -- the two classes
  linked to the unsafe-accept regression in section 1, independent of
  whether their individual boxes look flagged.

## 4. Recommended next experiment

Ranking the five candidates against the evidence above:

- **(D) tighten confidence_threshold/policy alone** would suppress every
  unsafe case found here (section 1), but is a mitigation, not a fix --
  it doesn't address the correctness drop (8->3 correct predictions) and
  a stricter threshold available today doesn't change what the model
  itself learned.
- **(C) reject/negative training or policy calibration** using
  unknown-domain images is a reasonable medium-term direction (it
  directly targets "don't fire on non-target clutter"), but is a bigger
  scope change than this dataset's regression currently justifies
  investigating.
- **(E) manual bbox labeling of all real_selected** is the most thorough
  fix but the most expensive for a 91-image proof-of-concept batch --
  premature before knowing whether the automated pseudo-labels are even
  the problem (section 2 suggests they largely aren't, by the hard-flag
  measure).
- **(B) downweight/cap real_selected per class** is plausible (91/~790
  images = ~11.5% of the training set is a lot of influence for a small,
  stylistically distinct source) but is a blunt instrument without first
  checking whether specific images are the actual cause.
- **(A) keep only GOOD-bbox real_selected images, retrain as
  real_aug_clean`** is now directly actionable: `results/real_selected_
  manual_review_list.md`'s tier 4 (10 images) and tier 5 (30 images,
  mostly paper) are exactly the images to eyeball first. This is the
  cheapest next step (no new data collection, no policy/infra change) and
  directly tests section 2's hypothesis -- if dropping a handful of
  loose/generic paper images from real_selected removes the unsafe-accept
  regression while keeping real_aug's detection-count gain, that
  confirms a small-bad-sample effect; if it doesn't, that's evidence
  favoring the generalization explanation instead (pointing toward B or
  C next).

**Recommendation: (A), with (D) as an immediate, low-cost complementary
mitigation.** Concretely:
1. Review tier 4 + tier 5 from `results/real_selected_manual_review_list
   .md` (40 images, previews for the 10-per-class sample already exist
   under `datasets/recycling_yolo_material_v1_real_aug/previews/
   real_selected_*`; the rest need a manual bbox check against their
   source image).
2. Drop/relabel any image whose box looks wrong, rebuild the dataset
   (`tools/build_recycling_material_v1_real_aug_dataset.py` supports
   re-running against a filtered `real_selected/` source folder, or add
   a manual exclude-list option if this becomes a recurring step).
3. Retrain as `real_aug_clean`, re-run `tools/analyze_material_v1_real_
   aug_comparison.py` (extend to 5 models) to check whether the
   correctness drop and unsafe-accept increase both resolve.
4. In parallel, since `confidence_threshold=0.5` already suppresses every
   unsafe case found so far, do not lower policy thresholds below the
   project default in production regardless of which retrain option is
   pursued next.

## Sources

- `results/material_v1_real_aug_comparison.md` -- the original 4-way
  comparison that surfaced this regression
- `results/real_aug_unsafe_unknown_cases.csv` / `.md`
- `results/real_selected_pseudolabel_quality.csv` / `.md`
- `results/real_selected_manual_review_list.csv` / `.md`
