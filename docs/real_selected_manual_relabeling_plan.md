# real_selected Manual Relabeling Plan

**Status: PLANNING ONLY.** No labeling tool has been installed as part
of this document, no dataset builder changes have been made, and no
training has happened. This documents the workflow before starting it.

## 1. Why 91 images need relabeling

`docs/real_selected_global_label_quality_correction.md` found that
real_selected's GroundingDINO pseudo-labels have bbox placement problems
-- not just in `paper` (see `docs/real_selected_paper_label_quality_
correction.md`) but in `plastic` and `glass` too, confirmed by direct
human review of the bbox overlays against the original photos. The
originally planned fix (`docs/material_v1_real_aug_label_clean_
experiment_plan.md`: keep only pseudo-labels a human marks `KEEP`,
drop the rest) is now considered a fallback rather than the first
choice, because a large enough fraction of the 91 images may need
`RELABEL_NEEDED` rather than a clean `KEEP`/`DROP` split -- discarding
every non-`KEEP` image could throw away most of real_selected's already
-small 91-image pool. Manually drawing correct boxes for all 91 images
directly (rather than filtering GroundingDINO's output) preserves the
full real-photo sample size while guaranteeing bbox correctness, and is
the more thorough fix implied by `docs/real_selected_global_label_
quality_correction.md`'s section 6 next-step framing.

This plan produces YOLO-format bboxes drawn by a human for all 91
`datasets/recycling_material_real_selected/` images, to replace
GroundingDINO's pseudo-labels for this source only (candidates_v1 and
v0_remapped are untouched -- see `docs/material_v1_real_aug_manual_
label_experiment_plan.md`).

## 2. GroundingDINO pseudo-label error types observed

From `results/real_selected_global_label_quality_review_sheet.csv`'s
`issue_type` taxonomy, carried forward here as the error categories a
human relabeler should watch for when deciding whether an old pseudo
-label can be trusted as a starting point or needs to be redrawn from
scratch:

- **box_outside_object** -- the box doesn't overlap the actual target
  object at all
- **wrong_target** -- the box is on a real object in the photo, but not
  the one the class label claims
- **missing_box** -- no bbox was produced at all (GroundingDINO scored
  below threshold)
- **too_much_background** -- the box technically includes the object but
  is loose enough that background dominates the labeled region
- **partial_object** -- the box only covers part of the object (e.g. a
  corner or edge), not the whole thing
- **wrong_class** -- the box is reasonably placed but tagged with the
  wrong material class

## 3. Labeling classes

Same 4-class material taxonomy as every material_v1 variant --
unchanged, no new classes:

| class_id | name    | bin mapping |
|---|---|---|
| 0 | plastic | plastic_bin |
| 1 | metal   | metal_bin   |
| 2 | glass   | glass_bin   |
| 3 | paper   | paper_bin   |

## 4. Per-class labeling criteria

- **plastic**: bottles, containers, clamshells, bags, food packaging
  made of plastic film or rigid plastic. If an object is ambiguous
  between plastic and another material (e.g. a plastic-coated paper
  carton), label by the DOMINANT visible material in the photographed
  view, and use `notes` in the tracker to flag the ambiguity.
- **metal**: cans, foil, metal containers/lids. Crushed/deformed cans
  still count as metal (deformation is not a reason to drop -- see
  `results/review_contact_sheets/real_selected_metal_contact_sheet.jpg`
  for examples already in the pool).
- **glass**: bottles, jars. Watch for reflections/transparency making
  the true object outline hard to see against a similarly light
  background -- draw the box from the object's actual edges, not from
  where reflections/shadows visually extend it.
- **paper**: paper, cardboard boxes, cardboard cups/sleeves. Per
  `docs/real_selected_manual_review_findings.md`'s finding, several
  paper-class photos in this pool visually resemble `test_images_real/
  unknown/` household clutter (plain white cartons, printed-text
  packaging) -- a correct bbox on a real paper object doesn't remove
  that visual-similarity risk, it only removes the bbox-quality
  confound so the two effects can be told apart in the next benchmark.

## 5. Bbox drawing criteria

- **Box the target object completely** -- all visible edges of the
  object should be inside the box, not just the "main body."
- **Minimize background** -- the box should hug the object; don't draw
  a loose box "to be safe."
- **No other objects inside the box** -- if two objects appear in frame,
  box only the one matching the photo's intended class (folder name in
  `datasets/recycling_material_real_selected/<class>/`), and use `notes`
  in the tracker if the second object is a source of ambiguity.
- **One box per image** -- matches every other source in this project's
  YOLO datasets (candidates_v1, v0, real_selected's original pseudo
  -labels all use exactly one box per image).
- **Too ambiguous -> DROP, don't guess** -- if it's genuinely unclear
  what the target object even is, or the object is barely visible/mostly
  occluded, mark it `DROP` in the tracker (`labeling_status=DROP`)
  instead of forcing a box. A dropped image is a smaller loss than a
  wrong label re-teaching the same mistake this whole relabeling effort
  exists to fix.

## 6. Label Studio vs. CVAT

| | Label Studio | CVAT |
|---|---|---|
| Local install | `pip install label-studio` in a venv, single process, browser UI on `localhost:8080` | Typically Docker Compose (multiple services: server, DB, Redis) |
| WSL fit today | Works with a plain pip install, no Docker needed | This machine's Docker Desktop is not WSL-integrated yet (`docker` not on PATH inside WSL) -- would need that set up first |
| YOLO export | Built-in YOLO export format | Built-in YOLO export format, generally considered more mature/battle-tested for CV-specific workflows |
| Setup overhead for 91 images, one labeler | Low -- single `pip install` + one `label-studio start` command | Higher -- multi-container stack is more than this task's scope needs |
| Best fit | Solo/small review batches, quick turnaround | Larger teams, video/interpolation, heavier production pipelines |

**Recommendation for this project: Label Studio.** 91 images, one
reviewer (you), and no current Docker-WSL integration make Label
Studio's single pip-install workflow the lower-friction choice. CVAT is
worth reconsidering later if this project's labeling volume grows
substantially or multiple people start labeling in parallel -- revisit
then, not now.

## 7. Label Studio workflow (this project)

Full install/run/export steps: `docs/label_studio_manual_labeling_
guide.md`. Summary of the workflow shape:

1. Install Label Studio in a dedicated venv (kept separate from
   `.venv-autolabel` and the ROS2 environment -- it's a labeling tool,
   not a training dependency).
2. Start it locally, create one "Object Detection with Bounding Boxes"
   project.
3. Import all 91 `datasets/recycling_material_real_selected/<class>/
   *.jpg` images (import each class subfolder, or all of them together
   -- either way, `results/real_selected_manual_relabeling_tracker.csv`
   is what tracks per-image status, not Label Studio's own folder
   structure).
4. Label every image using section 4/5's criteria above -- one box +
   one class per kept image, mark genuinely unlabelable images to skip
   (tracked as `DROP` in the tracker, not left half-done in Label
   Studio).
5. Export as YOLO format.
6. Update `results/real_selected_manual_relabeling_tracker.csv` per
   image: `labeling_status` (`LABELED`/`DROP`/`REVIEW`), `manual_label_
   path` (where the exported `.txt` ended up), `action`, `issue_type`
   (if relabeling from a specific pseudo-label defect), `notes`.

## 8. YOLO export details

Label Studio's YOLO export writes one `.txt` per labeled image
(`class_id x_center y_center width height`, normalized 0-1 -- same
format this project's other datasets already use) plus a `classes.txt`
listing label names in the order Label Studio assigned them to
`class_id`s. **Verify that order matches this project's taxonomy (`0
plastic, 1 metal, 2 glass, 3 paper`) before using the export** -- Label
Studio assigns `class_id` by the order labels were added to the project
config, which may not match this project's convention unless the
labeling-interface XML is written in that exact order (see `docs/label_
studio_manual_labeling_guide.md`'s label config for the exact XML to
use, which defines them in the correct order already). If `classes.txt`
ever comes out in a different order, remap `class_id`s in the exported
`.txt` files before use rather than assuming they already match.

## 9. Connecting manual labels to a future dataset builder

Not built yet -- `docs/material_v1_real_aug_manual_label_experiment_
plan.md` describes the target dataset composition. When that builder is
eventually written, the intended connection point is:

- For each real_selected image with `labeling_status=LABELED` in
  `results/real_selected_manual_relabeling_tracker.csv`, copy the image
  and its `manual_label_path` `.txt` directly into the new dataset's
  `images/<split>/` and `labels/<split>/` (no GroundingDINO call needed
  for this source anymore -- the whole point of this effort is to stop
  depending on pseudo-labels for real_selected).
- Images with `labeling_status` in (`DROP`, `TODO`, `REVIEW`) are
  excluded from that dataset build until they're `LABELED`.
- Train/val split: reuse the existing `--val-ratio`-style shuffle-and
  -split logic from `tools/build_recycling_material_v1_real_aug_dataset
  .py`, applied to the `LABELED` pool only, with the same seed
  convention for reproducibility.
- candidates_v1/v0_remapped composition stays completely unchanged --
  this only replaces how real_selected's labels are sourced.

## Sources

- `docs/real_selected_global_label_quality_correction.md`
- `docs/real_selected_paper_label_quality_correction.md`
- `results/real_selected_global_label_quality_review_sheet.csv` / `.md`
- `results/real_selected_manual_relabeling_tracker.csv` / `.md` (new)
- `docs/label_studio_manual_labeling_guide.md` (new)
- `docs/material_v1_real_aug_manual_label_experiment_plan.md` (new)
