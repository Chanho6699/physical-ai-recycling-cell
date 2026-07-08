# real_selected Manual Relabeling Tracker

Progress tracker for manually relabeling all 91 real_selected images in Label Studio -- see `docs/real_selected_manual_relabeling_plan.md` for the full plan and `docs/label_studio_manual_labeling_guide.md` for the labeling tool steps. Every row starts at `labeling_status=TODO`, `action=RELABEL` -- update per image as you work through Label Studio's queue.

**Re-running `tools/build_real_selected_manual_relabeling_tracker.py` OVERWRITES this file back to the all-TODO starting state -- do not re-run it once you've started filling this in, it does not merge with existing progress.**

## Column meanings

- `labeling_status`: `TODO`, `LABELED`, `DROP`, `REVIEW` -- `LABELED` once you've exported a YOLO box for this image from Label Studio and filled in `manual_label_path`; `DROP` if you decided (per `docs/real_selected_manual_relabeling_plan.md` section 5) the image is too ambiguous to label; `REVIEW` if you want a second look before deciding.
- `manual_label_path`: path to the exported YOLO `.txt` for this image once `labeling_status=LABELED` (see `docs/label_studio_manual_labeling_guide.md` section 7 for where exports land). Blank until then.
- `action`: defaults to `RELABEL` for every row (that's the point of this tracker). Change if you decide otherwise for a specific image (e.g. `DROP`).
- `issue_type`: optional -- if you know which GroundingDINO pseudo-label defect prompted the relabel (see `docs/real_selected_manual_relabeling_plan.md` section 2's categories), note it here for later analysis.
- `notes`: free text, e.g. class ambiguity, multiple objects in frame, anything worth remembering later.

## Per-class counts

| class | total images | TODO |
|---|---|---|
| plastic | 35 | 35 |
| metal | 24 | 24 |
| glass | 10 | 10 |
| paper | 22 | 22 |

## Tracker

| review_id | class_name | labeling_status | manual_label_path | action | issue_type | notes |
|---|---|---|---|---|---|---|
| glass_000041 | glass | TODO |  | RELABEL |  |  |
| glass_000042 | glass | TODO |  | RELABEL |  |  |
| glass_000161 | glass | TODO |  | RELABEL |  |  |
| glass_000162 | glass | TODO |  | RELABEL |  |  |
| glass_000163 | glass | TODO |  | RELABEL |  |  |
| glass_000164 | glass | TODO |  | RELABEL |  |  |
| glass_000165 | glass | TODO |  | RELABEL |  |  |
| glass_000166 | glass | TODO |  | RELABEL |  |  |
| glass_000167 | glass | TODO |  | RELABEL |  |  |
| glass_000168 | glass | TODO |  | RELABEL |  |  |
| metal_000041 | metal | TODO |  | RELABEL |  |  |
| metal_000042 | metal | TODO |  | RELABEL |  |  |
| metal_000043 | metal | TODO |  | RELABEL |  |  |
| metal_000044 | metal | TODO |  | RELABEL |  |  |
| metal_000045 | metal | TODO |  | RELABEL |  |  |
| metal_000161 | metal | TODO |  | RELABEL |  |  |
| metal_000162 | metal | TODO |  | RELABEL |  |  |
| metal_000163 | metal | TODO |  | RELABEL |  |  |
| metal_000164 | metal | TODO |  | RELABEL |  |  |
| metal_000165 | metal | TODO |  | RELABEL |  |  |
| metal_000166 | metal | TODO |  | RELABEL |  |  |
| metal_000167 | metal | TODO |  | RELABEL |  |  |
| metal_000168 | metal | TODO |  | RELABEL |  |  |
| metal_000169 | metal | TODO |  | RELABEL |  |  |
| metal_000170 | metal | TODO |  | RELABEL |  |  |
| metal_000171 | metal | TODO |  | RELABEL |  |  |
| metal_000172 | metal | TODO |  | RELABEL |  |  |
| metal_000173 | metal | TODO |  | RELABEL |  |  |
| metal_000174 | metal | TODO |  | RELABEL |  |  |
| metal_000175 | metal | TODO |  | RELABEL |  |  |
| metal_000176 | metal | TODO |  | RELABEL |  |  |
| metal_000177 | metal | TODO |  | RELABEL |  |  |
| metal_000178 | metal | TODO |  | RELABEL |  |  |
| metal_000179 | metal | TODO |  | RELABEL |  |  |
| paper_000021 | paper | TODO |  | RELABEL |  |  |
| paper_000022 | paper | TODO |  | RELABEL |  |  |
| paper_000023 | paper | TODO |  | RELABEL |  |  |
| paper_000024 | paper | TODO |  | RELABEL |  |  |
| paper_000081 | paper | TODO |  | RELABEL |  |  |
| paper_000082 | paper | TODO |  | RELABEL |  |  |
| paper_000083 | paper | TODO |  | RELABEL |  |  |
| paper_000084 | paper | TODO |  | RELABEL |  |  |
| paper_000085 | paper | TODO |  | RELABEL |  |  |
| paper_000086 | paper | TODO |  | RELABEL |  |  |
| paper_000087 | paper | TODO |  | RELABEL |  |  |
| paper_000088 | paper | TODO |  | RELABEL |  |  |
| paper_000089 | paper | TODO |  | RELABEL |  |  |
| paper_000090 | paper | TODO |  | RELABEL |  |  |
| paper_000091 | paper | TODO |  | RELABEL |  |  |
| paper_000092 | paper | TODO |  | RELABEL |  |  |
| paper_000093 | paper | TODO |  | RELABEL |  |  |
| paper_000094 | paper | TODO |  | RELABEL |  |  |
| paper_000095 | paper | TODO |  | RELABEL |  |  |
| paper_000096 | paper | TODO |  | RELABEL |  |  |
| paper_000097 | paper | TODO |  | RELABEL |  |  |
| paper_000098 | paper | TODO |  | RELABEL |  |  |
| plastic_000041 | plastic | TODO |  | RELABEL |  |  |
| plastic_000042 | plastic | TODO |  | RELABEL |  |  |
| plastic_000043 | plastic | TODO |  | RELABEL |  |  |
| plastic_000044 | plastic | TODO |  | RELABEL |  |  |
| plastic_000045 | plastic | TODO |  | RELABEL |  |  |
| plastic_000046 | plastic | TODO |  | RELABEL |  |  |
| plastic_000047 | plastic | TODO |  | RELABEL |  |  |
| plastic_000161 | plastic | TODO |  | RELABEL |  |  |
| plastic_000162 | plastic | TODO |  | RELABEL |  |  |
| plastic_000163 | plastic | TODO |  | RELABEL |  |  |
| plastic_000164 | plastic | TODO |  | RELABEL |  |  |
| plastic_000165 | plastic | TODO |  | RELABEL |  |  |
| plastic_000166 | plastic | TODO |  | RELABEL |  |  |
| plastic_000167 | plastic | TODO |  | RELABEL |  |  |
| plastic_000168 | plastic | TODO |  | RELABEL |  |  |
| plastic_000169 | plastic | TODO |  | RELABEL |  |  |
| plastic_000170 | plastic | TODO |  | RELABEL |  |  |
| plastic_000171 | plastic | TODO |  | RELABEL |  |  |
| plastic_000172 | plastic | TODO |  | RELABEL |  |  |
| plastic_000173 | plastic | TODO |  | RELABEL |  |  |
| plastic_000174 | plastic | TODO |  | RELABEL |  |  |
| plastic_000175 | plastic | TODO |  | RELABEL |  |  |
| plastic_000176 | plastic | TODO |  | RELABEL |  |  |
| plastic_000177 | plastic | TODO |  | RELABEL |  |  |
| plastic_000178 | plastic | TODO |  | RELABEL |  |  |
| plastic_000179 | plastic | TODO |  | RELABEL |  |  |
| plastic_000180 | plastic | TODO |  | RELABEL |  |  |
| plastic_000181 | plastic | TODO |  | RELABEL |  |  |
| plastic_000182 | plastic | TODO |  | RELABEL |  |  |
| plastic_000183 | plastic | TODO |  | RELABEL |  |  |
| plastic_000184 | plastic | TODO |  | RELABEL |  |  |
| plastic_000185 | plastic | TODO |  | RELABEL |  |  |
| plastic_000186 | plastic | TODO |  | RELABEL |  |  |
| plastic_000187 | plastic | TODO |  | RELABEL |  |  |
| plastic_000188 | plastic | TODO |  | RELABEL |  |  |

Once every row is `LABELED` or `DROP` (none left `TODO`/`REVIEW`), `docs/material_v1_real_aug_manual_label_experiment_plan.md` describes the planned next experiment -- not started yet.
