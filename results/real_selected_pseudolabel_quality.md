# real_selected Pseudo-label Quality Summary

Source: `datasets/recycling_yolo_material_v1_real_aug/autolabel_report.csv`, filtered to `source_type=real_selected` (91 images: plastic 35 / metal 24 / paper 22 / glass 10). Written to check whether real_selected's GroundingDINO pseudo-labels look obviously bad BEFORE assuming they explain real_aug's unsafe-accept regression (see results/real_aug_unsafe_unknown_cases.md for that side of the analysis).

## Per-class Summary

| class | total | labeled | no_box | excluded | error | too_large_bbox | too_small_bbox | low_confidence | boundary_clamped | avg_bbox_area_ratio | avg_confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| plastic | 35 | 35 | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 0.2867 | 0.9291 |
| metal | 24 | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.1238 | 0.9331 |
| glass | 10 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0.1311 | 0.7769 |
| paper | 22 | 22 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 0.3243 | 0.7478 |

**Headline:** 0 of 91 real_selected images triggered too_large_bbox/too_small_bbox/low_confidence (the three quality-risk flags), plus 18 boundary_clamped (box touched the image edge -- common and not inherently bad for a close-up real photo). This is a low warning rate -- real_selected's pseudo-label BBOXES do not look obviously bad by these metrics, which means the unsafe-accept regression is more likely a MODEL GENERALIZATION effect (training on cluttered/backgroundy real photos made the model fire more readily on similar clutter in test_images_real/unknown/) than a labeling-error effect -- see results/real_aug_unsafe_unknown_cases.md.

## Top 20 by bbox_area_ratio (largest boxes)

| output_image | target_class | bbox_area_ratio | confidence | warning_flags |
|---|---|---|---|---|
| paper_000093.jpg | paper | 0.7441 | 0.6674 | boundary_clamped |
| paper_000098.jpg | paper | 0.6383 | 0.6632 | boundary_clamped |
| plastic_000179.jpg | plastic | 0.6171 | 0.9498 | boundary_clamped |
| plastic_000172.jpg | plastic | 0.5604 | 0.7852 | (none) |
| plastic_000167.jpg | plastic | 0.5245 | 0.9462 | (none) |
| plastic_000162.jpg | plastic | 0.4974 | 0.9594 | (none) |
| paper_000097.jpg | paper | 0.4909 | 0.7832 | boundary_clamped |
| plastic_000171.jpg | plastic | 0.4556 | 0.9371 | boundary_clamped |
| plastic_000170.jpg | plastic | 0.4509 | 0.9132 | (none) |
| paper_000021.jpg | paper | 0.4254 | 0.7190 | boundary_clamped |
| paper_000086.jpg | paper | 0.4107 | 0.7756 | (none) |
| paper_000087.jpg | paper | 0.3935 | 0.8215 | (none) |
| paper_000096.jpg | paper | 0.3912 | 0.6532 | (none) |
| plastic_000183.jpg | plastic | 0.3861 | 0.9195 | boundary_clamped |
| paper_000091.jpg | paper | 0.3847 | 0.7502 | (none) |
| plastic_000168.jpg | plastic | 0.3769 | 0.9328 | boundary_clamped |
| paper_000082.jpg | paper | 0.3736 | 0.4504 | boundary_clamped |
| plastic_000165.jpg | plastic | 0.3647 | 0.9433 | (none) |
| paper_000024.jpg | paper | 0.3562 | 0.8401 | (none) |
| plastic_000184.jpg | plastic | 0.3540 | 0.9524 | boundary_clamped |

## Bottom 20 by confidence (least confident)

| output_image | target_class | confidence | bbox_area_ratio | warning_flags |
|---|---|---|---|---|
| paper_000082.jpg | paper | 0.4504 | 0.3736 | boundary_clamped |
| glass_000162.jpg | glass | 0.4960 | 0.1233 | (none) |
| glass_000163.jpg | glass | 0.6096 | 0.0726 | (none) |
| paper_000096.jpg | paper | 0.6532 | 0.3912 | (none) |
| paper_000081.jpg | paper | 0.6596 | 0.1228 | (none) |
| paper_000098.jpg | paper | 0.6632 | 0.6383 | boundary_clamped |
| paper_000093.jpg | paper | 0.6674 | 0.7441 | boundary_clamped |
| glass_000165.jpg | glass | 0.6741 | 0.0770 | (none) |
| glass_000161.jpg | glass | 0.6765 | 0.3180 | (none) |
| paper_000095.jpg | paper | 0.6880 | 0.0902 | (none) |
| paper_000083.jpg | paper | 0.6928 | 0.2347 | (none) |
| paper_000094.jpg | paper | 0.6945 | 0.1169 | (none) |
| paper_000084.jpg | paper | 0.6969 | 0.2815 | boundary_clamped |
| paper_000021.jpg | paper | 0.7190 | 0.4254 | boundary_clamped |
| paper_000089.jpg | paper | 0.7386 | 0.2901 | (none) |
| plastic_000166.jpg | plastic | 0.7390 | 0.3106 | (none) |
| paper_000091.jpg | paper | 0.7502 | 0.3847 | (none) |
| plastic_000164.jpg | plastic | 0.7667 | 0.2677 | (none) |
| glass_000164.jpg | glass | 0.7718 | 0.1040 | (none) |
| paper_000086.jpg | paper | 0.7756 | 0.4107 | (none) |
