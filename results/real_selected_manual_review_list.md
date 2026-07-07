# real_selected Manual Review Priority List

91 real_selected images from `datasets/recycling_yolo_material_v1_real_aug/autolabel_report.csv`, ranked into 6 priority tiers (1=highest). Within a tier, sorted by bbox_area_ratio descending (bigger boxes first) then confidence ascending (weaker matches first). See the module docstring of `tools/build_real_selected_manual_review_list.py` for the exact tier criteria.

## Tier Counts

| tier | criteria | count |
|---|---|---|
| 1 | too_large_bbox | 0 |
| 2 | low_confidence | 0 |
| 3 | boundary_clamped | 18 |
| 4 | large_bbox_area (>0.35, below too_large cutoff) | 10 |
| 5 | unsafe_linked_class (paper/plastic) | 30 |
| 6 | no_flags | 33 |

**Suggested review order:** work tiers 1 through 5 top to bottom (58 of 91 images) before assuming "real_selected labels are fine" -- tier 6 (no flags) is safe to skip on a first pass. Given results/real_selected_pseudolabel_quality.md found zero too_large_bbox/low_confidence hits, tiers 1-2 are expected to be empty or near-empty here -- tiers 4 and 5 are where the real signal is likely to be for this dataset.

## Tier 3: boundary_clamped (18)

| output_image | target_class | confidence | bbox_area_ratio | warning_flags | preview_path |
|---|---|---|---|---|---|
| paper_000093.jpg | paper | 0.6674 | 0.7441 | boundary_clamped | (not in preview sample) |
| paper_000098.jpg | paper | 0.6632 | 0.6383 | boundary_clamped | (not in preview sample) |
| plastic_000179.jpg | plastic | 0.9498 | 0.6171 | boundary_clamped | (not in preview sample) |
| paper_000097.jpg | paper | 0.7832 | 0.4909 | boundary_clamped | (not in preview sample) |
| plastic_000171.jpg | plastic | 0.9371 | 0.4556 | boundary_clamped | (not in preview sample) |
| paper_000021.jpg | paper | 0.7190 | 0.4254 | boundary_clamped | (not in preview sample) |
| plastic_000183.jpg | plastic | 0.9195 | 0.3861 | boundary_clamped | (not in preview sample) |
| plastic_000168.jpg | plastic | 0.9328 | 0.3769 | boundary_clamped | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_plastic_000168_preview.jpg |
| paper_000082.jpg | paper | 0.4504 | 0.3736 | boundary_clamped | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_paper_000082_preview.jpg |
| plastic_000184.jpg | plastic | 0.9524 | 0.3540 | boundary_clamped | (not in preview sample) |
| plastic_000175.jpg | plastic | 0.9574 | 0.3367 | boundary_clamped | (not in preview sample) |
| paper_000092.jpg | paper | 0.8307 | 0.3306 | boundary_clamped | (not in preview sample) |
| paper_000084.jpg | paper | 0.6969 | 0.2815 | boundary_clamped | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_paper_000084_preview.jpg |
| paper_000088.jpg | paper | 0.9047 | 0.2552 | boundary_clamped | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_paper_000088_preview.jpg |
| plastic_000042.jpg | plastic | 0.9644 | 0.2436 | boundary_clamped | (not in preview sample) |
| plastic_000187.jpg | plastic | 0.9622 | 0.2120 | boundary_clamped | (not in preview sample) |
| plastic_000041.jpg | plastic | 0.9571 | 0.2005 | boundary_clamped | (not in preview sample) |
| glass_000167.jpg | glass | 0.9606 | 0.0687 | boundary_clamped | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_glass_000167_preview.jpg |

## Tier 4: large_bbox_area (>0.35, below too_large cutoff) (10)

| output_image | target_class | confidence | bbox_area_ratio | warning_flags | preview_path |
|---|---|---|---|---|---|
| plastic_000172.jpg | plastic | 0.7852 | 0.5604 | (none) | (not in preview sample) |
| plastic_000167.jpg | plastic | 0.9462 | 0.5245 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_plastic_000167_preview.jpg |
| plastic_000162.jpg | plastic | 0.9594 | 0.4974 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_plastic_000162_preview.jpg |
| plastic_000170.jpg | plastic | 0.9132 | 0.4509 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_plastic_000170_preview.jpg |
| paper_000086.jpg | paper | 0.7756 | 0.4107 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_paper_000086_preview.jpg |
| paper_000087.jpg | paper | 0.8215 | 0.3935 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_paper_000087_preview.jpg |
| paper_000096.jpg | paper | 0.6532 | 0.3912 | (none) | (not in preview sample) |
| paper_000091.jpg | paper | 0.7502 | 0.3847 | (none) | (not in preview sample) |
| plastic_000165.jpg | plastic | 0.9433 | 0.3647 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_plastic_000165_preview.jpg |
| paper_000024.jpg | paper | 0.8401 | 0.3562 | (none) | (not in preview sample) |

## Tier 5: unsafe_linked_class (paper/plastic) (30)

| output_image | target_class | confidence | bbox_area_ratio | warning_flags | preview_path |
|---|---|---|---|---|---|
| plastic_000186.jpg | plastic | 0.9571 | 0.3189 | (none) | (not in preview sample) |
| plastic_000166.jpg | plastic | 0.7390 | 0.3106 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_plastic_000166_preview.jpg |
| paper_000089.jpg | paper | 0.7386 | 0.2901 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_paper_000089_preview.jpg |
| plastic_000178.jpg | plastic | 0.9139 | 0.2900 | (none) | (not in preview sample) |
| plastic_000045.jpg | plastic | 0.9624 | 0.2781 | (none) | (not in preview sample) |
| plastic_000047.jpg | plastic | 0.9595 | 0.2743 | (none) | (not in preview sample) |
| plastic_000164.jpg | plastic | 0.7667 | 0.2677 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_plastic_000164_preview.jpg |
| plastic_000161.jpg | plastic | 0.9695 | 0.2631 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_plastic_000161_preview.jpg |
| paper_000023.jpg | paper | 0.8656 | 0.2586 | (none) | (not in preview sample) |
| plastic_000046.jpg | plastic | 0.9305 | 0.2384 | (none) | (not in preview sample) |
| paper_000083.jpg | paper | 0.6928 | 0.2347 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_paper_000083_preview.jpg |
| plastic_000188.jpg | plastic | 0.8815 | 0.2320 | (none) | (not in preview sample) |
| plastic_000163.jpg | plastic | 0.9169 | 0.2220 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_plastic_000163_preview.jpg |
| plastic_000185.jpg | plastic | 0.9369 | 0.2162 | (none) | (not in preview sample) |
| paper_000090.jpg | paper | 0.8390 | 0.2069 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_paper_000090_preview.jpg |
| plastic_000182.jpg | plastic | 0.9567 | 0.2010 | (none) | (not in preview sample) |
| plastic_000174.jpg | plastic | 0.8794 | 0.1928 | (none) | (not in preview sample) |
| plastic_000181.jpg | plastic | 0.9682 | 0.1875 | (none) | (not in preview sample) |
| plastic_000180.jpg | plastic | 0.9111 | 0.1838 | (none) | (not in preview sample) |
| plastic_000177.jpg | plastic | 0.9581 | 0.1770 | (none) | (not in preview sample) |
| paper_000085.jpg | paper | 0.8174 | 0.1697 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_paper_000085_preview.jpg |
| paper_000022.jpg | paper | 0.8994 | 0.1685 | (none) | (not in preview sample) |
| plastic_000043.jpg | plastic | 0.9713 | 0.1512 | (none) | (not in preview sample) |
| plastic_000173.jpg | plastic | 0.9670 | 0.1341 | (none) | (not in preview sample) |
| plastic_000176.jpg | plastic | 0.9570 | 0.1230 | (none) | (not in preview sample) |
| paper_000081.jpg | paper | 0.6596 | 0.1228 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_paper_000081_preview.jpg |
| paper_000094.jpg | paper | 0.6945 | 0.1169 | (none) | (not in preview sample) |
| plastic_000044.jpg | plastic | 0.9716 | 0.1139 | (none) | (not in preview sample) |
| paper_000095.jpg | paper | 0.6880 | 0.0902 | (none) | (not in preview sample) |
| plastic_000169.jpg | plastic | 0.9634 | 0.0791 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_plastic_000169_preview.jpg |

## Tier 6: no_flags (33)

| output_image | target_class | confidence | bbox_area_ratio | warning_flags | preview_path |
|---|---|---|---|---|---|
| glass_000161.jpg | glass | 0.6765 | 0.3180 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_glass_000161_preview.jpg |
| metal_000165.jpg | metal | 0.9517 | 0.2103 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_metal_000165_preview.jpg |
| glass_000166.jpg | glass | 0.8717 | 0.1985 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_glass_000166_preview.jpg |
| metal_000164.jpg | metal | 0.8881 | 0.1857 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_metal_000164_preview.jpg |
| metal_000044.jpg | metal | 0.9331 | 0.1844 | (none) | (not in preview sample) |
| metal_000162.jpg | metal | 0.9285 | 0.1796 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_metal_000162_preview.jpg |
| metal_000168.jpg | metal | 0.9446 | 0.1620 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_metal_000168_preview.jpg |
| metal_000179.jpg | metal | 0.9454 | 0.1534 | (none) | (not in preview sample) |
| metal_000041.jpg | metal | 0.9518 | 0.1503 | (none) | (not in preview sample) |
| metal_000043.jpg | metal | 0.9472 | 0.1482 | (none) | (not in preview sample) |
| metal_000178.jpg | metal | 0.9543 | 0.1294 | (none) | (not in preview sample) |
| glass_000042.jpg | glass | 0.9060 | 0.1263 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_val_glass_000042_preview.jpg |
| metal_000174.jpg | metal | 0.9342 | 0.1243 | (none) | (not in preview sample) |
| glass_000162.jpg | glass | 0.4960 | 0.1233 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_glass_000162_preview.jpg |
| metal_000045.jpg | metal | 0.8949 | 0.1138 | (none) | (not in preview sample) |
| glass_000041.jpg | glass | 0.8333 | 0.1133 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_val_glass_000041_preview.jpg |
| metal_000163.jpg | metal | 0.9403 | 0.1108 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_metal_000163_preview.jpg |
| metal_000167.jpg | metal | 0.9353 | 0.1094 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_metal_000167_preview.jpg |
| glass_000168.jpg | glass | 0.9696 | 0.1091 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_glass_000168_preview.jpg |
| metal_000170.jpg | metal | 0.9374 | 0.1088 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_metal_000170_preview.jpg |
| metal_000169.jpg | metal | 0.9562 | 0.1077 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_metal_000169_preview.jpg |
| glass_000164.jpg | glass | 0.7718 | 0.1040 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_glass_000164_preview.jpg |
| metal_000173.jpg | metal | 0.9535 | 0.0999 | (none) | (not in preview sample) |
| metal_000171.jpg | metal | 0.9376 | 0.0953 | (none) | (not in preview sample) |
| metal_000172.jpg | metal | 0.9483 | 0.0920 | (none) | (not in preview sample) |
| metal_000042.jpg | metal | 0.9310 | 0.0854 | (none) | (not in preview sample) |
| metal_000177.jpg | metal | 0.9251 | 0.0845 | (none) | (not in preview sample) |
| metal_000175.jpg | metal | 0.9202 | 0.0843 | (none) | (not in preview sample) |
| metal_000166.jpg | metal | 0.9524 | 0.0843 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_metal_000166_preview.jpg |
| metal_000176.jpg | metal | 0.8459 | 0.0838 | (none) | (not in preview sample) |
| metal_000161.jpg | metal | 0.9365 | 0.0828 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_metal_000161_preview.jpg |
| glass_000165.jpg | glass | 0.6741 | 0.0770 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_glass_000165_preview.jpg |
| glass_000163.jpg | glass | 0.6096 | 0.0726 | (none) | datasets/recycling_yolo_material_v1_real_aug/previews/real_selected_train_glass_000163_preview.jpg |
