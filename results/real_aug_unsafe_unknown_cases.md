# real_aug Unsafe unknown -> ACCEPT_SORT Cases

Every case where a `test_images_real/unknown/` image (an image containing no plastic/metal/glass/paper object at all -- plastic bags, snack bags, tissues, a charger, a mouse, a book, etc.) got `ACCEPT_SORT`ed anyway, at any of the 4 benchmark thresholds. `perception_policy.evaluate_detections()` never sees ground truth -- every row below is a false-positive CLASS prediction on an object that is not actually plastic/metal/glass/paper, not a bounding-box localization error on a real target object.

## Models analyzed

- real_aug: `logs/vision_benchmark/test_images_real/vision_only/yolo11n_recycling_material_v1_real_aug_640_conf005.log`
- strong: `logs/vision_benchmark/test_images_real/vision_only/yolo11n_recycling_material_v1_aug_strong_640_conf005.log`
- baseline: `logs/vision_benchmark/test_images_real/vision_only/yolo11n_recycling_material_v1_640_conf005.log`

## All Cases

| source_model | image_path | threshold | predicted_class | confidence | bbox | decision | reason |
|---|---|---|---|---|---|---|---|
| baseline | unknown/unknown_bottle_cap_003.jpg.jpg | 0.3 | plastic | 0.37 | (1.0, -4.5, 2452.4, 2780.4) | ACCEPT_SORT | known_high_confidence |
| baseline | unknown/unknown_bottle_cap_003.jpg.jpg | 0.1 | plastic | 0.37 | (1.0, -4.5, 2452.4, 2780.4) | ACCEPT_SORT | known_high_confidence |
| baseline | unknown/unknown_bottle_cap_003.jpg.jpg | 0.05 | plastic | 0.37 | (1.0, -4.5, 2452.4, 2780.4) | ACCEPT_SORT | known_high_confidence |
| real_aug | unknown/unknown_bottle_cap_003.jpg.jpg | 0.3 | plastic | 0.31 | (91.9, 17.0, 2406.8, 2879.2) | ACCEPT_SORT | known_high_confidence |
| real_aug | unknown/unknown_bottle_cap_003.jpg.jpg | 0.1 | plastic | 0.31 | (91.9, 17.0, 2406.8, 2879.2) | ACCEPT_SORT | known_high_confidence |
| real_aug | unknown/unknown_bottle_cap_003.jpg.jpg | 0.05 | plastic | 0.31 | (91.9, 17.0, 2406.8, 2879.2) | ACCEPT_SORT | known_high_confidence |
| real_aug | unknown/unknown_charger_007.jpg.jpg | 0.1 | paper | 0.26 | (18.2, -14.3, 3014.6, 3533.8) | ACCEPT_SORT | known_high_confidence |
| real_aug | unknown/unknown_charger_007.jpg.jpg | 0.05 | paper | 0.26 | (18.2, -14.3, 3014.6, 3533.8) | ACCEPT_SORT | known_high_confidence |
| real_aug | unknown/unknown_snack_bag_002.jpg.jpg | 0.1 | paper | 0.27 | (980.6, 174.9, 2293.8, 2211.4) | ACCEPT_SORT | known_high_confidence |
| real_aug | unknown/unknown_snack_bag_002.jpg.jpg | 0.05 | paper | 0.27 | (980.6, 174.9, 2293.8, 2211.4) | ACCEPT_SORT | known_high_confidence |
| real_aug | unknown/unknown_tissue_001.jpg.jpg | 0.3 | paper | 0.44 | (578.0, 21.0, 2676.2, 2515.4) | ACCEPT_SORT | known_high_confidence |
| real_aug | unknown/unknown_tissue_001.jpg.jpg | 0.1 | paper | 0.44 | (578.0, 21.0, 2676.2, 2515.4) | ACCEPT_SORT | known_high_confidence |
| real_aug | unknown/unknown_tissue_001.jpg.jpg | 0.05 | paper | 0.44 | (578.0, 21.0, 2676.2, 2515.4) | ACCEPT_SORT | known_high_confidence |

## Analysis

### real_aug

10 unsafe (image, threshold) case(s) across 4 unique image(s): `unknown/unknown_bottle_cap_003.jpg.jpg`, `unknown/unknown_charger_007.jpg.jpg`, `unknown/unknown_snack_bag_002.jpg.jpg`, `unknown/unknown_tissue_001.jpg.jpg`.

**Predicted-class skew:** paper=7 (70%), plastic=3 (30%). Cases skew heavily toward `paper` (70% of unsafe cases) -- this is the class most likely to be over-triggering on non-target real-world clutter.

**Threshold spread:** cases per threshold: 0.3=2, 0.1=4, 0.05=4. All cases are below confidence_threshold=0.5 -- raising `confidence_threshold`/`policy_confidence_threshold` back toward 0.5 (the project default) would already suppress every case listed here.

**High-confidence risky cases (>=0.35, i.e. above perception_policy's own low_confidence_threshold):**
- `unknown/unknown_tissue_001.jpg.jpg` -> paper @ 0.44 (threshold=0.3)
- `unknown/unknown_tissue_001.jpg.jpg` -> paper @ 0.44 (threshold=0.1)
- `unknown/unknown_tissue_001.jpg.jpg` -> paper @ 0.44 (threshold=0.05)

### baseline

3 unsafe (image, threshold) case(s) across 1 unique image(s): `unknown/unknown_bottle_cap_003.jpg.jpg`.

**Predicted-class skew:** plastic=3 (100%). Cases skew heavily toward `plastic` (100% of unsafe cases) -- this is the class most likely to be over-triggering on non-target real-world clutter.

**Threshold spread:** cases per threshold: 0.3=1, 0.1=1, 0.05=1. All cases are below confidence_threshold=0.5 -- raising `confidence_threshold`/`policy_confidence_threshold` back toward 0.5 (the project default) would already suppress every case listed here.

**High-confidence risky cases (>=0.35, i.e. above perception_policy's own low_confidence_threshold):**
- `unknown/unknown_bottle_cap_003.jpg.jpg` -> plastic @ 0.37 (threshold=0.3)
- `unknown/unknown_bottle_cap_003.jpg.jpg` -> plastic @ 0.37 (threshold=0.1)
- `unknown/unknown_bottle_cap_003.jpg.jpg` -> plastic @ 0.37 (threshold=0.05)
