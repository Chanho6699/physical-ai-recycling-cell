#!/usr/bin/env python3
"""Build recycling_yolo_material_v1: a material-level (plastic/metal/
glass/paper) YOLO detection dataset combining two sources:

  1. datasets/recycling_yolo_candidates_v1/{plastic,metal,glass,paper}/ --
     freshly collected classification-style candidate images (varied
     backgrounds/framings), auto-labeled here with GroundingDINO.
  2. datasets/recycling_yolo_autolabel_v0/ -- the earlier v0 YOLO dataset
     (plastic/paper/can/glass_bottle), whose existing YOLO labels are
     REMAPPED into the material taxonomy and reused as-is (no re-running
     GroundingDINO on them) via --include-v0-remap.

Why this exists: custom_autolabel_v0 (plastic/paper/can/glass_bottle)
showed a severe "paper collapse" on real images -- 44/50 test_images_
real/ images were predicted as paper regardless of their real content
(see tools/analyze_custom_yolo_real_benchmark.py and
results/custom_yolo_v0_real_failure_analysis.md). v0's own "paper"
labels are a lead suspect (loose/oversized pseudo-label boxes from the
"paper"/"crumpled paper" prompts), so this script's v0-remap path
supports --exclude-v0-paper to leave that class out of the reused v0
data entirely, and the whole dataset moves to a coarser, hopefully more
visually-distinct material-level taxonomy (metal/glass/paper/plastic)
built from more varied source photos. See
docs/recycling_material_v1_dataset_plan.md for the full rationale.

This is a separate script from tools/autolabel_recycling_groundingdino.py
(the v0 builder) -- it does not import or modify it, so v0-related
tooling keeps working unchanged.

bbox quality warnings (recorded per-row in autolabel_report.csv, drawn
onto previews, and optionally used to filter labels out of the dataset):
  - too_large_bbox:  normalized bbox area ratio (width*height) > 0.75
  - too_small_bbox:  normalized bbox area ratio < 0.02
  - low_confidence:  detection confidence < 0.35
  - boundary_clamped: the raw bbox extended outside the image and had to
    be clamped (candidates_v1 only -- v0-remapped labels are already
    clamped/normalized from v0's own build, so this flag never applies
    to v0_remapped rows)

Requires autodistill + autodistill-grounding-dino + supervision (see
tools/requirements_autolabel.txt) for the candidates_v1 path -- run this
in the isolated venv used for the v0 builder, not the ROS2/ament Python
environment:

    python3 -m venv .venv-autolabel
    source .venv-autolabel/bin/activate
    pip install --upgrade pip
    pip install -r tools/requirements_autolabel.txt

If --include-v0-remap is used WITHOUT --candidate-root pointing at any
real candidates_v1 images (e.g. for a v0-only rebuild), GroundingDINO is
never loaded at all -- the import only happens when there is at least
one candidates_v1 image to label.

Usage:
    python3 tools/build_recycling_material_v1_dataset.py \\
        --include-v0-remap \\
        --exclude-v0-paper \\
        --exclude-too-large
"""
import argparse
import csv
import os
import random
import shutil
import sys
from collections import Counter, defaultdict

MATERIAL_CLASSES = ('plastic', 'metal', 'glass', 'paper')
MATERIAL_CLASS_ID = {name: idx for idx, name in enumerate(MATERIAL_CLASSES)}

# datasets/recycling_yolo_candidates_v1/<folder>/ -> material_v1 class.
# Identity today; kept explicit in case a future source uses different
# folder names than the target taxonomy.
CANDIDATES_V1_TO_MATERIAL = {
    'plastic': 'plastic',
    'metal': 'metal',
    'glass': 'glass',
    'paper': 'paper',
}

# datasets/recycling_yolo_autolabel_v0/recycling.yaml's own class_id order.
V0_CLASS_NAMES = ('plastic', 'paper', 'can', 'glass_bottle')

V0_TO_MATERIAL = {
    'plastic': 'plastic',
    'paper': 'paper',
    'can': 'metal',
    'glass_bottle': 'glass',
}

PROMPT_MAP = {
    'plastic': [
        'plastic bottle',
        'plastic container',
        'plastic cup',
        'plastic bag',
        'plastic packaging',
    ],
    'metal': [
        'aluminum can',
        'metal can',
        'tin can',
        'metal container',
    ],
    'glass': [
        'glass bottle',
        'glass jar',
        'glass container',
    ],
    'paper': [
        'paper',
        'cardboard box',
        'paper box',
        'magazine',
        'book',
    ],
}

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

TOO_LARGE_AREA_RATIO = 0.75
TOO_SMALL_AREA_RATIO = 0.02
LOW_CONFIDENCE_THRESHOLD = 0.35

CSV_FIELDNAMES = [
    'source_image',
    'output_image',
    'split',
    'source_type',
    'original_class',
    'target_class',
    'status',
    'confidence',
    'bbox_area_ratio',
    'warning_flags',
]

README_TEMPLATE = """\
# recycling_yolo_material_v1

## v0 paper collapse summary

custom_autolabel_v0 (plastic/paper/can/glass_bottle) was benchmarked
against test_images_real/ and showed a severe "paper collapse": 44/50
images were predicted as `paper` regardless of their real content --
80-100% of every non-paper ground-truth class (can, glass_bottle,
plastic_bottle, unknown) was misread as paper (see
`results/custom_yolo_v0_real_failure_analysis.md`). v0's own "paper"
pseudo-labels are a lead suspect: several were loose/oversized boxes
covering nearly the whole image. This dataset addresses that two ways:
(1) a coarser, hopefully more visually-separable material-level
taxonomy, and (2) an option (`--exclude-v0-paper`) to drop v0's paper
labels entirely from the reused data rather than carry the suspect class
forward unchanged.

## Purpose of material_v1

Move from an object-level taxonomy (plastic_bottle/paper_cup/can/
glass_bottle) to a material-level one (plastic/metal/glass/paper) using
a larger, more varied set of freshly collected candidate photos
(datasets/recycling_yolo_candidates_v1/), combined with v0's already
-labeled images (remapped into the new taxonomy) for extra volume.

## Final taxonomy

| class_id | name    | bin mapping |
|---|---|---|
| 0 | plastic | plastic_bin |
| 1 | metal   | metal_bin   |
| 2 | glass   | glass_bin   |
| 3 | paper   | paper_bin   |

## Source composition

- `candidates_v1`: datasets/recycling_yolo_candidates_v1/{{plastic,metal,
  glass,paper}}/, auto-labeled fresh with GroundingDINO using the prompt
  map below.
- `v0_remapped`: datasets/recycling_yolo_autolabel_v0/'s existing YOLO
  labels, remapped class_id -> material_v1 class_id (plastic->plastic,
  can->metal, glass_bottle->glass, paper->paper unless
  --exclude-v0-paper was used), reusing v0's own bboxes and (from v0's
  own autolabel_report.csv) confidence values -- GroundingDINO is not
  re-run on these images.

See `autolabel_report.csv`'s `source_type` column to tell the two apart
per image.

## bbox quality warning criteria

- `too_large_bbox`: normalized bbox area ratio (width * height) > 0.75
- `too_small_bbox`: normalized bbox area ratio < 0.02
- `low_confidence`: detection confidence < 0.35
- `boundary_clamped`: the raw GroundingDINO box extended outside the
  image and was clamped before normalizing (candidates_v1 only)

By default all labeled boxes are kept regardless of warnings; this run
may have used `--exclude-warned` and/or `--exclude-too-large` to drop
some -- check the command line this dataset was built with (also logged
below) and `autolabel_report.csv`'s `status` column (`excluded` rows had
a real detection that was filtered out of the final labels).

## Pseudo-label quality warning

**These labels are a draft, not ground truth**, same as v0 -- see
`docs/recycling_material_v1_dataset_plan.md` for the human-validation
plan before using this dataset for anything beyond pipeline validation.

## Files

- `images/train/`, `images/val/`, `labels/train/`, `labels/val/` -- YOLO
  format, same conventions as v0 (images with no kept label have no .txt
  file)
- `recycling_material_v1.yaml` -- Ultralytics dataset config
- `autolabel_report.csv` -- per-image status/source/confidence/bbox_area
  _ratio/warning_flags
- `previews/` -- bbox + class_name + confidence + warning_flags drawn on
  a sample of images from both source types

## Commands

Build:
```bash
python3 tools/build_recycling_material_v1_dataset.py \\
  --include-v0-remap \\
  --exclude-v0-paper \\
  --exclude-too-large
```

Train:
```bash
bash tools/train_recycling_yolo_material_v1.sh
```

Export to ONNX:
```bash
bash tools/export_recycling_yolo_material_v1_onnx.sh
```

## Built with

```
{build_command}
```
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--candidate-root', default='datasets/recycling_yolo_candidates_v1',
        help='Directory containing <class>/ subfolders of fresh v1 '
             'candidate images (default: '
             'datasets/recycling_yolo_candidates_v1)')
    parser.add_argument(
        '--output-root', default='datasets/recycling_yolo_material_v1',
        help='Directory to write the combined dataset into (default: '
             'datasets/recycling_yolo_material_v1)')
    parser.add_argument(
        '--include-v0-remap', action='store_true',
        help='Also reuse datasets/recycling_yolo_autolabel_v0/\'s '
             'existing YOLO labels, remapped into the material taxonomy '
             '(default: off -- candidates_v1 only)')
    parser.add_argument(
        '--v0-root', default='datasets/recycling_yolo_autolabel_v0',
        help='Path to the existing v0 YOLO dataset, used when '
             '--include-v0-remap is set (default: '
             'datasets/recycling_yolo_autolabel_v0)')
    parser.add_argument(
        '--exclude-v0-paper', action='store_true', default=False,
        help='Drop v0\'s "paper" class entirely from the reused v0 data '
             '(it is the lead suspect for the v0 paper-collapse failure '
             '-- see module docstring). Default: False (paper included) '
             '-- documented examples in this project use this flag on.')
    parser.add_argument(
        '--exclude-warned', action='store_true', default=False,
        help='Exclude any detection with ANY warning flag from the '
             'final labels (image is still copied, just without a '
             'label file). Default: False -- warned boxes are kept.')
    parser.add_argument(
        '--exclude-too-large', action='store_true', default=False,
        help='Exclude only detections flagged too_large_bbox from the '
             'final labels, regardless of other warnings. Default: '
             'False.')
    parser.add_argument(
        '--val-ratio', type=float, default=0.2,
        help='Fraction of each material class held out for validation, '
             'applied to the candidates_v1 pool (v0_remapped entries '
             'keep their original v0 train/val split). Default: 0.2')
    parser.add_argument(
        '--box-threshold', type=float, default=0.35,
        help='GroundingDINO box confidence threshold (default: 0.35)')
    parser.add_argument(
        '--text-threshold', type=float, default=0.25,
        help='GroundingDINO text-match threshold (default: 0.25)')
    parser.add_argument(
        '--preview-limit-per-class', type=int, default=10,
        help='Max number of preview (bbox-drawn) images saved per '
             'material class, shared across both source types '
             '(default: 10)')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for the candidates_v1 train/val split shuffle '
             '(default: 42)')
    parser.add_argument(
        '--limit-per-class', type=int, default=None,
        help='Optional cap on candidates_v1 images processed per class, '
             'for a quick smoke run before committing to the full '
             'dataset (default: no cap). Does not affect v0_remapped '
             'entries.')
    return parser.parse_args()


def discover_candidates_v1(candidate_root):
    images_by_class = {}
    for class_name in MATERIAL_CLASSES:
        class_dir = os.path.join(candidate_root, class_name)
        if not os.path.isdir(class_dir):
            print(f"WARN: expected class folder '{class_name}' not found "
                  f"under {candidate_root}, skipping")
            continue
        images = sorted(
            os.path.join(class_dir, name)
            for name in os.listdir(class_dir)
            if name.lower().endswith(IMAGE_EXTENSIONS)
        )
        images_by_class[class_name] = images
    return images_by_class


def load_v0_confidence_lookup(v0_root):
    """{(split, output_image): confidence} from v0's own autolabel_
    report.csv, so v0-remapped entries can carry over their original
    GroundingDINO confidence instead of losing it."""
    report_path = os.path.join(v0_root, 'autolabel_report.csv')
    lookup = {}
    if not os.path.isfile(report_path):
        print(f'WARN: {report_path} not found -- v0-remapped rows will '
              f'have no confidence value')
        return lookup
    with open(report_path, newline='') as report_file:
        for row in csv.DictReader(report_file):
            if row.get('status') == 'labeled' and row.get('confidence'):
                lookup[(row['split'], row['output_image'])] = \
                    float(row['confidence'])
    return lookup


def discover_v0_remap_entries(v0_root, exclude_v0_paper):
    """Every v0 image that already has a YOLO label, ready to be
    remapped into the material taxonomy. v0 images with no label file
    (v0 status=no_box) are skipped -- there is nothing to remap."""
    confidence_lookup = load_v0_confidence_lookup(v0_root)
    entries = []

    for split in ('train', 'val'):
        images_dir = os.path.join(v0_root, 'images', split)
        labels_dir = os.path.join(v0_root, 'labels', split)
        if not os.path.isdir(images_dir):
            continue

        for image_name in sorted(os.listdir(images_dir)):
            if not image_name.lower().endswith(IMAGE_EXTENSIONS):
                continue
            stem = os.path.splitext(image_name)[0]
            label_path = os.path.join(labels_dir, f'{stem}.txt')
            if not os.path.isfile(label_path):
                continue

            with open(label_path) as label_file:
                line = label_file.readline().strip()
            if not line:
                continue

            parts = line.split()
            v0_class_id = int(parts[0])
            x_center, y_center, width, height = (
                float(v) for v in parts[1:5])
            original_class = V0_CLASS_NAMES[v0_class_id]

            if original_class == 'paper' and exclude_v0_paper:
                continue

            entries.append({
                'source_image': os.path.join(images_dir, image_name),
                'split': split,
                'original_class': original_class,
                'target_class': V0_TO_MATERIAL[original_class],
                'confidence': confidence_lookup.get(
                    (split, image_name)),
                'bbox_norm': (x_center, y_center, width, height),
                'bbox_area_ratio': width * height,
            })

    return entries


def split_train_val(images, val_ratio, seed):
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    val_count = round(len(shuffled) * val_ratio)
    return shuffled[val_count:], shuffled[:val_count]


def clamp_bbox(x1, y1, x2, y2, img_w, img_h):
    cx1 = min(max(x1, 0.0), img_w)
    cy1 = min(max(y1, 0.0), img_h)
    cx2 = min(max(x2, 0.0), img_w)
    cy2 = min(max(y2, 0.0), img_h)
    clamped = (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2)
    return cx1, cy1, cx2, cy2, clamped


def normalize_bbox(x1, y1, x2, y2, img_w, img_h):
    x_center = ((x1 + x2) / 2.0) / img_w
    y_center = ((y1 + y2) / 2.0) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height


def compute_warning_flags(bbox_area_ratio, confidence, boundary_clamped):
    flags = []
    if bbox_area_ratio is not None:
        if bbox_area_ratio > TOO_LARGE_AREA_RATIO:
            flags.append('too_large_bbox')
        elif bbox_area_ratio < TOO_SMALL_AREA_RATIO:
            flags.append('too_small_bbox')
    if confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD:
        flags.append('low_confidence')
    if boundary_clamped:
        flags.append('boundary_clamped')
    return flags


class GroundingDINOLabeler:
    """One GroundingDINO model instance per material class, restricted
    to that class's own prompt list -- same design as
    tools/autolabel_recycling_groundingdino.py's labeler."""

    def __init__(self, box_threshold, text_threshold):
        from autodistill.detection import CaptionOntology
        from autodistill_grounding_dino import GroundingDINO

        self._models = {}
        self._prompts_by_class = {}
        for class_name, prompts in PROMPT_MAP.items():
            synthetic_labels = [f'{class_name}::{i}'
                                 for i in range(len(prompts))]
            ontology = CaptionOntology(
                dict(zip(prompts, synthetic_labels)))
            self._models[class_name] = GroundingDINO(
                ontology=ontology,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            self._prompts_by_class[class_name] = prompts

    def label(self, image_path, target_class):
        model = self._models[target_class]
        detections = model.predict(image_path)
        if detections is None or len(detections) == 0:
            return None

        best_idx = int(detections.confidence.argmax())
        x1, y1, x2, y2 = [float(v) for v in detections.xyxy[best_idx]]
        confidence = float(detections.confidence[best_idx])
        return {'confidence': confidence, 'bbox': (x1, y1, x2, y2)}


def draw_preview(image_bgr, bbox_px, class_name, confidence, warning_flags):
    import cv2

    x1, y1, x2, y2 = [int(round(v)) for v in bbox_px]
    preview = image_bgr.copy()
    color = (0, 0, 255) if warning_flags else (0, 255, 0)
    cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
    label_text = f'{class_name} {confidence:.2f}' if confidence is not None \
        else class_name
    # cv2.putText's y is the text baseline -- clamping to just max(0, ...)
    # still lets the glyphs render above row 0 (i.e. off-canvas) for a
    # box whose top edge is near the image top, which is common for
    # exactly the too_large_bbox cases this preview exists to surface.
    # 18px keeps the label fully on-canvas in that case.
    label_y = max(18, y1 - 8)
    cv2.putText(
        preview, label_text, (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if warning_flags:
        cv2.putText(
            preview, ','.join(warning_flags),
            (x1, min(preview.shape[0] - 5, y2 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return preview


def process_candidates_v1(labeler, images_by_class, val_ratio, seed,
                           limit_per_class):
    """Returns a list of raw entries (pre-copy) for candidates_v1,
    mirroring discover_v0_remap_entries()'s shape plus a 'status' key
    that's already resolved (labeled/no_box/error) and the raw pixel
    bbox + image dimensions needed to draw a preview."""
    import cv2

    entries = []
    for class_name, images in images_by_class.items():
        if limit_per_class is not None:
            images = images[:limit_per_class]
        train_images, val_images = split_train_val(images, val_ratio, seed)

        for split, split_images in (('train', train_images),
                                     ('val', val_images)):
            for image_path in split_images:
                entry = {
                    'source_image': image_path,
                    'split': split,
                    'original_class': class_name,
                    'target_class': CANDIDATES_V1_TO_MATERIAL[class_name],
                }
                try:
                    frame = cv2.imread(image_path)
                    if frame is None:
                        raise ValueError(
                            f'cv2.imread returned None for {image_path}')
                    img_h, img_w = frame.shape[:2]

                    result = labeler.label(image_path, entry['target_class'])
                    if result is None:
                        entry['status'] = 'no_box'
                    else:
                        x1, y1, x2, y2 = result['bbox']
                        cx1, cy1, cx2, cy2, clamped = clamp_bbox(
                            x1, y1, x2, y2, img_w, img_h)
                        x_center, y_center, width, height = normalize_bbox(
                            cx1, cy1, cx2, cy2, img_w, img_h)
                        entry.update({
                            'status': 'labeled',
                            'confidence': result['confidence'],
                            'bbox_norm': (x_center, y_center, width, height),
                            'bbox_area_ratio': width * height,
                            'boundary_clamped': clamped,
                            'bbox_px': (cx1, cy1, cx2, cy2),
                        })
                except Exception as exc:
                    entry['status'] = 'error'
                    entry['error'] = str(exc)

                entries.append(entry)

    return entries


def write_output(entries, output_root, exclude_warned, exclude_too_large,
                  preview_limit_per_class):
    import cv2

    for split in ('train', 'val'):
        os.makedirs(os.path.join(output_root, 'images', split),
                    exist_ok=True)
        os.makedirs(os.path.join(output_root, 'labels', split),
                    exist_ok=True)
    os.makedirs(os.path.join(output_root, 'previews'), exist_ok=True)

    report_rows = []
    preview_counts = defaultdict(int)
    stem_counters = defaultdict(int)

    for entry in entries:
        target_class = entry['target_class']
        split = entry['split']
        status = entry.get('status', 'error')

        stem_counters[(split, target_class)] += 1
        index = stem_counters[(split, target_class)]
        ext = os.path.splitext(entry['source_image'])[1].lower() or '.jpg'
        stem = f'{target_class}_{index:06d}'
        output_image_name = f'{stem}{ext}'

        row = {
            'source_image': entry['source_image'],
            'output_image': output_image_name,
            'split': split,
            'source_type': entry.get('source_type', 'unknown'),
            'original_class': entry['original_class'],
            'target_class': target_class,
            'status': status,
            'confidence': '',
            'bbox_area_ratio': '',
            'warning_flags': '',
        }

        if status == 'error':
            row['warning_flags'] = entry.get('error', '')
            report_rows.append(row)
            continue

        images_dir = os.path.join(output_root, 'images', split)
        labels_dir = os.path.join(output_root, 'labels', split)
        output_image_path = os.path.join(images_dir, output_image_name)

        bbox_norm = entry.get('bbox_norm')
        confidence = entry.get('confidence')
        bbox_area_ratio = entry.get('bbox_area_ratio')
        boundary_clamped = entry.get('boundary_clamped', False)

        warning_flags = []
        if status == 'labeled':
            warning_flags = compute_warning_flags(
                bbox_area_ratio, confidence, boundary_clamped)

        should_write_label = status == 'labeled'
        if should_write_label and exclude_warned and warning_flags:
            should_write_label = False
            status = 'excluded'
        if should_write_label and exclude_too_large \
                and 'too_large_bbox' in warning_flags:
            should_write_label = False
            status = 'excluded'

        try:
            shutil.copy(entry['source_image'], output_image_path)
        except Exception as exc:
            row['status'] = 'error'
            row['warning_flags'] = f'copy failed: {exc}'
            report_rows.append(row)
            continue

        row['status'] = status
        row['confidence'] = (
            f'{confidence:.4f}' if confidence is not None else '')
        row['bbox_area_ratio'] = (
            f'{bbox_area_ratio:.4f}' if bbox_area_ratio is not None else '')
        row['warning_flags'] = ';'.join(warning_flags)

        if should_write_label:
            x_center, y_center, width, height = bbox_norm
            label_path = os.path.join(labels_dir, f'{stem}.txt')
            with open(label_path, 'w') as label_file:
                label_file.write(
                    f'{MATERIAL_CLASS_ID[target_class]} {x_center:.6f} '
                    f'{y_center:.6f} {width:.6f} {height:.6f}\n')

        # Budget previews per (target_class, source_type), not just per
        # class -- candidates_v1 is processed first and vastly
        # outnumbers v0_remapped, so a single shared per-class budget
        # would always be spent before v0_remapped's (often more
        # warning-prone, and specifically what --exclude-v0-paper/
        # --exclude-too-large were added to investigate) examples ever
        # got a turn.
        preview_key = (target_class, entry.get('source_type', 'unknown'))
        if status in ('labeled', 'excluded') \
                and preview_counts[preview_key] < preview_limit_per_class:
            frame = cv2.imread(output_image_path)
            if frame is not None:
                img_h, img_w = frame.shape[:2]
                x_center, y_center, width, height = bbox_norm
                x1 = (x_center - width / 2.0) * img_w
                y1 = (y_center - height / 2.0) * img_h
                x2 = (x_center + width / 2.0) * img_w
                y2 = (y_center + height / 2.0) * img_h
                preview = draw_preview(
                    frame, (x1, y1, x2, y2), target_class, confidence,
                    warning_flags)
                preview_path = os.path.join(
                    output_root, 'previews', f'{split}_{stem}_preview.jpg')
                cv2.imwrite(preview_path, preview)
                preview_counts[preview_key] += 1

        report_rows.append(row)

    return report_rows


def write_yaml(output_root):
    yaml_path = os.path.join(output_root, 'recycling_material_v1.yaml')
    abs_output_root = os.path.abspath(output_root)
    names_lines = '\n'.join(
        f'  {idx}: {name}' for idx, name in enumerate(MATERIAL_CLASSES))
    with open(yaml_path, 'w') as yaml_file:
        yaml_file.write(
            f'path: {abs_output_root}\n'
            f'train: images/train\n'
            f'val: images/val\n'
            f'names:\n'
            f'{names_lines}\n'
        )
    return yaml_path


def write_readme(output_root, build_command):
    with open(os.path.join(output_root, 'README.md'), 'w') as readme_file:
        readme_file.write(
            README_TEMPLATE.format(build_command=build_command))


def write_report(output_root, report_rows):
    report_path = os.path.join(output_root, 'autolabel_report.csv')
    with open(report_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in report_rows:
            writer.writerow(row)
    return report_path


def print_summary(report_rows):
    print()
    print('== class-level summary (labeled/no_box/error/excluded) ==')
    by_class = defaultdict(Counter)
    for row in report_rows:
        by_class[row['target_class']][row['status']] += 1
    print(f'{"class":<10}{"total":>8}{"labeled":>10}{"no_box":>10}'
          f'{"excluded":>10}{"error":>8}')
    for class_name in MATERIAL_CLASSES:
        counts = by_class.get(class_name, Counter())
        total = sum(counts.values())
        print(f'{class_name:<10}{total:>8}{counts["labeled"]:>10}'
              f'{counts["no_box"]:>10}{counts["excluded"]:>10}'
              f'{counts["error"]:>8}')

    print()
    print('== source-type summary ==')
    by_source = Counter(row['source_type'] for row in report_rows)
    for source_type, count in by_source.most_common():
        print(f'  {source_type:<16} {count}')

    print()
    print('== warning summary (labeled + excluded rows) ==')
    warning_counter = Counter()
    for row in report_rows:
        if row['status'] not in ('labeled', 'excluded'):
            continue
        flags = row['warning_flags']
        warning_counter[flags if flags else '(none)'] += 1
    for flags, count in warning_counter.most_common():
        print(f'  {flags:<40} {count}')


def main():
    args = parse_args()
    build_command = 'python3 ' + ' '.join(
        [os.path.basename(__file__)] +
        [a for a in sys.argv[1:]])

    images_by_class = discover_candidates_v1(args.candidate_root)
    total_candidates = sum(len(v) for v in images_by_class.values())

    v0_entries = []
    if args.include_v0_remap:
        v0_entries = discover_v0_remap_entries(
            args.v0_root, args.exclude_v0_paper)
        for entry in v0_entries:
            entry['source_type'] = 'v0_remapped'
            entry['status'] = 'labeled'

    if total_candidates == 0 and not v0_entries:
        print(f'ERROR: no candidates_v1 images found under '
              f'{args.candidate_root} and no v0-remap entries available',
              file=sys.stderr)
        sys.exit(1)

    candidates_entries = []
    if total_candidates > 0:
        try:
            labeler = GroundingDINOLabeler(
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold)
        except ImportError as exc:
            print(
                'ERROR: autodistill / autodistill-grounding-dino / '
                'supervision are required to label candidates_v1 '
                'images. Install them in an isolated venv first:\n'
                '  python3 -m venv .venv-autolabel\n'
                '  source .venv-autolabel/bin/activate\n'
                '  pip install --upgrade pip\n'
                '  pip install -r tools/requirements_autolabel.txt\n'
                f'Original import error: {exc}', file=sys.stderr)
            sys.exit(1)

        candidates_entries = process_candidates_v1(
            labeler, images_by_class, args.val_ratio, args.seed,
            args.limit_per_class)
        for entry in candidates_entries:
            entry['source_type'] = 'candidates_v1'

    all_entries = candidates_entries + v0_entries

    report_rows = write_output(
        all_entries, args.output_root, args.exclude_warned,
        args.exclude_too_large, args.preview_limit_per_class)

    yaml_path = write_yaml(args.output_root)
    write_readme(args.output_root, build_command)
    report_path = write_report(args.output_root, report_rows)

    print_summary(report_rows)

    print()
    print(f'Wrote {report_path}')
    print(f'Dataset written to {args.output_root}')
    print(f'Config: {yaml_path}')


if __name__ == '__main__':
    main()
