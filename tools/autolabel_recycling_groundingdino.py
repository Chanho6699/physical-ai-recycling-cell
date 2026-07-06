#!/usr/bin/env python3
"""GroundingDINO zero-shot auto-labeling: turn the human-curated
classification-style candidate folders under datasets/recycling_yolo_
candidates/{plastic,paper,can,glass_bottle}/ into a YOLO detection
pseudo-label dataset (v0) for custom YOLO fine-tuning.

Why this exists: the Kaggle Garbage Classification dataset (and this
project's own human-curated subset of it) has no bounding-box
annotations -- it's an image classification dataset. YOLO detection
fine-tuning needs bbox labels. Since this is a v0 dataset meant to
validate the training/export/benchmark pipeline (not a final production
dataset), we generate pseudo-labels with GroundingDINO zero-shot
detection instead of hand-annotating ~2000 images up front. See
docs/groundingdino_autolabeling_plan.md for the full rationale and the
human-validation plan before these labels are trusted for anything beyond
pipeline validation.

For each candidate image, only the text prompts belonging to that image's
OWN expected class are used (a "can" image is only ever queried with
can-prompts, never with paper/plastic/glass prompts) -- this keeps the
auto-labeler from cross-contaminating classes even though GroundingDINO
itself is class-agnostic. Among all boxes GroundingDINO returns across
that class's prompts for one image, only the single highest-confidence
box is kept (one object per image, matching the "one candidate image usually
shows one primary recyclable item" assumption behind this dataset).

Requires autodistill + autodistill-grounding-dino + supervision (see
tools/requirements_autolabel.txt) -- these pull in their own torch/
transformers versions, so run this in an isolated virtualenv, not the
ROS2/ament Python environment:

    python3 -m venv .venv-autolabel
    source .venv-autolabel/bin/activate
    pip install --upgrade pip
    pip install -r tools/requirements_autolabel.txt

Usage:
    python3 tools/autolabel_recycling_groundingdino.py \\
        --input-dir datasets/recycling_yolo_candidates \\
        --output-dir datasets/recycling_yolo_autolabel_v0 \\
        --val-ratio 0.2 \\
        --box-threshold 0.35 \\
        --text-threshold 0.25 \\
        --preview-limit-per-class 10
"""
import argparse
import csv
import os
import random
import shutil
import sys

CLASS_NAMES = ('plastic', 'paper', 'can', 'glass_bottle')

CLASS_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Only this class's own prompts are ever used for its own images -- a
# "can" image is never queried against paper/plastic/glass prompts, so
# the auto-labeler can't cross-contaminate classes even though
# GroundingDINO itself has no notion of "this image is a can image".
PROMPT_MAP = {
    'plastic': [
        'plastic bottle',
        'plastic cup',
        'plastic container',
        'plastic waste',
    ],
    'paper': [
        'paper',
        'paper sheet',
        'crumpled paper',
    ],
    'can': [
        'aluminum can',
        'soda can',
        'metal can',
    ],
    'glass_bottle': [
        'glass bottle',
    ],
}

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

CSV_FIELDNAMES = [
    'image_path',
    'split',
    'expected_class',
    'selected_prompt',
    'status',
    'confidence',
    'x1',
    'y1',
    'x2',
    'y2',
    'output_image',
    'output_label',
]

README_TEMPLATE = """\
# recycling_yolo_autolabel_v0

## What this dataset is

The Kaggle Garbage Classification dataset is an **image classification**
dataset -- it has no bounding-box annotations. This project's robot
perception task needs YOLO **detection** training data, which requires
bbox labels. This dataset (`recycling_yolo_autolabel_v0`) is a
**GroundingDINO zero-shot pseudo-label v0 dataset**: bounding boxes were
generated automatically by prompting GroundingDINO with per-class text
descriptions, not drawn by a human annotator.

## Class taxonomy

The Kaggle classes (cardboard/glass/metal/paper/plastic/trash) were
re-mapped to match what the source images actually contain for this
project, not the original Kaggle label names:

| class_id | name         | bin mapping   | source folder meaning                |
|---|---|---|---|
| 0 | plastic      | plastic_bin | plastic waste generally, not just bottles |
| 1 | paper        | paper_bin   | paper/paper sheets, not paper cups        |
| 2 | can          | metal_bin   | metal cans (Kaggle "metal")               |
| 3 | glass_bottle | glass_bin   | glass bottles (Kaggle "glass")            |

`unknown` (not part of this dataset's 4 classes) still routes to
`reject_bin` at the task_manager level, unchanged from the earlier
pretrained-COCO setup.

## Pseudo-label quality warning

**These labels are a draft, not ground truth.** They were produced by an
automatic pipeline (`tools/autolabel_recycling_groundingdino.py`) with no
human review of individual boxes. Expect some fraction of:
- false positives (a box drawn where there's no real object of that class)
- `no_box` images (GroundingDINO found nothing above threshold)
- wrong object (box drawn around something other than the intended item)
- oversized/undersized boxes (loose or overly tight fit)

**Do not use this dataset for final robot grasp pose or end-to-end
sorting decisions without human correction of the labels first.** See
`docs/groundingdino_autolabeling_plan.md` for the human-validation plan.

## Purpose of this v0 dataset

To validate the custom-YOLO training -> ONNX export -> benchmark pipeline
end-to-end on the project's own class taxonomy, before investing in a
fully human-reviewed dataset.

## Files

- `images/train/`, `images/val/` -- copied candidate images
- `labels/train/`, `labels/val/` -- YOLO-format label .txt (class_id
  x_center y_center width height, normalized 0-1); images with no
  GroundingDINO detection above threshold have NO label file (contrast
  with an empty label file, which YOLO treats as "no objects", vs. a
  missing image entirely -- here the image is still included in
  images/, just without a paired label file, so `no_box` images stay
  visible in the report)
- `recycling.yaml` -- Ultralytics dataset config
- `autolabel_report.csv` -- per-image status (labeled/no_box/error),
  selected prompt, confidence, and bbox for every candidate image
- `previews/` -- a sample of images with the selected bbox drawn on
  them, for quick visual sanity-checking before training
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input-dir', default='datasets/recycling_yolo_candidates',
        help='Directory containing <class>/ subfolders of candidate '
             'images (default: datasets/recycling_yolo_candidates)')
    parser.add_argument(
        '--output-dir', default='datasets/recycling_yolo_autolabel_v0',
        help='Directory to write the YOLO dataset into (default: '
             'datasets/recycling_yolo_autolabel_v0)')
    parser.add_argument(
        '--val-ratio', type=float, default=0.2,
        help='Fraction of each class held out for validation (default: '
             '0.2)')
    parser.add_argument(
        '--box-threshold', type=float, default=0.35,
        help='GroundingDINO box confidence threshold (default: 0.35)')
    parser.add_argument(
        '--text-threshold', type=float, default=0.25,
        help='GroundingDINO text-match threshold (default: 0.25)')
    parser.add_argument(
        '--preview-limit-per-class', type=int, default=10,
        help='Max number of preview (bbox-drawn) images saved per class '
             '(default: 10)')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for the train/val split shuffle (default: 42)')
    parser.add_argument(
        '--limit-per-class', type=int, default=None,
        help='Optional cap on images processed per class, for a quick '
             'smoke run before committing to the full dataset (default: '
             'no cap)')
    return parser.parse_args()


def discover_candidate_images(input_dir):
    """Returns {class_name: [image_path, ...]} for every supported class
    folder found under input_dir, sorted by path for determinism."""
    images_by_class = {}
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"WARN: expected class folder '{class_name}' not found "
                  f"under {input_dir}, skipping")
            continue
        images = sorted(
            os.path.join(class_dir, name)
            for name in os.listdir(class_dir)
            if name.lower().endswith(IMAGE_EXTENSIONS)
        )
        images_by_class[class_name] = images
    return images_by_class


def split_train_val(images, val_ratio, seed):
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    val_count = round(len(shuffled) * val_ratio)
    val_images = shuffled[:val_count]
    train_images = shuffled[val_count:]
    return train_images, val_images


def bbox_to_yolo_line(class_id, x1, y1, x2, y2, img_w, img_h):
    # GroundingDINO boxes can land a few pixels outside the image bounds
    # (observed in practice, e.g. y1=-0.2) -- clamp before normalizing so
    # the label file is always valid YOLO format (0-1 inclusive).
    x1 = min(max(x1, 0.0), img_w)
    y1 = min(max(y1, 0.0), img_h)
    x2 = min(max(x2, 0.0), img_w)
    y2 = min(max(y2, 0.0), img_h)

    x_center = ((x1 + x2) / 2.0) / img_w
    y_center = ((y1 + y2) / 2.0) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return (f'{class_id} {x_center:.6f} {y_center:.6f} '
            f'{width:.6f} {height:.6f}')


class GroundingDINOLabeler:
    """One GroundingDINO model instance per expected_class, restricted to
    that class's own prompt list, so cross-class prompts never run on an
    image from a different class's folder.

    Each of a class's prompts is registered under its own synthetic
    ontology label (e.g. "plastic::0", "plastic::1", ...) instead of
    merging them all into one output label -- this is what lets us
    recover *which specific prompt* produced the winning box
    (autolabel_report.csv's selected_prompt column), since
    CaptionOntology otherwise only tells you the final merged class, not
    the source caption.
    """

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

    def label(self, image_path, expected_class):
        """Returns None if no detection, else a dict with confidence,
        bbox (x1, y1, x2, y2), and the winning prompt string."""
        model = self._models[expected_class]
        prompts = self._prompts_by_class[expected_class]

        detections = model.predict(image_path)
        if detections is None or len(detections) == 0:
            return None

        best_idx = int(detections.confidence.argmax())
        x1, y1, x2, y2 = [float(v) for v in detections.xyxy[best_idx]]
        confidence = float(detections.confidence[best_idx])
        prompt_index = int(detections.class_id[best_idx])
        selected_prompt = prompts[prompt_index]

        return {
            'confidence': confidence,
            'bbox': (x1, y1, x2, y2),
            'selected_prompt': selected_prompt,
        }


def draw_preview(image_bgr, bbox, label_text):
    import cv2

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    preview = image_bgr.copy()
    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        preview, label_text, (x1, max(0, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return preview


def process_split(labeler, images, split, class_name, output_dir,
                   preview_limit, preview_counts, report_rows):
    import cv2

    images_dir = os.path.join(output_dir, 'images', split)
    labels_dir = os.path.join(output_dir, 'labels', split)
    previews_dir = os.path.join(output_dir, 'previews')

    labeled_count = 0
    no_box_count = 0
    error_count = 0

    for index, image_path in enumerate(images, start=1):
        ext = os.path.splitext(image_path)[1].lower()
        stem = f'{class_name}_{index:06d}'
        output_image_name = f'{stem}{ext}'
        output_image_path = os.path.join(images_dir, output_image_name)

        row = {
            'image_path': image_path,
            'split': split,
            'expected_class': class_name,
            'selected_prompt': '',
            'status': '',
            'confidence': '',
            'x1': '', 'y1': '', 'x2': '', 'y2': '',
            'output_image': output_image_name,
            'output_label': '',
        }

        try:
            frame = cv2.imread(image_path)
            if frame is None:
                raise ValueError(f'cv2.imread returned None for '
                                  f'{image_path}')
            img_h, img_w = frame.shape[:2]

            result = labeler.label(image_path, class_name)
            shutil.copy(image_path, output_image_path)

            if result is None:
                row['status'] = 'no_box'
                no_box_count += 1
            else:
                x1, y1, x2, y2 = result['bbox']
                label_line = bbox_to_yolo_line(
                    CLASS_ID[class_name], x1, y1, x2, y2, img_w, img_h)
                label_name = f'{stem}.txt'
                label_path = os.path.join(labels_dir, label_name)
                with open(label_path, 'w') as label_file:
                    label_file.write(label_line + '\n')

                row.update({
                    'selected_prompt': result['selected_prompt'],
                    'status': 'labeled',
                    'confidence': f"{result['confidence']:.4f}",
                    'x1': f'{x1:.1f}', 'y1': f'{y1:.1f}',
                    'x2': f'{x2:.1f}', 'y2': f'{y2:.1f}',
                    'output_label': label_name,
                })
                labeled_count += 1

                if preview_counts[class_name] < preview_limit:
                    preview = draw_preview(
                        frame, (x1, y1, x2, y2),
                        f"{class_name} {result['confidence']:.2f}")
                    # previews/ is one flat directory shared by both
                    # splits, but `stem` restarts its index at 1 per
                    # split -- prefix with split to avoid train/val
                    # images silently overwriting each other's preview.
                    preview_path = os.path.join(
                        previews_dir, f'{split}_{stem}_preview.jpg')
                    cv2.imwrite(preview_path, preview)
                    preview_counts[class_name] += 1

        except Exception as exc:
            row['status'] = 'error'
            row['selected_prompt'] = str(exc)
            error_count += 1

        report_rows.append(row)

    return labeled_count, no_box_count, error_count


def write_yaml(output_dir):
    yaml_path = os.path.join(output_dir, 'recycling.yaml')
    abs_output_dir = os.path.abspath(output_dir)
    names_lines = '\n'.join(
        f'  {idx}: {name}' for idx, name in enumerate(CLASS_NAMES))
    content = (
        f'path: {abs_output_dir}\n'
        f'train: images/train\n'
        f'val: images/val\n'
        f'names:\n'
        f'{names_lines}\n'
    )
    with open(yaml_path, 'w') as yaml_file:
        yaml_file.write(content)


def write_readme(output_dir):
    with open(os.path.join(output_dir, 'README.md'), 'w') as readme_file:
        readme_file.write(README_TEMPLATE)


def write_report(output_dir, report_rows):
    report_path = os.path.join(output_dir, 'autolabel_report.csv')
    with open(report_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in report_rows:
            writer.writerow(row)
    return report_path


def main():
    args = parse_args()

    images_by_class = discover_candidate_images(args.input_dir)
    if not images_by_class:
        print(f'ERROR: no supported class folders found under '
              f'{args.input_dir} (expected one or more of {CLASS_NAMES})',
              file=sys.stderr)
        sys.exit(1)

    for split in ('train', 'val'):
        os.makedirs(
            os.path.join(args.output_dir, 'images', split), exist_ok=True)
        os.makedirs(
            os.path.join(args.output_dir, 'labels', split), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'previews'), exist_ok=True)

    try:
        labeler = GroundingDINOLabeler(
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold)
    except ImportError as exc:
        print(
            'ERROR: autodistill / autodistill-grounding-dino / '
            'supervision are required to run this script. Install them '
            'in an isolated venv first:\n'
            '  python3 -m venv .venv-autolabel\n'
            '  source .venv-autolabel/bin/activate\n'
            '  pip install --upgrade pip\n'
            '  pip install -r tools/requirements_autolabel.txt\n'
            f'Original import error: {exc}', file=sys.stderr)
        sys.exit(1)

    report_rows = []
    preview_counts = {name: 0 for name in CLASS_NAMES}
    summary = {}

    for class_name, images in images_by_class.items():
        if args.limit_per_class is not None:
            images = images[:args.limit_per_class]

        train_images, val_images = split_train_val(
            images, args.val_ratio, args.seed)

        total_labeled = 0
        total_no_box = 0
        total_error = 0

        for split, split_images in (('train', train_images),
                                     ('val', val_images)):
            labeled, no_box, error = process_split(
                labeler, split_images, split, class_name, args.output_dir,
                args.preview_limit_per_class, preview_counts, report_rows)
            total_labeled += labeled
            total_no_box += no_box
            total_error += error

        summary[class_name] = {
            'total': len(images),
            'labeled': total_labeled,
            'no_box': total_no_box,
            'error': total_error,
        }

    write_yaml(args.output_dir)
    write_readme(args.output_dir)
    report_path = write_report(args.output_dir, report_rows)

    print()
    print('== Auto-label summary ==')
    print(f'{"class":<14}{"total":>8}{"labeled":>10}{"no_box":>10}'
          f'{"error":>8}')
    for class_name in CLASS_NAMES:
        if class_name not in summary:
            continue
        stats = summary[class_name]
        print(f'{class_name:<14}{stats["total"]:>8}{stats["labeled"]:>10}'
              f'{stats["no_box"]:>10}{stats["error"]:>8}')
    print()
    print(f'Wrote {report_path}')
    print(f'Dataset written to {args.output_dir}')
    print(f"Config: {os.path.join(args.output_dir, 'recycling.yaml')}")


if __name__ == '__main__':
    main()
