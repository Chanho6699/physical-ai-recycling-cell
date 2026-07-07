#!/usr/bin/env python3
"""Extract every unsafe unknown->ACCEPT_SORT case from a material_v1
vision-benchmark log, at every confidence threshold in {0.5, 0.3, 0.1,
0.05} -- the follow-up to results/material_v1_real_aug_comparison.md's
finding that real_aug's unsafe-accept count (summed across thresholds)
jumped from strong's 0 to 10. That comparison only reported the COUNT;
this script reports which specific test_images_real/unknown/ images were
involved, what the model predicted, and at what confidence, so the root
cause (is it one image repeated across thresholds, or many different
ones? does it skew toward one predicted class?) can actually be
diagnosed. See docs/material_v1_real_aug_experiment_plan.md and
results/material_v1_real_aug_comparison.md for the experiment this is
following up on.

Root-cause note: perception_policy.evaluate_detections() never sees
ground truth -- ACCEPT_SORT on an `unknown/`-folder image simply means
the model's top1 detection was a known class (plastic/metal/glass/paper)
at or above confidence_threshold. So every row this script reports is
necessarily a false-positive class prediction on an image that contains
no plastic/metal/glass/paper object at all (test_images_real/unknown/
holds things like plastic bags, snack bags, tissues, a charger, a mouse,
a book -- see the folder for the exact contents).

Multiple models can be passed (each tagged by --model NAME=LOG_PATH) so
the same unknown/ images' behavior can be compared side by side (e.g.
did strong ALSO almost-trigger on the same image at a lower threshold?).
Defaults to real_aug only, matching the specific regression this script
exists to investigate.

Writes:
  - results/real_aug_unsafe_unknown_cases.csv (one row per
    model x threshold x unsafe image)
  - results/real_aug_unsafe_unknown_cases.md (the table plus a
    breakdown of which predicted class the false positives skew toward,
    whether they're threshold-specific, and which are high-confidence)

Usage:
    python3 tools/analyze_real_aug_unsafe_unknown_cases.py
    python3 tools/analyze_real_aug_unsafe_unknown_cases.py \\
        --model real_aug=logs/.../yolo11n_recycling_material_v1_real_aug_640_conf005.log \\
        --model strong=logs/.../yolo11n_recycling_material_v1_aug_strong_640_conf005.log
"""
import argparse
import csv
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'ros2_ws', 'src',
    'recycling_cell_vision', 'recycling_cell_vision'))
import perception_policy as pp  # noqa: E402

THRESHOLDS = (0.5, 0.3, 0.1, 0.05)

PROCESSING_RE = re.compile(
    r'Processing image \[(?P<index>\d+)/(?P<total>\d+)\]: (?P<image>\S+)'
)

ONNX_DETECTION_RE = re.compile(
    r'ONNX detection: object_id=(?P<object_id>\S+) '
    r'class_name=(?P<class_name>\S+) confidence=(?P<confidence>[\d.]+) '
    r'bbox=\((?P<x1>[-\d.]+), (?P<y1>[-\d.]+), (?P<x2>[-\d.]+), (?P<y2>[-\d.]+)\)'
)

DEFAULT_MODELS = {
    'real_aug': 'logs/vision_benchmark/test_images_real/vision_only/'
                'yolo11n_recycling_material_v1_real_aug_640_conf005.log',
}

CSV_FIELDNAMES = [
    'source_model',
    'image_path',
    'threshold',
    'predicted_class',
    'confidence',
    'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
    'decision',
    'reason',
    'num_detections',
    'preview_path',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--model', action='append', dest='models', default=[],
        metavar='NAME=LOG_PATH',
        help='Repeatable. A model name and its conf005 vision-benchmark '
             'log path, e.g. --model real_aug=logs/.../foo_conf005.log. '
             'Defaults to real_aug alone if not given at all.')
    parser.add_argument(
        '--output-csv',
        default='results/real_aug_unsafe_unknown_cases.csv')
    parser.add_argument(
        '--output-md',
        default='results/real_aug_unsafe_unknown_cases.md')
    return parser.parse_args()


def extract_ground_truth(image_path):
    parts = image_path.split('/')
    return parts[0] if len(parts) >= 2 else 'unrecognized'


def parse_log(log_path):
    per_image = {}
    current_image = None

    with open(log_path, 'r', errors='replace') as log_file:
        for line in log_file:
            match = PROCESSING_RE.search(line)
            if match:
                current_image = match.group('image')
                per_image[current_image] = {
                    'ground_truth': extract_ground_truth(current_image),
                    'detections': [],
                }
                continue

            match = ONNX_DETECTION_RE.search(line)
            if match and current_image is not None:
                per_image[current_image]['detections'].append({
                    'class_name': match.group('class_name'),
                    'confidence': float(match.group('confidence')),
                    'bbox': (
                        float(match.group('x1')), float(match.group('y1')),
                        float(match.group('x2')), float(match.group('y2')),
                    ),
                })

    return per_image


def find_unsafe_cases(model_name, per_image):
    """One row per (image, threshold) where ground_truth == 'unknown' and
    the policy decision at that threshold is ACCEPT_SORT."""
    rows = []
    for image, info in per_image.items():
        if info['ground_truth'] != 'unknown':
            continue
        for threshold in THRESHOLDS:
            filtered = [
                d for d in info['detections'] if d['confidence'] >= threshold
            ]
            if not filtered:
                continue
            policy_input = [
                {'class_name': d['class_name'], 'confidence': d['confidence']}
                for d in filtered
            ]
            result = pp.evaluate_detections(
                policy_input, confidence_threshold=threshold)
            if result['decision'] != 'ACCEPT_SORT':
                continue

            selected = filtered[result['selected_index']]
            preview_path = (
                f'test_images_real/{image}'
                if os.path.isfile(os.path.join('test_images_real', image))
                else '')
            rows.append({
                'source_model': model_name,
                'image_path': image,
                'threshold': threshold,
                'predicted_class': selected['class_name'],
                'confidence': selected['confidence'],
                'bbox_x1': round(selected['bbox'][0], 1),
                'bbox_y1': round(selected['bbox'][1], 1),
                'bbox_x2': round(selected['bbox'][2], 1),
                'bbox_y2': round(selected['bbox'][3], 1),
                'decision': result['decision'],
                'reason': result['reason'],
                'num_detections': result['num_detections'],
                'preview_path': preview_path,
            })
    return rows


def write_csv(rows, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in sorted(
                rows, key=lambda r: (r['source_model'], r['image_path'],
                                      -r['threshold'])):
            writer.writerow(row)


def write_markdown(rows, models, md_path):
    lines = []
    lines.append('# real_aug Unsafe unknown -> ACCEPT_SORT Cases')
    lines.append('')
    lines.append(
        'Every case where a `test_images_real/unknown/` image (an image '
        'containing no plastic/metal/glass/paper object at all -- plastic '
        'bags, snack bags, tissues, a charger, a mouse, a book, etc.) got '
        '`ACCEPT_SORT`ed anyway, at any of the 4 benchmark thresholds. '
        '`perception_policy.evaluate_detections()` never sees ground '
        'truth -- every row below is a false-positive CLASS prediction on '
        'an object that is not actually plastic/metal/glass/paper, not a '
        'bounding-box localization error on a real target object.'
    )
    lines.append('')
    lines.append('## Models analyzed')
    lines.append('')
    for name, log_path in models.items():
        lines.append(f'- {name}: `{log_path}`')
    lines.append('')

    if not rows:
        lines.append('No unsafe unknown->ACCEPT_SORT cases found for the '
                      'model(s) analyzed.')
        with open(md_path, 'w') as md_file:
            md_file.write('\n'.join(lines))
        return

    lines.append('## All Cases')
    lines.append('')
    lines.append(
        '| source_model | image_path | threshold | predicted_class | '
        'confidence | bbox | decision | reason |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for row in sorted(
            rows, key=lambda r: (r['source_model'], r['image_path'],
                                  -r['threshold'])):
        bbox = (f'({row["bbox_x1"]}, {row["bbox_y1"]}, '
                f'{row["bbox_x2"]}, {row["bbox_y2"]})')
        lines.append(
            f'| {row["source_model"]} | {row["image_path"]} | '
            f'{row["threshold"]} | {row["predicted_class"]} | '
            f'{row["confidence"]} | {bbox} | {row["decision"]} | '
            f'{row["reason"]} |')
    lines.append('')

    lines.append('## Analysis')
    lines.append('')

    by_model = {}
    for row in rows:
        by_model.setdefault(row['source_model'], []).append(row)

    for model_name, model_rows in by_model.items():
        lines.append(f'### {model_name}')
        lines.append('')

        unique_images = sorted({r['image_path'] for r in model_rows})
        lines.append(
            f'{len(model_rows)} unsafe (image, threshold) case(s) across '
            f'{len(unique_images)} unique image(s): '
            + ', '.join(f'`{img}`' for img in unique_images) + '.'
        )
        lines.append('')

        class_counter = Counter(r['predicted_class'] for r in model_rows)
        total = sum(class_counter.values())
        class_lines = ', '.join(
            f'{cls}={count} ({100*count/total:.0f}%)'
            for cls, count in class_counter.most_common())
        dominant_class, dominant_count = class_counter.most_common(1)[0]
        lines.append(
            f'**Predicted-class skew:** {class_lines}. '
            + (f'Cases skew heavily toward `{dominant_class}` '
               f'({100*dominant_count/total:.0f}% of unsafe cases) -- '
               f'this is the class most likely to be over-triggering on '
               f'non-target real-world clutter.'
               if dominant_count / total >= 0.5
               else 'No single class dominates -- the false positives are '
                    'spread across classes rather than concentrated in one.')
        )
        lines.append('')

        threshold_counter = Counter(r['threshold'] for r in model_rows)
        only_low_thresholds = all(
            t <= 0.3 for t in threshold_counter)
        appears_at_05 = 0.5 in threshold_counter
        lines.append(
            f'**Threshold spread:** cases per threshold: '
            + ', '.join(f'{t}={threshold_counter[t]}'
                        for t in THRESHOLDS if t in threshold_counter)
            + '. '
            + ('At least one case survives confidence_threshold=0.5 -- this '
               'is not just a low-threshold artifact, it would happen at '
               'the project\'s default policy_confidence_threshold too.'
               if appears_at_05 else
               'All cases are below confidence_threshold=0.5 -- raising '
               '`confidence_threshold`/`policy_confidence_threshold` back '
               'toward 0.5 (the project default) would already suppress '
               'every case listed here.')
        )
        lines.append('')

        high_conf = sorted(
            [r for r in model_rows if r['confidence'] >= 0.35],
            key=lambda r: -r['confidence'])
        if high_conf:
            lines.append(
                f'**High-confidence risky cases (>=0.35, i.e. above '
                f'perception_policy\'s own low_confidence_threshold):**')
            for r in high_conf:
                lines.append(
                    f'- `{r["image_path"]}` -> {r["predicted_class"]} '
                    f'@ {r["confidence"]} (threshold={r["threshold"]})')
        else:
            lines.append(
                '**High-confidence risky cases (>=0.35):** none -- every '
                'unsafe case here is a low-confidence false positive '
                '(highest observed: '
                f'{max(r["confidence"] for r in model_rows):.2f}), which '
                'is the least alarming version of this failure mode but '
                'still crossed ACCEPT_SORT at low enough thresholds.'
            )
        lines.append('')

    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, 'w') as md_file:
        md_file.write('\n'.join(lines))


def main():
    args = parse_args()

    models = DEFAULT_MODELS.copy() if not args.models else {}
    for spec in args.models:
        if '=' not in spec:
            print(f'ERROR: --model must be NAME=LOG_PATH, got: {spec}',
                  file=sys.stderr)
            sys.exit(1)
        name, log_path = spec.split('=', 1)
        models[name] = log_path

    all_rows = []
    for model_name, log_path in models.items():
        if not os.path.isfile(log_path):
            print(f'ERROR: log not found for {model_name}: {log_path}',
                  file=sys.stderr)
            sys.exit(1)
        per_image = parse_log(log_path)
        rows = find_unsafe_cases(model_name, per_image)
        print(f'{model_name}: {len(rows)} unsafe unknown->ACCEPT_SORT '
              f'case(s) across {len(per_image)} image(s) parsed from '
              f'{log_path}')
        all_rows.extend(rows)

    write_csv(all_rows, args.output_csv)
    print(f'Wrote {args.output_csv} ({len(all_rows)} rows)')

    write_markdown(all_rows, models, args.output_md)
    print(f'Wrote {args.output_md}')


if __name__ == '__main__':
    main()
