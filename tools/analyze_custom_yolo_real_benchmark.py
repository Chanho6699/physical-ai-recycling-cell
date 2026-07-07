#!/usr/bin/env python3
"""Analyze a custom_autolabel_v0 (model_class_mode=recycling_custom)
vision_only benchmark log against test_images_real/ for ground_truth vs.
predicted_class confusion -- built to investigate a specific observed
failure mode: many images collapsing to a single predicted_class
("paper") regardless of their real content.

This is a separate, standalone tool from tools/analyze_real_image_
detections.py (which analyzes the pretrained-COCO model's correct/
no_detection/misclassified/correct_reject/false_known taxonomy) --
it does not import or share state with that script, so the two never
conflict. Ground_truth/predicted_class here are compared as plain
strings; see the "Ground truth vs. taxonomy" note in the generated
Markdown for a caveat about test_images_real/'s folder names predating
the custom model's plastic/paper taxonomy.

Reads one vision_perception_node log (produced by
tools/run_vision_size_benchmark.sh with BENCHMARK_MODE=vision_only and
MODEL_CLASS_MODE=recycling_custom) and for each image:
  - "Processing image [n/50]: <path>" marks the start of a new image;
    <path>'s first folder segment is its ground_truth (e.g.
    "can/can_angle_view_002.jpg.jpg" -> "can")
  - "ONNX detection: object_id=... class_name=<c> confidence=<p> ..."
    lines (0 or more) between that and the next PerceptionPolicy line are
    this image's detections
  - "No ONNX detections above threshold" means zero detections for this
    image (implicit: no ONNX detection lines will have been seen)
  - "[PerceptionPolicy] image=<path> decision=<d> reason=<r> ..." finalizes
    the record for <path>: predicted_class is the highest-confidence
    detection's class_name (or "no_detection" if there were none)

Writes:
  - results/custom_yolo_v0_real_image_predictions.csv (one row per image:
    image, ground_truth, predicted_class, confidence, num_detections,
    policy_decision, policy_reason)
  - results/custom_yolo_v0_real_confusion_summary.csv (ground_truth,
    predicted_class, count)
  - results/custom_yolo_v0_real_failure_analysis.md (overall accuracy,
    no_detection count, ground_truth-wise predicted_class distribution,
    an automated "paper collapse" check, and unsafe_accept examples)

Usage:
    python3 tools/analyze_custom_yolo_real_benchmark.py \\
        --log-file logs/vision_benchmark/test_images_real/vision_only/yolo11n_recycling_autolabel_v0_640.log
"""
import argparse
import csv
import os
import re
from collections import Counter, defaultdict

PROCESSING_RE = re.compile(
    r'Processing image \[(?P<index>\d+)/(?P<total>\d+)\]: (?P<image>\S+)'
)

ONNX_DETECTION_RE = re.compile(
    r'ONNX detection: object_id=(?P<object_id>\S+) '
    r'class_name=(?P<class_name>\S+) confidence=(?P<confidence>[\d.]+)'
)

POLICY_RE = re.compile(
    r'\[PerceptionPolicy\] image=(?P<image>\S+) '
    r'decision=(?P<decision>\S+) reason=(?P<reason>\S+) '
    r'selected_class=(?P<selected_class>\S+) conf=(?P<conf>\S+) '
    r'num_detections=(?P<num_detections>\d+) '
    r'recommended_action=(?P<recommended_action>\S+)'
)

NO_DETECTION_LABEL = 'no_detection'

PREDICTIONS_CSV_FIELDNAMES = [
    'image', 'ground_truth', 'predicted_class', 'confidence',
    'num_detections', 'policy_decision', 'policy_reason',
]

CONFUSION_CSV_FIELDNAMES = ['ground_truth', 'predicted_class', 'count']


def extract_ground_truth(image_path):
    parts = image_path.split('/')
    if len(parts) < 2:
        return 'unrecognized'
    return parts[0]


def parse_log(log_path):
    """Parse one vision_only benchmark log into a list of per-image
    prediction records."""
    records = []
    pending_detections = []

    with open(log_path, 'r', errors='replace') as log_file:
        for line in log_file:
            match = PROCESSING_RE.search(line)
            if match:
                # A new image starting resets the buffer -- defensive
                # only, since the PerceptionPolicy line for the previous
                # image should always have consumed it first.
                pending_detections = []
                continue

            match = ONNX_DETECTION_RE.search(line)
            if match:
                pending_detections.append((
                    match.group('class_name'),
                    float(match.group('confidence')),
                ))
                continue

            match = POLICY_RE.search(line)
            if match:
                image = match.group('image')
                ground_truth = extract_ground_truth(image)
                detections = pending_detections
                pending_detections = []

                if detections:
                    predicted_class, confidence = max(
                        detections, key=lambda pair: pair[1])
                else:
                    predicted_class, confidence = NO_DETECTION_LABEL, None

                records.append({
                    'image': image,
                    'ground_truth': ground_truth,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'num_detections': len(detections),
                    'policy_decision': match.group('decision'),
                    'policy_reason': match.group('reason'),
                    'policy_selected_class': match.group('selected_class'),
                    'policy_conf': match.group('conf'),
                })

    return records


def write_predictions_csv(records, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=PREDICTIONS_CSV_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow({
                'image': record['image'],
                'ground_truth': record['ground_truth'],
                'predicted_class': record['predicted_class'],
                'confidence': (
                    f"{record['confidence']:.2f}"
                    if record['confidence'] is not None else ''),
                'num_detections': record['num_detections'],
                'policy_decision': record['policy_decision'],
                'policy_reason': record['policy_reason'],
            })


def compute_confusion(records):
    counter = Counter(
        (r['ground_truth'], r['predicted_class']) for r in records)
    return counter


def write_confusion_csv(confusion, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=CONFUSION_CSV_FIELDNAMES)
        writer.writeheader()
        for (ground_truth, predicted_class), count in sorted(
                confusion.items()):
            writer.writerow({
                'ground_truth': ground_truth,
                'predicted_class': predicted_class,
                'count': count,
            })


def compute_paper_collapse(records, ground_truths, collapse_threshold=0.5):
    """Checks whether predictions collapsed onto a single dominant class
    (named for the specific "paper" case observed, but computed
    generically off whichever predicted_class is most common)."""
    total = len(records)
    predicted_counter = Counter(r['predicted_class'] for r in records)
    if not predicted_counter:
        return None

    dominant_class, dominant_count = predicted_counter.most_common(1)[0]
    overall_rate = dominant_count / total if total else 0.0

    non_dominant_gt_total = sum(
        1 for r in records if r['ground_truth'] != dominant_class)
    non_dominant_gt_predicted_as_dominant = sum(
        1 for r in records
        if r['ground_truth'] != dominant_class
        and r['predicted_class'] == dominant_class)
    cross_class_rate = safe_ratio(
        non_dominant_gt_predicted_as_dominant, non_dominant_gt_total)

    return {
        'dominant_class': dominant_class,
        'dominant_count': dominant_count,
        'overall_rate': overall_rate,
        'non_dominant_gt_total': non_dominant_gt_total,
        'non_dominant_gt_predicted_as_dominant':
            non_dominant_gt_predicted_as_dominant,
        'cross_class_rate': cross_class_rate,
        'collapsed': cross_class_rate >= collapse_threshold,
        'threshold': collapse_threshold,
    }


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def write_markdown(records, confusion, log_path, md_path):
    total = len(records)
    ground_truths = sorted({r['ground_truth'] for r in records})
    predicted_classes = sorted({r['predicted_class'] for r in records})

    correct = sum(
        1 for r in records if r['predicted_class'] == r['ground_truth'])
    accuracy = safe_ratio(correct, total)
    no_detection_count = sum(
        1 for r in records if r['predicted_class'] == NO_DETECTION_LABEL)

    collapse = compute_paper_collapse(records, ground_truths)

    unsafe_accepts = [
        r for r in records
        if r['policy_decision'] == 'ACCEPT_SORT'
        and r['predicted_class'] != r['ground_truth']
    ]

    lines = []
    lines.append('# Custom YOLO v0 Real-Image Failure Analysis')
    lines.append('')

    lines.append('## Experiment Purpose')
    lines.append('')
    lines.append(
        'Investigate an observed failure mode of the custom_autolabel_v0 '
        'model (model_class_mode=recycling_custom) on test_images_real/: '
        'a large fraction of images being predicted as a single class '
        '("paper") regardless of their actual content. This cross-'
        'tabulates ground_truth (from the folder name) against '
        'predicted_class (the highest-confidence ONNX detection) to '
        'quantify exactly how many can/glass_bottle/plastic_bottle/'
        'unknown images were misread as paper.'
    )
    lines.append('')

    lines.append('## Ground truth vs. taxonomy')
    lines.append('')
    lines.append(
        '**Note:** test_images_real/ still uses the older folder names '
        '(`can`, `glass_bottle`, `paper_cup`, `plastic_bottle`, '
        '`unknown`), predating the custom model\'s 4-class taxonomy '
        '(`plastic`, `paper`, `can`, `glass_bottle`). ground_truth below '
        'is the raw folder name, compared to predicted_class as plain '
        'strings -- so a `paper_cup` image correctly predicted as '
        '`paper`, or a `plastic_bottle` image correctly predicted as '
        '`plastic`, will NOT count as "correct" in the accuracy number '
        'below (exact string match only). Only `can` and `glass_bottle` '
        'are named identically in both. This does not affect the '
        'ground_truth=`unknown` rows or the paper-collapse analysis, '
        'which are unambiguous regardless of taxonomy.'
    )
    lines.append('')

    lines.append('## Dataset')
    lines.append('')
    lines.append(f'- Log file: `{log_path}`')
    lines.append(f'- Total images parsed: {total}')
    lines.append(f'- Ground truth classes: {", ".join(ground_truths)}')
    lines.append(
        f'- Predicted classes observed: {", ".join(predicted_classes)}')
    lines.append('')

    lines.append('## Overall Accuracy')
    lines.append('')
    lines.append(
        f'- Exact-match accuracy (predicted_class == ground_truth): '
        f'{correct}/{total} = {accuracy * 100:.1f}% (see taxonomy note '
        f'above -- this undercounts paper_cup/plastic_bottle correctness)')
    lines.append(
        f'- no_detection count: {no_detection_count}/{total} '
        f'({safe_ratio(no_detection_count, total) * 100:.1f}%)')
    lines.append('')

    lines.append('## Ground-truth-wise Predicted-class Distribution')
    lines.append('')
    header = ['ground_truth'] + predicted_classes + ['total']
    lines.append('| ' + ' | '.join(header) + ' |')
    lines.append('|' + '---|' * len(header))
    for ground_truth in ground_truths:
        row_total = sum(
            count for (gt, _pc), count in confusion.items() if gt == ground_truth)
        cells = [ground_truth]
        for predicted_class in predicted_classes:
            cells.append(str(confusion.get((ground_truth, predicted_class), 0)))
        cells.append(str(row_total))
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')

    lines.append('## Paper Collapse Check')
    lines.append('')
    if collapse is None:
        lines.append('_No records to analyze._')
    else:
        verdict = 'CONFIRMED' if collapse['collapsed'] else 'not confirmed'
        lines.append(
            f'- Most-predicted class overall: **{collapse["dominant_class"]}** '
            f'-- {collapse["dominant_count"]}/{total} of all predictions '
            f'({collapse["overall_rate"] * 100:.1f}%)')
        lines.append(
            f'- Of the {collapse["non_dominant_gt_total"]} images whose '
            f'ground_truth is NOT `{collapse["dominant_class"]}`, '
            f'{collapse["non_dominant_gt_predicted_as_dominant"]} '
            f'({collapse["cross_class_rate"] * 100:.1f}%) were still '
            f'predicted as `{collapse["dominant_class"]}`')
        lines.append(
            f'- Collapse verdict (>= {collapse["threshold"] * 100:.0f}% '
            f'cross-class rate): **{verdict}**')
        if collapse['collapsed']:
            per_class_breakdown = []
            for ground_truth in ground_truths:
                if ground_truth == collapse['dominant_class']:
                    continue
                gt_total = sum(
                    count for (gt, _pc), count in confusion.items()
                    if gt == ground_truth)
                gt_to_dominant = confusion.get(
                    (ground_truth, collapse['dominant_class']), 0)
                if gt_total:
                    per_class_breakdown.append(
                        f'{ground_truth}: {gt_to_dominant}/{gt_total} '
                        f'({gt_to_dominant / gt_total * 100:.0f}%)')
            lines.append(
                f'- Per-ground_truth breakdown of images predicted as '
                f'`{collapse["dominant_class"]}`: '
                f'{"; ".join(per_class_breakdown)}')
    lines.append('')

    lines.append('## Unsafe ACCEPT_SORT Examples')
    lines.append('')
    if unsafe_accepts:
        lines.append(
            f'{len(unsafe_accepts)} image(s) were auto-accepted '
            f'(policy_decision=ACCEPT_SORT) despite predicted_class not '
            f'matching ground_truth (exact-string comparison -- see '
            f'taxonomy note above for the paper_cup/plastic_bottle '
            f'caveat):')
        lines.append('')
        lines.append(
            '| image | ground_truth | predicted_class | confidence |')
        lines.append('|---|---|---|---|')
        for record in unsafe_accepts[:15]:
            lines.append(
                f'| {record["image"]} | {record["ground_truth"]} | '
                f'{record["predicted_class"]} | '
                f'{record["confidence"]:.2f} |')
        if len(unsafe_accepts) > 15:
            lines.append('')
            lines.append(
                f'... and {len(unsafe_accepts) - 15} more (see '
                f'`results/custom_yolo_v0_real_image_predictions.csv`).')
    else:
        lines.append(
            '_No unsafe ACCEPT_SORT cases found (by exact-string '
            'comparison)._')
    lines.append('')

    lines.append('## Limitations')
    lines.append('')
    lines.append(
        '- Exact-string ground_truth/predicted_class comparison only -- '
        'see the taxonomy note above; a real per-class semantic mapping '
        '(paper_cup -> paper, plastic_bottle -> plastic) would give a '
        'more meaningful accuracy number.')
    lines.append(
        '- No ground-truth bounding boxes -- this is an image-level '
        'class comparison, not an IoU/mAP evaluation.')
    lines.append(
        '- Single run at one input_size/confidence_threshold/box_'
        'threshold combination; the paper-collapse behavior has not '
        'been checked across other sizes/thresholds in this report.')
    lines.append(
        '- Root cause is not diagnosed here (e.g., whether it is a '
        'training-data imbalance, a GroundingDINO pseudo-label bias '
        'toward loose/oversized "paper"-prompt boxes, or a training '
        'convergence issue) -- this tool only quantifies the symptom.')
    lines.append('')

    lines.append('## Next Steps')
    lines.append('')
    lines.append(
        '- Inspect training data balance and pseudo-label bbox quality '
        'for the "paper" class specifically (see `datasets/recycling_'
        'yolo_autolabel_v0/previews/` and `autolabel_report.csv`) for a '
        'systematic bias (e.g., oversized boxes nearly covering the '
        'whole image, which several of the "paper" detections in this '
        'run showed).')
    lines.append(
        '- Re-run this analysis after retraining/rebalancing to confirm '
        'the collapse is resolved, not just shifted to a different '
        'dominant class.')
    lines.append(
        '- Cross-check against `results/policy_effectiveness_summary.md`'
        '-style metrics (unsafe_accept_rate) once the custom model is '
        'included in a full threshold sweep.')
    lines.append('')

    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, 'w') as md_file:
        md_file.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--log-file', required=True,
        help='Path to a vision_perception_node vision_only benchmark log '
             '(model_class_mode=recycling_custom)')
    parser.add_argument(
        '--output-csv',
        default='results/custom_yolo_v0_real_image_predictions.csv')
    parser.add_argument(
        '--output-confusion-csv',
        default='results/custom_yolo_v0_real_confusion_summary.csv')
    parser.add_argument(
        '--output-md',
        default='results/custom_yolo_v0_real_failure_analysis.md')
    args = parser.parse_args()

    if not os.path.isfile(args.log_file):
        parser.error(f'log file not found: {args.log_file}')

    records = parse_log(args.log_file)
    print(f'Parsed {args.log_file}: {len(records)} image record(s)')

    if not records:
        print('No records parsed; nothing to write.')
        return

    write_predictions_csv(records, args.output_csv)
    print(f'Wrote {args.output_csv} ({len(records)} rows)')

    confusion = compute_confusion(records)
    write_confusion_csv(confusion, args.output_confusion_csv)
    print(f'Wrote {args.output_confusion_csv} ({len(confusion)} rows)')

    write_markdown(records, confusion, args.log_file, args.output_md)
    print(f'Wrote {args.output_md}')

    print()
    predicted_counter = Counter(r['predicted_class'] for r in records)
    print('== predicted_class distribution ==')
    for predicted_class, count in predicted_counter.most_common():
        print(f'  {predicted_class:<15} {count}')


if __name__ == '__main__':
    main()
