#!/usr/bin/env python3
"""Aggregate a confidence_threshold x input_size sweep (produced by
tools/run_vision_threshold_sweep.sh) into one CSV/Markdown summary.

Reads logs under
  {sweep-dir}/conf_<threshold>/yolo11n_<size>.log
(one directory per confidence_threshold, one log per input_size inside
each) and reuses analyze_real_image_detections.py's log parser
(same [VisionPerf]/ONNX detection/[PerceptionPolicy] line formats,
same expected_class/condition/status rules) so this script stays
consistent with the single-threshold analysis instead of re-implementing
the parsing logic.

For each (threshold, input_size) combination, computes:
  - image-level detection stability: total_images, correct_or_reject,
    no_detection, misclassified_or_false_known, image_level_success_rate
  - policy decision counts: ACCEPT_SORT, ROUTE_TO_REJECT,
    SKIP_NO_DETECTION, RETRY_VIEW, MANUAL_REVIEW

Writes:
  - results/threshold_sweep_summary.csv (one row per threshold x
    input_size combination)
  - results/threshold_sweep_summary.md  (report with sections:
    Experiment Purpose, Dataset, Thresholds and Input Sizes, Overall
    Summary Table, Policy-level Summary Table, Key Findings,
    Limitations, Next Steps)

Usage:
    python3 tools/analyze_threshold_sweep_results.py
    python3 tools/analyze_threshold_sweep_results.py \\
        --sweep-dir logs/vision_benchmark/test_images_real/vision_only_threshold_sweep \\
        --output-csv results/threshold_sweep_summary.csv \\
        --output-md results/threshold_sweep_summary.md
"""
import argparse
import csv
import os
import re
from collections import Counter, defaultdict

import analyze_real_image_detections as ard

CONF_DIR_RE = re.compile(r'^conf_(?P<threshold>[\d.]+)$')

CSV_FIELDNAMES = [
    'threshold',
    'input_size',
    'total_images',
    'correct_or_reject',
    'no_detection',
    'misclassified_or_false_known',
    'image_level_success_rate',
    'ACCEPT_SORT',
    'ROUTE_TO_REJECT',
    'SKIP_NO_DETECTION',
    'RETRY_VIEW',
    'MANUAL_REVIEW',
]


def discover_threshold_dirs(sweep_dir):
    thresholds = {}
    for name in sorted(os.listdir(sweep_dir)):
        match = CONF_DIR_RE.match(name)
        full_path = os.path.join(sweep_dir, name)
        if match and os.path.isdir(full_path):
            thresholds[float(match.group('threshold'))] = full_path
    return thresholds


def parse_sweep(sweep_dir, sizes, model_stem='yolo11n'):
    """Returns a flat list of per-image records, each tagged with
    'threshold', reusing analyze_real_image_detections.parse_log() so the
    expected_class/condition/status/policy parsing rules stay identical
    to the single-threshold analysis."""
    threshold_dirs = discover_threshold_dirs(sweep_dir)
    if not threshold_dirs:
        raise SystemExit(
            f"No 'conf_<threshold>' directories found under {sweep_dir}")

    all_records = []
    for threshold, conf_dir in sorted(threshold_dirs.items()):
        for size in sizes:
            log_path = os.path.join(conf_dir, f'{model_stem}_{size}.log')
            if not os.path.isfile(log_path):
                print(f'SKIP threshold={threshold} size={size}: log not '
                      f'found at {log_path}')
                continue
            records = ard.parse_log(log_path)
            for record in records:
                record['threshold'] = threshold
            print(f'Parsed {log_path}: {len(records)} image record(s)')
            all_records.extend(records)
    return all_records


def compute_summary(records):
    by_group = defaultdict(list)
    for record in records:
        by_group[(record['threshold'], record['input_size'])].append(
            record)

    summary = {}
    for key, group in by_group.items():
        total = len(group)
        correct_or_reject = sum(
            1 for r in group if r['status'] in ('correct', 'correct_reject'))
        no_detection = sum(1 for r in group if r['status'] == 'no_detection')
        misclassified_or_false_known = sum(
            1 for r in group
            if r['status'] in ('misclassified', 'false_known'))
        policy_counts = Counter(
            r['policy_decision'] for r in group if r['policy_decision'])

        summary[key] = {
            'total_images': total,
            'correct_or_reject': correct_or_reject,
            'no_detection': no_detection,
            'misclassified_or_false_known': misclassified_or_false_known,
            'image_level_success_rate':
                correct_or_reject / total if total else 0.0,
            'policy_counts': policy_counts,
        }
    return summary


def build_csv_row(threshold, input_size, stats):
    counts = stats['policy_counts']
    return {
        'threshold': threshold,
        'input_size': input_size,
        'total_images': stats['total_images'],
        'correct_or_reject': stats['correct_or_reject'],
        'no_detection': stats['no_detection'],
        'misclassified_or_false_known':
            stats['misclassified_or_false_known'],
        'image_level_success_rate':
            round(stats['image_level_success_rate'], 3),
        'ACCEPT_SORT': counts.get('ACCEPT_SORT', 0),
        'ROUTE_TO_REJECT': counts.get('ROUTE_TO_REJECT', 0),
        'SKIP_NO_DETECTION': counts.get('SKIP_NO_DETECTION', 0),
        'RETRY_VIEW': counts.get('RETRY_VIEW', 0),
        'MANUAL_REVIEW': counts.get('MANUAL_REVIEW', 0),
    }


def write_csv(summary, thresholds, sizes, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for threshold in thresholds:
            for size in sizes:
                stats = summary.get((threshold, size))
                if not stats:
                    continue
                writer.writerow(build_csv_row(threshold, size, stats))


def aggregate_by_threshold(summary, thresholds, sizes):
    """Sums stability/policy counts across input sizes for each threshold,
    used for the threshold-direction findings below."""
    totals = {}
    for threshold in thresholds:
        agg = defaultdict(int)
        for size in sizes:
            stats = summary.get((threshold, size))
            if not stats:
                continue
            agg['total_images'] += stats['total_images']
            agg['no_detection'] += stats['no_detection']
            agg['misclassified_or_false_known'] += \
                stats['misclassified_or_false_known']
            for decision, count in stats['policy_counts'].items():
                agg[decision] += count
        totals[threshold] = agg
    return totals


def compute_key_findings(summary, thresholds, sizes):
    findings = []
    if not summary:
        return ['- Not enough data to compute automated findings (no '
                'records parsed).']

    def describe_combo(key):
        threshold, size = key
        return f'threshold={threshold}, input_size={size}'

    # 1. highest image_level_success_rate combo(s)
    best_rate = max(s['image_level_success_rate'] for s in summary.values())
    best_keys = sorted(
        k for k, s in summary.items()
        if s['image_level_success_rate'] == best_rate)
    findings.append(
        f'- Highest image_level_success_rate: '
        f'{", ".join(describe_combo(k) for k in best_keys)} at '
        f'{best_rate * 100:.1f}%.')

    # 2. lowest no_detection combo(s)
    min_no_detection = min(s['no_detection'] for s in summary.values())
    min_nd_keys = sorted(
        k for k, s in summary.items()
        if s['no_detection'] == min_no_detection)
    findings.append(
        f'- Lowest no_detection count: '
        f'{", ".join(describe_combo(k) for k in min_nd_keys)} with '
        f'{min_no_detection} no_detection image(s).')

    # 3. most ACCEPT_SORT combo(s)
    accept_counts = {
        k: s['policy_counts'].get('ACCEPT_SORT', 0)
        for k, s in summary.items()}
    max_accept = max(accept_counts.values())
    max_accept_keys = sorted(
        k for k, v in accept_counts.items() if v == max_accept)
    findings.append(
        f'- Most ACCEPT_SORT decisions: '
        f'{", ".join(describe_combo(k) for k in max_accept_keys)} with '
        f'{max_accept} ACCEPT_SORT image(s).')

    # 4. most SKIP_NO_DETECTION combo(s)
    skip_counts = {
        k: s['policy_counts'].get('SKIP_NO_DETECTION', 0)
        for k, s in summary.items()}
    max_skip = max(skip_counts.values())
    max_skip_keys = sorted(
        k for k, v in skip_counts.items() if v == max_skip)
    findings.append(
        f'- Most SKIP_NO_DETECTION decisions: '
        f'{", ".join(describe_combo(k) for k in max_skip_keys)} with '
        f'{max_skip} SKIP_NO_DETECTION image(s).')

    # 5/6. threshold-direction trends, aggregated across input sizes.
    thresholds_sorted = sorted(thresholds)
    if len(thresholds_sorted) >= 2:
        totals = aggregate_by_threshold(summary, thresholds_sorted, sizes)
        lowest_t, highest_t = thresholds_sorted[0], thresholds_sorted[-1]
        low = totals[lowest_t]
        high = totals[highest_t]

        nd_delta = low['no_detection'] - high['no_detection']
        mis_delta = low['misclassified_or_false_known'] - \
            high['misclassified_or_false_known']
        nd_direction = 'lower' if nd_delta < 0 else (
            'higher' if nd_delta > 0 else 'the same')
        mis_direction = 'lower' if mis_delta < 0 else (
            'higher' if mis_delta > 0 else 'the same')
        pattern_confirmed = nd_delta < 0 and mis_delta > 0
        findings.append(
            f'- Lowering the threshold from {highest_t} to {lowest_t} '
            f'(summed across all input sizes): no_detection went from '
            f'{high["no_detection"]} to {low["no_detection"]} ({nd_direction} '
            f'at the lower threshold), misclassified_or_false_known went '
            f'from {high["misclassified_or_false_known"]} to '
            f'{low["misclassified_or_false_known"]} ({mis_direction} at the '
            f'lower threshold). This '
            f'{"confirms" if pattern_confirmed else "does NOT confirm"} '
            f'the expected pattern of "lower threshold -> fewer '
            f'no_detection, more misclassified" on this dataset.')

        accept_low = low.get('ACCEPT_SORT', 0)
        accept_high = high.get('ACCEPT_SORT', 0)
        reject_or_skip_low = (
            low.get('SKIP_NO_DETECTION', 0) + low.get('ROUTE_TO_REJECT', 0))
        reject_or_skip_high = (
            high.get('SKIP_NO_DETECTION', 0)
            + high.get('ROUTE_TO_REJECT', 0))
        accept_delta = accept_high - accept_low
        reject_delta = reject_or_skip_high - reject_or_skip_low
        pattern_confirmed_2 = accept_delta < 0 and reject_delta > 0
        findings.append(
            f'- Raising the threshold from {lowest_t} to {highest_t} '
            f'(summed across all input sizes): ACCEPT_SORT went from '
            f'{accept_low} to {accept_high} '
            f'({"down" if accept_delta < 0 else "up" if accept_delta > 0 else "unchanged"}), '
            f'SKIP_NO_DETECTION+ROUTE_TO_REJECT went from '
            f'{reject_or_skip_low} to {reject_or_skip_high} '
            f'({"up" if reject_delta > 0 else "down" if reject_delta < 0 else "unchanged"}). '
            f'This '
            f'{"confirms" if pattern_confirmed_2 else "does NOT confirm"} '
            f'the expected pattern of "higher threshold -> fewer '
            f'ACCEPT_SORT, more SKIP/REJECT" on this dataset.')

    return findings


def write_markdown(summary, thresholds, sizes, sweep_dir, md_path):
    thresholds_sorted = sorted(thresholds)
    sizes_sorted = sorted(sizes, reverse=True)

    lines = []
    lines.append('# Confidence Threshold Sweep Summary')
    lines.append('')

    lines.append('## Experiment Purpose')
    lines.append('')
    lines.append(
        'The real-image vision_only benchmark and detection-stability '
        'analysis so far used a single confidence_threshold=0.5. This '
        'sweep runs the same test_images_real/ dataset at multiple '
        'confidence_threshold values (with policy_confidence_threshold '
        'matched to the same value) across all three input sizes, to see '
        'how no_detection, misclassified, and the failure-aware policy '
        'decisions (ACCEPT_SORT/ROUTE_TO_REJECT/SKIP_NO_DETECTION/'
        'RETRY_VIEW/MANUAL_REVIEW) actually shift as the threshold moves, '
        'instead of assuming the trend from first principles.'
    )
    lines.append('')

    lines.append('## Dataset')
    lines.append('')
    lines.append(f'- Sweep log directory: `{sweep_dir}`')
    lines.append(
        '- Dataset: test_images_real/ (recursive_image_folder=true), '
        'same 50 images (5 classes x 10 shooting conditions) used in the '
        'single-threshold real-image analysis')
    lines.append('- benchmark_mode=vision_only for every run (no '
                  'task_manager/MoveIt)')
    lines.append('')

    lines.append('## Thresholds and Input Sizes')
    lines.append('')
    lines.append(
        f'- confidence_threshold values swept: '
        f'{", ".join(str(t) for t in thresholds_sorted)} '
        f'(policy_confidence_threshold set equal to each)')
    lines.append(
        f'- input_size values swept: '
        f'{", ".join(str(s) for s in sizes_sorted)}')
    lines.append(
        f'- Total combinations: {len(thresholds_sorted) * len(sizes_sorted)} '
        f'({len(thresholds_sorted)} thresholds x {len(sizes_sorted)} sizes)')
    lines.append('')

    lines.append('## Overall Summary Table')
    lines.append('')
    lines.append(
        '| threshold | input_size | total_images | correct_or_reject | '
        'no_detection | misclassified_or_false_known | '
        'image_level_success_rate |')
    lines.append('|---|---|---|---|---|---|---|')
    for threshold in thresholds_sorted:
        for size in sizes_sorted:
            stats = summary.get((threshold, size))
            if not stats:
                continue
            lines.append(
                f'| {threshold} | {size} | {stats["total_images"]} | '
                f'{stats["correct_or_reject"]} | {stats["no_detection"]} | '
                f'{stats["misclassified_or_false_known"]} | '
                f'{stats["image_level_success_rate"] * 100:.1f}% |')
    lines.append('')

    lines.append('## Policy-level Summary Table')
    lines.append('')
    lines.append(
        '| threshold | input_size | ' +
        ' | '.join(ard.POLICY_DECISIONS) + ' |')
    lines.append('|---|---|' + '---|' * len(ard.POLICY_DECISIONS))
    for threshold in thresholds_sorted:
        for size in sizes_sorted:
            stats = summary.get((threshold, size))
            if not stats:
                continue
            counts = stats['policy_counts']
            lines.append(
                f'| {threshold} | {size} | ' +
                ' | '.join(str(counts.get(d, 0))
                           for d in ard.POLICY_DECISIONS) + ' |')
    lines.append('')

    lines.append('## Key Findings')
    lines.append('')
    for finding in compute_key_findings(summary, thresholds, sizes):
        lines.append(finding)
    lines.append('')

    lines.append('## Limitations')
    lines.append('')
    lines.append(
        '- Same 50-image dataset as the single-threshold analysis -- '
        'still no ground-truth bounding boxes, and 10 images per class is '
        'a small sample per (threshold, input_size, class) cell.')
    lines.append(
        '- Only 3 threshold values (0.3/0.5/0.7) were tested; the real '
        'decision boundary between "too strict" and "too loose" could '
        'sit anywhere between them and wasn\'t swept continuously.')
    lines.append(
        '- policy_confidence_threshold was always kept equal to '
        'confidence_threshold in this sweep, so this cannot separate the '
        'effect of the ONNX postprocess filter from the effect of the '
        'perception policy\'s own confidence gate -- a follow-up sweep '
        'would need to vary them independently.')
    lines.append(
        '- can/glass_bottle still cannot score ACCEPT_SORT at any '
        'threshold (see the single-threshold analysis\'s mapping-gap '
        'finding) -- this sweep does not change that structural gap.')
    lines.append(
        '- Each (threshold, input_size) combination was only run once; no '
        'repeated trials to quantify run-to-run variance.')
    lines.append('')

    lines.append('## Next Steps')
    lines.append('')
    lines.append(
        '- Narrow the sweep around whichever threshold in this run looked '
        'best (see Key Findings) with finer steps (e.g. 0.4, 0.45, 0.55, '
        '0.6) to find the actual local optimum instead of just comparing '
        'three coarse points.')
    lines.append(
        '- Sweep confidence_threshold and policy_confidence_threshold '
        'independently to see whether the ONNX-level filter or the '
        'policy-level gate is the bigger lever on ACCEPT_SORT/RETRY_VIEW '
        'rates.')
    lines.append(
        '- Once a preferred threshold is chosen, validate it with a '
        'benchmark_mode=end_to_end run on a representative subset to '
        'confirm the sorting pipeline still behaves as expected end to '
        'end, not just at the vision-only/policy level.')
    lines.append(
        '- Extend the sweep to a larger/more varied image set before '
        'treating any single threshold as final, per the sample-size '
        'limitations already noted in the single-threshold analysis.')
    lines.append('')

    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, 'w') as md_file:
        md_file.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--sweep-dir',
        default=(
            'logs/vision_benchmark/test_images_real/'
            'vision_only_threshold_sweep'),
        help="Directory containing conf_<threshold>/yolo11n_<size>.log")
    parser.add_argument(
        '--sizes', type=int, nargs='+', default=[640, 416, 320],
        help='Input sizes to look for (default: 640 416 320)')
    parser.add_argument(
        '--model-stem', default='yolo11n',
        help="Log filename stem, matching <stem>_<size>.log (default: "
             "'yolo11n')")
    parser.add_argument(
        '--output-csv', default='results/threshold_sweep_summary.csv')
    parser.add_argument(
        '--output-md', default='results/threshold_sweep_summary.md')
    args = parser.parse_args()

    records = parse_sweep(args.sweep_dir, args.sizes, args.model_stem)
    if not records:
        print('No logs parsed; nothing to write.')
        return

    thresholds = sorted({record['threshold'] for record in records})
    sizes = sorted({record['input_size'] for record in records},
                   reverse=True)
    summary = compute_summary(records)

    write_csv(summary, thresholds, sizes, args.output_csv)
    print(f'Wrote {args.output_csv} ({len(thresholds) * len(sizes)} rows)')

    write_markdown(summary, thresholds, sizes, args.sweep_dir,
                    args.output_md)
    print(f'Wrote {args.output_md}')


if __name__ == '__main__':
    main()
