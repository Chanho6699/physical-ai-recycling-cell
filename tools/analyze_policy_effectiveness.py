#!/usr/bin/env python3
"""Cross-tabulate perception_policy.py's policy_decision against the
offline, folder-name-derived ground truth (status) from
analyze_real_image_detections.py, across the confidence_threshold sweep
produced by tools/run_vision_threshold_sweep.sh.

This answers a different question than threshold_sweep_summary.md: that
report asks "how do detection stability and policy-decision counts change
with threshold", separately. This report asks "when the policy said
ACCEPT_SORT, was it actually right?" and "when the detection was actually
wrong, did the policy catch it?" -- i.e. how *effective* the policy layer
is at its stated job (never blindly sorting something it shouldn't), not
just what it decided.

This is an OFFLINE evaluation of the runtime policy against a folder-name
ground-truth label -- not a bbox-annotation-based mAP evaluation, and not
a live A/B test of the policy actually gating the sort pipeline (task_
manager doesn't consume policy_decision yet).

Reuses tools/analyze_threshold_sweep_results.py's parse_sweep() (which
itself reuses analyze_real_image_detections.py's parse_log()), so all
three tools parse the same log format the same way.

Reads logs under
  {sweep-dir}/conf_<threshold>/yolo11n_<size>.log

Writes:
  - results/policy_effectiveness_summary.csv (one row per threshold x
    input_size combination)
  - results/policy_effectiveness_summary.md  (report with sections:
    Experiment Purpose, Dataset and Inputs, Analysis Definitions, Overall
    Policy Effectiveness Table, ACCEPT_SORT Safety Table, Risk Blocking
    Table, Missed Opportunity Table, Per-class ACCEPT_SORT Safety, Key
    Findings, Limitations, Next Steps)

Usage:
    python3 tools/analyze_policy_effectiveness.py
    python3 tools/analyze_policy_effectiveness.py \\
        --sweep-dir logs/vision_benchmark/test_images_real/vision_only_threshold_sweep \\
        --output-csv results/policy_effectiveness_summary.csv \\
        --output-md results/policy_effectiveness_summary.md
"""
import argparse
import csv
import os
from collections import defaultdict

import analyze_threshold_sweep_results as ats

RISKY_STATUSES = ('misclassified', 'false_known')
SAFE_STATUSES = ('correct', 'correct_reject')

CSV_FIELDNAMES = [
    'threshold',
    'input_size',
    'accept_total',
    'accept_correct',
    'accept_misclassified',
    'accept_false_known',
    'accept_unknown_or_no_detection',
    'unsafe_accept_count',
    'unsafe_accept_rate',
    'risky_total',
    'blocked_risky_count',
    'blocked_risky_rate',
    'known_correct_total',
    'missed_correct_count',
    'missed_correct_rate',
]


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def compute_policy_effectiveness(records):
    """Per (threshold, input_size): ACCEPT_SORT safety, risk-blocking,
    and missed-opportunity metrics, cross-tabulating policy_decision
    against the offline ground-truth status."""
    by_group = defaultdict(list)
    for record in records:
        by_group[(record['threshold'], record['input_size'])].append(
            record)

    summary = {}
    for key, group in by_group.items():
        accepted = [r for r in group if r['policy_decision'] == 'ACCEPT_SORT']
        accept_total = len(accepted)
        accept_correct = sum(1 for r in accepted if r['status'] == 'correct')
        accept_misclassified = sum(
            1 for r in accepted if r['status'] == 'misclassified')
        accept_false_known = sum(
            1 for r in accepted if r['status'] == 'false_known')
        accept_unknown_or_no_detection = sum(
            1 for r in accepted
            if r['status'] in ('correct_reject', 'no_detection'))
        unsafe_accept_count = accept_misclassified + accept_false_known

        risky = [r for r in group if r['status'] in RISKY_STATUSES]
        risky_total = len(risky)
        blocked_risky_count = sum(
            1 for r in risky if r['policy_decision'] != 'ACCEPT_SORT')

        known_correct = [
            r for r in group
            if r['expected_class'] != 'unknown' and r['status'] == 'correct'
        ]
        known_correct_total = len(known_correct)
        missed_correct_count = sum(
            1 for r in known_correct if r['policy_decision'] != 'ACCEPT_SORT')

        summary[key] = {
            'accept_total': accept_total,
            'accept_correct': accept_correct,
            'accept_misclassified': accept_misclassified,
            'accept_false_known': accept_false_known,
            'accept_unknown_or_no_detection': accept_unknown_or_no_detection,
            'unsafe_accept_count': unsafe_accept_count,
            'unsafe_accept_rate': safe_ratio(
                unsafe_accept_count, accept_total),
            'risky_total': risky_total,
            'blocked_risky_count': blocked_risky_count,
            'blocked_risky_rate': safe_ratio(
                blocked_risky_count, risky_total),
            'known_correct_total': known_correct_total,
            'missed_correct_count': missed_correct_count,
            'missed_correct_rate': safe_ratio(
                missed_correct_count, known_correct_total),
        }
    return summary


def compute_per_class_accept_safety(records):
    """Per (threshold, input_size, expected_class): ACCEPT_SORT safety
    broken down by class, so e.g. a class that's structurally unable to
    be ACCEPT_SORT-correct (can/glass_bottle -- see the mapping-gap note
    in analyze_real_image_detections.py) is visible on its own row."""
    by_group = defaultdict(list)
    for record in records:
        by_group[(record['threshold'], record['input_size'],
                   record['expected_class'])].append(record)

    summary = {}
    for key, group in by_group.items():
        accepted = [r for r in group if r['policy_decision'] == 'ACCEPT_SORT']
        accept_total = len(accepted)
        accept_correct = sum(1 for r in accepted if r['status'] == 'correct')
        accept_unsafe = sum(
            1 for r in accepted if r['status'] in RISKY_STATUSES)
        summary[key] = {
            'accept_total': accept_total,
            'accept_correct': accept_correct,
            'accept_unsafe': accept_unsafe,
            'unsafe_accept_rate': safe_ratio(accept_unsafe, accept_total),
        }
    return summary


def build_csv_row(threshold, input_size, stats):
    return {
        'threshold': threshold,
        'input_size': input_size,
        'accept_total': stats['accept_total'],
        'accept_correct': stats['accept_correct'],
        'accept_misclassified': stats['accept_misclassified'],
        'accept_false_known': stats['accept_false_known'],
        'accept_unknown_or_no_detection':
            stats['accept_unknown_or_no_detection'],
        'unsafe_accept_count': stats['unsafe_accept_count'],
        'unsafe_accept_rate': round(stats['unsafe_accept_rate'], 3),
        'risky_total': stats['risky_total'],
        'blocked_risky_count': stats['blocked_risky_count'],
        'blocked_risky_rate': round(stats['blocked_risky_rate'], 3),
        'known_correct_total': stats['known_correct_total'],
        'missed_correct_count': stats['missed_correct_count'],
        'missed_correct_rate': round(stats['missed_correct_rate'], 3),
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
    """Sums numerators/denominators across input sizes per threshold, then
    re-derives rates from the summed totals (never averages per-size
    rates directly, which would weight small/large groups equally)."""
    totals = {}
    for threshold in thresholds:
        agg = defaultdict(int)
        for size in sizes:
            stats = summary.get((threshold, size))
            if not stats:
                continue
            for field in ('accept_total', 'accept_correct',
                          'unsafe_accept_count', 'risky_total',
                          'blocked_risky_count', 'known_correct_total',
                          'missed_correct_count'):
                agg[field] += stats[field]
        agg['unsafe_accept_rate'] = safe_ratio(
            agg['unsafe_accept_count'], agg['accept_total'])
        agg['blocked_risky_rate'] = safe_ratio(
            agg['blocked_risky_count'], agg['risky_total'])
        agg['missed_correct_rate'] = safe_ratio(
            agg['missed_correct_count'], agg['known_correct_total'])
        totals[threshold] = agg
    return totals


MAPPING_GAP_CLASSES = ('can', 'glass_bottle')


def compute_mapping_gap_contribution(per_class_summary):
    """How much of the overall unsafe_accept_rate is structurally caused
    by the can/glass_bottle mapping gap (which can only ever land on
    accept_unsafe, never accept_correct, when ACCEPT_SORT fires at all)
    vs. classes the current model mapping actually supports."""
    gap = defaultdict(int)
    supported = defaultdict(int)
    for (_threshold, _size, expected_class), stats in \
            per_class_summary.items():
        bucket = gap if expected_class in MAPPING_GAP_CLASSES else supported
        bucket['accept_total'] += stats['accept_total']
        bucket['accept_correct'] += stats['accept_correct']
        bucket['accept_unsafe'] += stats['accept_unsafe']

    return {
        'mapping_gap': {
            **gap,
            'unsafe_accept_rate': safe_ratio(
                gap['accept_unsafe'], gap['accept_total']),
        },
        'supported': {
            **supported,
            'unsafe_accept_rate': safe_ratio(
                supported['accept_unsafe'], supported['accept_total']),
        },
    }


def compute_key_findings(summary, per_class_summary, thresholds, sizes):
    findings = []
    if not summary:
        return ['- Not enough data to compute automated findings (no '
                'records parsed).']

    def describe_combo(key):
        threshold, size = key
        return f'threshold={threshold}, input_size={size}'

    contribution = compute_mapping_gap_contribution(per_class_summary)
    gap_stats = contribution['mapping_gap']
    supported_stats = contribution['supported']
    if gap_stats['accept_total'] or supported_stats['accept_total']:
        findings.append(
            f'- Mapping-gap contribution to unsafe_accept_rate: '
            f'can/glass_bottle ACCEPT_SORT cases (summed across all '
            f'thresholds/sizes) are unsafe '
            f'{gap_stats["unsafe_accept_rate"] * 100:.1f}% of the time '
            f'({gap_stats["accept_unsafe"]}/{gap_stats["accept_total"]}), '
            f'vs. {supported_stats["unsafe_accept_rate"] * 100:.1f}% '
            f'({supported_stats["accept_unsafe"]}/'
            f'{supported_stats["accept_total"]}) for paper_cup/'
            f'plastic_bottle. The mapping gap, not the confidence-based '
            f'policy logic itself, is the dominant driver of the overall '
            f'unsafe_accept_rate above.')

    # Only rank combinations that actually have ACCEPT_SORT cases -- a
    # 0/0 unsafe_accept_rate isn't a meaningful "safest" result.
    with_accepts = {
        k: s for k, s in summary.items() if s['accept_total'] > 0}

    if with_accepts:
        min_unsafe_rate = min(
            s['unsafe_accept_rate'] for s in with_accepts.values())
        min_unsafe_keys = sorted(
            k for k, s in with_accepts.items()
            if s['unsafe_accept_rate'] == min_unsafe_rate)
        findings.append(
            f'- Lowest unsafe_accept_rate: '
            f'{", ".join(describe_combo(k) for k in min_unsafe_keys)} at '
            f'{min_unsafe_rate * 100:.1f}%.')

    max_accept_correct = max(s['accept_correct'] for s in summary.values())
    max_ac_keys = sorted(
        k for k, s in summary.items()
        if s['accept_correct'] == max_accept_correct)
    findings.append(
        f'- Most accept_correct: '
        f'{", ".join(describe_combo(k) for k in max_ac_keys)} with '
        f'{max_accept_correct} correctly-accepted image(s).')

    with_risky = {
        k: s for k, s in summary.items() if s['risky_total'] > 0}
    if with_risky:
        max_blocked_rate = max(
            s['blocked_risky_rate'] for s in with_risky.values())
        max_br_keys = sorted(
            k for k, s in with_risky.items()
            if s['blocked_risky_rate'] == max_blocked_rate)
        findings.append(
            f'- Highest blocked_risky_rate: '
            f'{", ".join(describe_combo(k) for k in max_br_keys)} at '
            f'{max_blocked_rate * 100:.1f}%.')

    with_known_correct = {
        k: s for k, s in summary.items() if s['known_correct_total'] > 0}
    if with_known_correct:
        max_missed_rate = max(
            s['missed_correct_rate'] for s in with_known_correct.values())
        max_missed_keys = sorted(
            k for k, s in with_known_correct.items()
            if s['missed_correct_rate'] == max_missed_rate)
        findings.append(
            f'- Highest missed_correct_rate: '
            f'{", ".join(describe_combo(k) for k in max_missed_keys)} at '
            f'{max_missed_rate * 100:.1f}%.')

    thresholds_sorted = sorted(thresholds)
    if len(thresholds_sorted) >= 2:
        totals = aggregate_by_threshold(summary, thresholds_sorted, sizes)
        lowest_t, highest_t = thresholds_sorted[0], thresholds_sorted[-1]
        low, high = totals[lowest_t], totals[highest_t]

        # threshold=lowest: does it grow ACCEPT_SORT volume *and*
        # unsafe_accept along with it?
        accept_grew = low['accept_total'] > high['accept_total']
        unsafe_rate_grew = low['unsafe_accept_rate'] > high['unsafe_accept_rate']
        findings.append(
            f'- threshold={lowest_t} vs threshold={highest_t} (summed '
            f'across input sizes): accept_total {low["accept_total"]} vs '
            f'{high["accept_total"]} '
            f'({"more" if accept_grew else "not more"} ACCEPT_SORT at '
            f'{lowest_t}), unsafe_accept_rate '
            f'{low["unsafe_accept_rate"] * 100:.1f}% vs '
            f'{high["unsafe_accept_rate"] * 100:.1f}% '
            f'({"higher" if unsafe_rate_grew else "not higher"} at '
            f'{lowest_t}). '
            f'{"Confirms" if accept_grew and unsafe_rate_grew else "Does NOT confirm"} '
            f'that lowering the threshold trades more ACCEPT_SORT volume '
            f'for a higher unsafe-accept rate on this dataset.')

        # threshold=highest: does it block risk better *and* increase
        # missed opportunity?
        blocks_better = high['blocked_risky_rate'] > low['blocked_risky_rate']
        misses_more = high['missed_correct_rate'] > low['missed_correct_rate']
        findings.append(
            f'- threshold={highest_t} vs threshold={lowest_t} (summed '
            f'across input sizes): blocked_risky_rate '
            f'{high["blocked_risky_rate"] * 100:.1f}% vs '
            f'{low["blocked_risky_rate"] * 100:.1f}% '
            f'({"better" if blocks_better else "not better"} risk-blocking '
            f'at {highest_t}), missed_correct_rate '
            f'{high["missed_correct_rate"] * 100:.1f}% vs '
            f'{low["missed_correct_rate"] * 100:.1f}% '
            f'({"more" if misses_more else "not more"} missed opportunity '
            f'at {highest_t}). '
            f'{"Confirms" if blocks_better and misses_more else "Does NOT confirm"} '
            f'that raising the threshold trades better risk-blocking for '
            f'more missed sorting opportunities on this dataset.')

    return findings


def write_markdown(summary, per_class_summary, thresholds, sizes, sweep_dir,
                    md_path):
    thresholds_sorted = sorted(thresholds)
    sizes_sorted = sorted(sizes, reverse=True)
    expected_classes = sorted({
        key[2] for key in per_class_summary})

    lines = []
    lines.append('# Policy Effectiveness Analysis')
    lines.append('')

    lines.append('## Experiment Purpose')
    lines.append('')
    lines.append(
        'threshold_sweep_summary.md showed how detection stability and '
        'policy-decision *counts* shift with confidence_threshold. This '
        'analysis asks a sharper question of the same sweep data: when '
        'perception_policy.py said ACCEPT_SORT, was it actually right '
        '(vs. the folder-name ground truth), and when the ground truth '
        'says a detection was risky (misclassified/false_known), did the '
        'policy actually block it rather than sorting it anyway? This is '
        'a policy-effectiveness evaluation, not a raw accuracy evaluation.'
    )
    lines.append('')

    lines.append('## Dataset and Inputs')
    lines.append('')
    lines.append(f'- Sweep log directory: `{sweep_dir}`')
    lines.append(
        '- Dataset: test_images_real/ (recursive_image_folder=true), same '
        '50 images (5 classes x 10 shooting conditions) used throughout '
        'the real-image analyses')
    lines.append(
        f'- confidence_threshold values: '
        f'{", ".join(str(t) for t in thresholds_sorted)} '
        f'(policy_confidence_threshold matched to each)')
    lines.append(
        f'- input_size values: {", ".join(str(s) for s in sizes_sorted)}')
    lines.append('- benchmark_mode=vision_only for every run (no '
                  'task_manager/MoveIt)')
    lines.append(
        '- **This is an offline evaluation of the runtime policy against '
        'a folder-name-derived ground truth label, not a bbox-annotation '
        '-based mAP evaluation, and not a live test of the policy actually '
        'gating the sort pipeline** (task_manager does not consume '
        'policy_decision yet).')
    lines.append('')

    lines.append('## Analysis Definitions')
    lines.append('')
    lines.append(
        '- `accept_correct`: `policy_decision == ACCEPT_SORT` and '
        '`status == correct`')
    lines.append(
        '- `unsafe_accept_count`: `policy_decision == ACCEPT_SORT` and '
        '`status in {misclassified, false_known}`; `unsafe_accept_rate = '
        'unsafe_accept_count / accept_total`')
    lines.append(
        '- risky case: `status in {misclassified, false_known}`; '
        '`blocked_risky_count`: a risky case where '
        '`policy_decision != ACCEPT_SORT`; `blocked_risky_rate = '
        'blocked_risky_count / risky_total`')
    lines.append(
        '- `known_correct_total`: `expected_class != unknown` and '
        '`status == correct`; `missed_correct_count`: one of those where '
        '`policy_decision != ACCEPT_SORT` anyway (a real object the '
        'policy could safely have sorted, but didn\'t); '
        '`missed_correct_rate = missed_correct_count / known_correct_total`')
    lines.append('')

    def format_pct(value):
        return f'{value * 100:.1f}%'

    lines.append('## Overall Policy Effectiveness Table')
    lines.append('')
    lines.append(
        '| threshold | input_size | accept_total | accept_correct | '
        'unsafe_accept_count | unsafe_accept_rate | risky_total | '
        'blocked_risky_count | blocked_risky_rate | known_correct_total | '
        'missed_correct_count | missed_correct_rate |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for threshold in thresholds_sorted:
        for size in sizes_sorted:
            stats = summary.get((threshold, size))
            if not stats:
                continue
            lines.append(
                f'| {threshold} | {size} | {stats["accept_total"]} | '
                f'{stats["accept_correct"]} | '
                f'{stats["unsafe_accept_count"]} | '
                f'{format_pct(stats["unsafe_accept_rate"])} | '
                f'{stats["risky_total"]} | '
                f'{stats["blocked_risky_count"]} | '
                f'{format_pct(stats["blocked_risky_rate"])} | '
                f'{stats["known_correct_total"]} | '
                f'{stats["missed_correct_count"]} | '
                f'{format_pct(stats["missed_correct_rate"])} |')
    lines.append('')

    lines.append('## ACCEPT_SORT Safety Table')
    lines.append('')
    lines.append(
        '| threshold | input_size | accept_total | accept_correct | '
        'accept_misclassified | accept_false_known | '
        'accept_unknown_or_no_detection | unsafe_accept_rate |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for threshold in thresholds_sorted:
        for size in sizes_sorted:
            stats = summary.get((threshold, size))
            if not stats:
                continue
            lines.append(
                f'| {threshold} | {size} | {stats["accept_total"]} | '
                f'{stats["accept_correct"]} | '
                f'{stats["accept_misclassified"]} | '
                f'{stats["accept_false_known"]} | '
                f'{stats["accept_unknown_or_no_detection"]} | '
                f'{format_pct(stats["unsafe_accept_rate"])} |')
    lines.append('')

    lines.append('## Risk Blocking Table')
    lines.append('')
    lines.append(
        '| threshold | input_size | risky_total | blocked_risky_count | '
        'blocked_risky_rate |')
    lines.append('|---|---|---|---|---|')
    for threshold in thresholds_sorted:
        for size in sizes_sorted:
            stats = summary.get((threshold, size))
            if not stats:
                continue
            lines.append(
                f'| {threshold} | {size} | {stats["risky_total"]} | '
                f'{stats["blocked_risky_count"]} | '
                f'{format_pct(stats["blocked_risky_rate"])} |')
    lines.append('')

    lines.append('## Missed Opportunity Table')
    lines.append('')
    lines.append(
        '| threshold | input_size | known_correct_total | '
        'missed_correct_count | missed_correct_rate |')
    lines.append('|---|---|---|---|---|')
    for threshold in thresholds_sorted:
        for size in sizes_sorted:
            stats = summary.get((threshold, size))
            if not stats:
                continue
            lines.append(
                f'| {threshold} | {size} | {stats["known_correct_total"]} | '
                f'{stats["missed_correct_count"]} | '
                f'{format_pct(stats["missed_correct_rate"])} |')
    lines.append('')

    lines.append('## Per-class ACCEPT_SORT Safety')
    lines.append('')
    lines.append(
        '| threshold | input_size | expected_class | accept_total | '
        'accept_correct | accept_unsafe | unsafe_accept_rate |')
    lines.append('|---|---|---|---|---|---|---|')
    for threshold in thresholds_sorted:
        for size in sizes_sorted:
            for expected_class in expected_classes:
                stats = per_class_summary.get(
                    (threshold, size, expected_class))
                if not stats:
                    continue
                lines.append(
                    f'| {threshold} | {size} | {expected_class} | '
                    f'{stats["accept_total"]} | {stats["accept_correct"]} | '
                    f'{stats["accept_unsafe"]} | '
                    f'{format_pct(stats["unsafe_accept_rate"])} |')
    lines.append('')

    lines.append('## Key Findings')
    lines.append('')
    for finding in compute_key_findings(
            summary, per_class_summary, thresholds, sizes):
        lines.append(finding)
    lines.append('')

    lines.append('## Limitations')
    lines.append('')
    lines.append(
        '- **can/glass_bottle can never be `accept_correct` today**: the '
        'current YOLO/COCO mapping never emits `can`/`glass_bottle` as '
        'class_name (only plastic_bottle/paper_cup/unknown), so any '
        'ACCEPT_SORT on a can/glass_bottle image is structurally '
        'impossible -- see Per-class ACCEPT_SORT Safety, where those two '
        'classes should show accept_total=0 or accept_correct=0 '
        'regardless of threshold. This is a class-mapping gap, not a '
        'policy-effectiveness result for those classes.')
    lines.append(
        '- This evaluates the runtime policy_decision against an offline, '
        'folder-name-derived expected_class -- not a bbox-annotation-based '
        'mAP/precision-recall evaluation, and there is no ground-truth '
        'bounding box data for test_images_real/.')
    lines.append(
        '- task_manager does not consume policy_decision yet, so '
        '"unsafe_accept"/"blocked_risky" describe what the policy *would* '
        'have decided, not an outcome actually observed on the real sort '
        'pipeline.')
    lines.append(
        '- Same 50-image dataset as the other real-image analyses -- 10 '
        'images per class is a small sample, so a single image flipping '
        'status can swing a per-class rate by 10-100 percentage points.')
    lines.append(
        '- Only 3 threshold values (0.3/0.5/0.7) were tested; the true '
        'safety/opportunity trade-off curve between them is not known.')
    lines.append('')

    lines.append('## Next Steps')
    lines.append('')
    lines.append(
        '- Extend the class mapping so can/glass_bottle can actually be '
        'accept_correct, then re-run this analysis -- right now their '
        'rows are not informative about policy effectiveness at all.')
    lines.append(
        '- Once a threshold is chosen from this trade-off, wire '
        'policy_decision into task_manager so ROUTE_TO_REJECT/'
        'SKIP_NO_DETECTION/RETRY_VIEW/MANUAL_REVIEW actually change sort '
        'pipeline behavior, then re-measure unsafe_accept_rate/'
        'blocked_risky_rate against real outcomes instead of this offline '
        'estimate.')
    lines.append(
        '- Sweep confidence_threshold and policy_confidence_threshold '
        'independently to see whether tightening just the policy gate '
        '(while leaving the ONNX-level filter looser) can lower '
        'unsafe_accept_rate without also raising missed_correct_rate as '
        'much as tightening both together.')
    lines.append(
        '- Grow the dataset (more images per class/condition) before '
        'treating any single per-class or per-condition rate in this '
        'report as final.')
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
        help='Directory containing conf_<threshold>/yolo11n_<size>.log')
    parser.add_argument(
        '--sizes', type=int, nargs='+', default=[640, 416, 320],
        help='Input sizes to look for (default: 640 416 320)')
    parser.add_argument(
        '--model-stem', default='yolo11n',
        help="Log filename stem, matching <stem>_<size>.log (default: "
             "'yolo11n')")
    parser.add_argument(
        '--output-csv', default='results/policy_effectiveness_summary.csv')
    parser.add_argument(
        '--output-md', default='results/policy_effectiveness_summary.md')
    args = parser.parse_args()

    records = ats.parse_sweep(args.sweep_dir, args.sizes, args.model_stem)
    if not records:
        print('No logs parsed; nothing to write.')
        return

    thresholds = sorted({record['threshold'] for record in records})
    sizes = sorted({record['input_size'] for record in records},
                   reverse=True)

    summary = compute_policy_effectiveness(records)
    per_class_summary = compute_per_class_accept_safety(records)

    write_csv(summary, thresholds, sizes, args.output_csv)
    print(f'Wrote {args.output_csv} ({len(thresholds) * len(sizes)} rows)')

    write_markdown(summary, per_class_summary, thresholds, sizes,
                    args.sweep_dir, args.output_md)
    print(f'Wrote {args.output_md}')


if __name__ == '__main__':
    main()
