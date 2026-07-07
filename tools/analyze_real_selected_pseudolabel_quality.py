#!/usr/bin/env python3
"""Quality-check summary for the real_selected source only, pulled out of
recycling_yolo_material_v1_real_aug's autolabel_report.csv -- the
follow-up to results/material_v1_real_aug_comparison.md's finding that
real_aug increased unsafe unknown->ACCEPT_SORT cases 0 -> 10 versus
strong. Before assuming "real_selected's pseudo-labels are bad", this
summarizes exactly what the labeler actually produced for real_selected:
per-class counts, warning-flag breakdown, confidence/bbox-area-ratio
distributions, and the specific images worth a human look first.

Only source_type=real_selected rows are considered -- candidates_v1 and
v0_remapped are a separate, already-validated (via material_v1's own
benchmark history) source and are out of scope here.

Writes:
  - results/real_selected_pseudolabel_quality.csv (one row per class:
    counts, warning breakdown, confidence/area stats)
  - results/real_selected_pseudolabel_quality.md (the table plus the
    two ranked lists: highest bbox_area_ratio and lowest confidence)

Usage:
    python3 tools/analyze_real_selected_pseudolabel_quality.py
    python3 tools/analyze_real_selected_pseudolabel_quality.py \\
        --report datasets/recycling_yolo_material_v1_real_aug/autolabel_report.csv
"""
import argparse
import csv
import os

MATERIAL_CLASSES = ('plastic', 'metal', 'glass', 'paper')

WARNING_FLAGS = (
    'too_large_bbox', 'too_small_bbox', 'low_confidence', 'boundary_clamped',
)

TOP_N = 20

CSV_FIELDNAMES = [
    'target_class',
    'total_images',
    'labeled', 'no_box', 'excluded', 'error',
    'too_large_bbox', 'too_small_bbox', 'low_confidence', 'boundary_clamped',
    'avg_bbox_area_ratio',
    'avg_confidence',
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--report',
        default='datasets/recycling_yolo_material_v1_real_aug/'
                'autolabel_report.csv')
    parser.add_argument(
        '--output-csv',
        default='results/real_selected_pseudolabel_quality.csv')
    parser.add_argument(
        '--output-md',
        default='results/real_selected_pseudolabel_quality.md')
    return parser.parse_args()


def load_real_selected_rows(report_path):
    with open(report_path, newline='') as report_file:
        return [row for row in csv.DictReader(report_file)
                if row['source_type'] == 'real_selected']


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_class(class_name, rows):
    class_rows = [r for r in rows if r['target_class'] == class_name]
    status_counts = {'labeled': 0, 'no_box': 0, 'excluded': 0, 'error': 0}
    warning_counts = {flag: 0 for flag in WARNING_FLAGS}
    area_ratios = []
    confidences = []

    for row in class_rows:
        status = row['status'] if row['status'] in status_counts else 'error'
        status_counts[status] += 1

        flags = row['warning_flags'].split(';') if row['warning_flags'] else []
        for flag in flags:
            if flag in warning_counts:
                warning_counts[flag] += 1

        area_ratio = to_float(row['bbox_area_ratio'])
        if area_ratio is not None:
            area_ratios.append(area_ratio)
        confidence = to_float(row['confidence'])
        if confidence is not None:
            confidences.append(confidence)

    return {
        'target_class': class_name,
        'total_images': len(class_rows),
        'labeled': status_counts['labeled'],
        'no_box': status_counts['no_box'],
        'excluded': status_counts['excluded'],
        'error': status_counts['error'],
        'too_large_bbox': warning_counts['too_large_bbox'],
        'too_small_bbox': warning_counts['too_small_bbox'],
        'low_confidence': warning_counts['low_confidence'],
        'boundary_clamped': warning_counts['boundary_clamped'],
        'avg_bbox_area_ratio': (
            round(sum(area_ratios) / len(area_ratios), 4)
            if area_ratios else None),
        'avg_confidence': (
            round(sum(confidences) / len(confidences), 4)
            if confidences else None),
    }


def write_csv(class_summaries, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for summary in class_summaries:
            writer.writerow(summary)


def write_markdown(class_summaries, rows, md_path, report_path):
    lines = []
    lines.append('# real_selected Pseudo-label Quality Summary')
    lines.append('')
    lines.append(
        f'Source: `{report_path}`, filtered to `source_type=real_selected` '
        f'({len(rows)} images: plastic 35 / metal 24 / paper 22 / glass '
        f'10). Written to check whether real_selected\'s GroundingDINO '
        f'pseudo-labels look obviously bad BEFORE assuming they explain '
        f'real_aug\'s unsafe-accept regression (see '
        f'results/real_aug_unsafe_unknown_cases.md for that side of the '
        f'analysis).'
    )
    lines.append('')

    lines.append('## Per-class Summary')
    lines.append('')
    lines.append(
        '| class | total | labeled | no_box | excluded | error | '
        'too_large_bbox | too_small_bbox | low_confidence | '
        'boundary_clamped | avg_bbox_area_ratio | avg_confidence |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for s in class_summaries:
        lines.append(
            f'| {s["target_class"]} | {s["total_images"]} | {s["labeled"]} '
            f'| {s["no_box"]} | {s["excluded"]} | {s["error"]} | '
            f'{s["too_large_bbox"]} | {s["too_small_bbox"]} | '
            f'{s["low_confidence"]} | {s["boundary_clamped"]} | '
            f'{s["avg_bbox_area_ratio"]} | {s["avg_confidence"]} |')
    lines.append('')

    total_warned = sum(
        s['too_large_bbox'] + s['too_small_bbox'] + s['low_confidence']
        for s in class_summaries)
    total_boundary = sum(s['boundary_clamped'] for s in class_summaries)
    lines.append(
        f'**Headline:** {total_warned} of {len(rows)} real_selected images '
        f'triggered too_large_bbox/too_small_bbox/low_confidence (the '
        f'three quality-risk flags), plus {total_boundary} boundary_clamped '
        f'(box touched the image edge -- common and not inherently bad for '
        f'a close-up real photo). '
        + ('This is a low warning rate -- real_selected\'s pseudo-label '
           'BBOXES do not look obviously bad by these metrics, which means '
           'the unsafe-accept regression is more likely a MODEL '
           'GENERALIZATION effect (training on cluttered/backgroundy real '
           'photos made the model fire more readily on similar clutter in '
           'test_images_real/unknown/) than a labeling-error effect -- see '
           'results/real_aug_unsafe_unknown_cases.md.'
           if total_warned / len(rows) < 0.15 else
           'This is a meaningful fraction of the dataset -- bad bboxes are '
           'a plausible contributor to real_aug\'s regression and worth '
           'reviewing before the next retrain.')
    )
    lines.append('')

    lines.append(f'## Top {TOP_N} by bbox_area_ratio (largest boxes)')
    lines.append('')
    lines.append(
        '| output_image | target_class | bbox_area_ratio | confidence | '
        'warning_flags |')
    lines.append('|---|---|---|---|---|')
    ranked_by_area = sorted(
        [r for r in rows if to_float(r['bbox_area_ratio']) is not None],
        key=lambda r: -to_float(r['bbox_area_ratio']))
    for row in ranked_by_area[:TOP_N]:
        lines.append(
            f'| {row["output_image"]} | {row["target_class"]} | '
            f'{row["bbox_area_ratio"]} | {row["confidence"]} | '
            f'{row["warning_flags"] or "(none)"} |')
    lines.append('')

    lines.append(f'## Bottom {TOP_N} by confidence (least confident)')
    lines.append('')
    lines.append(
        '| output_image | target_class | confidence | bbox_area_ratio | '
        'warning_flags |')
    lines.append('|---|---|---|---|---|')
    ranked_by_conf = sorted(
        [r for r in rows if to_float(r['confidence']) is not None],
        key=lambda r: to_float(r['confidence']))
    for row in ranked_by_conf[:TOP_N]:
        lines.append(
            f'| {row["output_image"]} | {row["target_class"]} | '
            f'{row["confidence"]} | {row["bbox_area_ratio"]} | '
            f'{row["warning_flags"] or "(none)"} |')
    lines.append('')

    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, 'w') as md_file:
        md_file.write('\n'.join(lines))


def main():
    args = parse_args()
    rows = load_real_selected_rows(args.report)
    if not rows:
        raise SystemExit(
            f'ERROR: no source_type=real_selected rows found in '
            f'{args.report}')

    class_summaries = [
        summarize_class(class_name, rows) for class_name in MATERIAL_CLASSES
    ]

    write_csv(class_summaries, args.output_csv)
    print(f'Wrote {args.output_csv} ({len(class_summaries)} rows)')

    write_markdown(class_summaries, rows, args.output_md, args.report)
    print(f'Wrote {args.output_md}')


if __name__ == '__main__':
    main()
