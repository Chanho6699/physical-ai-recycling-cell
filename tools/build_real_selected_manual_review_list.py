#!/usr/bin/env python3
"""Rank real_selected images (recycling_yolo_material_v1_real_aug's
autolabel_report.csv, source_type=real_selected) into a manual-review
priority list -- the third piece of the real_aug regression follow-up,
after results/real_aug_unsafe_unknown_cases.md (which images/classes the
model false-positives on) and results/real_selected_pseudolabel_quality.md
(real_selected's own warning-flag/confidence/bbox-area distribution).

Priority tiers (highest first; each image gets its single highest tier):
  1. too_large_bbox    -- normalized bbox area > 0.75, likely covers
     background/table along with (or instead of) the real object
  2. low_confidence     -- GroundingDINO confidence < 0.35, weak match to
     the target-class prompt
  3. boundary_clamped   -- raw box extended past the image edge and was
     clamped; usually benign for a close-up photo but worth a glance
  4. large_bbox_area    -- bbox area ratio > LARGE_AREA_THRESHOLD (0.35)
     but below the too_large_bbox cutoff (0.75) -- a softer version of
     tier 1, since a box covering a third+ of the frame is still a
     plausible "grabbed background too" case even without crossing the
     hard warning threshold
  5. unsafe_linked_class -- target_class is `paper` or `plastic`, the two
     classes results/real_aug_unsafe_unknown_cases.md found driving
     real_aug's unsafe unknown->ACCEPT_SORT regression (paper 70%,
     plastic 30% of cases) -- reviewing these checks whether a handful of
     loose/generic real_selected boxes taught the model an overly
     generic "paper-like" or "plastic-like" texture cue

Images matching no tier are listed last as tier 6 (no flags raised).

Writes:
  - results/real_selected_manual_review_list.csv (one row per
    real_selected image, tier + reason + all the underlying fields)
  - results/real_selected_manual_review_list.md (grouped by tier, with
    counts and a suggested review order)

Usage:
    python3 tools/build_real_selected_manual_review_list.py
"""
import argparse
import csv
import os

LARGE_AREA_THRESHOLD = 0.35
UNSAFE_LINKED_CLASSES = ('paper', 'plastic')

TIER_LABELS = {
    1: 'too_large_bbox',
    2: 'low_confidence',
    3: 'boundary_clamped',
    4: 'large_bbox_area (>0.35, below too_large cutoff)',
    5: 'unsafe_linked_class (paper/plastic)',
    6: 'no_flags',
}

CSV_FIELDNAMES = [
    'review_tier',
    'tier_reason',
    'output_image',
    'target_class',
    'confidence',
    'bbox_area_ratio',
    'warning_flags',
    'split',
    'preview_path',
    'source_image',
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--report',
        default='datasets/recycling_yolo_material_v1_real_aug/'
                'autolabel_report.csv')
    parser.add_argument(
        '--previews-dir',
        default='datasets/recycling_yolo_material_v1_real_aug/previews')
    parser.add_argument(
        '--output-csv',
        default='results/real_selected_manual_review_list.csv')
    parser.add_argument(
        '--output-md',
        default='results/real_selected_manual_review_list.md')
    return parser.parse_args()


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find_preview_path(previews_dir, split, output_image):
    stem = os.path.splitext(output_image)[0]
    candidate = os.path.join(
        previews_dir, f'real_selected_{split}_{stem}_preview.jpg')
    return candidate if os.path.isfile(candidate) else ''


def assign_tier(row):
    flags = row['warning_flags'].split(';') if row['warning_flags'] else []
    area_ratio = to_float(row['bbox_area_ratio'])

    if 'too_large_bbox' in flags:
        return 1, 'too_large_bbox flag set'
    if 'low_confidence' in flags:
        return 2, 'low_confidence flag set'
    if 'boundary_clamped' in flags:
        return 3, 'boundary_clamped flag set'
    if area_ratio is not None and area_ratio > LARGE_AREA_THRESHOLD:
        return 4, f'bbox_area_ratio={area_ratio:.4f} > {LARGE_AREA_THRESHOLD}'
    if row['target_class'] in UNSAFE_LINKED_CLASSES:
        return 5, (f'target_class={row["target_class"]} matches the class '
                    f'driving real_aug\'s unsafe ACCEPT_SORT cases')
    return 6, 'no quality/risk flags raised'


def load_and_rank(report_path, previews_dir):
    with open(report_path, newline='') as report_file:
        rows = [row for row in csv.DictReader(report_file)
                if row['source_type'] == 'real_selected']

    ranked = []
    for row in rows:
        tier, reason = assign_tier(row)
        ranked.append({
            'review_tier': tier,
            'tier_reason': reason,
            'output_image': row['output_image'],
            'target_class': row['target_class'],
            'confidence': row['confidence'],
            'bbox_area_ratio': row['bbox_area_ratio'],
            'warning_flags': row['warning_flags'],
            'split': row['split'],
            'preview_path': find_preview_path(
                previews_dir, row['split'], row['output_image']),
            'source_image': row['source_image'],
        })

    ranked.sort(key=lambda r: (
        r['review_tier'],
        -(to_float(r['bbox_area_ratio']) or 0),
        to_float(r['confidence']) or 0,
    ))
    return ranked


def write_csv(ranked, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in ranked:
            writer.writerow(row)


def write_markdown(ranked, md_path, report_path):
    lines = []
    lines.append('# real_selected Manual Review Priority List')
    lines.append('')
    lines.append(
        f'{len(ranked)} real_selected images from `{report_path}`, ranked '
        f'into 6 priority tiers (1=highest). Within a tier, sorted by '
        f'bbox_area_ratio descending (bigger boxes first) then confidence '
        f'ascending (weaker matches first). See the module docstring of '
        f'`tools/build_real_selected_manual_review_list.py` for the exact '
        f'tier criteria.'
    )
    lines.append('')

    by_tier = {}
    for row in ranked:
        by_tier.setdefault(row['review_tier'], []).append(row)

    lines.append('## Tier Counts')
    lines.append('')
    lines.append('| tier | criteria | count |')
    lines.append('|---|---|---|')
    for tier in sorted(TIER_LABELS):
        count = len(by_tier.get(tier, []))
        lines.append(f'| {tier} | {TIER_LABELS[tier]} | {count} |')
    lines.append('')

    reviewable = sum(len(by_tier.get(t, [])) for t in (1, 2, 3, 4, 5))
    lines.append(
        f'**Suggested review order:** work tiers 1 through 5 top to '
        f'bottom ({reviewable} of {len(ranked)} images) before assuming '
        f'"real_selected labels are fine" -- tier 6 (no flags) is safe to '
        f'skip on a first pass. Given results/real_selected_pseudolabel_'
        f'quality.md found zero too_large_bbox/low_confidence hits, tiers '
        f'1-2 are expected to be empty or near-empty here -- tiers 4 and 5 '
        f'are where the real signal is likely to be for this dataset.'
    )
    lines.append('')

    for tier in sorted(TIER_LABELS):
        rows = by_tier.get(tier, [])
        if not rows:
            continue
        lines.append(f'## Tier {tier}: {TIER_LABELS[tier]} ({len(rows)})')
        lines.append('')
        lines.append(
            '| output_image | target_class | confidence | bbox_area_ratio '
            '| warning_flags | preview_path |')
        lines.append('|---|---|---|---|---|---|')
        for row in rows:
            lines.append(
                f'| {row["output_image"]} | {row["target_class"]} | '
                f'{row["confidence"]} | {row["bbox_area_ratio"]} | '
                f'{row["warning_flags"] or "(none)"} | '
                f'{row["preview_path"] or "(not in preview sample)"} |')
        lines.append('')

    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, 'w') as md_file:
        md_file.write('\n'.join(lines))


def main():
    args = parse_args()
    ranked = load_and_rank(args.report, args.previews_dir)
    if not ranked:
        raise SystemExit(
            f'ERROR: no source_type=real_selected rows found in '
            f'{args.report}')

    write_csv(ranked, args.output_csv)
    print(f'Wrote {args.output_csv} ({len(ranked)} rows)')

    write_markdown(ranked, args.output_md, args.report)
    print(f'Wrote {args.output_md}')


if __name__ == '__main__':
    main()
