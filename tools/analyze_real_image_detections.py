#!/usr/bin/env python3
"""Analyze real-image benchmark_mode=vision_only logs for detection
stability, broken down by input_size / expected class / shooting condition.

This is NOT a bbox-annotation-based mAP/precision-recall evaluation -- there
is no ground-truth bounding box data for test_images_real/. It is an
image-level detection *stability* analysis: for each image, did the model
detect (at least one instance of) the class we expect it to, given only the
dataset's folder-name convention as the label?

Reads logs/vision_benchmark/test_images_real/vision_only/yolo11n_<size>.log
(produced by tools/run_vision_size_benchmark.sh with BENCHMARK_MODE=
vision_only) and for each image line:
  - takes the image path + input_size + detections count from the
    "[VisionPerf] source=... image=<path> input_size=<n> detections=<k> ..."
    line
  - takes class_name/confidence from the "ONNX detection: object_id=...
    class_name=<c> confidence=<p> ..." lines that were logged for that image
    just before its [VisionPerf] line (the node logs all of an image's
    per-detection lines first, then one [VisionPerf] summary line, so
    detections are buffered until that line arrives)
  - takes policy_decision/policy_reason/recommended_action from the
    "[PerceptionPolicy] image=<path> decision=<d> reason=<r> ...
    recommended_action=<a>" line the node logs for every image
    (perception_policy.py) -- matched by its own image= field rather than
    by ordering, and left blank for older logs that predate this feature

expected_class is the image path's first folder segment (e.g.
"can/can_dark_light_003.jpg.jpg" -> "can"). condition is the filename with
the expected_class prefix, trailing "_<index>", and image extension(s)
stripped (e.g. "can_dark_light_003.jpg.jpg" -> "dark_light").

Writes:
  - results/real_image_detection_analysis.csv (one row per image per
    input_size)
  - results/real_image_detection_analysis.md  (report with sections:
    Experiment Purpose, Dataset, Analysis Rule, Input-size Summary,
    Class-level Summary, Condition-level Summary, Policy-level Summary,
    Key Findings, Limitations, Next Steps)

Usage:
    python3 tools/analyze_real_image_detections.py
    python3 tools/analyze_real_image_detections.py \\
        --log-dir logs/vision_benchmark/test_images_real/vision_only
"""
import argparse
import csv
import os
import re
from collections import Counter, defaultdict

VISION_PERF_RE = re.compile(
    r'\[VisionPerf\] source=(?P<source>\S+) image=(?P<image>\S+) '
    r'input_size=(?P<input_size>\d+) detections=(?P<detections>\d+)'
)

ONNX_DETECTION_RE = re.compile(
    r'ONNX detection: object_id=(?P<object_id>\S+) '
    r'class_name=(?P<class_name>\S+) confidence=(?P<confidence>[\d.]+)'
)

POLICY_RE = re.compile(
    r'\[PerceptionPolicy\] image=(?P<image>\S+) '
    r'decision=(?P<decision>\S+) reason=(?P<reason>\S+) '
    r'selected_class=\S+ conf=\S+ num_detections=\d+ '
    r'recommended_action=(?P<recommended_action>\S+)'
)

# Must stay in sync with recycling_cell_vision/perception_policy.py's
# DECISIONS -- duplicated here (instead of importing the ROS2 package) so
# this script stays runnable with plain python3, no ROS2 environment needed.
POLICY_DECISIONS = (
    'ACCEPT_SORT', 'ROUTE_TO_REJECT', 'SKIP_NO_DETECTION', 'RETRY_VIEW',
    'MANUAL_REVIEW',
)

# can/glass_bottle have no confident COCO equivalent in the current
# COCO_CLASS_ID_TO_PROJECT_CLASS mapping (see vision_perception_node.py),
# so they can never actually be emitted as a detected class_name today --
# they're still tracked here as "known" so that IF a future model mapping
# started emitting them, a real object mistakenly labeled as one of these
# would still correctly count as false_known rather than silently passing.
KNOWN_CLASSES = {'paper_cup', 'plastic_bottle', 'can', 'glass_bottle'}

IMAGE_EXT_RE = re.compile(r'(?:\.jpe?g|\.png)+$', re.IGNORECASE)
TRAILING_INDEX_RE = re.compile(r'_\d+$')

CSV_FIELDNAMES = [
    'input_size',
    'image',
    'expected_class',
    'condition',
    'detections_count',
    'detected_classes',
    'detected_confidences',
    'status',
    'policy_decision',
    'policy_reason',
    'recommended_action',
]


def extract_expected_class_and_condition(image_path):
    parts = image_path.split('/')
    if len(parts) < 2:
        # This dataset always nests images under a class folder; fall back
        # to something visibly wrong rather than silently mislabeling.
        return 'unrecognized', os.path.splitext(image_path)[0]

    expected_class = parts[0]
    basename = parts[-1]

    stem = IMAGE_EXT_RE.sub('', basename)
    stem = TRAILING_INDEX_RE.sub('', stem)

    prefix = expected_class + '_'
    condition = stem[len(prefix):] if stem.startswith(prefix) else stem
    return expected_class, condition


def determine_status(expected_class, detected_classes, detections_count):
    if expected_class != 'unknown':
        if detections_count == 0:
            return 'no_detection'
        if expected_class in detected_classes:
            return 'correct'
        return 'misclassified'

    if detections_count == 0:
        return 'correct_reject'
    if any(c in KNOWN_CLASSES for c in detected_classes):
        return 'false_known'
    return 'correct_reject'


def parse_log(log_path):
    """Parse one benchmark log into a list of per-image detection records."""
    records = []
    pending_detections = []
    policy_by_image = {}

    with open(log_path, 'r', errors='replace') as log_file:
        for line in log_file:
            match = POLICY_RE.search(line)
            if match:
                policy_by_image[match.group('image')] = {
                    'policy_decision': match.group('decision'),
                    'policy_reason': match.group('reason'),
                    'recommended_action': match.group('recommended_action'),
                }
                continue

            match = ONNX_DETECTION_RE.search(line)
            if match:
                pending_detections.append((
                    match.group('class_name'),
                    float(match.group('confidence')),
                ))
                continue

            match = VISION_PERF_RE.search(line)
            if match:
                image = match.group('image')
                input_size = int(match.group('input_size'))
                detections_count = int(match.group('detections'))
                detections = list(pending_detections)
                pending_detections = []

                expected_class, condition = \
                    extract_expected_class_and_condition(image)
                detected_classes = [c for c, _ in detections]
                status = determine_status(
                    expected_class, detected_classes, detections_count)

                # Older logs predate [PerceptionPolicy] -- leave it blank
                # rather than fabricating a decision for them.
                policy = policy_by_image.pop(image, {
                    'policy_decision': '',
                    'policy_reason': '',
                    'recommended_action': '',
                })

                records.append({
                    'input_size': input_size,
                    'image': image,
                    'expected_class': expected_class,
                    'condition': condition,
                    'detections_count': detections_count,
                    'detected_classes': detected_classes,
                    'detected_confidences': [c for _, c in detections],
                    'status': status,
                    'policy_decision': policy['policy_decision'],
                    'policy_reason': policy['policy_reason'],
                    'recommended_action': policy['recommended_action'],
                })

    return records


def build_csv_row(record):
    return {
        'input_size': record['input_size'],
        'image': record['image'],
        'expected_class': record['expected_class'],
        'condition': record['condition'],
        'detections_count': record['detections_count'],
        'detected_classes': '+'.join(record['detected_classes']),
        'detected_confidences': '+'.join(
            f'{c:.2f}' for c in record['detected_confidences']),
        'status': record['status'],
        'policy_decision': record['policy_decision'],
        'policy_reason': record['policy_reason'],
        'recommended_action': record['recommended_action'],
    }


def write_csv(records, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(build_csv_row(record))


def compute_input_size_summary(records):
    by_size = defaultdict(list)
    for record in records:
        by_size[record['input_size']].append(record)

    summary = {}
    for input_size, group in by_size.items():
        total = len(group)
        correct_or_reject = sum(
            1 for r in group if r['status'] in ('correct', 'correct_reject'))
        no_detection = sum(1 for r in group if r['status'] == 'no_detection')
        misclassified_or_false_known = sum(
            1 for r in group
            if r['status'] in ('misclassified', 'false_known'))
        summary[input_size] = {
            'total_images': total,
            'correct_or_reject': correct_or_reject,
            'no_detection': no_detection,
            'misclassified_or_false_known': misclassified_or_false_known,
            'image_level_success_rate':
                correct_or_reject / total if total else 0.0,
        }
    return summary


def compute_class_summary(records):
    by_group = defaultdict(list)
    for record in records:
        by_group[(record['input_size'], record['expected_class'])].append(
            record)

    summary = {}
    for (input_size, expected_class), group in by_group.items():
        detected_counter = Counter()
        for record in group:
            detected_counter.update(record['detected_classes'])

        main_detected = ','.join(
            f'{cls}:{count}'
            for cls, count in detected_counter.most_common(3)) or 'none'

        summary[(input_size, expected_class)] = {
            'total': len(group),
            'correct': sum(1 for r in group if r['status'] == 'correct'),
            'no_detection': sum(
                1 for r in group if r['status'] == 'no_detection'),
            'misclassified': sum(
                1 for r in group if r['status'] == 'misclassified'),
            'correct_reject': sum(
                1 for r in group if r['status'] == 'correct_reject'),
            'false_known': sum(
                1 for r in group if r['status'] == 'false_known'),
            'main_detected_classes': main_detected,
        }
    return summary


def compute_condition_summary(records):
    by_group = defaultdict(list)
    for record in records:
        by_group[(record['input_size'], record['condition'])].append(record)

    summary = {}
    for (input_size, condition), group in by_group.items():
        total = len(group)
        correct_or_reject = sum(
            1 for r in group if r['status'] in ('correct', 'correct_reject'))
        failures = total - correct_or_reject
        failure_counter = Counter(
            r['status'] for r in group
            if r['status'] in ('no_detection', 'misclassified',
                                'false_known'))
        common_failure_modes = ','.join(
            f'{mode}:{count}'
            for mode, count in failure_counter.most_common()) or 'none'

        summary[(input_size, condition)] = {
            'total': total,
            'correct_or_reject': correct_or_reject,
            'failures': failures,
            'failure_rate': failures / total if total else 0.0,
            'common_failure_modes': common_failure_modes,
        }
    return summary


def compute_policy_summary(records):
    by_size = defaultdict(Counter)
    for record in records:
        decision = record['policy_decision']
        if decision:
            by_size[record['input_size']][decision] += 1
    return by_size


def compute_key_findings(records, class_summary, input_sizes):
    findings = []
    expected_classes = sorted({record['expected_class']
                                for record in records})

    # 1. Most unstable class: aggregate correct_or_reject/total across all
    # input sizes per expected_class, then rank ascending.
    class_totals = defaultdict(lambda: {'total': 0, 'ok': 0})
    for (input_size, expected_class), stats in class_summary.items():
        class_totals[expected_class]['total'] += stats['total']
        class_totals[expected_class]['ok'] += (
            stats['correct'] + stats['correct_reject'])

    class_rates = {
        cls: (vals['ok'] / vals['total'] if vals['total'] else 0.0)
        for cls, vals in class_totals.items()
    }
    if class_rates:
        worst_rate = min(class_rates.values())
        worst_classes = sorted(
            c for c, rate in class_rates.items() if rate == worst_rate)
        worst_label = ' and '.join(worst_classes) if len(worst_classes) <= 2 \
            else ', '.join(worst_classes[:-1]) + f', and {worst_classes[-1]}'
        tie_note = ' (tied)' if len(worst_classes) > 1 else ''
        findings.append(
            f'- Least stable class{"es" if len(worst_classes) > 1 else ""}: '
            f'**{worst_label}**{tie_note} with an overall '
            f'{worst_rate * 100:.0f}% correct/correct_reject rate across '
            f'all input sizes, the lowest of all {len(class_rates)} classes '
            f'({", ".join(f"{c}={class_rates[c] * 100:.0f}%" for c in sorted(class_rates))}).'
        )

    # 2. can/glass_bottle domain gap: current mapping never emits these as
    # class_name, so any image of them can only end up no_detection or
    # misclassified -- confirm this directly from the data.
    domain_gap_classes = [
        cls for cls in ('can', 'glass_bottle') if cls in class_totals
    ]
    if domain_gap_classes:
        ever_detected_as_known = set()
        for record in records:
            if record['expected_class'] in domain_gap_classes:
                ever_detected_as_known.update(
                    c for c in record['detected_classes']
                    if c in domain_gap_classes)
        gap_parts = ', '.join(
            f'{cls} {class_totals[cls]["ok"]}/{class_totals[cls]["total"]} '
            f'correct ({class_rates[cls] * 100:.0f}%)'
            for cls in domain_gap_classes)
        if not ever_detected_as_known:
            findings.append(
                f'- Domain gap confirmed for {" and ".join(domain_gap_classes)}: '
                f'{gap_parts}. The current YOLO/COCO class mapping '
                f'(COCO_CLASS_ID_TO_PROJECT_CLASS in vision_perception_node.py) '
                f'never emits a "can" or "glass_bottle" class_name at all '
                f'(only plastic_bottle/paper_cup/unknown), so these classes '
                f'can only ever land on no_detection or misclassified, '
                f'independent of input_size -- this is a labeling/model '
                f'coverage gap, not a resolution problem.'
            )
        else:
            findings.append(
                f'- {" and ".join(domain_gap_classes)}: {gap_parts}; '
                f'detected as their own known class '
                f'{sorted(ever_detected_as_known)} at least once, so the '
                f'mapping gap is partial, not total.'
            )

    # 3. Worst condition (aggregated across sizes).
    condition_totals = defaultdict(lambda: {'total': 0, 'failures': 0})
    condition_modes = defaultdict(Counter)
    for record in records:
        key = record['condition']
        condition_totals[key]['total'] += 1
        if record['status'] in ('no_detection', 'misclassified',
                                 'false_known'):
            condition_totals[key]['failures'] += 1
            condition_modes[key][record['status']] += 1

    condition_rates = {
        cond: (vals['failures'] / vals['total'] if vals['total'] else 0.0)
        for cond, vals in condition_totals.items()
    }
    if condition_rates:
        worst_conditions = sorted(
            condition_rates, key=condition_rates.get, reverse=True)[:3]
        worst_conditions = [c for c in worst_conditions
                             if condition_rates[c] > 0]
        if worst_conditions:
            parts = []
            for cond in worst_conditions:
                modes = ','.join(
                    f'{mode}:{count}'
                    for mode, count in condition_modes[cond].most_common())
                parts.append(
                    f'{cond} ({condition_totals[cond]["failures"]}/'
                    f'{condition_totals[cond]["total"]} failed, {modes})')
            findings.append(
                f'- Conditions with the most failures across all input '
                f'sizes: {"; ".join(parts)}.')
        else:
            findings.append(
                '- No shooting condition had any failures across all '
                'input sizes and classes.')

    # 4. input_size=320 stability trend vs. the largest size run.
    if 320 in input_sizes and max(input_sizes) in input_sizes \
            and max(input_sizes) != 320:
        largest = max(input_sizes)
        size_summary = compute_input_size_summary(records)
        rate_320 = size_summary[320]['image_level_success_rate']
        rate_largest = size_summary[largest]['image_level_success_rate']
        delta = (rate_320 - rate_largest) * 100
        if abs(delta) < 1.0:
            trend = 'stayed essentially flat'
        elif delta > 0:
            trend = f'*improved* by {delta:.1f} points'
        else:
            trend = f'*dropped* by {abs(delta):.1f} points'
        findings.append(
            f'- input_size=320 vs. {largest}: image-level success rate '
            f'{trend} ({rate_320 * 100:.0f}% vs. {rate_largest * 100:.0f}%). '
            f'Combined with the throughput results in '
            f'results/vision_benchmark_real_vision_only_summary.md '
            f'(320 is fastest), this indicates 320 is '
            f'{"a reasonable speed/stability trade-off" if delta >= -1.0 else "faster but measurably less stable, so the speed gain has a real detection-quality cost"} '
            f'on this dataset.'
        )

    if not findings:
        findings.append(
            '- Not enough data to compute automated findings (no records '
            'parsed).')
    return findings


def write_markdown(records, class_summary, condition_summary, md_path,
                    log_dir):
    input_sizes = sorted({record['input_size'] for record in records},
                          reverse=True)
    expected_classes = sorted({record['expected_class']
                                for record in records})
    conditions = sorted({record['condition'] for record in records})
    size_summary = compute_input_size_summary(records)
    total_images_per_size = {
        size: size_summary[size]['total_images'] for size in input_sizes}

    lines = []
    lines.append('# Real-Image Detection Stability Analysis')
    lines.append('')

    lines.append('## Experiment Purpose')
    lines.append('')
    lines.append(
        'Break down the benchmark_mode=vision_only real-image run '
        '(test_images_real/, 5 classes x 10 shooting conditions) by '
        'input_size to see not just latency/FPS but whether detection '
        '*stability* changes with input_size -- per class and per shooting '
        'condition, not just in aggregate.'
    )
    lines.append(
        '**This is an image-level detection stability analysis, not a '
        'bbox-annotation-based mAP/precision-recall evaluation** -- '
        'test_images_real/ has no ground-truth bounding boxes, only a '
        'folder-name convention (`<class>/<class>_<condition>_<index>.jpg'
        '.jpg`) used as a per-image expected label.'
    )
    lines.append('')

    lines.append('## Dataset')
    lines.append('')
    lines.append(f'- Log directory: `{log_dir}`')
    lines.append(
        f'- Classes ({len(expected_classes)}): {", ".join(expected_classes)}')
    lines.append(f'- Shooting conditions observed ({len(conditions)}): '
                  f'{", ".join(conditions)}')
    lines.append(
        f'- Images per input_size: '
        f'{", ".join(f"{size}={total_images_per_size[size]}" for size in input_sizes)}')
    lines.append(
        '- expected_class = first path segment of the image (e.g. '
        '`can/can_dark_light_003.jpg.jpg` -> `can`)')
    lines.append(
        '- condition = filename with the expected_class prefix, trailing '
        '`_<index>`, and `.jpg`/`.jpg.jpg` extension stripped (e.g. '
        '`can_dark_light_003.jpg.jpg` -> `dark_light`)')
    lines.append('')

    lines.append('## Analysis Rule')
    lines.append('')
    lines.append('For `expected_class != unknown`:')
    lines.append('- `correct`: at least one detection has '
                  '`class_name == expected_class`')
    lines.append('- `no_detection`: `detections_count == 0`')
    lines.append('- `misclassified`: detections exist, but none match '
                  '`expected_class`')
    lines.append('')
    lines.append('For `expected_class == unknown`:')
    lines.append('- `correct_reject`: no detections, or every detection '
                  'is `class_name == unknown`')
    lines.append(
        f'- `false_known`: at least one detection is a known class '
        f'({", ".join(sorted(KNOWN_CLASSES))})')
    lines.append(
        '- Note: the current YOLO/COCO mapping only ever emits '
        '`plastic_bottle`/`paper_cup`/`unknown` as class_name -- `can` and '
        '`glass_bottle` are still included in the known-class set for '
        'false_known checks so the rule stays correct if that mapping is '
        'extended later.')
    lines.append('')

    lines.append('## Input-size Summary')
    lines.append('')
    lines.append(
        '| input_size | total_images | correct_or_reject | no_detection | '
        'misclassified_or_false_known | image_level_success_rate |')
    lines.append('|---|---|---|---|---|---|')
    for size in input_sizes:
        stats = size_summary[size]
        lines.append(
            f'| {size} | {stats["total_images"]} | '
            f'{stats["correct_or_reject"]} | {stats["no_detection"]} | '
            f'{stats["misclassified_or_false_known"]} | '
            f'{stats["image_level_success_rate"] * 100:.1f}% |')
    lines.append('')

    lines.append('## Class-level Summary')
    lines.append('')
    lines.append(
        '| input_size | expected_class | total | correct | no_detection | '
        'misclassified | correct_reject | false_known | '
        'main_detected_classes |')
    lines.append('|---|---|---|---|---|---|---|---|---|')
    for size in input_sizes:
        for expected_class in expected_classes:
            stats = class_summary.get((size, expected_class))
            if not stats:
                continue
            lines.append(
                f'| {size} | {expected_class} | {stats["total"]} | '
                f'{stats["correct"]} | {stats["no_detection"]} | '
                f'{stats["misclassified"]} | {stats["correct_reject"]} | '
                f'{stats["false_known"]} | {stats["main_detected_classes"]} |')
    lines.append('')

    lines.append('## Condition-level Summary')
    lines.append('')
    lines.append(
        '| input_size | condition | total | correct_or_reject | failures | '
        'failure_rate | common_failure_modes |')
    lines.append('|---|---|---|---|---|---|---|')
    for size in input_sizes:
        for condition in conditions:
            stats = condition_summary.get((size, condition))
            if not stats:
                continue
            lines.append(
                f'| {size} | {condition} | {stats["total"]} | '
                f'{stats["correct_or_reject"]} | {stats["failures"]} | '
                f'{stats["failure_rate"] * 100:.1f}% | '
                f'{stats["common_failure_modes"]} |')
    lines.append('')

    lines.append('## Policy-level Summary')
    lines.append('')
    policy_summary = compute_policy_summary(records)
    if any(policy_summary.values()):
        lines.append(
            '| input_size | ' + ' | '.join(POLICY_DECISIONS) + ' |')
        lines.append('|---|' + '---|' * len(POLICY_DECISIONS))
        for size in input_sizes:
            counts = policy_summary.get(size, Counter())
            lines.append(
                f'| {size} | ' +
                ' | '.join(str(counts.get(d, 0))
                           for d in POLICY_DECISIONS) + ' |')
        lines.append('')
        lines.append(
            'Counts come from the [PerceptionPolicy] log line '
            '(recycling_cell_vision/perception_policy.py), which runs '
            'independently of what actually gets published -- it reflects '
            'what a failure-aware policy layer would decide for each '
            'image\'s detections, not the current publish behavior.')
    else:
        lines.append(
            '_No [PerceptionPolicy] lines found in these logs -- they '
            'predate the perception_policy.py integration, so this '
            'section has nothing to summarize._')
    lines.append('')

    lines.append('## Key Findings')
    lines.append('')
    for finding in compute_key_findings(records, class_summary, input_sizes):
        lines.append(finding)
    lines.append('')

    lines.append('## Limitations')
    lines.append('')
    lines.append(
        '- No ground-truth bounding boxes -- this measures whether the '
        'expected class name shows up anywhere in an image\'s detections, '
        'not localization accuracy (IoU/mAP).')
    lines.append(
        '- 10 images per class per condition-family is still a small '
        'sample; single misclassified/no_detection images swing a '
        'per-condition rate by 100 percentage points at this size.')
    lines.append(
        '- can/glass_bottle can structurally never score `correct` today '
        '(see Key Findings) -- their no_detection/misclassified counts '
        'reflect a class-mapping gap, not necessarily a harder visual '
        'case, and should not be read as "the model is bad at bottles".')
    lines.append(
        '- confidence_threshold=0.5 and CPUExecutionProvider only; no '
        'sweep over threshold or provider in this analysis.')
    lines.append(
        '- Each image was only run once per input_size (from the '
        'vision_only benchmark), so there is no repeat-run variance data '
        'for borderline-confidence detections.')
    lines.append('')

    lines.append('## Next Steps')
    lines.append('')
    lines.append(
        '- Extend the ONNX/COCO class mapping (or fine-tune on the '
        'project\'s own classes) so can/glass_bottle can actually be '
        'emitted as class_name, then re-run this analysis to see their '
        'real detection stability instead of a guaranteed miss.')
    lines.append(
        '- Collect more images per condition (10 is thin) to make the '
        'condition-level failure_rate numbers less sensitive to a single '
        'image flipping status.')
    lines.append(
        '- Add bounding-box ground truth for a subset of images to move '
        'from this image-level stability check to a real IoU/mAP '
        'evaluation.')
    lines.append(
        '- Re-run at additional confidence_threshold values to see '
        'whether misclassified/no_detection images are near-miss '
        '(just under threshold) or genuinely undetected.')
    lines.append('')

    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, 'w') as md_file:
        md_file.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--log-dir',
        default='logs/vision_benchmark/test_images_real/vision_only',
        help='Directory containing yolo11n_<size>.log files')
    parser.add_argument(
        '--sizes', type=int, nargs='+', default=[640, 416, 320],
        help='Input sizes to look for (default: 640 416 320)')
    parser.add_argument(
        '--model-stem', default='yolo11n',
        help="Log filename stem, matching <stem>_<size>.log (default: "
             "'yolo11n')")
    parser.add_argument(
        '--output-csv',
        default='results/real_image_detection_analysis.csv')
    parser.add_argument(
        '--output-md',
        default='results/real_image_detection_analysis.md')
    args = parser.parse_args()

    all_records = []
    for size in args.sizes:
        log_path = os.path.join(
            args.log_dir, f'{args.model_stem}_{size}.log')
        if not os.path.isfile(log_path):
            print(f'SKIP size={size}: log not found at {log_path}')
            continue
        records = parse_log(log_path)
        print(f'Parsed {log_path}: {len(records)} image record(s)')
        all_records.extend(records)

    if not all_records:
        print('No logs parsed; nothing to write.')
        return

    write_csv(all_records, args.output_csv)
    print(f'Wrote {args.output_csv} ({len(all_records)} rows)')

    class_summary = compute_class_summary(all_records)
    condition_summary = compute_condition_summary(all_records)
    write_markdown(
        all_records, class_summary, condition_summary, args.output_md,
        args.log_dir)
    print(f'Wrote {args.output_md}')


if __name__ == '__main__':
    main()
