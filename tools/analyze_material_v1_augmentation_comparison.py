#!/usr/bin/env python3
"""Compare baseline / medium-augmentation / strong-augmentation
recycling_yolo_material_v1 models against test_images_real/, at multiple
confidence thresholds, on metrics beyond validation mAP -- built to
answer a specific question: does heavier training-time augmentation
actually reduce the domain gap (near-zero real-world detection despite
strong validation mAP) documented in
docs/material_v1_augmentation_experiment_plan.md, or does it just make
things worse?

Efficiency trick: each model only needs to be RUN ONCE against
test_images_real/, at the LOWEST threshold being compared
(confidence_threshold=0.05) -- a detection's confidence doesn't change
with the threshold, only whether it survives the threshold filter. So
rather than 3 models x 4 thresholds = 12 live ROS runs, this script
takes one log per model (already run at confidence_threshold=0.05,
which is a superset of what any higher threshold would keep) and
re-derives every higher threshold's results by filtering the SAME
per-image detections in Python, matching exactly what a live run at
that threshold would have logged.

Because of that, the failure-aware policy decision (ACCEPT_SORT/
ROUTE_TO_REJECT/SKIP_NO_DETECTION/RETRY_VIEW/MANUAL_REVIEW) also can't
be read from the log directly -- the log's own [PerceptionPolicy] line
reflects whatever detections survived the ACTUAL 0.05 run, not what
would have survived a stricter virtual threshold. So this script
re-evaluates the policy itself (importing recycling_cell_vision's
perception_policy.py directly -- it's pure Python with no ROS2/rclpy
dependency, so no ROS environment needs to be sourced to run this
script) against the threshold-filtered detections for every virtual
threshold, with confidence_threshold=policy_confidence_threshold=that
threshold (matching how tools/run_vision_size_benchmark.sh configures
a real run).

Reads three per-model vision_only logs (see
tools/run_vision_size_benchmark.sh / direct `ros2 run` invocations, all
at confidence_threshold=0.05 against test_images_real/) and for each
model x threshold in {0.5, 0.3, 0.1, 0.05}, computes:
  - images_with_detection (out of 50)
  - predicted_class distribution
  - correct-case counts for can->metal, glass_bottle->glass,
    plastic_bottle->plastic (the material_v1 taxonomy's remap targets),
    plus paper_cup->paper as bonus context
  - unknown ground-truth images that got ACCEPT_SORT anyway (the
    specific "false accept" risk this project's policy layer exists to
    catch)
  - avg inference_ms / avg total_ms (threshold-independent, computed
    once per model from all per-image [VisionPerf] lines)
  - a class-collapse check (does one predicted class dominate
    regardless of ground truth, the same failure shape as
    custom_autolabel_v0's earlier "paper collapse")

Writes:
  - results/material_v1_augmentation_comparison.csv (one row per model x
    threshold)
  - results/material_v1_augmentation_comparison.md (comparison tables +
    an explicit medium-vs-strong-vs-baseline verdict)

Usage:
    python3 tools/analyze_material_v1_augmentation_comparison.py \\
        --baseline-log logs/vision_benchmark/test_images_real/vision_only/yolo11n_recycling_material_v1_640_conf005.log \\
        --medium-log logs/vision_benchmark/test_images_real/vision_only/yolo11n_recycling_material_v1_aug_medium_640_conf005.log \\
        --strong-log logs/vision_benchmark/test_images_real/vision_only/yolo11n_recycling_material_v1_aug_strong_640_conf005.log
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'ros2_ws', 'src',
    'recycling_cell_vision', 'recycling_cell_vision'))
import perception_policy as pp  # noqa: E402

import re  # noqa: E402

PROCESSING_RE = re.compile(
    r'Processing image \[(?P<index>\d+)/(?P<total>\d+)\]: (?P<image>\S+)'
)

ONNX_DETECTION_RE = re.compile(
    r'ONNX detection: object_id=(?P<object_id>\S+) '
    r'class_name=(?P<class_name>\S+) confidence=(?P<confidence>[\d.]+)'
)

VISION_PERF_RE = re.compile(
    r'\[VisionPerf\] source=\S+ image=\S+ input_size=\d+ detections=\d+ '
    r'acquire=[\d.]+ms preprocess=[\d.]+ms inference=(?P<inference_ms>[\d.]+)ms '
    r'postprocess=[\d.]+ms total=(?P<total_ms>[\d.]+)ms'
)

THRESHOLDS = (0.5, 0.3, 0.1, 0.05)

# material_v1's v0-remap rule (see tools/build_recycling_material_v1_
# dataset.py) inverted: test_images_real/'s folder names are the OLD
# object-level taxonomy, so this maps ground_truth -> the material_v1
# class a CORRECT prediction should be.
CORRECT_CASE_MAP = {
    'can': 'metal',
    'glass_bottle': 'glass',
    'plastic_bottle': 'plastic',
    'paper_cup': 'paper',  # bonus context, not one of the 3 explicitly asked for
}

CSV_FIELDNAMES = [
    'model',
    'threshold',
    'total_images',
    'images_with_detection',
    'images_with_detection_pct',
    'predicted_class_distribution',
    'can_to_metal_correct', 'can_total',
    'glass_bottle_to_glass_correct', 'glass_bottle_total',
    'plastic_bottle_to_plastic_correct', 'plastic_bottle_total',
    'paper_cup_to_paper_correct', 'paper_cup_total',
    'unknown_accept_sort_risky', 'unknown_total',
    'avg_inference_ms',
    'avg_total_ms',
    'dominant_predicted_class',
    'dominant_class_share',
    'class_collapse',
]


def extract_ground_truth(image_path):
    parts = image_path.split('/')
    return parts[0] if len(parts) >= 2 else 'unrecognized'


def parse_log(log_path):
    """Returns (per_image, perf_samples):
    per_image: {image_path: {'ground_truth':, 'detections': [(class_name,
      confidence), ...]}}  -- detections are ALL raw ONNX detections
      logged (i.e. everything that passed whatever confidence_threshold
      the log was actually run with -- expected to be 0.05).
    perf_samples: list of (inference_ms, total_ms) per image.
    """
    per_image = {}
    perf_samples = []
    current_image = None
    pending_detections = []

    with open(log_path, 'r', errors='replace') as log_file:
        for line in log_file:
            match = PROCESSING_RE.search(line)
            if match:
                current_image = match.group('image')
                pending_detections = []
                per_image[current_image] = {
                    'ground_truth': extract_ground_truth(current_image),
                    'detections': [],
                }
                continue

            match = ONNX_DETECTION_RE.search(line)
            if match and current_image is not None:
                pending_detections.append((
                    match.group('class_name'),
                    float(match.group('confidence')),
                ))
                per_image[current_image]['detections'] = pending_detections
                continue

            match = VISION_PERF_RE.search(line)
            if match:
                perf_samples.append((
                    float(match.group('inference_ms')),
                    float(match.group('total_ms')),
                ))

    return per_image, perf_samples


def evaluate_at_threshold(per_image, threshold):
    """Re-derives predicted_class + policy decision for every image at a
    virtual confidence threshold, by filtering the already-parsed raw
    detections and re-running perception_policy.evaluate_detections()
    directly (not trusting the log's own [PerceptionPolicy] line, which
    reflects the log's ACTUAL run threshold, not this virtual one)."""
    results = {}
    for image, info in per_image.items():
        filtered = [
            {'class_name': c, 'confidence': conf}
            for c, conf in info['detections'] if conf >= threshold
        ]
        if filtered:
            predicted_class = max(filtered, key=lambda d: d['confidence'])['class_name']
        else:
            predicted_class = 'no_detection'

        policy_result = pp.evaluate_detections(
            filtered, confidence_threshold=threshold)

        results[image] = {
            'ground_truth': info['ground_truth'],
            'predicted_class': predicted_class,
            'policy_decision': policy_result['decision'],
        }
    return results


def compute_row(model_name, threshold, results):
    total_images = len(results)
    images_with_detection = sum(
        1 for r in results.values() if r['predicted_class'] != 'no_detection')

    predicted_counter = Counter(r['predicted_class'] for r in results.values())
    distribution = ';'.join(
        f'{cls}:{count}' for cls, count in predicted_counter.most_common())

    correct_counts = {}
    totals = {}
    for ground_truth, target_class in CORRECT_CASE_MAP.items():
        matching = [r for r in results.values()
                    if r['ground_truth'] == ground_truth]
        totals[ground_truth] = len(matching)
        correct_counts[ground_truth] = sum(
            1 for r in matching if r['predicted_class'] == target_class)

    unknown_images = [r for r in results.values()
                       if r['ground_truth'] == 'unknown']
    unknown_accept_sort_risky = sum(
        1 for r in unknown_images if r['policy_decision'] == 'ACCEPT_SORT')

    if predicted_counter:
        dominant_class, dominant_count = predicted_counter.most_common(1)[0]
        # Collapse is only meaningful among classes OTHER than
        # no_detection (a mostly-no_detection run isn't a "collapse").
        non_nd = Counter({c: n for c, n in predicted_counter.items()
                           if c != 'no_detection'})
        if non_nd:
            dominant_class, dominant_count = non_nd.most_common(1)[0]
            dominant_share = dominant_count / sum(non_nd.values())
        else:
            dominant_class, dominant_share = 'no_detection', 0.0
    else:
        dominant_class, dominant_share = 'no_detection', 0.0

    # Collapse flag: only meaningful with enough detections to judge --
    # 1-2 detections trivially have a "100% dominant" class by definition.
    class_collapse = (
        sum(non_nd.values()) >= 5 and dominant_share >= 0.7
        if predicted_counter and non_nd else False
    )

    return {
        'model': model_name,
        'threshold': threshold,
        'total_images': total_images,
        'images_with_detection': images_with_detection,
        'images_with_detection_pct': round(
            100 * images_with_detection / total_images, 1)
            if total_images else 0.0,
        'predicted_class_distribution': distribution,
        'can_to_metal_correct': correct_counts['can'],
        'can_total': totals['can'],
        'glass_bottle_to_glass_correct': correct_counts['glass_bottle'],
        'glass_bottle_total': totals['glass_bottle'],
        'plastic_bottle_to_plastic_correct': correct_counts['plastic_bottle'],
        'plastic_bottle_total': totals['plastic_bottle'],
        'paper_cup_to_paper_correct': correct_counts['paper_cup'],
        'paper_cup_total': totals['paper_cup'],
        'unknown_accept_sort_risky': unknown_accept_sort_risky,
        'unknown_total': len(unknown_images),
        'dominant_predicted_class': dominant_class,
        'dominant_class_share': round(dominant_share, 3),
        'class_collapse': class_collapse,
    }


def compute_perf(perf_samples):
    if not perf_samples:
        return {'avg_inference_ms': None, 'avg_total_ms': None}
    avg_inference = sum(p[0] for p in perf_samples) / len(perf_samples)
    avg_total = sum(p[1] for p in perf_samples) / len(perf_samples)
    return {
        'avg_inference_ms': round(avg_inference, 2),
        'avg_total_ms': round(avg_total, 2),
    }


def write_csv(rows, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compute_verdict(rows_by_model):
    """Aggregates across thresholds (sum of images_with_detection and
    correct-case counts) per model, then compares medium/strong against
    baseline to produce an explicit improved/worse/mixed verdict."""
    totals = {}
    for model, rows in rows_by_model.items():
        detection_sum = sum(r['images_with_detection'] for r in rows)
        correct_sum = sum(
            r['can_to_metal_correct'] + r['glass_bottle_to_glass_correct']
            + r['plastic_bottle_to_plastic_correct'] for r in rows)
        risky_sum = sum(r['unknown_accept_sort_risky'] for r in rows)
        collapse_any = any(r['class_collapse'] for r in rows)
        totals[model] = {
            'detection_sum': detection_sum,
            'correct_sum': correct_sum,
            'risky_sum': risky_sum,
            'collapse_any': collapse_any,
        }
    return totals


def write_markdown(rows, rows_by_model, perf_by_model, md_path,
                    baseline_log, medium_log, strong_log):
    models = ['baseline', 'medium', 'strong']
    verdict_totals = compute_verdict(rows_by_model)

    lines = []
    lines.append('# material_v1 Augmentation Comparison: baseline vs. medium vs. strong')
    lines.append('')

    lines.append('## Experiment Purpose')
    lines.append('')
    lines.append(
        'The material_v1 baseline had strong validation mAP but detected '
        'almost nothing on test_images_real/ (see '
        '`docs/material_v1_augmentation_experiment_plan.md`). This '
        'compares baseline against medium- and strong-augmentation '
        'retrains of the SAME dataset, on real-image behavior -- not '
        'validation mAP -- across four confidence thresholds (0.5/0.3/'
        '0.1/0.05), to determine whether heavier augmentation actually '
        'closes the domain gap or just adds noise.'
    )
    lines.append('')

    lines.append('## Source logs')
    lines.append('')
    lines.append(f'- baseline: `{baseline_log}`')
    lines.append(f'- medium: `{medium_log}`')
    lines.append(f'- strong: `{strong_log}`')
    lines.append(
        '- All three logged at confidence_threshold=0.05 against the '
        'same 50-image test_images_real/ set; every other threshold\'s '
        'results below are re-derived by filtering those same raw '
        'per-image detections and re-running the actual failure-aware '
        'policy (perception_policy.evaluate_detections) at each virtual '
        'threshold, not read from a separately re-run log.'
    )
    lines.append('')

    lines.append('## Inference Speed (threshold-independent)')
    lines.append('')
    lines.append('| model | avg_inference_ms | avg_total_ms |')
    lines.append('|---|---|---|')
    for model in models:
        perf = perf_by_model[model]
        lines.append(
            f'| {model} | {perf["avg_inference_ms"]} | '
            f'{perf["avg_total_ms"]} |')
    lines.append('')

    lines.append('## Images With Detection, by Threshold')
    lines.append('')
    lines.append(
        '| threshold | baseline | medium | strong |')
    lines.append('|---|---|---|---|')
    for threshold in THRESHOLDS:
        cells = []
        for model in models:
            row = next(r for r in rows_by_model[model]
                       if r['threshold'] == threshold)
            cells.append(
                f'{row["images_with_detection"]}/{row["total_images"]} '
                f'({row["images_with_detection_pct"]}%)')
        lines.append(f'| {threshold} | ' + ' | '.join(cells) + ' |')
    lines.append('')

    lines.append('## Predicted-class Distribution, by Threshold')
    lines.append('')
    for threshold in THRESHOLDS:
        lines.append(f'**threshold={threshold}**')
        lines.append('')
        lines.append('| model | distribution |')
        lines.append('|---|---|')
        for model in models:
            row = next(r for r in rows_by_model[model]
                       if r['threshold'] == threshold)
            lines.append(
                f'| {model} | {row["predicted_class_distribution"]} |')
        lines.append('')

    lines.append('## Correct-case Counts (v0-remap targets), by Threshold')
    lines.append('')
    lines.append(
        '| threshold | model | can->metal | glass_bottle->glass | '
        'plastic_bottle->plastic | paper_cup->paper (bonus) |')
    lines.append('|---|---|---|---|---|---|')
    for threshold in THRESHOLDS:
        for model in models:
            row = next(r for r in rows_by_model[model]
                       if r['threshold'] == threshold)
            lines.append(
                f'| {threshold} | {model} | '
                f'{row["can_to_metal_correct"]}/{row["can_total"]} | '
                f'{row["glass_bottle_to_glass_correct"]}/{row["glass_bottle_total"]} | '
                f'{row["plastic_bottle_to_plastic_correct"]}/{row["plastic_bottle_total"]} | '
                f'{row["paper_cup_to_paper_correct"]}/{row["paper_cup_total"]} |')
    lines.append('')

    lines.append('## Unsafe ACCEPT_SORT on unknown Ground Truth, by Threshold')
    lines.append('')
    lines.append(
        '| threshold | baseline | medium | strong |')
    lines.append('|---|---|---|---|')
    for threshold in THRESHOLDS:
        cells = []
        for model in models:
            row = next(r for r in rows_by_model[model]
                       if r['threshold'] == threshold)
            cells.append(
                f'{row["unknown_accept_sort_risky"]}/{row["unknown_total"]}')
        lines.append(f'| {threshold} | ' + ' | '.join(cells) + ' |')
    lines.append('')

    lines.append('## Class Collapse Check, by Threshold')
    lines.append('')
    lines.append(
        '| threshold | model | dominant_class | dominant_share | '
        'collapse_flag |')
    lines.append('|---|---|---|---|---|')
    for threshold in THRESHOLDS:
        for model in models:
            row = next(r for r in rows_by_model[model]
                       if r['threshold'] == threshold)
            lines.append(
                f'| {threshold} | {model} | '
                f'{row["dominant_predicted_class"]} | '
                f'{row["dominant_class_share"] * 100:.0f}% | '
                f'{row["class_collapse"]} |')
    lines.append('')
    lines.append(
        '`collapse_flag` only triggers with >=5 non-empty detections and '
        'a dominant share >=70% -- with this few real detections overall, '
        'most cells here are too small a sample for the flag to be '
        'meaningful and are left `False` by design rather than reporting '
        'a spurious 100%-of-1 "collapse".'
    )
    lines.append('')

    lines.append('## Verdict: medium/strong vs. baseline')
    lines.append('')
    base = verdict_totals['baseline']
    for model in ('medium', 'strong'):
        stats = verdict_totals[model]
        detection_delta = stats['detection_sum'] - base['detection_sum']
        correct_delta = stats['correct_sum'] - base['correct_sum']
        risky_delta = stats['risky_sum'] - base['risky_sum']
        lines.append(
            f'- **{model}** vs baseline (summed across all 4 '
            f'thresholds): images_with_detection '
            f'{base["detection_sum"]} -> {stats["detection_sum"]} '
            f'({"+" if detection_delta >= 0 else ""}{detection_delta}), '
            f'correct v0-remap-target predictions '
            f'{base["correct_sum"]} -> {stats["correct_sum"]} '
            f'({"+" if correct_delta >= 0 else ""}{correct_delta}), '
            f'unsafe unknown->ACCEPT_SORT '
            f'{base["risky_sum"]} -> {stats["risky_sum"]} '
            f'({"+" if risky_delta >= 0 else ""}{risky_delta}), '
            f'class collapse observed: {stats["collapse_any"]}.'
        )

    medium_stats = verdict_totals['medium']
    strong_stats = verdict_totals['strong']
    medium_improved = (
        medium_stats['detection_sum'] > base['detection_sum']
        and medium_stats['correct_sum'] >= base['correct_sum'])
    strong_improved = (
        strong_stats['detection_sum'] > base['detection_sum']
        and strong_stats['correct_sum'] >= base['correct_sum'])
    strong_vs_medium_better = (
        strong_stats['correct_sum'] > medium_stats['correct_sum']
        or (strong_stats['correct_sum'] == medium_stats['correct_sum']
            and strong_stats['detection_sum'] > medium_stats['detection_sum']))

    lines.append('')
    if medium_improved and strong_improved and strong_vs_medium_better:
        conclusion = (
            'Augmentation helps, and **strong outperforms medium**: both '
            'reduce the domain gap versus baseline, and pushing '
            'augmentation further kept helping rather than degrading '
            'results on this dataset.'
        )
    elif medium_improved and strong_improved and not strong_vs_medium_better:
        conclusion = (
            'Augmentation helps, but **medium is the better trade-off**: '
            'both improve over baseline, but strong did not add further '
            'gains over medium (and may cost a bit of precision) -- '
            'diminishing (or slightly negative) returns from pushing '
            'augmentation harder on this ~700-image dataset.'
        )
    elif medium_improved and not strong_improved:
        conclusion = (
            '**Strong augmentation is too much for this dataset size**: '
            'medium improved over baseline, but strong regressed below '
            'baseline on real-image detection/correctness -- pushing '
            'augmentation further broke more than it fixed. Medium is '
            'the better choice of the three.'
        )
    elif not medium_improved and not strong_improved:
        conclusion = (
            'Neither medium nor strong augmentation meaningfully '
            'improved on baseline\'s real-image behavior -- the domain '
            'gap does not appear to be primarily an augmentation-strength '
            'problem. See docs/recycling_material_v1_dataset_plan.md\'s '
            'next-steps for data-side options (more/more-varied training '
            'images) instead.'
        )
    else:
        conclusion = (
            'Mixed result: the three models trade off differently across '
            'metrics (see the deltas above) rather than one clearly '
            'dominating -- judgment call depends on which metric '
            '(detection count vs. correctness vs. safety) matters most '
            'for the next step.'
        )
    lines.append(f'**Conclusion:** {conclusion}')
    lines.append('')

    lines.append('## Limitations')
    lines.append('')
    lines.append(
        '- Real detection counts are very low in absolute terms (single '
        'digits out of 50 images even at threshold=0.05) -- every metric '
        'above is working with a small sample, so differences of 1-2 '
        'detections should not be over-interpreted as a robust trend.')
    lines.append(
        '- Each model was trained once (no repeated runs), so there is '
        'no variance estimate to know whether a different random seed '
        'would shift these results as much as the augmentation change '
        'did.')
    lines.append(
        '- ground_truth is test_images_real/\'s folder name compared to '
        'predicted_class as a plain string via the v0-remap mapping -- '
        'no bounding-box IoU is checked, only whether the right class '
        'name appears at all.')
    lines.append('')

    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, 'w') as md_file:
        md_file.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-log', required=True)
    parser.add_argument('--medium-log', required=True)
    parser.add_argument('--strong-log', required=True)
    parser.add_argument(
        '--output-csv',
        default='results/material_v1_augmentation_comparison.csv')
    parser.add_argument(
        '--output-md',
        default='results/material_v1_augmentation_comparison.md')
    args = parser.parse_args()

    logs = {
        'baseline': args.baseline_log,
        'medium': args.medium_log,
        'strong': args.strong_log,
    }

    all_rows = []
    rows_by_model = {}
    perf_by_model = {}

    for model, log_path in logs.items():
        if not os.path.isfile(log_path):
            parser.error(f'{model} log not found: {log_path}')
        per_image, perf_samples = parse_log(log_path)
        print(f'Parsed {log_path}: {len(per_image)} image record(s), '
              f'{len(perf_samples)} perf sample(s)')

        perf = compute_perf(perf_samples)
        perf_by_model[model] = perf

        model_rows = []
        for threshold in THRESHOLDS:
            results = evaluate_at_threshold(per_image, threshold)
            row = compute_row(model, threshold, results)
            row['avg_inference_ms'] = perf['avg_inference_ms']
            row['avg_total_ms'] = perf['avg_total_ms']
            model_rows.append(row)
            all_rows.append(row)
        rows_by_model[model] = model_rows

    write_csv(all_rows, args.output_csv)
    print(f'Wrote {args.output_csv} ({len(all_rows)} rows)')

    write_markdown(
        all_rows, rows_by_model, perf_by_model, args.output_md,
        args.baseline_log, args.medium_log, args.strong_log)
    print(f'Wrote {args.output_md}')


if __name__ == '__main__':
    main()
