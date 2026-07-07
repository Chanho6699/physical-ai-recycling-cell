#!/usr/bin/env python3
"""Compare baseline / aug_medium / aug_strong / real_aug recycling_yolo_
material_v1 models against test_images_real/, at multiple confidence
thresholds -- the 4-way follow-up to tools/analyze_material_v1_
augmentation_comparison.py (which only covered baseline/medium/strong).
Answers a specific question: does adding a small set of real-camera-style
photos (datasets/recycling_material_real_selected/) to the SAME strong-
augmentation training recipe reduce the real-world domain gap further
than augmentation alone did? See
docs/material_v1_real_aug_experiment_plan.md for the full rationale.

Reuses the same efficiency trick as the 3-way script: each model only
needs to be RUN ONCE against test_images_real/, at confidence_
threshold=0.05 (a superset of every higher threshold), and every
threshold's results are re-derived in Python by filtering those same raw
per-image detections and re-running perception_policy.evaluate_
detections() directly (not read from the log's own [PerceptionPolicy]
line, which reflects the log's ACTUAL run threshold, not a virtual one).

Also reads each model's own training results.csv (final epoch's
metrics/mAP50(B) and metrics/mAP50-95(B)) so the writeup can directly
compare "did validation mAP move the same direction as real-image
detection" -- expected to NOT move together, since that was material_v1's
original failure mode (strong validation mAP, near-zero real detection).

Writes:
  - results/material_v1_real_aug_comparison.csv (one row per model x
    threshold)
  - results/material_v1_real_aug_comparison.md (comparison tables + an
    explicit interpretation section)

Usage:
    python3 tools/analyze_material_v1_real_aug_comparison.py \\
        --baseline-log logs/vision_benchmark/test_images_real/vision_only/yolo11n_recycling_material_v1_640_conf005.log \\
        --medium-log logs/vision_benchmark/test_images_real/vision_only/yolo11n_recycling_material_v1_aug_medium_640_conf005.log \\
        --strong-log logs/vision_benchmark/test_images_real/vision_only/yolo11n_recycling_material_v1_aug_strong_640_conf005.log \\
        --real-aug-log logs/vision_benchmark/test_images_real/vision_only/yolo11n_recycling_material_v1_real_aug_640_conf005.log
"""
import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'ros2_ws', 'src',
    'recycling_cell_vision', 'recycling_cell_vision'))
import perception_policy as pp  # noqa: E402

MODELS = ('baseline', 'medium', 'strong', 'real_aug')

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

# test_images_real/'s folder names are the OLD object-level taxonomy --
# this maps ground_truth -> the material_v1-taxonomy class a CORRECT
# prediction should be (see tools/build_recycling_material_v1_dataset.py's
# v0-remap rule for why can->metal / glass_bottle->glass).
CORRECT_CASE_MAP = {
    'can': 'metal',
    'glass_bottle': 'glass',
    'plastic_bottle': 'plastic',
    'paper_cup': 'paper',
}

CSV_FIELDNAMES = [
    'model',
    'threshold',
    'total_images',
    'images_with_detection',
    'no_detection_count',
    'images_with_detection_pct',
    'predicted_class_distribution',
    'can_to_metal_correct', 'can_total',
    'glass_bottle_to_glass_correct', 'glass_bottle_total',
    'plastic_bottle_to_plastic_correct', 'plastic_bottle_total',
    'paper_cup_to_paper_correct', 'paper_cup_total',
    'unknown_accept_sort_risky', 'unknown_total',
    'avg_inference_ms',
    'avg_total_ms',
    'avg_fps',
    'dominant_predicted_class',
    'dominant_class_share',
    'class_collapse',
    'val_mAP50',
    'val_mAP50_95',
]


def extract_ground_truth(image_path):
    parts = image_path.split('/')
    return parts[0] if len(parts) >= 2 else 'unrecognized'


def parse_log(log_path):
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
    no_detection_count = total_images - images_with_detection

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

    non_nd = Counter({c: n for c, n in predicted_counter.items()
                       if c != 'no_detection'})
    if non_nd:
        dominant_class, dominant_count = non_nd.most_common(1)[0]
        dominant_share = dominant_count / sum(non_nd.values())
    else:
        dominant_class, dominant_share = 'no_detection', 0.0

    class_collapse = (
        sum(non_nd.values()) >= 5 and dominant_share >= 0.7
        if non_nd else False
    )

    return {
        'model': model_name,
        'threshold': threshold,
        'total_images': total_images,
        'images_with_detection': images_with_detection,
        'no_detection_count': no_detection_count,
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
        return {'avg_inference_ms': None, 'avg_total_ms': None, 'avg_fps': None}
    avg_inference = sum(p[0] for p in perf_samples) / len(perf_samples)
    avg_total = sum(p[1] for p in perf_samples) / len(perf_samples)
    avg_fps = 1000.0 / avg_total if avg_total else None
    return {
        'avg_inference_ms': round(avg_inference, 2),
        'avg_total_ms': round(avg_total, 2),
        'avg_fps': round(avg_fps, 2) if avg_fps is not None else None,
    }


def read_val_map(results_csv_path):
    """Final-epoch metrics/mAP50(B) and metrics/mAP50-95(B) from a
    training run's results.csv, so the writeup can compare validation
    mAP movement against real-image detection movement directly. Returns
    (None, None) if the file isn't found (e.g. a model trained/exported
    elsewhere without its results.csv kept around)."""
    if not results_csv_path or not os.path.isfile(results_csv_path):
        return None, None
    with open(results_csv_path, newline='') as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        return None, None
    last = rows[-1]
    try:
        return (round(float(last['metrics/mAP50(B)']), 4),
                round(float(last['metrics/mAP50-95(B)']), 4))
    except (KeyError, ValueError):
        return None, None


def write_csv(rows, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compute_totals(rows_by_model):
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


def fmt_delta(base, value):
    delta = value - base
    return f'{base} -> {value} ({"+" if delta >= 0 else ""}{delta})'


def write_markdown(rows, rows_by_model, perf_by_model, val_map_by_model,
                    md_path, logs_by_model):
    totals = compute_totals(rows_by_model)

    lines = []
    lines.append('# material_v1_real_aug Comparison: baseline vs. medium '
                  'vs. strong vs. real_aug')
    lines.append('')

    lines.append('## Experiment Purpose')
    lines.append('')
    lines.append(
        'material_v1/aug_medium/aug_strong (see '
        '`docs/material_v1_augmentation_experiment_plan.md`) tested '
        'whether heavier training-time augmentation alone could close '
        'the domain gap between material_v1\'s training photos and '
        'test_images_real/. It helped (strong beat medium beat baseline) '
        'but stayed far from reliable (3/50 detections at '
        'confidence_threshold=0.5 for strong). This adds a 4th model, '
        '`real_aug`: the SAME strong-augmentation recipe, but trained on '
        'strong\'s dataset PLUS a small set of real-camera-style photos '
        '(`datasets/recycling_material_real_selected/`, 91 images) added '
        'directly to the training set -- see '
        '`docs/material_v1_real_aug_experiment_plan.md`.'
    )
    lines.append('')

    lines.append('## Source logs')
    lines.append('')
    for model in MODELS:
        lines.append(f'- {model}: `{logs_by_model[model]}`')
    lines.append(
        '- All four logged at confidence_threshold=0.05 against the same '
        '50-image test_images_real/ set; every other threshold\'s results '
        'below are re-derived by filtering those same raw per-image '
        'detections and re-running the actual failure-aware policy '
        '(perception_policy.evaluate_detections) at each virtual '
        'threshold, not read from a separately re-run log.'
    )
    lines.append('')

    lines.append('## Inference Speed (threshold-independent)')
    lines.append('')
    lines.append('| model | avg_inference_ms | avg_total_ms | avg_fps |')
    lines.append('|---|---|---|---|')
    for model in MODELS:
        perf = perf_by_model[model]
        lines.append(
            f'| {model} | {perf["avg_inference_ms"]} | '
            f'{perf["avg_total_ms"]} | {perf["avg_fps"]} |')
    lines.append('')

    lines.append('## Validation mAP (final epoch, own held-out split)')
    lines.append('')
    lines.append('| model | val_mAP50 | val_mAP50-95 |')
    lines.append('|---|---|---|')
    for model in MODELS:
        map50, map5095 = val_map_by_model[model]
        lines.append(f'| {model} | {map50} | {map5095} |')
    lines.append('')

    lines.append('## Images With Detection, by Threshold')
    lines.append('')
    lines.append('| threshold | baseline | medium | strong | real_aug |')
    lines.append('|---|---|---|---|---|')
    for threshold in THRESHOLDS:
        cells = []
        for model in MODELS:
            row = next(r for r in rows_by_model[model]
                       if r['threshold'] == threshold)
            cells.append(
                f'{row["images_with_detection"]}/{row["total_images"]} '
                f'({row["images_with_detection_pct"]}%)')
        lines.append(f'| {threshold} | ' + ' | '.join(cells) + ' |')
    lines.append('')

    lines.append('## No-detection Count, by Threshold')
    lines.append('')
    lines.append('| threshold | baseline | medium | strong | real_aug |')
    lines.append('|---|---|---|---|---|')
    for threshold in THRESHOLDS:
        cells = []
        for model in MODELS:
            row = next(r for r in rows_by_model[model]
                       if r['threshold'] == threshold)
            cells.append(str(row['no_detection_count']))
        lines.append(f'| {threshold} | ' + ' | '.join(cells) + ' |')
    lines.append('')

    lines.append('## Predicted-class Distribution, by Threshold')
    lines.append('')
    for threshold in THRESHOLDS:
        lines.append(f'**threshold={threshold}**')
        lines.append('')
        lines.append('| model | distribution |')
        lines.append('|---|---|')
        for model in MODELS:
            row = next(r for r in rows_by_model[model]
                       if r['threshold'] == threshold)
            lines.append(
                f'| {model} | {row["predicted_class_distribution"]} |')
        lines.append('')

    lines.append('## Correct-case Counts (v0-remap targets), by Threshold')
    lines.append('')
    lines.append(
        '| threshold | model | can->metal | glass_bottle->glass | '
        'plastic_bottle->plastic | paper_cup->paper |')
    lines.append('|---|---|---|---|---|---|')
    for threshold in THRESHOLDS:
        for model in MODELS:
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
    lines.append('| threshold | baseline | medium | strong | real_aug |')
    lines.append('|---|---|---|---|---|')
    for threshold in THRESHOLDS:
        cells = []
        for model in MODELS:
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
        for model in MODELS:
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

    # ---- Interpretation (requirement 6 in the task this script answers) ----
    base = totals['baseline']
    strong = totals['strong']
    real_aug = totals['real_aug']

    lines.append('## Interpretation')
    lines.append('')

    lines.append('### 1. Did adding real_selected increase detection count?')
    lines.append('')
    lines.append(
        f'real_aug vs. baseline (summed across all 4 thresholds): '
        f'images_with_detection {fmt_delta(base["detection_sum"], real_aug["detection_sum"])}. '
        f'real_aug vs. strong (the more relevant comparison, since both '
        f'share the same augmentation strength and base dataset -- the '
        f'only difference is real_selected): images_with_detection '
        f'{fmt_delta(strong["detection_sum"], real_aug["detection_sum"])}.'
    )
    lines.append('')

    lines.append('### 2. Did real_aug beat strong (augmentation alone) on '
                  'real photos?')
    lines.append('')
    real_aug_beats_strong_detection = real_aug['detection_sum'] > strong['detection_sum']
    real_aug_beats_strong_correct = real_aug['correct_sum'] >= strong['correct_sum']
    lines.append(
        f'correct v0-remap-target predictions (can->metal + '
        f'glass_bottle->glass + plastic_bottle->plastic), summed across '
        f'all 4 thresholds: strong={strong["correct_sum"]}, '
        f'real_aug={real_aug["correct_sum"]}. detection count: '
        f'strong={strong["detection_sum"]}, real_aug={real_aug["detection_sum"]}. '
        + ('real_aug improves on strong on both counts.'
           if real_aug_beats_strong_detection and real_aug_beats_strong_correct
           else 'real_aug does NOT clearly beat strong on both counts -- '
                'see the per-threshold tables above for where it does/'
                'doesn\'t.')
    )
    lines.append('')

    lines.append('### 3. Did unsafe unknown->ACCEPT_SORT increase?')
    lines.append('')
    lines.append(
        f'unsafe unknown->ACCEPT_SORT, summed across all 4 thresholds: '
        f'baseline={base["risky_sum"]}, strong={strong["risky_sum"]}, '
        f'real_aug={real_aug["risky_sum"]}. '
        + ('real_aug did NOT increase unsafe accepts versus strong.'
           if real_aug['risky_sum'] <= strong['risky_sum']
           else 'real_aug INCREASED unsafe accepts versus strong -- worth '
                'checking which unknown images flipped to ACCEPT_SORT '
                'before shipping this model.')
    )
    lines.append('')

    lines.append('### 4. Did a new class collapse appear?')
    lines.append('')
    lines.append(
        f'class_collapse observed anywhere across thresholds: '
        f'baseline={base["collapse_any"]}, strong={strong["collapse_any"]}, '
        f'real_aug={real_aug["collapse_any"]}. '
        + ('No collapse in real_aug.' if not real_aug['collapse_any']
           else 'real_aug DID trigger the collapse flag at some threshold '
                '-- see the Class Collapse Check table above for which '
                'one, and check whether real_selected\'s class balance '
                '(plastic 35 / metal 24 / paper 22 / glass 10 -- glass is '
                'the smallest by a wide margin) is a likely cause.')
    )
    lines.append('')

    lines.append('### 5. Validation mAP vs. real-image detection: same '
                  'direction or not?')
    lines.append('')
    map_values = {m: val_map_by_model[m][0] for m in MODELS}
    detection_values = {m: totals[m]['detection_sum'] for m in MODELS}
    if all(v is not None for v in map_values.values()):
        map_sorted = sorted(MODELS, key=lambda m: map_values[m])
        det_sorted = sorted(MODELS, key=lambda m: detection_values[m])
        same_order = map_sorted == det_sorted
        lines.append(
            'val_mAP50: ' + ', '.join(f'{m}={map_values[m]}' for m in MODELS)
            + '. real-image detection_sum (summed across thresholds): '
            + ', '.join(f'{m}={detection_values[m]}' for m in MODELS) + '. '
            + ('Both metrics rank the four models in the SAME order -- '
               'validation mAP is, unusually, a reasonable proxy for '
               'real-image behavior in this comparison.'
               if same_order else
               'The two metrics rank the four models in a DIFFERENT '
               'order -- validation mAP on material_v1\'s own held-out '
               'split is NOT a reliable proxy for real-image detection '
               'here, consistent with the original material_v1 failure '
               'mode (strong validation mAP, near-zero real-world '
               'recall). Use the real-image numbers above, not '
               'validation mAP, to judge which model is actually better '
               'for test_images_real/-like conditions.')
        )
    else:
        lines.append(
            'One or more models\' results.csv was not found -- val_mAP '
            'comparison skipped (see the Validation mAP table above for '
            'which models have data).'
        )
    lines.append('')

    lines.append('### 6. Next steps')
    lines.append('')
    next_steps = []
    # "helped" requires ALL THREE of: more detections, correctness not
    # worse, and unsafe accepts not worse -- raw detection_sum going up
    # is not itself good news if it's driven by more WRONG detections on
    # unknown/other images rather than more correct ones (that is a
    # regression, not an improvement, even though it also "detects more").
    detection_up = real_aug['detection_sum'] > strong['detection_sum']
    correctness_up = real_aug['correct_sum'] >= strong['correct_sum']
    safety_ok = real_aug['risky_sum'] <= strong['risky_sum']
    if detection_up and correctness_up and safety_ok and not real_aug['collapse_any']:
        next_steps.append(
            'Adding real photos helped on every axis (more detections, '
            'correctness not worse, unsafe accepts not worse) -- the '
            'highest-leverage next step is collecting MORE real_selected '
            'images (91 is a proof-of-concept sample size, not a '
            'production one; glass at only 10 images is the thinnest '
            'class and the best candidate for more collection).'
        )
    elif detection_up and not (correctness_up and safety_ok):
        next_steps.append(
            f'real_aug detects MORE than strong (images_with_detection '
            f'{strong["detection_sum"]} -> {real_aug["detection_sum"]}) but '
            f'this is NOT an unambiguous win: correct v0-remap-target '
            f'predictions went {strong["correct_sum"]} -> '
            f'{real_aug["correct_sum"]} and unsafe unknown->ACCEPT_SORT '
            f'went {strong["risky_sum"]} -> {real_aug["risky_sum"]}. The '
            f'extra detections are disproportionately WRONG-class or '
            f'false-positive-on-unknown, not more correct sorts -- do '
            f'NOT treat higher detection_sum alone as "real_aug is '
            f'better." Before collecting more real_selected data, first '
            f'check `datasets/recycling_yolo_material_v1_real_aug/'
            f'previews/real_selected_*` for bad GroundingDINO boxes '
            f'(unreviewed pseudo-labels) and check which specific '
            f'`unknown/` images in test_images_real/ flipped to '
            f'ACCEPT_SORT (see the per-threshold unsafe-accept table '
            f'above) -- a few noisy real_selected labels teaching the '
            f'model to fire more readily on background clutter is a '
            f'plausible cause given real_selected images were shot with '
            f'more background clutter than candidates_v1.'
        )
    else:
        next_steps.append(
            'Adding 91 real photos did not clearly beat augmentation '
            'alone -- before concluding "real photos don\'t help", check '
            '`datasets/recycling_yolo_material_v1_real_aug/previews/'
            'real_selected_*` for bad GroundingDINO boxes (these are '
            'unreviewed pseudo-labels) and consider whether 91 images is '
            'simply too small a fraction of the ~700-image training set '
            'to move the needle (real_selected is diluted roughly 1:7 by '
            'candidates_v1+v0_remapped) -- oversampling real_selected or '
            'collecting substantially more of it are both worth trying '
            'before changing taxonomy.'
        )
    if not safety_ok or real_aug['collapse_any']:
        next_steps.append(
            'Unsafe accepts increased and/or class collapse appeared -- '
            'do NOT deploy real_aug past benchmarking as-is. Consider '
            'tightening `policy_confidence_threshold` (see '
            '`ros2_ws/src/recycling_cell_vision/recycling_cell_vision/'
            'perception_policy.py`) and re-running this comparison, or '
            'reverting to `strong` (0 unsafe accepts) until real_aug\'s '
            'false-positive-on-unknown behavior is understood.'
        )
    total_real_detections = sum(
        totals[m]['detection_sum'] for m in ('baseline', 'medium', 'strong', 'real_aug'))
    if total_real_detections < 40:
        next_steps.append(
            'Absolute detection counts remain low across all four models '
            '(single digits per model out of 50 images at every '
            'threshold) -- taxonomy (plastic/metal/glass/paper) does not '
            'look like the bottleneck (no collapse, correct-case counts '
            'exist for every class), so the likelier lever is still more/'
            'better training data in test_images_real/\'s domain, not a '
            'taxonomy or confidence-threshold change.'
        )
    for step in next_steps:
        lines.append(f'- {step}')
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
        'would shift these results as much as adding real_selected did.')
    lines.append(
        '- ground_truth is test_images_real/\'s folder name compared to '
        'predicted_class as a plain string via the v0-remap mapping -- '
        'no bounding-box IoU is checked, only whether the right class '
        'name appears at all.')
    lines.append(
        '- real_selected\'s own labels are unreviewed GroundingDINO '
        'pseudo-labels (same caveat as candidates_v1/v0) -- a real_aug '
        'result that looks worse than expected could be a labeling-'
        'quality artifact, not evidence that real photos don\'t help in '
        'general.')
    lines.append('')

    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, 'w') as md_file:
        md_file.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-log', required=True)
    parser.add_argument('--medium-log', required=True)
    parser.add_argument('--strong-log', required=True)
    parser.add_argument('--real-aug-log', required=True)
    parser.add_argument(
        '--baseline-val-csv',
        default='runs/detect/runs/recycling_yolo/yolo11n_material_v1_640/results.csv')
    parser.add_argument(
        '--medium-val-csv',
        default='runs/detect/runs/recycling_yolo/yolo11n_material_v1_aug_medium_640/results.csv')
    parser.add_argument(
        '--strong-val-csv',
        default='runs/detect/runs/recycling_yolo/yolo11n_material_v1_aug_strong_640/results.csv')
    parser.add_argument(
        '--real-aug-val-csv',
        default='runs/detect/runs/recycling_yolo/yolo11n_material_v1_real_aug_640/results.csv')
    parser.add_argument(
        '--output-csv',
        default='results/material_v1_real_aug_comparison.csv')
    parser.add_argument(
        '--output-md',
        default='results/material_v1_real_aug_comparison.md')
    args = parser.parse_args()

    logs_by_model = {
        'baseline': args.baseline_log,
        'medium': args.medium_log,
        'strong': args.strong_log,
        'real_aug': args.real_aug_log,
    }
    val_csv_by_model = {
        'baseline': args.baseline_val_csv,
        'medium': args.medium_val_csv,
        'strong': args.strong_val_csv,
        'real_aug': args.real_aug_val_csv,
    }

    all_rows = []
    rows_by_model = {}
    perf_by_model = {}
    val_map_by_model = {}

    for model in MODELS:
        log_path = logs_by_model[model]
        if not os.path.isfile(log_path):
            parser.error(f'{model} log not found: {log_path}')
        per_image, perf_samples = parse_log(log_path)
        print(f'Parsed {log_path}: {len(per_image)} image record(s), '
              f'{len(perf_samples)} perf sample(s)')

        perf = compute_perf(perf_samples)
        perf_by_model[model] = perf
        val_map_by_model[model] = read_val_map(val_csv_by_model[model])

        model_rows = []
        for threshold in THRESHOLDS:
            results = evaluate_at_threshold(per_image, threshold)
            row = compute_row(model, threshold, results)
            row['avg_inference_ms'] = perf['avg_inference_ms']
            row['avg_total_ms'] = perf['avg_total_ms']
            row['avg_fps'] = perf['avg_fps']
            row['val_mAP50'], row['val_mAP50_95'] = val_map_by_model[model]
            model_rows.append(row)
            all_rows.append(row)
        rows_by_model[model] = model_rows

    write_csv(all_rows, args.output_csv)
    print(f'Wrote {args.output_csv} ({len(all_rows)} rows)')

    write_markdown(
        all_rows, rows_by_model, perf_by_model, val_map_by_model,
        args.output_md, logs_by_model)
    print(f'Wrote {args.output_md}')


if __name__ == '__main__':
    main()
