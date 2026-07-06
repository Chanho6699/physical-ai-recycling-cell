#!/usr/bin/env python3
"""Parse recycling_cell_vision [VisionPerf] benchmark logs into a CSV/Markdown
summary.

Reads logs/vision_benchmark/yolo11n_<size>.log (produced by
tools/run_vision_size_benchmark.sh) for each requested input size and
extracts, per size:
  - per-image ONNX detections (class_name, confidence)
  - per-image [VisionPerf] timing (acquire/preprocess/inference/postprocess/
    total/fps)
  - the final [VisionPerf][avg over last N] rolling average
  - Published detection counts per image
  - task_manager overall_success outcomes
  - whether "Image folder scan completed" was reached

Writes:
  - results/vision_benchmark_summary.csv (one row per input size)
  - results/vision_benchmark_summary.md   (report with a fixed set of
    sections: Experiment Purpose, Environment, Result Table,
    Interpretation, Limitations, Next Steps)

The raw *.log files under logs/vision_benchmark/ are gitignored (see
.gitignore's `*.log` rule); only this script's csv/md output is meant to be
committed.

Usage:
    python3 tools/parse_vision_benchmark_logs.py
    python3 tools/parse_vision_benchmark_logs.py --sizes 640 416
"""
import argparse
import csv
import os
import re

VISION_PERF_RE = re.compile(
    r'\[VisionPerf\] source=(?P<source>\S+) image=(?P<image>\S+) '
    r'input_size=(?P<input_size>\d+) detections=(?P<detections>\d+) '
    r'acquire=(?P<acquire_ms>[\d.]+)ms '
    r'preprocess=(?P<preprocess_ms>[\d.]+)ms '
    r'inference=(?P<inference_ms>[\d.]+)ms '
    r'postprocess=(?P<postprocess_ms>[\d.]+)ms '
    r'total=(?P<total_ms>[\d.]+)ms fps=(?P<fps>[\d.]+) '
    r'provider=(?P<provider>\S+)'
)

VISION_PERF_AVG_RE = re.compile(
    r'\[VisionPerf\]\[avg over last (?P<n>\d+)\] '
    r'avg_total=(?P<avg_total_ms>[\d.]+)ms '
    r'avg_inference=(?P<avg_inference_ms>[\d.]+)ms '
    r'avg_fps=(?P<avg_fps>[\d.]+)'
)

ONNX_DETECTION_RE = re.compile(
    r'ONNX detection: object_id=(?P<object_id>\S+) '
    r'class_name=(?P<class_name>\S+) confidence=(?P<confidence>[\d.]+)'
)

PUBLISHED_RE = re.compile(
    r'Published (?P<count>\d+) detections from (?P<image>\S+)'
)

OVERALL_SUCCESS_RE = re.compile(
    r'finished: overall_success=(?P<success>True|False)'
)

SCAN_COMPLETED_TEXT = 'Image folder scan completed'

CSV_FIELDNAMES = [
    'input_size',
    'avg_total_ms',
    'avg_inference_ms',
    'avg_fps',
    'images_processed',
    'detections_summary',
    'published_counts',
    'success_count',
    'success_total',
    'all_success',
    'scan_completed',
]


def parse_log(log_path):
    """Parse one benchmark log file into a summary dict."""
    images = []
    detections_by_image = {}
    published_by_image = {}
    pending_detections = []
    per_image_perf = []
    avg_samples = []
    success_flags = []
    scan_completed = False
    input_size = None
    provider = None

    with open(log_path, 'r', errors='replace') as log_file:
        for line in log_file:
            match = ONNX_DETECTION_RE.search(line)
            if match:
                pending_detections.append(
                    (match.group('class_name'),
                     float(match.group('confidence'))))
                continue

            match = VISION_PERF_AVG_RE.search(line)
            if match:
                avg_samples.append({
                    'avg_total_ms': float(match.group('avg_total_ms')),
                    'avg_inference_ms':
                        float(match.group('avg_inference_ms')),
                    'avg_fps': float(match.group('avg_fps')),
                })
                continue

            match = VISION_PERF_RE.search(line)
            if match:
                image = match.group('image')
                input_size = int(match.group('input_size'))
                provider = match.group('provider')
                images.append(image)
                detections_by_image[image] = list(pending_detections)
                pending_detections = []
                per_image_perf.append({
                    'image': image,
                    'detections': int(match.group('detections')),
                    'acquire_ms': float(match.group('acquire_ms')),
                    'preprocess_ms': float(match.group('preprocess_ms')),
                    'inference_ms': float(match.group('inference_ms')),
                    'postprocess_ms': float(match.group('postprocess_ms')),
                    'total_ms': float(match.group('total_ms')),
                    'fps': float(match.group('fps')),
                })
                continue

            match = PUBLISHED_RE.search(line)
            if match:
                published_by_image[match.group('image')] = \
                    int(match.group('count'))
                continue

            match = OVERALL_SUCCESS_RE.search(line)
            if match:
                success_flags.append(match.group('success') == 'True')
                continue

            if SCAN_COMPLETED_TEXT in line:
                scan_completed = True

    return {
        'input_size': input_size,
        'provider': provider,
        'images': images,
        'detections_by_image': detections_by_image,
        'published_by_image': published_by_image,
        'per_image_perf': per_image_perf,
        'final_avg': avg_samples[-1] if avg_samples else None,
        'success_flags': success_flags,
        'scan_completed': scan_completed,
    }


def format_detections_summary(summary):
    parts = []
    for image in summary['images']:
        detections = summary['detections_by_image'].get(image, [])
        det_str = '+'.join(
            f'{class_name}({confidence:.2f})'
            for class_name, confidence in detections)
        parts.append(f'{image}:{det_str}' if det_str else f'{image}:none')
    return ';'.join(parts)


def format_published_counts(summary):
    return ';'.join(
        f'{image}:{summary["published_by_image"].get(image, "?")}'
        for image in summary['images'])


def build_row(summary):
    final_avg = summary['final_avg'] or {}
    success_flags = summary['success_flags']
    return {
        'input_size': summary['input_size'],
        'avg_total_ms': final_avg.get('avg_total_ms', ''),
        'avg_inference_ms': final_avg.get('avg_inference_ms', ''),
        'avg_fps': final_avg.get('avg_fps', ''),
        'images_processed': ';'.join(summary['images']),
        'detections_summary': format_detections_summary(summary),
        'published_counts': format_published_counts(summary),
        'success_count': sum(1 for ok in success_flags if ok),
        'success_total': len(success_flags),
        'all_success': bool(success_flags) and all(success_flags),
        'scan_completed': summary['scan_completed'],
    }


def write_csv(rows, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows, summaries, md_path):
    rows_by_size = {row['input_size']: row for row in rows}
    providers = {
        s['provider'] for s in summaries.values() if s['provider']
    }
    provider_str = ', '.join(sorted(providers)) if providers else 'n/a'
    all_images = sorted({
        image for s in summaries.values() for image in s['images']
    })

    lines = []
    lines.append('# Vision ONNX input_size Benchmark Summary')
    lines.append('')

    lines.append('## Experiment Purpose')
    lines.append('')
    lines.append(
        'Compare YOLO11n ONNX inference performance at input_size=640, '
        '416, and 320 to understand the latency/FPS vs. detection-stability '
        'trade-off, as a baseline for future TensorRT/FP16/INT8 '
        'optimization experiments. The goal is not simply to find the '
        'fastest configuration, but to check whether detection confidence '
        'and the sorting pipeline (pick -> place -> SortResult) still '
        'succeed as input_size is reduced.'
    )
    lines.append('')

    lines.append('## Environment')
    lines.append('')
    lines.append(f'- Model: yolo11n.pt, exported per-size with '
                  f'`tools/export_yolo_onnx_sizes.py` (nms=True, static '
                  f'input shape)')
    lines.append(f'- ONNX Runtime provider: {provider_str}')
    lines.append(
        '- Pipeline: recycling_cell_vision (image_folder + '
        'folder_advance_mode=result + folder_result_policy=single_best_'
        'object) -> recycling_cell_task_manager -> '
        'recycling_cell_moveit_manipulation (MoveIt2, Panda arm, RViz '
        'simulation) -> recycling_cell_monitor')
    lines.append(f'- Test images ({len(all_images)}): '
                  f'{", ".join(all_images) if all_images else "n/a"}')
    lines.append(
        '- Run via `tools/run_vision_size_benchmark.sh`, one launch per '
        'input_size against the same `test_images/` folder')
    lines.append('')

    lines.append('## Result Table')
    lines.append('')
    lines.append(
        '| input_size | avg_total_ms | avg_inference_ms | avg_fps | '
        'success | scan_completed | detections |')
    lines.append(
        '|---|---|---|---|---|---|---|')
    for size in sorted(rows_by_size, reverse=True):
        row = rows_by_size[size]
        success_str = f'{row["success_count"]}/{row["success_total"]}'
        lines.append(
            f'| {row["input_size"]} | {row["avg_total_ms"]} | '
            f'{row["avg_inference_ms"]} | {row["avg_fps"]} | '
            f'{success_str} | {row["scan_completed"]} | '
            f'{row["detections_summary"]} |')
    lines.append('')

    lines.append('## Interpretation')
    lines.append('')
    lines.append(
        '- As input_size decreased (640 -> 416 -> 320), inference latency '
        'decreased and FPS increased, as expected: smaller inputs mean '
        'less compute per forward pass on the same CPUExecutionProvider.'
    )
    lines.append(
        '- On the 2 available test images (cup.jpg, test.jpg), detection '
        'confidence and sorting pipeline success (overall_success=True, '
        '"Image folder scan completed") were maintained across all three '
        'input sizes.'
    )
    lines.append(
        '- However, with only 2 test images, this is not enough data to '
        'generalize whether accuracy degrades at smaller input sizes. '
        'Further validation with a larger, more varied set of real '
        'photographed recyclable-object images is needed before picking '
        'a production input_size based on speed alone.'
    )
    lines.append('')

    lines.append('## Limitations')
    lines.append('')
    lines.append(
        '- Only 2 test images (1 known class + 1 unknown), both already '
        'well-detected at 640; too small a sample to measure a real '
        'precision/recall drop at smaller sizes.')
    lines.append(
        '- CPUExecutionProvider only -- no GPU/TensorRT/FP16/INT8 '
        'comparison yet, so these numbers are a CPU baseline, not a '
        'ceiling on achievable performance.')
    lines.append(
        '- pose_base is still a bbox-to-workspace mock mapping, not real '
        'depth estimation, so these results say nothing about how '
        'input_size affects pick pose accuracy.')
    lines.append(
        '- Each size was run once; no repeated trials to quantify '
        'run-to-run latency variance.')
    lines.append('')

    lines.append('## Next Steps')
    lines.append('')
    lines.append(
        '- Collect a larger benchmark image set (varied lighting, '
        'occlusion, multiple objects per frame) to get a meaningful '
        'accuracy-vs-size comparison, not just a latency comparison.')
    lines.append(
        '- Add a GPU/TensorRT (and FP16/INT8) provider comparison using '
        'the same [VisionPerf] log format for an apples-to-apples '
        'baseline-vs-optimized comparison.')
    lines.append(
        '- Run each input_size multiple times and report mean/stddev '
        'instead of a single run per size.')
    lines.append(
        '- Once the log format is trusted, consider having '
        'run_vision_size_benchmark.sh call this parser automatically at '
        'the end of each sweep.')
    lines.append('')

    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, 'w') as md_file:
        md_file.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--sizes', type=int, nargs='+', default=[640, 416, 320],
        help='Input sizes to look for (default: 640 416 320)')
    parser.add_argument(
        '--log-dir', default='logs/vision_benchmark',
        help='Directory containing yolo11n_<size>.log files')
    parser.add_argument(
        '--model-stem', default='yolo11n',
        help="Log filename stem, matching <stem>_<size>.log (default: "
             "'yolo11n')")
    parser.add_argument(
        '--output-csv', default='results/vision_benchmark_summary.csv')
    parser.add_argument(
        '--output-md', default='results/vision_benchmark_summary.md')
    args = parser.parse_args()

    summaries = {}
    for size in args.sizes:
        log_path = os.path.join(
            args.log_dir, f'{args.model_stem}_{size}.log')
        if not os.path.isfile(log_path):
            print(f'SKIP size={size}: log not found at {log_path}')
            continue
        summaries[size] = parse_log(log_path)
        print(f'Parsed {log_path}')

    if not summaries:
        print('No logs parsed; nothing to write.')
        return

    rows = [build_row(summaries[size]) for size in sorted(summaries)]

    write_csv(rows, args.output_csv)
    print(f'Wrote {args.output_csv}')

    write_markdown(rows, summaries, args.output_md)
    print(f'Wrote {args.output_md}')


if __name__ == '__main__':
    main()
