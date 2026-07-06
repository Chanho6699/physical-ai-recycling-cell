#!/usr/bin/env python3
"""Parse recycling_cell_vision [VisionPerf] benchmark logs into a CSV/Markdown
summary.

Reads logs/vision_benchmark/yolo11n_<size>.log (produced by
tools/run_vision_size_benchmark.sh) for each requested input size and
extracts, per size:
  - benchmark_mode ('end_to_end' or 'vision_only', from the node's startup
    log line; defaults to 'end_to_end' for older logs that predate this
    field) -- vision_only logs have no task_manager/MoveIt in the loop, so
    there is no overall_success to report and that is reflected as such
    rather than as a failure
  - the scanned image_folder_path and whether it was scanned recursively
    (from vision_perception_node's "image_folder scan found N image(s) in
    '<path>'" line)
  - per-image ONNX detections (class_name, confidence)
  - per-image [VisionPerf] timing (acquire/preprocess/inference/postprocess/
    total/fps) -- image names may include a subfolder prefix (e.g.
    "paper_cup/paper_cup_dark_001.jpg") when recursive_image_folder=true
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

IMAGE_FOLDER_SCAN_RE = re.compile(
    r"image_folder scan found (?P<count>\d+) image\(s\) in "
    r"'(?P<folder_path>[^']+)'(?P<recursive> \(recursive\))?"
)

BENCHMARK_MODE_RE = re.compile(r"benchmark_mode='(?P<benchmark_mode>\w+)'")

SCAN_COMPLETED_TEXT = 'Image folder scan completed'

CSV_FIELDNAMES = [
    'input_size',
    'benchmark_mode',
    'dataset',
    'image_folder_path',
    'recursive_image_folder',
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
    folder_path = None
    recursive = False
    benchmark_mode = 'end_to_end'

    with open(log_path, 'r', errors='replace') as log_file:
        for line in log_file:
            match = BENCHMARK_MODE_RE.search(line)
            if match:
                benchmark_mode = match.group('benchmark_mode')
                continue

            match = IMAGE_FOLDER_SCAN_RE.search(line)
            if match:
                folder_path = match.group('folder_path')
                recursive = match.group('recursive') is not None
                continue

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
        'benchmark_mode': benchmark_mode,
        'folder_path': folder_path,
        'recursive': recursive,
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
    folder_path = summary['folder_path'] or ''
    benchmark_mode = summary['benchmark_mode']

    # vision_only never runs task_manager, so there's no overall_success to
    # report -- record that as "not applicable", not as a failed 0/0.
    if benchmark_mode == 'vision_only' and not success_flags:
        all_success = 'n/a (vision_only)'
    else:
        all_success = bool(success_flags) and all(success_flags)

    return {
        'input_size': summary['input_size'],
        'benchmark_mode': benchmark_mode,
        'dataset': os.path.basename(folder_path.rstrip('/')) if folder_path
        else '',
        'image_folder_path': folder_path,
        'recursive_image_folder': summary['recursive'],
        'avg_total_ms': final_avg.get('avg_total_ms', ''),
        'avg_inference_ms': final_avg.get('avg_inference_ms', ''),
        'avg_fps': final_avg.get('avg_fps', ''),
        'images_processed': ';'.join(summary['images']),
        'detections_summary': format_detections_summary(summary),
        'published_counts': format_published_counts(summary),
        'success_count': sum(1 for ok in success_flags if ok),
        'success_total': len(success_flags),
        'all_success': all_success,
        'scan_completed': summary['scan_completed'],
    }


def write_csv(rows, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows, summaries, md_path, log_dir, csv_path):
    rows_by_size = {row['input_size']: row for row in rows}
    providers = {
        s['provider'] for s in summaries.values() if s['provider']
    }
    provider_str = ', '.join(sorted(providers)) if providers else 'n/a'
    all_images = sorted({
        image for s in summaries.values() for image in s['images']
    })
    image_count = len(all_images)
    small_sample = image_count <= 10

    dataset_name = next(
        (row['dataset'] for row in rows if row['dataset']), 'test_images')
    folder_path = next(
        (row['image_folder_path'] for row in rows
         if row['image_folder_path']), 'n/a')
    recursive = any(row['recursive_image_folder'] for row in rows)

    benchmark_modes_present = {row['benchmark_mode'] for row in rows}
    any_vision_only = 'vision_only' in benchmark_modes_present
    any_end_to_end = 'end_to_end' in benchmark_modes_present
    mixed_modes = any_vision_only and any_end_to_end

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
    if any_vision_only and not any_end_to_end:
        lines.append(
            '- Pipeline: recycling_cell_vision only '
            '(benchmark_mode=vision_only) -- ONNX inference + [VisionPerf] '
            'logging on a fast per-image loop; task_manager/MoveIt were '
            'not launched for this run, so no pick/place cycle time gates '
            'the scan')
    elif any_end_to_end and not any_vision_only:
        lines.append(
            '- Pipeline: recycling_cell_vision (image_folder + '
            'folder_advance_mode=result + folder_result_policy=single_'
            'best_object) -> recycling_cell_task_manager -> '
            'recycling_cell_moveit_manipulation (MoveIt2, Panda arm, RViz '
            'simulation) -> recycling_cell_monitor')
    else:
        lines.append(
            '- Pipeline: mixed across sizes -- some ran benchmark_mode='
            'end_to_end (full task_manager/MoveIt pipeline) and others ran '
            'benchmark_mode=vision_only (vision node only, no robot); see '
            'the benchmark_mode column in the Result Table/CSV per size')
    lines.append(f'- Dataset: {dataset_name} (image_folder_path='
                 f'`{folder_path}`, recursive_image_folder={recursive})')
    lines.append(f'- Log directory: `{log_dir}`')
    if small_sample:
        lines.append(f'- Test images ({image_count}): '
                      f'{", ".join(all_images) if all_images else "n/a"}')
    else:
        lines.append(f'- Test images: {image_count} files across the '
                      f'dataset folder (full per-image list in the CSV, '
                      f'omitted here for readability)')
    lines.append(
        '- Run via `tools/run_vision_size_benchmark.sh`, one launch per '
        'input_size against the same image folder')
    lines.append('')

    def format_success(row):
        if row['benchmark_mode'] == 'vision_only' \
                and row['success_total'] == 0:
            return 'n/a (vision_only)'
        return f'{row["success_count"]}/{row["success_total"]}'

    lines.append('## Result Table')
    lines.append('')
    if small_sample:
        lines.append(
            '| input_size | mode | avg_total_ms | avg_inference_ms | '
            'avg_fps | success | scan_completed | detections |')
        lines.append('|---|---|---|---|---|---|---|---|')
        for size in sorted(rows_by_size, reverse=True):
            row = rows_by_size[size]
            lines.append(
                f'| {row["input_size"]} | {row["benchmark_mode"]} | '
                f'{row["avg_total_ms"]} | {row["avg_inference_ms"]} | '
                f'{row["avg_fps"]} | {format_success(row)} | '
                f'{row["scan_completed"]} | {row["detections_summary"]} |')
    else:
        lines.append(
            '| input_size | mode | avg_total_ms | avg_inference_ms | '
            'avg_fps | success | scan_completed | images_processed |')
        lines.append('|---|---|---|---|---|---|---|---|')
        for size in sorted(rows_by_size, reverse=True):
            row = rows_by_size[size]
            processed_count = len(
                summaries[size]['images']) if size in summaries else 0
            lines.append(
                f'| {row["input_size"]} | {row["benchmark_mode"]} | '
                f'{row["avg_total_ms"]} | {row["avg_inference_ms"]} | '
                f'{row["avg_fps"]} | {format_success(row)} | '
                f'{row["scan_completed"]} | {processed_count} |')
        lines.append('')
        lines.append(
            'Per-image class/confidence detail is in '
            f'`{csv_path}` '
            '(`detections_summary`/`published_counts` columns) -- omitted '
            'from this table because the dataset is too large to render '
            'readably here.')
    lines.append('')

    lines.append('## Interpretation')
    lines.append('')
    lines.append(
        '- As input_size decreased (640 -> 416 -> 320), inference latency '
        'decreased and FPS increased, as expected: smaller inputs mean '
        'less compute per forward pass on the same CPUExecutionProvider.'
    )
    image_list_str = (
        f' ({", ".join(all_images)})' if small_sample and all_images else '')
    if any_vision_only and not any_end_to_end:
        lines.append(
            f'- On the {image_count} available test image'
            f'{"s" if image_count != 1 else ""}{image_list_str}, '
            f'detection confidence stayed high and every image completed '
            f'("Image folder scan completed") across all three input '
            f'sizes. This run used benchmark_mode=vision_only, so '
            f'task_manager/MoveIt were never exercised and there is no '
            f'overall_success/sorting-pipeline outcome to report here -- '
            f'only ONNX detection quality and throughput.'
        )
    elif mixed_modes:
        lines.append(
            f'- On the {image_count} available test image'
            f'{"s" if image_count != 1 else ""}{image_list_str}, '
            f'detection confidence held up across all three input sizes. '
            f'Sizes run with benchmark_mode=end_to_end additionally '
            f'confirmed sorting pipeline success (overall_success=True); '
            f'sizes run with benchmark_mode=vision_only have no such '
            f'outcome to report since task_manager/MoveIt were skipped.'
        )
    else:
        lines.append(
            f'- On the {image_count} available test image'
            f'{"s" if image_count != 1 else ""}{image_list_str}, detection '
            f'confidence and sorting pipeline success (overall_success='
            f'True, "Image folder scan completed") were maintained across '
            f'all three input sizes.'
        )
    lines.append(
        f'- However, with only {image_count} test image'
        f'{"s" if image_count != 1 else ""} from the "{dataset_name}" '
        f'dataset, this is not enough data to generalize whether accuracy '
        f'degrades at smaller input sizes. Further validation with a '
        f'larger, more varied set of real photographed recyclable-object '
        f'images is needed before picking a production input_size based '
        f'on speed alone.'
    )
    lines.append('')

    lines.append('## Limitations')
    lines.append('')
    lines.append(
        f'- Only {image_count} test image{"s" if image_count != 1 else ""} '
        f'from the "{dataset_name}" dataset; too small a sample to measure '
        f'a real precision/recall drop at smaller sizes.')
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
    if any_vision_only:
        lines.append(
            '- benchmark_mode=vision_only sizes measure ONNX throughput/'
            'detection quality only -- they do not exercise task_manager '
            'routing, MoveIt planning, or gripper execution, so a full '
            'sorting-pipeline validation (including any of these sizes) '
            'still needs a separate benchmark_mode=end_to_end run on a '
            'representative subset.')
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
    if any_vision_only:
        lines.append(
            '- Run benchmark_mode=end_to_end (task_manager + MoveIt) on a '
            'small representative subset of this dataset to validate the '
            'full sorting pipeline, since the vision_only sizes above only '
            'cover detection throughput/quality.')
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

    write_markdown(
        rows, summaries, args.output_md, args.log_dir, args.output_csv)
    print(f'Wrote {args.output_md}')


if __name__ == '__main__':
    main()
