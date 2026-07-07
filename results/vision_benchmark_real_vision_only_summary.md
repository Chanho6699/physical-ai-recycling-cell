# Vision ONNX input_size Benchmark Summary

## Experiment Purpose

Compare YOLO11n ONNX inference performance at input_size=640, 416, and 320 to understand the latency/FPS vs. detection-stability trade-off, as a baseline for future TensorRT/FP16/INT8 optimization experiments. The goal is not simply to find the fastest configuration, but to check whether detection confidence and the sorting pipeline (pick -> place -> SortResult) still succeed as input_size is reduced.

## Environment

- Model: yolo11n.pt, exported per-size with `tools/export_yolo_onnx_sizes.py` (nms=True, static input shape)
- ONNX Runtime provider: CPUExecutionProvider
- Pipeline: recycling_cell_vision only (benchmark_mode=vision_only) -- ONNX inference + [VisionPerf] logging on a fast per-image loop; task_manager/MoveIt were not launched for this run, so no pick/place cycle time gates the scan
- Dataset: test_images_real (image_folder_path=`/home/rlack/Projects/physical-ai-recycling-cell/test_images_real`, recursive_image_folder=True)
- Log directory: `logs/vision_benchmark/test_images_real/vision_only`
- Test images: 50 files across the dataset folder (full per-image list in the CSV, omitted here for readability)
- Run via `tools/run_vision_size_benchmark.sh`, one launch per input_size against the same image folder

## Result Table

| input_size | mode | avg_total_ms | avg_inference_ms | avg_fps | success | scan_completed | images_processed |
|---|---|---|---|---|---|---|---|
| 640 | vision_only | 127.6 | 29.2 | 7.9 | n/a (vision_only) | True | 50 |
| 416 | vision_only | 105.8 | 13.1 | 9.5 | n/a (vision_only) | True | 50 |
| 320 | vision_only | 95.4 | 7.9 | 10.5 | n/a (vision_only) | True | 50 |

Per-image class/confidence detail is in `results/vision_benchmark_real_vision_only_summary.csv` (`detections_summary`/`published_counts` columns) -- omitted from this table because the dataset is too large to render readably here.

## Interpretation

- As input_size decreased (640 -> 416 -> 320), inference latency decreased and FPS increased, as expected: smaller inputs mean less compute per forward pass on the same CPUExecutionProvider.
- On the 50 available test images, detection confidence stayed high and every image completed ("Image folder scan completed") across all three input sizes. This run used benchmark_mode=vision_only, so task_manager/MoveIt were never exercised and there is no overall_success/sorting-pipeline outcome to report here -- only ONNX detection quality and throughput.
- However, with only 50 test images from the "test_images_real" dataset, this is not enough data to generalize whether accuracy degrades at smaller input sizes. Further validation with a larger, more varied set of real photographed recyclable-object images is needed before picking a production input_size based on speed alone.

## Limitations

- Only 50 test images from the "test_images_real" dataset; too small a sample to measure a real precision/recall drop at smaller sizes.
- CPUExecutionProvider only -- no GPU/TensorRT/FP16/INT8 comparison yet, so these numbers are a CPU baseline, not a ceiling on achievable performance.
- pose_base is still a bbox-to-workspace mock mapping, not real depth estimation, so these results say nothing about how input_size affects pick pose accuracy.
- Each size was run once; no repeated trials to quantify run-to-run latency variance.
- benchmark_mode=vision_only sizes measure ONNX throughput/detection quality only -- they do not exercise task_manager routing, MoveIt planning, or gripper execution, so a full sorting-pipeline validation (including any of these sizes) still needs a separate benchmark_mode=end_to_end run on a representative subset.

## Next Steps

- Collect a larger benchmark image set (varied lighting, occlusion, multiple objects per frame) to get a meaningful accuracy-vs-size comparison, not just a latency comparison.
- Add a GPU/TensorRT (and FP16/INT8) provider comparison using the same [VisionPerf] log format for an apples-to-apples baseline-vs-optimized comparison.
- Run each input_size multiple times and report mean/stddev instead of a single run per size.
- Once the log format is trusted, consider having run_vision_size_benchmark.sh call this parser automatically at the end of each sweep.
- Run benchmark_mode=end_to_end (task_manager + MoveIt) on a small representative subset of this dataset to validate the full sorting pipeline, since the vision_only sizes above only cover detection throughput/quality.
