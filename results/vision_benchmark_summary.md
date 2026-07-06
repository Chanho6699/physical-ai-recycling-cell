# Vision ONNX input_size Benchmark Summary

## Experiment Purpose

Compare YOLO11n ONNX inference performance at input_size=640, 416, and 320 to understand the latency/FPS vs. detection-stability trade-off, as a baseline for future TensorRT/FP16/INT8 optimization experiments. The goal is not simply to find the fastest configuration, but to check whether detection confidence and the sorting pipeline (pick -> place -> SortResult) still succeed as input_size is reduced.

## Environment

- Model: yolo11n.pt, exported per-size with `tools/export_yolo_onnx_sizes.py` (nms=True, static input shape)
- ONNX Runtime provider: CPUExecutionProvider
- Pipeline: recycling_cell_vision (image_folder + folder_advance_mode=result + folder_result_policy=single_best_object) -> recycling_cell_task_manager -> recycling_cell_moveit_manipulation (MoveIt2, Panda arm, RViz simulation) -> recycling_cell_monitor
- Test images (2): cup.jpg, test.jpg
- Run via `tools/run_vision_size_benchmark.sh`, one launch per input_size against the same `test_images/` folder

## Result Table

| input_size | avg_total_ms | avg_inference_ms | avg_fps | success | scan_completed | detections |
|---|---|---|---|---|---|---|
| 640 | 59.5 | 46.8 | 16.8 | 2/2 | True | cup.jpg:paper_cup(0.86);test.jpg:unknown(0.87)+unknown(0.60) |
| 416 | 29.7 | 25.6 | 37.3 | 2/2 | True | cup.jpg:paper_cup(0.95);test.jpg:unknown(0.95)+unknown(0.63) |
| 320 | 22.6 | 15.0 | 54.3 | 2/2 | True | cup.jpg:paper_cup(0.94);test.jpg:unknown(0.92) |

## Interpretation

- As input_size decreased (640 -> 416 -> 320), inference latency decreased and FPS increased, as expected: smaller inputs mean less compute per forward pass on the same CPUExecutionProvider.
- On the 2 available test images (cup.jpg, test.jpg), detection confidence and sorting pipeline success (overall_success=True, "Image folder scan completed") were maintained across all three input sizes.
- However, with only 2 test images, this is not enough data to generalize whether accuracy degrades at smaller input sizes. Further validation with a larger, more varied set of real photographed recyclable-object images is needed before picking a production input_size based on speed alone.

## Limitations

- Only 2 test images (1 known class + 1 unknown), both already well-detected at 640; too small a sample to measure a real precision/recall drop at smaller sizes.
- CPUExecutionProvider only -- no GPU/TensorRT/FP16/INT8 comparison yet, so these numbers are a CPU baseline, not a ceiling on achievable performance.
- pose_base is still a bbox-to-workspace mock mapping, not real depth estimation, so these results say nothing about how input_size affects pick pose accuracy.
- Each size was run once; no repeated trials to quantify run-to-run latency variance.

## Next Steps

- Collect a larger benchmark image set (varied lighting, occlusion, multiple objects per frame) to get a meaningful accuracy-vs-size comparison, not just a latency comparison.
- Add a GPU/TensorRT (and FP16/INT8) provider comparison using the same [VisionPerf] log format for an apples-to-apples baseline-vs-optimized comparison.
- Run each input_size multiple times and report mean/stddev instead of a single run per size.
- Once the log format is trusted, consider having run_vision_size_benchmark.sh call this parser automatically at the end of each sweep.
