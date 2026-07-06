#!/usr/bin/env python3
"""Export a YOLO .pt checkpoint to per-input-size ONNX models.

Each exported model keeps NMS baked in (nms=True), so its output stays
shaped (1, N, 6) -- the same post-NMS format
recycling_cell_vision's vision_perception_node.postprocess_yolo() already
decodes. This matches the input shape check done before writing this
script: models/yolo11n.onnx has a static input shape [1, 3, 640, 640], so
comparing input sizes means exporting one ONNX file per size rather than
resizing a single dynamic model at runtime.

Export runs in an isolated temp directory (Ultralytics names its output
after the weights file, e.g. "yolo11n.onnx", regardless of imgsz) so this
never overwrites the existing models/yolo11n.onnx.

Usage:
    python3 tools/export_yolo_onnx_sizes.py
    python3 tools/export_yolo_onnx_sizes.py --weights models/yolo11n.pt \\
        --sizes 640 416 320 --output-dir models
"""
import argparse
import os
import shutil
import sys
import tempfile

from ultralytics import YOLO


def export_one(weights_path, size, output_dir):
    stem = os.path.splitext(os.path.basename(weights_path))[0]
    target_path = os.path.join(output_dir, f'{stem}_{size}.onnx')

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_weights = os.path.join(tmp_dir, os.path.basename(weights_path))
        shutil.copy(weights_path, tmp_weights)

        model = YOLO(tmp_weights)
        exported_path = model.export(
            format='onnx', imgsz=size, nms=True, dynamic=False)
        exported_path = os.path.join(tmp_dir, os.path.basename(exported_path))

        shutil.move(exported_path, target_path)

    return target_path


def print_model_shapes(onnx_path):
    import onnxruntime as ort

    session = ort.InferenceSession(
        onnx_path, providers=['CPUExecutionProvider'])
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    print(f'  input:  name={input_info.name} shape={input_info.shape}')
    print(f'  output: name={output_info.name} shape={output_info.shape}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--weights', default='models/yolo11n.pt',
        help='Source .pt checkpoint (default: models/yolo11n.pt)')
    parser.add_argument(
        '--sizes', type=int, nargs='+', default=[640, 416, 320],
        help='Square input sizes to export (default: 640 416 320)')
    parser.add_argument(
        '--output-dir', default='models',
        help='Directory to write <stem>_<size>.onnx into (default: models)')
    args = parser.parse_args()

    if not os.path.isfile(args.weights):
        print(f'ERROR: weights file not found: {args.weights}',
              file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    for size in args.sizes:
        print(f'== Exporting {os.path.basename(args.weights)} at '
              f'imgsz={size} ==')
        target_path = export_one(args.weights, size, args.output_dir)
        print(f'  saved to {target_path}')
        print_model_shapes(target_path)
        print()

    print('Done.')


if __name__ == '__main__':
    main()
