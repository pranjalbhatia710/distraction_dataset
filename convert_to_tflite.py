#!/usr/bin/env python3
"""
Convert the PyTorch distraction model to TFLite and generate a C header
for Arduino deployment.

Usage:
  pip install torch torchvision onnx onnx-tf tensorflow
  python convert_to_tflite.py
  python convert_to_tflite.py --model best_distraction_model_v2.pth --quantize
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path


def load_pytorch_model(path):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 3)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def export_to_onnx(model, onnx_path):
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
        opset_version=13,
    )
    print(f"  ONNX exported: {onnx_path}")


def convert_onnx_to_tflite(onnx_path, tflite_path, quantize=False):
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf

    # ONNX → TF SavedModel
    onnx_model = onnx.load(str(onnx_path))
    tf_rep = prepare(onnx_model)
    saved_model_dir = str(onnx_path.parent / "tf_saved_model")
    tf_rep.export_graph(saved_model_dir)
    print(f"  TF SavedModel: {saved_model_dir}")

    # TF SavedModel → TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if quantize:
        print("  Applying int8 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(1, 224, 224, 3).astype(np.float32)
                yield [data]

        converter.representative_dataset = representative_dataset
    else:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"  TFLite model: {tflite_path} ({size_kb:.1f} KB)")
    return tflite_model


def generate_c_header(tflite_path, header_path):
    with open(tflite_path, "rb") as f:
        data = f.read()

    with open(header_path, "w") as f:
        f.write("// Auto-generated — do not edit\n")
        f.write(f"// Source: {tflite_path.name}\n")
        f.write(f"// Size: {len(data)} bytes\n\n")
        f.write("#ifndef MODEL_DATA_H\n")
        f.write("#define MODEL_DATA_H\n\n")
        f.write("#include <cstdint>\n\n")
        f.write(f"alignas(8) const unsigned char g_distraction_model[] = {{\n")

        for i in range(0, len(data), 12):
            chunk = data[i:i+12]
            hex_vals = ", ".join(f"0x{b:02x}" for b in chunk)
            f.write(f"    {hex_vals},\n")

        f.write("};\n\n")
        f.write(f"const unsigned int g_distraction_model_len = {len(data)};\n\n")
        f.write("#endif // MODEL_DATA_H\n")

    print(f"  C header: {header_path}")


def main():
    p = argparse.ArgumentParser(description="Convert PyTorch model to TFLite + C header")
    p.add_argument("--model", default="best_distraction_model.pth", help="Input .pth file")
    p.add_argument("--quantize", action="store_true", help="Apply int8 quantization (smaller, faster)")
    p.add_argument("--output-dir", default=".", help="Output directory")
    args = p.parse_args()

    out = Path(args.output_dir)
    pth_path = Path(args.model)
    onnx_path = out / "distraction_model.onnx"
    tflite_path = out / "distraction_model.tflite"
    header_path = out / "arduino" / "distraction_detector" / "model_data.h"

    print(f"\n  Converting: {pth_path}")
    print(f"  Quantize:   {args.quantize}\n")

    print("[1/4] Loading PyTorch model...")
    model = load_pytorch_model(pth_path)

    print("[2/4] Exporting to ONNX...")
    export_to_onnx(model, onnx_path)

    print("[3/4] Converting to TFLite...")
    convert_onnx_to_tflite(onnx_path, tflite_path, args.quantize)

    print("[4/4] Generating C header...")
    header_path.parent.mkdir(parents=True, exist_ok=True)
    generate_c_header(tflite_path, header_path)

    print(f"\nDone! Copy the 'arduino/distraction_detector/' folder to your Arduino sketches.")


if __name__ == "__main__":
    main()
