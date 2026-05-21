"""
export_onnx.py — Export PyTorch YOLOv8 weights (best.pt) to ONNX for WebGPU / WASM SIMD.

Usage:
    python export_onnx.py
"""

import os
from pathlib import Path
from ultralytics import YOLO

def export_model():
    current_dir = Path(__file__).resolve().parent
    model_path = current_dir / "models" / "best.pt"
    output_onnx = current_dir / "models" / "best.onnx"

    if not model_path.exists():
        raise FileNotFoundError(f"Source model not found at {model_path}")

    print(f"[EXPORT] Loading PyTorch model from {model_path}...")
    model = YOLO(str(model_path))

    print("[EXPORT] Exporting to ONNX format (imgsz=640, opset=18, dynamic=False)...")
    exported_path = model.export(
        format="onnx",
        imgsz=640,
        opset=18,
        dynamic=False,
        simplify=True
    )

    print(f"[EXPORT] Successfully generated ONNX model: {exported_path}")
    return exported_path

if __name__ == "__main__":
    export_model()
