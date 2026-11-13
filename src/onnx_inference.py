"""
onnx_inference.py
=================
Deliverable 8  -  CPU inference / benchmark of ALL exported ONNX models with ONNX Runtime.

Output: results/onnx_benchmark.csv (+ onnx_predictions.csv)
Usage:  python src/onnx_inference.py
        python src/inference.py --model qat --backend onnx --num-images 8   # single-model demo
"""
import argparse
import os

from common import DATA_DIR, ONNX_DIR, ONNX_FILES, RES_DIR
from benchmark_lib import run_benchmark

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    a = ap.parse_args()
    names = [n for n, f in ONNX_FILES.items() if os.path.exists(os.path.join(ONNX_DIR, f))]
    # keep fp32 first so deltas/speed-ups are relative to the FP32 ONNX model
    names.sort(key=lambda n: ["fp32", "ptq", "qat", "unstructured", "structured"].index(n))
    run_benchmark(names, "onnx", "onnx", RES_DIR, a.data_dir)
