"""
benchmark_q1.py
===============
Q1.5  -  Automatically evaluates FP32, PTQ-INT8 and QAT-INT8 on the CIFAR-10 test set.

Output (results/):
    q1_benchmark.csv    accuracy, size, latency, throughput, speed-ups per model
    q1_predictions.csv  per-image test predictions / confidence / correctness of every model

Usage:  python src/benchmark_q1.py [--backend torch|onnx]
"""
import argparse

from common import DATA_DIR, RES_DIR
from benchmark_lib import run_benchmark

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx"])
    ap.add_argument("--data-dir", default=DATA_DIR)
    a = ap.parse_args()
    run_benchmark(["fp32", "ptq", "qat"], a.backend, "q1" if a.backend == "torch" else "q1_onnx",
                  RES_DIR, a.data_dir)
