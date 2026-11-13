"""
benchmark_q2.py
===============
Q2.6  -  Automatically evaluates the original, unstructured-pruned and structured-pruned
ResNet-18 on the CIFAR-10 test set (accuracy, parameters, sparsity, MACs, size, latency,
throughput).

Output (results/):  q2_benchmark.csv, q2_predictions.csv

Usage:  python src/benchmark_q2.py [--backend torch|onnx]
"""
import argparse

from common import DATA_DIR, RES_DIR
from benchmark_lib import run_benchmark

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx"])
    ap.add_argument("--data-dir", default=DATA_DIR)
    a = ap.parse_args()
    run_benchmark(["fp32", "unstructured", "structured"], a.backend,
                  "q2" if a.backend == "torch" else "q2_onnx", RES_DIR, a.data_dir)
