"""
benchmark_lib.py
================
Shared engine used by benchmark_q1.py, benchmark_q2.py and onnx_inference.py.

For every model it measures (always on CPU, on the FULL 10,000-image CIFAR-10 test set)
  * top-1 accuracy, per-image predictions + confidence
  * model file size (MB) and gzip-compressed size
  * single-image latency (batch 1, 1 thread):  mean / p50 / p95
  * batch throughput (batch 256, all threads):  images / second
  * parameter count, non-zero parameters, sparsity and MACs (float models only)
"""
import os
import time

import numpy as np
import pandas as pd
import torch

from common import (CLASSES, DATA_DIR, count_macs, count_params, file_size_mb, gzip_size_mb,
                    load_cifar10, load_deployable, measure_latency)
from inference import Predictor, softmax


def model_stats(name):
    """Static complexity numbers (float models only; INT8 TorchScript has no dense weights)."""
    if name in ("ptq", "qat"):
        return {}
    m = load_deployable(name)
    total, nz = count_params(m)
    return {"params_total": total, "params_nonzero": nz, "sparsity_pct": round(100 * (1 - nz / total), 2),
            "macs_dense_M": round(count_macs(m) / 1e6, 2),
            "macs_nonzero_M": round(count_macs(m, density_aware=True) / 1e6, 2)}


def benchmark_one(name, backend, xte, yte, lat_images=300, bs=256, threads=None):
    n_threads = threads or max(1, os.cpu_count() or 1)
    # ---- accuracy + throughput: full test set, batch 256, all threads -----------------------
    p = Predictor(name, backend, threads=n_threads)
    p(xte[:bs])                                              # warm-up
    t0 = time.perf_counter()
    logits = p.predict_batches(xte, bs)
    wall = time.perf_counter() - t0
    prob = softmax(logits)
    pred = prob.argmax(1)
    acc = 100 * (pred == yte.numpy()).mean()
    # ---- single-image latency: batch 1, 1 thread ---------------------------------------------
    p1 = Predictor(name, backend, threads=1)
    sample = xte[:1]
    lat = measure_latency(p1, sample, warmup=30, iters=lat_images)
    row = {"model": name, "backend": backend, "test_accuracy_pct": round(acc, 2),
           "model_file": os.path.basename(p.path), "size_mb": round(file_size_mb(p.path), 2),
           "size_gzip_mb": round(gzip_size_mb(p.path), 2),
           "latency_bs1_mean_ms": round(lat["mean"], 3), "latency_bs1_p50_ms": round(lat["p50"], 3),
           "latency_bs1_p95_ms": round(lat["p95"], 3),
           "throughput_bs256_img_per_s": round(len(xte) / wall, 1), "threads_throughput": n_threads}
    if backend == "torch":
        row.update(model_stats(name))
    return row, pred, prob.max(1)


def run_benchmark(names, backend, out_prefix, res_dir, data_dir=DATA_DIR, lat_images=300):
    """Benchmark `names`; write <out_prefix>_benchmark.csv and <out_prefix>_predictions.csv."""
    xte, yte = load_cifar10(data_dir, False)
    rows, preds = [], {}
    for name in names:
        print(f"benchmarking {name} ({backend}) ...", flush=True)
        row, pred, conf = benchmark_one(name, backend, xte, yte, lat_images)
        rows.append(row)
        preds[name] = (pred, conf)
        print(f"   acc {row['test_accuracy_pct']:.2f}% | size {row['size_mb']} MB | "
              f"lat {row['latency_bs1_mean_ms']} ms | thr {row['throughput_bs256_img_per_s']} img/s", flush=True)
    df = pd.DataFrame(rows)
    base = df.iloc[0]
    df["acc_delta_vs_first_pts"] = (df["test_accuracy_pct"] - base["test_accuracy_pct"]).round(2)
    df["size_reduction_x"] = (base["size_mb"] / df["size_mb"]).round(2)
    df["gzip_size_reduction_x"] = (base["size_gzip_mb"] / df["size_gzip_mb"]).round(2)
    df["latency_speedup_x"] = (base["latency_bs1_mean_ms"] / df["latency_bs1_mean_ms"]).round(2)
    df["throughput_speedup_x"] = (df["throughput_bs256_img_per_s"] / base["throughput_bs256_img_per_s"]).round(2)
    first = names[0]
    df["agreement_with_first_pct"] = [round(100 * (preds[n][0] == preds[first][0]).mean(), 2) for n in names]
    df.to_csv(os.path.join(res_dir, f"{out_prefix}_benchmark.csv"), index=False)

    # per-image predictions of all models  (wide format)
    pdf = pd.DataFrame({"image_index": np.arange(len(yte)), "true_label": yte.numpy(),
                        "true_class": [CLASSES[i] for i in yte.numpy()]})
    for n in names:
        pr, cf = preds[n]
        pdf[f"{n}_pred"] = pr
        pdf[f"{n}_class"] = [CLASSES[i] for i in pr]
        pdf[f"{n}_confidence"] = np.round(cf, 4)
        pdf[f"{n}_correct"] = (pr == yte.numpy()).astype(int)
    pdf.to_csv(os.path.join(res_dir, f"{out_prefix}_predictions.csv"), index=False)
    print(f"\nWrote {out_prefix}_benchmark.csv and {out_prefix}_predictions.csv in {res_dir}")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.drop(columns=["model_file", "threads_throughput"]).to_string(index=False))
    return df
