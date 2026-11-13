"""
make_report_assets.py
=====================
Builds the figures used in the technical report from the CSV / log files that the pipeline
produced (results/figures/*.png) and renders the real console logs as "screenshot" images
(results/screenshots/*.png).  Nothing here is hand-edited: every number comes from results/.
"""
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import LOG_DIR, RES_DIR

FIG = os.path.join(RES_DIR, "figures")
SHOT = os.path.join(RES_DIR, "screenshots")
os.makedirs(FIG, exist_ok=True)
os.makedirs(SHOT, exist_ok=True)
COL = {"fp32": "#4C72B0", "ptq": "#DD8452", "qat": "#55A868", "unstructured": "#C44E52",
       "structured": "#8172B3"}
LABEL = {"fp32": "FP32", "ptq": "PTQ-INT8", "qat": "QAT-INT8", "unstructured": "Unstructured\n(85% sparse)",
         "structured": "Structured\n(channels)"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def csv(name):
    p = os.path.join(RES_DIR, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def bars(df, cols, titles, fname, fmt="{:.2f}"):
    fig, axes = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 3.6))
    for ax, c, t in zip(axes, cols, titles):
        vals = df[c].astype(float).values
        ax.bar([LABEL[m] for m in df["model"]], vals, color=[COL[m] for m in df["model"]])
        for i, v in enumerate(vals):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
        ax.set_title(t, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=160)
    plt.close(fig)


def console(path, out, title, max_lines=34, width=118):
    """Render a text log as a terminal-style image."""
    if not os.path.exists(path):
        return
    lines = []
    for raw in open(path).read().splitlines():
        lines += textwrap.wrap(raw, width) or [""]
    if len(lines) > max_lines:
        lines = lines[:6] + ["   ..."] + lines[-(max_lines - 7):]
    fig = plt.figure(figsize=(11, 0.205 * len(lines) + 0.7), facecolor="#1e1e1e")
    fig.text(0.01, 1 - 0.3 / fig.get_figheight(), f"$ {title}", color="#8fbf7f", family="monospace", fontsize=8.5)
    for i, ln in enumerate(lines):
        fig.text(0.01, 1 - (0.62 + 0.2 * i) / fig.get_figheight(), ln, color="#d4d4d4",
                 family="monospace", fontsize=8)
    fig.savefig(os.path.join(SHOT, out), dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    # ---- baseline training curve ------------------------------------------------------------
    h = csv("baseline_training_history.csv")
    if h is not None:
        fig, ax = plt.subplots(1, 2, figsize=(8, 3.3))
        ax[0].plot(h.epoch, h.train_loss, "o-", label="train"); ax[0].plot(h.epoch, h.test_loss, "s-", label="test")
        ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
        ax[1].plot(h.epoch, h.train_acc, "o-", label="train"); ax[1].plot(h.epoch, h.test_acc, "s-", label="test")
        ax[1].set_title("Accuracy (%)"); ax[1].set_xlabel("epoch"); ax[1].legend()
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_baseline_training.png"), dpi=160); plt.close(fig)

    # ---- Q1 comparison ----------------------------------------------------------------------
    q1 = csv("q1_benchmark.csv")
    if q1 is not None:
        bars(q1, ["test_accuracy_pct", "size_mb", "latency_bs1_mean_ms", "throughput_bs256_img_per_s"],
             ["Test accuracy (%)", "Model size (MB)", "Latency, batch 1 (ms)", "Throughput (img/s)"],
             "fig_q1_comparison.png")
    a = csv("q1_ptq_calibration_ablation.csv")
    if a is not None:
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        ax.plot(a.calibration_images, a.int8_test_acc, "o-", color=COL["ptq"]); ax.set_xscale("log")
        ax.set_xlabel("calibration images"); ax.set_ylabel("INT8 test acc (%)"); ax.set_title("PTQ: calibration-set size")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_ptq_calibration.png"), dpi=160); plt.close(fig)
    t = csv("q1_qat_training_history.csv")
    if t is not None:
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        ax.plot(t.step, t.train_loss, "o-", color=COL["qat"]); ax.set_xlabel("step"); ax.set_ylabel("train loss")
        ax.set_title("QAT fine-tuning loss"); fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig_qat_loss.png"), dpi=160); plt.close(fig)

    # ---- Q2 comparison ----------------------------------------------------------------------
    q2 = csv("q2_benchmark.csv")
    if q2 is not None:
        d = q2.copy()
        d["nz_M"] = d["params_nonzero"] / 1e6
        bars(d, ["test_accuracy_pct", "nz_M", "macs_nonzero_M", "latency_bs1_mean_ms"],
             ["Test accuracy (%)", "Non-zero params (M)", "Effective MACs (M)", "Latency, batch 1 (ms)"],
             "fig_q2_comparison.png")
    it = csv("q2_unstructured_iterations.csv")
    if it is not None:
        it = it[it.iteration != "final_ft"]
        fig, ax = plt.subplots(figsize=(5, 3.3))
        ax.plot(it.actual_sparsity * 100, it.acc_after_prune, "o--", label="right after pruning", color="#999")
        ax.plot(it.actual_sparsity * 100, it.acc_after_finetune, "s-", label="after fine-tuning", color=COL["unstructured"])
        ax.set_xlabel("global sparsity (%)"); ax.set_ylabel("test acc (%)"); ax.legend(); ax.set_title("Iterative pruning")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_unstructured_iterations.png"), dpi=160); plt.close(fig)
    ls = csv("q2_unstructured_layer_sparsity.csv")
    if ls is not None:
        fig, ax = plt.subplots(figsize=(8.5, 3.4))
        ax.bar(range(len(ls)), ls.sparsity_pct, color=COL["unstructured"])
        ax.set_xticks(range(len(ls))); ax.set_xticklabels(ls.layer, rotation=90, fontsize=7)
        ax.set_ylabel("sparsity (%)"); ax.set_title("Per-layer sparsity after global magnitude pruning")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_layer_sparsity.png"), dpi=160); plt.close(fig)
    sc = os.path.join(RES_DIR, "q2_structured_channels.csv")
    if os.path.exists(sc):
        d = pd.read_csv(sc, nrows=8)
        fig, ax = plt.subplots(figsize=(6.5, 3.3))
        x = range(len(d))
        ax.bar([i - 0.2 for i in x], d.orig_hidden_channels, 0.4, label="original", color="#bbb")
        ax.bar([i + 0.2 for i in x], d.kept_hidden_channels, 0.4, label="kept", color=COL["structured"])
        ax.set_xticks(list(x)); ax.set_xticklabels(d.block, rotation=45, fontsize=8)
        ax.set_ylabel("hidden channels"); ax.legend(); ax.set_title("Structured pruning: channels per block")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_structured_channels.png"), dpi=160); plt.close(fig)

    # ---- ONNX -------------------------------------------------------------------------------
    ox = csv("onnx_benchmark.csv")
    if ox is not None:
        bars(ox, ["test_accuracy_pct", "size_mb", "latency_bs1_mean_ms", "throughput_bs256_img_per_s"],
             ["ONNX: test acc (%)", "ONNX: size (MB)", "ONNX: latency b=1 (ms)", "ONNX: throughput (img/s)"],
             "fig_onnx_comparison.png")

    # compact, readable benchmark tables for the screenshots
    short = {"test_accuracy_pct": "acc%", "size_mb": "MB", "size_gzip_mb": "gzMB", "latency_bs1_mean_ms": "lat_ms",
             "throughput_bs256_img_per_s": "img/s", "params_nonzero": "nz_params", "macs_nonzero_M": "MACs_M",
             "sparsity_pct": "sparse%", "latency_speedup_x": "lat_x", "throughput_speedup_x": "thr_x"}
    for csvname, out, cmd in (("q1_benchmark.csv", "q1_benchmark_compact.txt", "python src/benchmark_q1.py"),
                              ("q2_benchmark.csv", "q2_benchmark_compact.txt", "python src/benchmark_q2.py"),
                              ("onnx_benchmark.csv", "onnx_benchmark_compact.txt", "python src/onnx_inference.py")):
        d = csv(csvname)
        if d is not None:
            keep = ["model"] + [c for c in short if c in d.columns]
            open(os.path.join(LOG_DIR, out), "w").write(d[keep].rename(columns=short).to_string(index=False))

    # ---- console "screenshots" (rendered from the real logs) ---------------------------------
    L = lambda n: os.path.join(LOG_DIR, n)
    console(L("q1_q2_train_baseline.log"), "shot_baseline_training.png", "python src/train_baseline.py --epochs 8")
    console(L("q1_ptq.log"), "shot_ptq.png", "python src/ptq.py --calib-size 1000 --ablation")
    console(L("q1_qat.log"), "shot_qat.png", "python src/qat.py --epochs 2", max_lines=30)
    console(L("q2_prune_unstructured.log"), "shot_prune_unstructured.png",
            "python src/prune_unstructured.py --sparsity 0.85 --iterations 5", max_lines=40)
    console(L("q2_prune_structured.log"), "shot_prune_structured.png",
            "python src/prune_structured.py --ratio 0.5 --ft-epochs 3")
    console(L("q1_benchmark_compact.txt"), "shot_benchmark_q1.png", "python src/benchmark_q1.py")
    console(L("q2_benchmark_compact.txt"), "shot_benchmark_q2.png", "python src/benchmark_q2.py")
    console(L("onnx_benchmark_compact.txt"), "shot_onnx_benchmark.png", "python src/onnx_inference.py")
    console(L("inference_demo.txt"), "shot_inference_demo.png", "python src/inference.py --model all --num-images 5",
            max_lines=48)
    print("figures ->", FIG, "\nscreenshots ->", SHOT)


if __name__ == "__main__":
    main()
