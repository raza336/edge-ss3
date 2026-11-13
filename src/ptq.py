"""
ptq.py
======
Q1.2  -  Post-Training STATIC Quantization (PTQ) of the FP32 ResNet-18.

Pipeline
  1. load trained FP32 checkpoint
  2. fuse Conv+BN(+ReLU) blocks                (fuse_model)
  3. configure the quantization back-end       (fbgemm on x86 / qnnpack on ARM)
  4. attach qconfig, insert observers          (torch.ao.quantization.prepare)
  5. CALIBRATE on a subset of TRAIN data       (observers record activation ranges)
  6. convert to a real INT8 model              (torch.ao.quantization.convert)
  7. save + re-load + verify the deployable (TorchScript) INT8 model

Usage:  python src/ptq.py --calib-size 1000            [--ablation]
"""
import argparse
import copy
import csv
import os

import torch
from torch.ao.quantization import convert, get_default_qconfig, prepare

from common import (CKPT_DIR, DATA_DIR, LOG_DIR, MODEL_FILES, RES_DIR, Logger,
                    calibration_subset, evaluate, file_size_mb, fuse_model, load_cifar10,
                    load_fp32_checkpoint, normalize, select_quant_backend, set_seed)


def calibrate(prepared, x_calib_u8, batch_size=100):
    """Calibration pipeline: forward-only passes that let observers record min/max
    (histogram) statistics of every activation tensor. No gradients, no weight updates."""
    prepared.eval()
    with torch.no_grad():
        for i in range(0, x_calib_u8.shape[0], batch_size):
            prepared(normalize(x_calib_u8[i:i + batch_size]))


def quantize_ptq(fp32_model, x_calib_u8, backend):
    """FP32 model -> INT8 model. Returns the converted model (input is not modified)."""
    model = copy.deepcopy(fp32_model).eval()
    fuse_model(model, qat=False)                              # step 2
    model.qconfig = get_default_qconfig(backend)              # step 3/4 (per-channel weights,
    prepare(model, inplace=True)                              #  histogram act. observers)
    calibrate(model, x_calib_u8)                              # step 5
    return convert(model, inplace=False)                      # step 6


def save_deployable(int8_model, path):
    """Serialise as TorchScript so the file can be executed without the class definition."""
    example = normalize(torch.zeros(1, 3, 32, 32, dtype=torch.uint8))
    scripted = torch.jit.trace(int8_model, example)
    torch.jit.save(scripted, path)
    return scripted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-size", type=int, default=1000, help="# train images for calibration")
    ap.add_argument("--fp32", default=os.path.join(CKPT_DIR, MODEL_FILES["fp32"]))
    ap.add_argument("--out", default=os.path.join(CKPT_DIR, MODEL_FILES["ptq"]))
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--ablation", action="store_true",
                    help="also sweep the calibration-set size (report table)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    log = Logger(os.path.join(LOG_DIR, "q1_ptq.log"))
    backend = select_quant_backend()
    log(f"=== PTQ | backend={backend} | {vars(args)}")

    xtr, ytr = load_cifar10(args.data_dir, True)
    xte, yte = load_cifar10(args.data_dir, False)
    fp32, ck = load_fp32_checkpoint(args.fp32)
    acc_fp32, _ = evaluate(fp32, xte, yte)
    log(f"FP32 test accuracy: {acc_fp32:.2f}%   ({file_size_mb(args.fp32):.1f} MB)")

    xc, _ = calibration_subset(xtr, ytr, args.calib_size)
    log(f"Calibrating with {xc.shape[0]} training images ...")
    int8 = quantize_ptq(fp32, xc, backend)
    acc, _ = evaluate(int8, xte, yte)
    log(f"PTQ INT8 test accuracy: {acc:.2f}%  (drop {acc_fp32 - acc:+.2f} pts)")

    scripted = save_deployable(int8, args.out)
    reloaded = torch.jit.load(args.out)
    acc_re, _ = evaluate(reloaded, xte, yte)
    log(f"Saved deployable INT8 model -> {args.out} ({file_size_mb(args.out):.1f} MB); "
        f"re-loaded accuracy {acc_re:.2f}%")
    assert abs(acc_re - acc) < 0.05, "re-loaded model differs from in-memory model"

    if args.ablation:
        rows = []
        for n in (10, 50, 100, 250, 500, 1000, 2000, 5000):
            xc, _ = calibration_subset(xtr, ytr, n)
            a, _ = evaluate(quantize_ptq(fp32, xc, backend), xte, yte)
            rows.append({"calibration_images": n, "int8_test_acc": round(a, 2)})
            log(f"  calibration size {n:5d} -> INT8 acc {a:.2f}%")
        with open(os.path.join(RES_DIR, "q1_ptq_calibration_ablation.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["calibration_images", "int8_test_acc"])
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
