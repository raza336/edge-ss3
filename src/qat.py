"""
qat.py
======
Q1.3  -  Quantization-Aware Training (QAT) of ResNet-18.

Pipeline
  1. initialise from the PRE-TRAINED FP32 checkpoint
  2. fuse Conv+BN(+ReLU) (QAT variant keeps BN trainable)
  3. attach the QAT qconfig and INSERT FAKE-QUANT modules  (prepare_qat)
  4. FINE-TUNE with fake quantization in the forward pass (straight-through estimator)
       - BN statistics are frozen and observers disabled late in training (standard recipe)
  5. CONVERT the trained fake-quant model to a real INT8 model
  6. EXPORT the INT8 model (TorchScript) and a QDQ-ONNX version of the fake-quant model

Usage:  python src/qat.py --epochs 2
"""
import argparse
import copy
import csv
import os
import time

import torch
import torch.nn as nn
from torch.ao.quantization import (convert, disable_observer, get_default_qat_qconfig,
                                   prepare_qat)
from torch.ao.nn.intrinsic.qat import freeze_bn_stats

from common import (CKPT_DIR, DATA_DIR, LOG_DIR, MODEL_FILES, RES_DIR, Logger, evaluate,
                    file_size_mb, fuse_model, load_cifar10, load_fp32_checkpoint,
                    normalize, select_quant_backend, set_seed, train_batches)
from ptq import save_deployable


def build_qat_model(fp32_model, backend):
    """FP32 model -> fake-quant (QAT-prepared) model in train mode."""
    model = copy.deepcopy(fp32_model).train()
    fuse_model(model, qat=True)
    model.qconfig = get_default_qat_qconfig(backend)
    prepare_qat(model, inplace=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=2.0, help="QAT fine-tuning epochs (may be fractional)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--freeze-bn-frac", type=float, default=0.6, help="freeze BN stats after this fraction of steps")
    ap.add_argument("--freeze-obs-frac", type=float, default=0.8, help="freeze observers after this fraction")
    ap.add_argument("--fp32", default=os.path.join(CKPT_DIR, MODEL_FILES["fp32"]))
    ap.add_argument("--out", default=os.path.join(CKPT_DIR, MODEL_FILES["qat"]))
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-onnx", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    log = Logger(os.path.join(LOG_DIR, "q1_qat.log"))
    backend = select_quant_backend()
    log(f"=== QAT | backend={backend} | {vars(args)}")

    xtr, ytr = load_cifar10(args.data_dir, True)
    xte, yte = load_cifar10(args.data_dir, False)
    fp32, _ = load_fp32_checkpoint(args.fp32)
    acc_fp32, _ = evaluate(fp32, xte, yte)
    log(f"FP32 starting point: {acc_fp32:.2f}%")

    qat = build_qat_model(fp32, backend)
    n_fq = sum(1 for m in qat.modules() if "FakeQuantize" in type(m).__name__)
    log(f"fake-quant modules inserted: {n_fq}")

    steps_per_epoch = xtr.shape[0] // args.batch_size
    total = max(1, int(args.epochs * steps_per_epoch))
    opt = torch.optim.SGD(qat.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=args.lr * 0.01)
    crit = nn.CrossEntropyLoss()

    hist, t0, run_loss, run_n = [], time.time(), 0.0, 0
    bn_frozen = obs_frozen = False
    qat.train()
    for step, (xb, yb) in enumerate(train_batches(xtr, ytr, args.batch_size, total), 1):
        if not bn_frozen and step > args.freeze_bn_frac * total:
            qat.apply(freeze_bn_stats)
            bn_frozen = True
            log(f"step {step}: BN statistics frozen")
        if not obs_frozen and step > args.freeze_obs_frac * total:
            qat.apply(disable_observer)
            obs_frozen = True
            log(f"step {step}: observers frozen (quantization ranges fixed)")
        opt.zero_grad(set_to_none=True)
        loss = crit(qat(xb), yb)
        loss.backward()
        opt.step()
        sched.step()
        run_loss += loss.item() * yb.numel()
        run_n += yb.numel()
        if step % 100 == 0 or step == total:
            log(f"step {step:5d}/{total} loss {run_loss / run_n:.4f} lr {sched.get_last_lr()[0]:.5f} "
                f"| {(time.time() - t0) / 60:.1f} min")
            hist.append({"step": step, "train_loss": run_loss / run_n})
            run_loss = run_n = 0

    qat.eval()
    acc_fake, _ = evaluate(qat, xte, yte)
    log(f"QAT fake-quant model test accuracy: {acc_fake:.2f}%")

    # ---- convert to a real INT8 model and export -------------------------------------------
    int8 = convert(copy.deepcopy(qat).cpu().eval(), inplace=False)
    acc_int8, _ = evaluate(int8, xte, yte)
    log(f"QAT INT8 test accuracy: {acc_int8:.2f}%  (vs FP32 {acc_fp32:.2f}%, drop {acc_fp32 - acc_int8:+.2f})")
    save_deployable(int8, args.out)
    acc_re, _ = evaluate(torch.jit.load(args.out), xte, yte)
    log(f"Saved deployable INT8 model -> {args.out} ({file_size_mb(args.out):.1f} MB); reloaded acc {acc_re:.2f}%")
    assert abs(acc_re - acc_int8) < 0.05

    with open(os.path.join(RES_DIR, "q1_qat_training_history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "train_loss"])
        w.writeheader()
        w.writerows(hist)
    with open(os.path.join(RES_DIR, "q1_qat_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fp32_acc", "qat_fakequant_acc", "qat_int8_acc", "epochs", "lr"])
        w.writerow([round(acc_fp32, 2), round(acc_fake, 2), round(acc_int8, 2), args.epochs, args.lr])

    if not args.no_onnx:
        from export_onnx import export_qat_qdq
        export_qat_qdq(qat)


if __name__ == "__main__":
    main()
