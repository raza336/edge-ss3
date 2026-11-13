"""
train_baseline.py
=================
Q1.1 / Q2.1  -  Train the FP32 ResNet-18 baseline on CIFAR-10 and save the checkpoint.

The same FP32 checkpoint is the starting point for PTQ, QAT, unstructured pruning and
structured pruning, so every compressed model is compared against exactly one baseline.

Usage:  python src/train_baseline.py --epochs 8
"""
import argparse
import csv
import os
import time

import torch
import torch.nn as nn

from common import (CKPT_DIR, DATA_DIR, LOG_DIR, RES_DIR, MODEL_FILES, Logger, build_model,
                    evaluate, load_cifar10, save_checkpoint, set_seed, train_batches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1, help="peak LR of the one-cycle schedule")
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out", default=os.path.join(CKPT_DIR, MODEL_FILES["fp32"]))
    ap.add_argument("--threads", type=int, default=0, help="0 = torch default")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    log = Logger(os.path.join(LOG_DIR, "q1_q2_train_baseline.log"))
    log(f"=== Baseline training | {vars(args)}")

    xtr, ytr = load_cifar10(args.data_dir, train=True)
    xte, yte = load_cifar10(args.data_dir, train=False)

    model = build_model()
    log(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    steps_per_epoch = xtr.shape[0] // args.batch_size
    total = steps_per_epoch * args.epochs
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True,
                          weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total,
                                                pct_start=0.25)
    crit = nn.CrossEntropyLoss()

    hist = []
    t0 = time.time()
    step = 0
    run_loss = run_correct = run_n = 0
    model.train()
    for xb, yb in train_batches(xtr, ytr, args.batch_size, total, aug=True):
        opt.zero_grad(set_to_none=True)
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        run_loss += loss.item() * yb.numel()
        run_correct += (out.argmax(1) == yb).sum().item()
        run_n += yb.numel()
        if step % steps_per_epoch == 0:
            ep = step // steps_per_epoch
            acc, tl = evaluate(model, xte, yte)
            model.train()
            row = dict(epoch=ep, train_loss=run_loss / run_n, train_acc=100 * run_correct / run_n,
                       test_loss=tl, test_acc=acc, lr=sched.get_last_lr()[0],
                       elapsed_min=(time.time() - t0) / 60)
            hist.append(row)
            log(f"epoch {ep:2d}/{args.epochs} train_loss {row['train_loss']:.4f} "
                f"train_acc {row['train_acc']:.2f} | test_loss {tl:.4f} test_acc {acc:.2f} "
                f"| {row['elapsed_min']:.1f} min")
            run_loss = run_correct = run_n = 0
            save_checkpoint(model, args.out, arch={"name": "resnet18", "num_classes": 10},
                            test_acc=acc, epochs_done=ep)

    with open(os.path.join(RES_DIR, "baseline_training_history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader()
        w.writerows(hist)
    log(f"Saved FP32 checkpoint -> {args.out}  (final test acc {hist[-1]['test_acc']:.2f}%)")


if __name__ == "__main__":
    main()
