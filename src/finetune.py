"""
finetune.py
===========
Q2.4  -  Fine-tuning pipeline shared by the unstructured and the structured pruning scripts.

`finetune()` recovers accuracy after pruning.  It works for
  * masked models (torch.nn.utils.prune re-parametrisation keeps the zeros at zero), and
  * physically rebuilt (smaller) models.

It can also be run stand-alone on any float checkpoint:
    python src/finetune.py --ckpt checkpoints/pruned_structured_resnet18.pth --epochs 1
"""
import argparse
import os
import time

import torch
import torch.nn as nn

from common import (DATA_DIR, LOG_DIR, Logger, evaluate, load_cifar10, load_fp32_checkpoint,
                    save_checkpoint, set_seed, train_batches)


def finetune(model, xtr, ytr, xte, yte, epochs, lr=0.01, wd=5e-4, batch_size=128, log=print,
             tag="finetune", min_lr_frac=0.01):
    """SGD + cosine-annealed fine-tuning for `epochs` (fractional allowed).
    Returns the final test accuracy (%).  The test set is only used for reporting."""
    steps_per_epoch = xtr.shape[0] // batch_size
    total = max(1, int(round(epochs * steps_per_epoch)))
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=lr * min_lr_frac)
    crit = nn.CrossEntropyLoss()
    model.train()
    t0, run, n = time.time(), 0.0, 0
    for step, (xb, yb) in enumerate(train_batches(xtr, ytr, batch_size, total), 1):
        opt.zero_grad(set_to_none=True)
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
        sched.step()
        run += loss.item() * yb.numel()
        n += yb.numel()
        if step % 100 == 0 or step == total:
            log(f"  [{tag}] step {step:4d}/{total} loss {run / n:.4f} | {(time.time() - t0) / 60:.1f} min")
            run = n = 0
    acc, _ = evaluate(model, xte, yte)
    log(f"  [{tag}] done -> test acc {acc:.2f}%")
    return acc


def main():
    ap = argparse.ArgumentParser(description="Fine-tune a float / pruned checkpoint")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None, help="default: overwrite --ckpt")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--data-dir", default=DATA_DIR)
    args = ap.parse_args()
    set_seed(42)
    log = Logger(os.path.join(LOG_DIR, "q2_finetune_standalone.log"))
    xtr, ytr = load_cifar10(args.data_dir, True)
    xte, yte = load_cifar10(args.data_dir, False)
    model, ck = load_fp32_checkpoint(args.ckpt)
    acc0, _ = evaluate(model, xte, yte)
    log(f"before: {acc0:.2f}%")
    acc = finetune(model, xtr, ytr, xte, yte, args.epochs, args.lr, log=log)
    extra = {k: v for k, v in ck.items() if k not in ("state_dict", "arch")}
    extra["test_acc"] = acc
    save_checkpoint(model, args.out or args.ckpt, ck.get("arch"), **extra)


if __name__ == "__main__":
    main()
