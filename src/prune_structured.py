"""
prune_structured.py
===================
Q2.3  -  Structured CHANNEL pruning of ResNet-18 with physical architecture reconstruction.

Which channels are pruned?
  Inside every BasicBlock the hidden feature map  conv1 -> bn1 -> ReLU -> conv2  can be thinned
  freely, whereas the block OUTPUT channels are tied to the residual stream (identity / 1x1
  shortcut) and therefore are kept.  So for each block we remove hidden channel k, which means
        conv1.weight[k]            (an output filter)          -> removed
        bn1.{gamma,beta,mean,var}[k]                           -> removed
        conv2.weight[:, k]         (the matching input slice)  -> removed
  All three tensors are sliced consistently, so tensor shapes stay valid across the
  interconnected layers  (conv1.out == bn1.C == conv2.in).

Importance criterion
  L1-norm of the filter  ||W1[k]||_1  (Li et al., "Pruning Filters for Efficient ConvNets").
  Scores are divided by the layer mean so one global threshold can be used across layers with
  different weight scales; each layer keeps at least `--min-keep` of its channels.

Reconstruction
  A NEW, smaller ResNet18(inner_channels=[...]) is instantiated and the surviving weights are
  copied into it - no masks, no zero-valued channels at run time. The channel configuration is
  stored inside the checkpoint so the model can be rebuilt anywhere.

Usage:  python src/prune_structured.py --ratio 0.5 --ft-epochs 3
"""
import argparse
import csv
import math
import os

import torch

from common import (CKPT_DIR, DATA_DIR, LOG_DIR, MODEL_FILES, RES_DIR, Logger, ResNet18,
                    count_macs, count_params, evaluate, file_size_mb, load_cifar10,
                    load_fp32_checkpoint, save_checkpoint, set_seed)
from finetune import finetune


def filter_importance(block):
    """L1 norm of each conv1 output filter  -> tensor [mid_channels]."""
    return block.conv1.weight.detach().abs().sum(dim=(1, 2, 3))


def select_channels(model, ratio, min_keep=0.25):
    """Global ranking of normalised filter importance. Returns list (one per block) of
    sorted index tensors = channels to KEEP."""
    blocks = model.blocks()
    scores = [filter_importance(b) for b in blocks]
    norm = [s / s.mean() for s in scores]
    flat = torch.cat(norm)
    n_prune = int(round(ratio * flat.numel()))
    thr = torch.sort(flat)[0][n_prune - 1] if n_prune > 0 else -1.0
    keep = []
    for s, ns in zip(scores, norm):
        k = (ns > thr).nonzero().flatten()
        floor = max(8, math.ceil(min_keep * s.numel()))
        if k.numel() < floor:                       # never let a layer collapse
            k = torch.topk(ns, floor).indices
        keep.append(torch.sort(k)[0])
    return keep


@torch.no_grad()
def rebuild_pruned(model, keep):
    """Create the physically smaller network and copy the surviving weights into it."""
    new = ResNet18(num_classes=model.fc.out_features, inner_channels=[len(k) for k in keep])
    # layers whose shapes do not change: stem and classifier
    for name in ("conv1", "bn1", "fc"):
        getattr(new, name).load_state_dict(getattr(model, name).state_dict())
    # residual blocks: slice conv1 / bn1 / conv2 consistently, copy the rest unchanged
    for ob, nb, k in zip(model.blocks(), new.blocks(), keep):
        nb.conv1.weight.copy_(ob.conv1.weight[k])                       # output filters
        for attr in ("weight", "bias", "running_mean", "running_var"):  # BN channels
            getattr(nb.bn1, attr).copy_(getattr(ob.bn1, attr)[k])
        nb.bn1.num_batches_tracked.copy_(ob.bn1.num_batches_tracked)
        nb.conv2.weight.copy_(ob.conv2.weight[:, k])                    # matching input slices
        nb.bn2.load_state_dict(ob.bn2.state_dict())
        if len(ob.shortcut) > 0:
            nb.shortcut.load_state_dict(ob.shortcut.state_dict())
    return new.eval()


@torch.no_grad()
def check_equivalence(model, pruned, keep, x):
    """Zero the conv2 input slices of the pruned channels in a copy of the original and
    compare the logits with the rebuilt network (they must be identical)."""
    import copy
    ref = copy.deepcopy(model).eval()
    for b, k in zip(ref.blocks(), keep):
        drop = torch.ones(b.conv2.weight.shape[1], dtype=torch.bool)
        drop[k] = False
        b.conv2.weight[:, drop] = 0
    return (ref(x) - pruned(x)).abs().max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=0.5, help="fraction of hidden channels to remove")
    ap.add_argument("--min-keep", type=float, default=0.25, help="min fraction of channels kept per layer")
    ap.add_argument("--ft-epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--fp32", default=os.path.join(CKPT_DIR, MODEL_FILES["fp32"]))
    ap.add_argument("--out", default=os.path.join(CKPT_DIR, MODEL_FILES["structured"]))
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    log = Logger(os.path.join(LOG_DIR, "q2_prune_structured.log"))
    log(f"=== Structured channel pruning | {vars(args)}")
    xtr, ytr = load_cifar10(args.data_dir, True)
    xte, yte = load_cifar10(args.data_dir, False)
    model, _ = load_fp32_checkpoint(args.fp32)
    base_acc, _ = evaluate(model, xte, yte)
    p0, _ = count_params(model)
    m0 = count_macs(model)
    log(f"Baseline: acc {base_acc:.2f}% | params {p0:,} | MACs {m0 / 1e6:.1f} M")

    keep = select_channels(model, args.ratio, args.min_keep)
    widths = [len(k) for k in keep]
    orig = [b.conv1.out_channels for b in model.blocks()]
    log("hidden channels per block (orig -> kept):", [f"{o}->{w}" for o, w in zip(orig, widths)])

    pruned = rebuild_pruned(model, keep)
    err = check_equivalence(model, pruned, keep, xte[:64].float() / 255)
    log(f"Reconstruction check: max |logit difference| vs. masked original = {err:.2e}")
    assert err < 1e-3, "rebuilt network does not match the masked network"

    acc_pruned, _ = evaluate(pruned, xte, yte)
    p1, _ = count_params(pruned)
    m1 = count_macs(pruned)
    log(f"After pruning (no fine-tune): acc {acc_pruned:.2f}% | params {p1:,} ({100 * p1 / p0:.1f}%) "
        f"| MACs {m1 / 1e6:.1f} M ({100 * m1 / m0:.1f}%)")

    acc_ft = finetune(pruned, xtr, ytr, xte, yte, args.ft_epochs, args.lr, log=log, tag="structured")
    pruned.eval()
    arch = {"name": "resnet18", "num_classes": 10, "inner_channels": widths}
    save_checkpoint(pruned, args.out, arch, test_acc=acc_ft, prune_ratio=args.ratio)
    log(f"Saved reconstructed model -> {args.out} ({file_size_mb(args.out):.1f} MB); final acc {acc_ft:.2f}%")

    # reload test: the checkpoint must be self-describing
    re, _ = load_fp32_checkpoint(args.out)
    acc_re, _ = evaluate(re, xte, yte)
    assert abs(acc_re - acc_ft) < 0.01
    with open(os.path.join(RES_DIR, "q2_structured_channels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block", "orig_hidden_channels", "kept_hidden_channels", "pruned_pct"])
        for i, (o, k) in enumerate(zip(orig, widths)):
            w.writerow([f"layer{i // 2 + 1}.{i % 2}", o, k, round(100 * (1 - k / o), 1)])
        w.writerow([])
        w.writerow(["acc_baseline", "acc_after_prune", "acc_after_finetune", "params_ratio", "macs_ratio"])
        w.writerow([round(base_acc, 2), round(acc_pruned, 2), round(acc_ft, 2),
                    round(p1 / p0, 4), round(m1 / m0, 4)])


if __name__ == "__main__":
    main()
