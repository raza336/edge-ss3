"""
prune_unstructured.py
=====================
Q2.2  -  Iterative, global, magnitude-based UNSTRUCTURED pruning of ResNet-18.

  * every Conv2d and the final Linear layer is a pruning target
  * criterion : L1 magnitude of individual weights, ranked GLOBALLY over all target layers
  * iterative : sparsity follows a cubic schedule over N iterations
                (prune -> fine-tune -> prune -> fine-tune ...), so the network can adapt
  * masking   : torch.nn.utils.prune keeps `weight_orig` + a binary `weight_mask`;
                the mask is re-applied in every forward pass so pruned weights stay at zero
  * permanent : at the end `prune.remove` folds the mask into the weight tensor
                (weight = weight_orig * mask) and drops the re-parametrisation
  * export    : dense checkpoint (loads into the ordinary ResNet18)  +  a really smaller
                bit-mask/values checkpoint that stores only the non-zero weights

Usage:  python src/prune_unstructured.py --sparsity 0.85 --iterations 5
"""
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import prune

from common import (CKPT_DIR, DATA_DIR, LOG_DIR, MODEL_FILES, RES_DIR, Logger, build_model,
                    count_params, evaluate, file_size_mb, gzip_size_mb, load_cifar10,
                    load_fp32_checkpoint, save_checkpoint, set_seed)
from finetune import finetune


# ----------------------------------------------------------------------------------------- #
def prunable_layers(model):
    """All Conv2d + Linear layers (the weights that get pruned)."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def global_sparsity(model):
    """Fraction of zero weights over all conv/linear weight tensors (works with or without masks)."""
    zeros = total = 0
    for _, m in prunable_layers(model):
        w = m.weight
        zeros += (w == 0).sum().item()
        total += w.numel()
    return zeros / total


def layer_sparsity_table(model):
    return [{"layer": n, "shape": "x".join(map(str, m.weight.shape)), "params": m.weight.numel(),
             "sparsity_pct": round(100 * (m.weight == 0).float().mean().item(), 2)}
            for n, m in prunable_layers(model)]


def cubic_schedule(final_sparsity, iterations):
    """Zhu & Gupta (2017) cubic schedule: s_i = s_f * (1 - (1 - i/N)^3)."""
    return [final_sparsity * (1 - (1 - i / iterations) ** 3) for i in range(1, iterations + 1)]


def prune_step(model, current, target):
    """Globally prune so that overall sparsity becomes `target`.
    torch's `amount` is relative to the weights that are still alive, hence the conversion."""
    amount = (target - current) / (1.0 - current)
    params = [(m, "weight") for _, m in prunable_layers(model)]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)


def make_permanent(model):
    """Fold masks into the weights: weight <- weight_orig * weight_mask, remove re-parametrisation."""
    for _, m in prunable_layers(model):
        if prune.is_pruned(m):
            prune.remove(m, "weight")


# ----------------------------------------------------------------------------------------- #
# Compact on-disk representation (bit-mask + non-zero values)
# ----------------------------------------------------------------------------------------- #
def save_sparse_checkpoint(model, path, arch=None):
    sd = model.state_dict()
    packed, weight_keys = {}, set()
    for name, m in prunable_layers(model):
        w = m.weight.detach().flatten()
        mask = (w != 0).numpy()
        packed[name] = {"shape": list(m.weight.shape), "mask": torch.from_numpy(np.packbits(mask)),
                        "values": w[torch.from_numpy(mask)].clone()}
        weight_keys.add(name + ".weight")
    dense = {k: v for k, v in sd.items() if k not in weight_keys}
    torch.save({"sparse": packed, "dense": dense, "arch": arch or {}}, path)


def load_sparse_checkpoint(path):
    ck = torch.load(path, map_location="cpu")
    sd = dict(ck["dense"])
    for name, p in ck["sparse"].items():
        n = int(np.prod(p["shape"]))
        mask = torch.from_numpy(np.unpackbits(p["mask"].numpy())[:n].astype(bool))
        w = torch.zeros(n)
        w[mask] = p["values"]
        sd[name + ".weight"] = w.view(*p["shape"])
    model = build_model(ck["arch"])
    model.load_state_dict(sd)
    return model.eval()


# ----------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparsity", type=float, default=0.85, help="final global sparsity")
    ap.add_argument("--iterations", type=int, default=5, help="# prune/fine-tune rounds")
    ap.add_argument("--ft-epochs", type=float, default=0.4, help="fine-tune epochs after each round")
    ap.add_argument("--final-ft-epochs", type=float, default=1.0, help="extra fine-tune with the final mask")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--fp32", default=os.path.join(CKPT_DIR, MODEL_FILES["fp32"]))
    ap.add_argument("--out", default=os.path.join(CKPT_DIR, MODEL_FILES["unstructured"]))
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    log = Logger(os.path.join(LOG_DIR, "q2_prune_unstructured.log"))
    log(f"=== Unstructured pruning | {vars(args)}")
    xtr, ytr = load_cifar10(args.data_dir, True)
    xte, yte = load_cifar10(args.data_dir, False)
    model, _ = load_fp32_checkpoint(args.fp32)
    base_acc, _ = evaluate(model, xte, yte)
    log(f"Baseline FP32 accuracy: {base_acc:.2f}%")

    schedule = cubic_schedule(args.sparsity, args.iterations)
    log("sparsity schedule:", [round(s, 3) for s in schedule])
    rows, current = [], 0.0
    for it, target in enumerate(schedule, 1):
        prune_step(model, current, target)
        current = global_sparsity(model)
        acc_pruned, _ = evaluate(model, xte, yte)
        log(f"[iter {it}/{args.iterations}] target {target:.3f} actual {current:.3f} | "
            f"acc right after pruning {acc_pruned:.2f}%")
        acc_ft = finetune(model, xtr, ytr, xte, yte, args.ft_epochs, args.lr, log=log,
                          tag=f"iter{it}")
        rows.append({"iteration": it, "target_sparsity": round(target, 4),
                     "actual_sparsity": round(current, 4), "acc_after_prune": round(acc_pruned, 2),
                     "acc_after_finetune": round(acc_ft, 2)})

    log("Final fine-tuning with the final mask ...")
    acc_final = finetune(model, xtr, ytr, xte, yte, args.final_ft_epochs, args.lr, log=log, tag="final")
    rows.append({"iteration": "final_ft", "target_sparsity": args.sparsity,
                 "actual_sparsity": round(global_sparsity(model), 4),
                 "acc_after_prune": "", "acc_after_finetune": round(acc_final, 2)})

    # ---- make pruning permanent + verify ---------------------------------------------------
    sp_before = global_sparsity(model)
    acc_before, _ = evaluate(model, xte, yte)
    make_permanent(model)
    assert not any(prune.is_pruned(m) for _, m in prunable_layers(model)), "masks still attached"
    assert not any(n.endswith(("weight_orig", "weight_mask")) for n, _ in model.named_buffers()), "mask buffers left"
    acc_after, _ = evaluate(model, xte, yte)
    sp_after = global_sparsity(model)
    log(f"Made permanent: sparsity {sp_before:.4f} -> {sp_after:.4f}; acc {acc_before:.2f}% -> {acc_after:.2f}%")
    assert abs(sp_before - sp_after) < 1e-6 and abs(acc_before - acc_after) < 0.01

    total, nz = count_params(model)
    log(f"Parameters: total {total:,}, non-zero {nz:,} ({100 * nz / total:.1f}% dense)")
    arch = {"name": "resnet18", "num_classes": 10}
    save_checkpoint(model, args.out, arch, test_acc=acc_after, sparsity=sp_after)
    sparse_path = args.out.replace(".pth", "_sparse_compact.pth")
    save_sparse_checkpoint(model, sparse_path, arch)
    re = load_sparse_checkpoint(sparse_path)
    acc_re, _ = evaluate(re, xte, yte)
    log(f"Saved dense checkpoint  -> {args.out} ({file_size_mb(args.out):.1f} MB, gzip {gzip_size_mb(args.out):.1f} MB)")
    log(f"Saved compact checkpoint-> {sparse_path} ({file_size_mb(sparse_path):.1f} MB); reload acc {acc_re:.2f}%")
    assert abs(acc_re - acc_after) < 0.01

    with open(os.path.join(RES_DIR, "q2_unstructured_iterations.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tab = layer_sparsity_table(model)
    with open(os.path.join(RES_DIR, "q2_unstructured_layer_sparsity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tab[0].keys()))
        w.writeheader()
        w.writerows(tab)
    log("per-layer sparsity (%):", {r["layer"]: r["sparsity_pct"] for r in tab})


if __name__ == "__main__":
    main()
