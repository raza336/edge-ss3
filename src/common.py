"""
common.py
=========
Shared utilities for the Edge Computing & Intelligent Systems - Assignment 3.

Contains
  * CIFAR-10 loading (kept as uint8 tensors in RAM) + GPU-free batch augmentation
  * A quantization-friendly ResNet-18 (CIFAR variant) that can ALSO be built with
    reduced inner widths (needed by the structured-pruning question)
  * Training / evaluation helpers
  * Model registry + loaders for all five deployable models
  * Small profiling helpers (parameter counts, MACs, latency)

Every other script imports from this file so that all experiments use exactly the
same data pipeline, architecture and evaluation code.
"""
import os
import random
import warnings
import time
import gzip
import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization import QuantStub, DeQuantStub
from torchvision.datasets import CIFAR10

# torch.ao eager-mode quantization prints migration/deprecation notices (still fully supported)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
ONNX_DIR = os.path.join(ROOT, "onnx_models")
RES_DIR = os.path.join(ROOT, "results")
LOG_DIR = os.path.join(RES_DIR, "logs")
for _d in (CKPT_DIR, ONNX_DIR, RES_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

# Registry of the five deployable models (name -> default file)
MODEL_FILES = {
    "fp32": "fp32_resnet18.pth",
    "ptq": "ptq_int8_resnet18.pt",
    "qat": "qat_int8_resnet18.pt",
    "unstructured": "pruned_unstructured_resnet18.pth",
    "structured": "pruned_structured_resnet18.pth",
}

# ONNX versions of the five models (QAT = QDQ graph, PTQ = ONNX-Runtime static INT8 QDQ)
ONNX_FILES = {
    "fp32": "fp32_resnet18.onnx",
    "ptq": "ptq_int8_resnet18.onnx",
    "qat": "qat_int8_qdq_resnet18.onnx",
    "unstructured": "pruned_unstructured_resnet18.onnx",
    "structured": "pruned_structured_resnet18.onnx",
}


# --------------------------------------------------------------------------- #
# Reproducibility / environment
# --------------------------------------------------------------------------- #
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_quant_backend():
    """Configure the quantized-kernel backend (fbgemm for x86, qnnpack for ARM)."""
    engines = torch.backends.quantized.supported_engines
    for name in ("fbgemm", "x86", "qnnpack"):
        if name in engines:
            torch.backends.quantized.engine = name
            return name
    raise RuntimeError(f"No usable quantization backend in {engines}")


class Logger:
    """Print to stdout and append to a log file."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        with open(self.path, "a") as f:
            f.write(msg + "\n")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_cifar10(data_dir=DATA_DIR, train=True):
    """Return (uint8 images [N,3,32,32], int64 labels [N]). Downloads if missing."""
    ds = CIFAR10(root=data_dir, train=train, download=True)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(ds.targets, dtype=torch.long)
    return x, y


def normalize(x_u8):
    return (x_u8.float() / 255.0 - MEAN) / STD


def augment(x_u8):
    """Random crop (pad 4) + horizontal flip on a uint8 batch."""
    n = x_u8.shape[0]
    padded = F.pad(x_u8, (4, 4, 4, 4))
    ox = torch.randint(0, 9, (n,))
    oy = torch.randint(0, 9, (n,))
    out = torch.empty_like(x_u8)
    for i in range(n):
        out[i] = padded[i, :, oy[i]:oy[i] + 32, ox[i]:ox[i] + 32]
    flip = torch.rand(n) < 0.5
    out[flip] = out[flip].flip(-1)
    return out


def train_batches(x_u8, y, batch_size, steps, aug=True):
    """Yield exactly `steps` shuffled (augmented, normalised) mini-batches."""
    n = x_u8.shape[0]
    done = 0
    while done < steps:
        perm = torch.randperm(n)
        for i in range(0, n - batch_size + 1, batch_size):
            if done >= steps:
                return
            idx = perm[i:i + batch_size]
            xb = x_u8[idx]
            if aug:
                xb = augment(xb)
            yield normalize(xb), y[idx]
            done += 1


def calibration_subset(x_u8, y, n=1000, seed=0):
    """Deterministic class-agnostic subset of the TRAIN set used for calibration."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x_u8.shape[0], generator=g)[:n]
    return x_u8[idx], y[idx]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    """ResNet basic block.  `mid_c` (inner width) can be smaller than `out_c`
    which is what structured channel pruning produces."""

    def __init__(self, in_c, out_c, stride=1, mid_c=None):
        super().__init__()
        mid_c = mid_c or out_c
        self.conv1 = nn.Conv2d(in_c, mid_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(mid_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c))
        else:
            self.shortcut = nn.Sequential()
        # FloatFunctional lets the residual add (+ReLU) be quantized in eager mode
        self.add_relu = FloatFunctional()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.add_relu.add_relu(out, self.shortcut(x))


class ResNet18(nn.Module):
    """ResNet-18 for 32x32 inputs (3x3 stem, stride-2, no max-pool, 4 stages x 2 blocks).

    inner_channels : optional list of 8 ints = width of conv1->conv2 hidden layer of
                     each BasicBlock (default: same as the block's output width).
    """
    STAGES = [(64, 1), (128, 2), (256, 2), (512, 2)]

    def __init__(self, num_classes=10, inner_channels=None):
        super().__init__()
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        blocks, in_c, k = [], 64, 0
        for out_c, stride in self.STAGES:
            stage = []
            for b in range(2):
                mid = inner_channels[k] if inner_channels else None
                stage.append(BasicBlock(in_c, out_c, stride if b == 0 else 1, mid))
                in_c = out_c
                k += 1
            blocks.append(nn.Sequential(*stage))
        self.layer1, self.layer2, self.layer3, self.layer4 = blocks
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        return self.dequant(self.fc(x))

    def blocks(self):
        return [b for stage in (self.layer1, self.layer2, self.layer3, self.layer4) for b in stage]


def fuse_model(model, qat=False):
    """Fuse Conv+BN(+ReLU) groups (required before static PTQ / QAT)."""
    from torch.ao.quantization import fuse_modules, fuse_modules_qat
    fuse = fuse_modules_qat if qat else fuse_modules
    groups = [["conv1", "bn1", "relu"]]
    names = [(f"layer{s}", b) for s in range(1, 5) for b in range(2)]
    for layer, b in names:
        p = f"{layer}.{b}"
        groups += [[f"{p}.conv1", f"{p}.bn1", f"{p}.relu1"], [f"{p}.conv2", f"{p}.bn2"]]
        if len(getattr(model, layer)[b].shortcut) > 0:
            groups.append([f"{p}.shortcut.0", f"{p}.shortcut.1"])
    fuse(model, groups, inplace=True)
    return model


def build_model(arch=None):
    arch = arch or {}
    return ResNet18(arch.get("num_classes", 10), arch.get("inner_channels"))


def save_checkpoint(model, path, arch=None, **extra):
    torch.save({"state_dict": model.state_dict(), "arch": arch or {}, **extra}, path)


def load_fp32_checkpoint(path, map_location="cpu"):
    """Load a float checkpoint (FP32 / pruned) -> eval-mode model."""
    ck = torch.load(path, map_location=map_location)
    model = build_model(ck.get("arch"))
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck


def load_deployable(name, ckpt_dir=CKPT_DIR):
    """Load any of the five deployable models by registry name -> eval-mode module."""
    path = os.path.join(ckpt_dir, MODEL_FILES[name])
    if name in ("ptq", "qat"):
        select_quant_backend()
        return torch.jit.load(path, map_location="cpu").eval()
    return load_fp32_checkpoint(path)[0]


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, x_u8, batch_size=256):
    """Return (logits [N,10]) for a uint8 image tensor (CPU)."""
    model.eval()
    out = []
    for i in range(0, x_u8.shape[0], batch_size):
        out.append(model(normalize(x_u8[i:i + batch_size])).float())
    return torch.cat(out)


def evaluate(model, x_u8, y, batch_size=256):
    logits = predict(model, x_u8, batch_size)
    loss = F.cross_entropy(logits, y).item()
    acc = (logits.argmax(1) == y).float().mean().item() * 100
    return acc, loss


# --------------------------------------------------------------------------- #
# Profiling helpers
# --------------------------------------------------------------------------- #
def count_params(model):
    """(total_params, nonzero_params) over conv/linear weights + BN/bias."""
    total = nz = 0
    for p in model.parameters():
        total += p.numel()
        nz += (p != 0).sum().item()
    return total, nz


def count_macs(model, density_aware=False):
    """Multiply-accumulates for one 3x32x32 image (conv + linear).
    density_aware=True scales each layer by its fraction of non-zero weights,
    i.e. the MACs a sparse-aware engine would need."""
    macs = [0.0]
    hooks = []

    def conv_hook(m, i, o):
        dens = (m.weight != 0).float().mean().item() if density_aware else 1.0
        macs[0] += o.shape[2] * o.shape[3] * m.weight.numel() * dens

    def lin_hook(m, i, o):
        dens = (m.weight != 0).float().mean().item() if density_aware else 1.0
        macs[0] += m.weight.numel() * dens

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, 32, 32))
    for h in hooks:
        h.remove()
    return macs[0]


def file_size_mb(path):
    return os.path.getsize(path) / 1e6


def gzip_size_mb(path):
    with open(path, "rb") as f:
        return len(gzip.compress(f.read(), compresslevel=6)) / 1e6


def measure_latency(fn, sample, warmup=20, iters=200):
    """Latency (ms) of fn(sample) -> dict(mean, p50, p95). sample is a ready tensor/array."""
    for _ in range(warmup):
        fn(sample)
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn(sample)
        ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts)
    return {"mean": ts.mean(), "p50": np.percentile(ts, 50), "p95": np.percentile(ts, 95)}
