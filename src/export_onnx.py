"""
export_onnx.py
==============
Deliverable 8  -  ONNX export of the models (for CPU deployment with ONNX Runtime).

  fp32 / unstructured / structured : torch.onnx.export of the float network
  qat                              : the trained FAKE-QUANT model is exported, every fake-quant
                                     becomes QuantizeLinear/DequantizeLinear (QDQ) nodes; ONNX
                                     Runtime executes them with INT8 kernels (called from qat.py)
  ptq                              : the FP32 ONNX graph is statically quantized to INT8 (QDQ) with
                                     onnxruntime.quantization.quantize_static using the SAME
                                     calibration subset of the training data as ptq.py

(PyTorch eager-mode INT8 modules use fbgemm packed ops that have no ONNX equivalent, which is why
 the INT8 ONNX models are produced via QDQ graphs as described above.)

Usage:  python src/export_onnx.py          # fp32, pruned x2 and ptq;  qat is exported by qat.py
"""
import argparse
import os
import warnings

import numpy as np
import torch

from common import (CKPT_DIR, DATA_DIR, MODEL_FILES, ONNX_DIR, ONNX_FILES, calibration_subset,
                    file_size_mb, load_cifar10, load_fp32_checkpoint, normalize)

warnings.filterwarnings("ignore")
OPSET = 13


def _export(model, path):
    model = model.eval().cpu()
    dummy = torch.randn(1, 3, 32, 32)
    torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=OPSET, do_constant_folding=True, dynamo=False)
    return path


def check_onnx(path, model, tol=1e-3):
    """Run ONNX Runtime on random input and compare with the PyTorch module."""
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(path))
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        ref = model.eval()(x).numpy()
    out = sess.run(None, {"input": x.numpy()})[0]
    return float(np.abs(ref - out).max())


def export_float(name):
    model, _ = load_fp32_checkpoint(os.path.join(CKPT_DIR, MODEL_FILES[name]))
    path = _export(model, os.path.join(ONNX_DIR, ONNX_FILES[name]))
    err = check_onnx(path, model)
    print(f"[onnx] {name:13s} -> {path} ({file_size_mb(path):.1f} MB) | max|torch-ort| = {err:.2e}")
    return path


class _ExportFQ(torch.nn.Module):
    """Plain (export-friendly) fake-quant with frozen scale / zero-point.
    PyTorch's training-time FusedMovingAvgObsFakeQuantize has no ONNX symbolic, whereas
    torch.fake_quantize_*_affine maps to QuantizeLinear + DequantizeLinear."""

    def __init__(self, fq):
        super().__init__()
        scale = fq.scale.detach().clone().float()
        zp = fq.zero_point.detach().clone().to(torch.int32)
        qmin, qmax = int(fq.quant_min), int(fq.quant_max)
        if (qmin, qmax) == (0, 127):
            # fbgemm "reduce_range" activations use 7 bits [0,127]; ONNX uint8 needs [0,255].
            # Re-express the SAME real-valued range [-zp*s, (127-zp)*s] on 256 levels.
            zp = torch.round(zp.float() * 255.0 / 127.0).to(torch.int32)
            scale = scale * 127.0 / 255.0
            qmax = 255
        self.per_channel = fq.qscheme in (torch.per_channel_symmetric, torch.per_channel_affine)
        if not self.per_channel:            # ONNX Runtime requires scalar (0-d) per-tensor qparams
            scale, zp = scale.reshape(()), zp.reshape(())
        self.register_buffer("scale", scale)
        self.register_buffer("zp", zp)
        self.qmin, self.qmax = qmin, qmax
        self.axis = int(getattr(fq, "ch_axis", 0))

    def forward(self, x):
        if self.per_channel:
            return torch.fake_quantize_per_channel_affine(x, self.scale, self.zp, self.axis, self.qmin, self.qmax)
        return torch.fake_quantize_per_tensor_affine(x, self.scale, self.zp, self.qmin, self.qmax)


class _FoldedQATConv(torch.nn.Module):
    """QAT ConvBn(ReLU) with the BatchNorm folded into the conv weights:
         W' = W * gamma/sqrt(var+eps),   b' = beta - mean*gamma/sqrt(var+eps)
       followed by weight fake-quant (per-channel), conv, optional ReLU and activation fake-quant.
       (This is exactly what the QAT module computes in eval mode, minus the extra BN nodes.)"""

    def __init__(self, mod):
        super().__init__()
        bn = mod.bn
        s = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
        self.weight = torch.nn.Parameter(mod.weight.detach() * s.reshape(-1, 1, 1, 1), requires_grad=False)
        bias = bn.bias.detach() - bn.running_mean * s
        if mod.bias is not None:
            bias = bias + mod.bias.detach() * s
        self.bias = torch.nn.Parameter(bias, requires_grad=False)
        self.stride, self.padding, self.dilation, self.groups = mod.stride, mod.padding, mod.dilation, mod.groups
        self.relu = "ReLU" in type(mod).__name__
        self.wfq = _ExportFQ(mod.weight_fake_quant)
        self.afq = _ExportFQ(mod.activation_post_process)

    def forward(self, x):
        y = torch.nn.functional.conv2d(x, self.wfq(self.weight), self.bias, self.stride,
                                       self.padding, self.dilation, self.groups)
        return self.afq(torch.relu(y) if self.relu else y)


def _set_module(root, dotted, new):
    parent = root
    *path, leaf = dotted.split(".")
    for p in path:
        parent = getattr(parent, p)
    setattr(parent, leaf, new)


def make_export_friendly(qat_model):
    """Copy of a trained QAT model that is ONNX-exportable:
       1. ConvBn(ReLU) QAT modules -> BN-folded conv + plain fake-quant
       2. every remaining FusedMovingAvgObsFakeQuantize -> _ExportFQ"""
    import copy
    m = copy.deepcopy(qat_model).cpu().eval()
    for name, mod in list(m.named_modules()):
        if type(mod).__name__ in ("ConvBn2d", "ConvBnReLU2d"):
            _set_module(m, name, _FoldedQATConv(mod))
    for name, mod in list(m.named_modules()):
        if "FakeQuantize" in type(mod).__name__:
            _set_module(m, name, _ExportFQ(mod))
    return m


def export_qat_qdq(qat_model):
    """Export the trained fake-quant (QAT) model as a QDQ ONNX graph (INT8 on ONNX Runtime)."""
    path = os.path.join(ONNX_DIR, ONNX_FILES["qat"])
    try:
        friendly = make_export_friendly(qat_model)
        _export(friendly, path)
        import onnx
        graph = onnx.load(path).graph
        onnx.checker.check_model(onnx.load(path))
        ops = [n.op_type for n in graph.node]
        err = check_onnx(path, friendly)
        print(f"[onnx] qat           -> {path} ({file_size_mb(path):.1f} MB) | "
              f"QuantizeLinear x{ops.count('QuantizeLinear')}, DequantizeLinear x{ops.count('DequantizeLinear')} "
              f"| max|torch-ort| = {err:.2e}")
    except Exception as e:  # keep the training pipeline alive if export is unsupported
        print(f"[onnx] QAT export failed: {type(e).__name__}: {e}")


class _CalibReader:
    def __init__(self, x_u8, batch=50):
        self.data = [normalize(x_u8[i:i + batch]).numpy() for i in range(0, len(x_u8), batch)]
        self.it = iter(self.data)

    def get_next(self):
        b = next(self.it, None)
        return None if b is None else {"input": b}


def export_ptq_onnx(calib_size=1000, data_dir=DATA_DIR):
    """FP32 ONNX -> static INT8 (QDQ, per-channel weights) using ONNX Runtime."""
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    src = os.path.join(ONNX_DIR, ONNX_FILES["fp32"])
    pre = src.replace(".onnx", "_preprocessed.onnx")
    quant_pre_process(src, pre)
    xtr, ytr = load_cifar10(data_dir, True)
    xc, _ = calibration_subset(xtr, ytr, calib_size)
    dst = os.path.join(ONNX_DIR, ONNX_FILES["ptq"])
    quantize_static(pre, dst, _CalibReader(xc), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True, calibrate_method=CalibrationMethod.MinMax)
    os.remove(pre)
    print(f"[onnx] ptq           -> {dst} ({file_size_mb(dst):.1f} MB)")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-size", type=int, default=1000)
    args = ap.parse_args()
    for n in ("fp32", "unstructured", "structured"):
        if os.path.exists(os.path.join(CKPT_DIR, MODEL_FILES[n])):
            export_float(n)
    if os.path.exists(os.path.join(ONNX_DIR, ONNX_FILES["fp32"])):
        export_ptq_onnx(args.calib_size)
    print("(QAT ONNX is written by qat.py right after training)")


if __name__ == "__main__":
    main()
