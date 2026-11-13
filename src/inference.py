"""
inference.py
============
Q1.4 / Q2.5 / Deliverable 6  -  ONE inference pipeline for all five deployable models:

      fp32 | ptq | qat | unstructured | structured          (PyTorch / TorchScript back-end)
      same names, ONNX files                                (ONNX Runtime back-end, CPU)

Examples
  python src/inference.py --model fp32 --num-images 8
  python src/inference.py --model qat --image my_photo.png
  python src/inference.py --model all --num-images 5
  python src/inference.py --model structured --backend onnx --num-images 8

`Predictor` is also imported by the benchmark scripts, so benchmarking and deployment use
exactly the same code path.
"""
import argparse
import os

import numpy as np
import torch

from common import (CLASSES, CKPT_DIR, DATA_DIR, MODEL_FILES, ONNX_DIR, ONNX_FILES, load_cifar10,
                    load_deployable, normalize, select_quant_backend)


class Predictor:
    """Uniform wrapper:  predictor(uint8_images[N,3,32,32]) -> float32 logits [N,10]."""

    def __init__(self, name, backend="torch", threads=None):
        self.name, self.backend = name, backend
        if backend == "torch":
            if threads:
                torch.set_num_threads(threads)
            select_quant_backend()
            self.path = os.path.join(CKPT_DIR, MODEL_FILES[name])
            self.model = load_deployable(name)
        elif backend == "onnx":
            import onnxruntime as ort
            self.path = os.path.join(ONNX_DIR, ONNX_FILES[name])
            so = ort.SessionOptions()
            if threads:
                so.intra_op_num_threads = threads
            self.sess = ort.InferenceSession(self.path, so, providers=["CPUExecutionProvider"])
        else:
            raise ValueError(backend)

    @torch.no_grad()
    def __call__(self, x_u8):
        x = normalize(x_u8)
        if self.backend == "torch":
            return self.model(x).float().numpy()
        return self.sess.run(None, {"input": x.numpy()})[0]

    def predict_batches(self, x_u8, batch_size=256):
        return np.concatenate([self(x_u8[i:i + batch_size]) for i in range(0, len(x_u8), batch_size)])


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def load_image(path):
    """Any image file -> uint8 tensor [1,3,32,32] (resized to CIFAR resolution)."""
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((32, 32))
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).unsqueeze(0).contiguous()


def main():
    ap = argparse.ArgumentParser(description="Run inference with any of the five models")
    ap.add_argument("--model", default="fp32", help="fp32|ptq|qat|unstructured|structured|all")
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx"])
    ap.add_argument("--image", help="path to an image file (otherwise CIFAR-10 test images are used)")
    ap.add_argument("--num-images", type=int, default=8)
    ap.add_argument("--data-dir", default=DATA_DIR)
    args = ap.parse_args()

    names = list(MODEL_FILES) if args.model == "all" else [args.model]
    if args.image:
        x, y = load_image(args.image), None
    else:
        xte, yte = load_cifar10(args.data_dir, False)
        x, y = xte[:args.num_images], yte[:args.num_images]

    for name in names:
        pred = Predictor(name, args.backend)
        prob = softmax(pred(x))
        top = prob.argmax(1)
        print(f"\n=== model: {name} | backend: {args.backend} | file: {os.path.basename(pred.path)}")
        print(f"{'#':>3} {'prediction':<12} {'conf':>6}  {'truth':<12} ok")
        for i in range(len(top)):
            truth = CLASSES[y[i]] if y is not None else "-"
            ok = "" if y is None else ("Y" if top[i] == y[i].item() else "N")
            print(f"{i:>3} {CLASSES[top[i]]:<12} {prob[i, top[i]]:6.1%}  {truth:<12} {ok}")
        if y is not None:
            print(f"accuracy on these {len(top)} images: {100 * (top == y.numpy()).mean():.1f}%")


if __name__ == "__main__":
    main()
