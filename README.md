# Assignment 3 - Edge Computing & Intelligent Systems
Quantization (PTQ, QAT) and pruning (unstructured, structured) of ResNet-18 on CIFAR-10, PyTorch, CPU only.

## 1. Installation
```bash
python -m venv venv && source venv/bin/activate      # optional
pip install -r requirements.txt                       # torch, torchvision, onnx, onnxruntime, pandas, pillow, matplotlib
```
CIFAR-10 is downloaded automatically by torchvision into `./data` on first use (about 170 MB). Everything runs on CPU; no GPU needed. Tested with Python 3.13 / PyTorch 2.15 on x86-64 Linux (quantization backend `fbgemm`; falls back to `qnnpack` on ARM).

## 2. Run everything (reproducible pipeline)
```bash
./run_all.sh            # full pipeline used for the submitted results (~2 h on 2 CPU cores)
QUICK=1 ./run_all.sh    # tiny smoke test (minutes) - overwrites results, use a copy of the folder
```
Seeds are fixed to 42. Multi-threaded CPU maths can still move accuracies by a few tenths of a point between machines.

## 3. Step by step
| Step | Command | Output |
|---|---|---|
| Baseline FP32 (Q1.1, Q2.1) | `python src/train_baseline.py --epochs 8` | `checkpoints/fp32_resnet18.pth` |
| PTQ (Q1.2) | `python src/ptq.py --calib-size 1000 --ablation` | `checkpoints/ptq_int8_resnet18.pt` |
| QAT (Q1.3) | `python src/qat.py --epochs 2` | `checkpoints/qat_int8_resnet18.pt`, `onnx_models/qat_int8_qdq_resnet18.onnx` |
| Unstructured pruning (Q2.2) | `python src/prune_unstructured.py --sparsity 0.85 --iterations 5` | `checkpoints/pruned_unstructured_resnet18.pth` (+ `_sparse_compact.pth`) |
| Structured pruning (Q2.3) | `python src/prune_structured.py --ratio 0.5 --ft-epochs 3` | `checkpoints/pruned_structured_resnet18.pth` |
| Fine-tuning (Q2.4) | used inside both pruning scripts; stand-alone: `python src/finetune.py --ckpt <file>` | - |
| ONNX export | `python src/export_onnx.py` (QAT ONNX is written by `qat.py`) | `onnx_models/*.onnx` |
| Inference | `python src/inference.py --model all --num-images 5` | console |
| Benchmark Q1 | `python src/benchmark_q1.py` | `results/q1_benchmark.csv`, `results/q1_predictions.csv` |
| Benchmark Q2 | `python src/benchmark_q2.py` | `results/q2_benchmark.csv`, `results/q2_predictions.csv` |
| ONNX CPU inference | `python src/onnx_inference.py` | `results/onnx_benchmark.csv`, `results/onnx_predictions.csv` |
| Report figures | `python src/make_report_assets.py` | `results/figures`, `results/screenshots` |

Order matters: the baseline must exist before everything else; run QAT before `export_onnx.py` if you want the ONNX benchmark to include QAT.

## 4. Inference examples
```bash
python src/inference.py --model fp32 --num-images 8                 # first 8 CIFAR-10 test images
python src/inference.py --model qat --image my_photo.png            # any image (resized to 32x32)
python src/inference.py --model structured --backend onnx           # ONNX Runtime, CPU
```
Models: `fp32 | ptq | qat | unstructured | structured | all`.

## 5. Folder structure
```
src/          all code (common.py = data, ResNet-18, helpers; one script per task, see table)
checkpoints/  FP32, PTQ-INT8, QAT-INT8 (TorchScript), unstructured-pruned, structured-pruned
onnx_models/  ONNX versions of the five models
results/      CSVs, logs/, figures/, screenshots/
report/       Technical_Report.docx / .pdf
run_all.sh, requirements.txt, README.md
```

## 6. Results (CIFAR-10 test set, 10,000 images, CPU)
| Model | Acc % | Size MB | Latency b=1 ms | Throughput img/s |
|---|---|---|---|---|
| FP32 | 86.33 | 44.8 | 6.37 | 862 |
| PTQ-INT8 | 86.30 | 11.4 | 1.16 | 2695 |
| QAT-INT8 | 87.30 | 11.4 | 1.29 | 2735 |
| Unstructured 85% sparse | 86.97 | 44.8 (8.2 compact) | 7.36 | 895 |
| Structured 50% channels | 86.82 | 23.9 | 4.20 | 1454 |

The baseline is deliberately short-trained (8 epochs, stride-2 stem) because of CPU-only training, so QAT/pruned models gain from their extra fine-tuning epochs; see the report for discussion.

## 7. Notes
- INT8 PyTorch models are saved as TorchScript (executable without the class definition) and need an x86 build with fbgemm (or qnnpack on ARM).
- Pruned float checkpoints carry their architecture (`arch`), so the structured model can be rebuilt from the file alone.
- `torch.ao` eager-mode quantization is deprecated in newer PyTorch in favour of torchao but works in the tested version; its notices are silenced in `common.py`.
