#!/usr/bin/env bash
# =============================================================================
# run_all.sh  -  reproduces EVERY result of Assignment 3 end-to-end (CPU only)
#
#   ./run_all.sh          full pipeline (the settings used for the submitted results)
#   QUICK=1 ./run_all.sh  tiny smoke-test run (a few minutes) to verify the setup
#
# CIFAR-10 is downloaded automatically into ./data on the first run.
# All seeds are fixed (42); CPU multi-threading can still cause tiny floating-point
# differences between machines (typically +-0.1-0.3 accuracy points).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${QUICK:-0}" == "1" ]]; then
  BASE_EP=1; QAT_EP=0.1; PRUNE_IT=2; PRUNE_FT=0.05; PRUNE_FINAL=0.05; STRUCT_FT=0.1
else
  BASE_EP=8; QAT_EP=2; PRUNE_IT=5; PRUNE_FT=0.4; PRUNE_FINAL=1.0; STRUCT_FT=3
fi

### [Q1/Q2.1] baseline FP32 ResNet-18";   python src/train_baseline.py --epochs "$BASE_EP"
### [Q1.2]    post-training quantization"; python src/ptq.py --calib-size 1000 --ablation
### [Q1.3]    quantization-aware training"; python src/qat.py --epochs "$QAT_EP"
### [Q2.2]    unstructured pruning";       python src/prune_unstructured.py --sparsity 0.85 \
        --iterations "$PRUNE_IT" --ft-epochs "$PRUNE_FT" --final-ft-epochs "$PRUNE_FINAL"
### [Q2.3]    structured channel pruning"; python src/prune_structured.py --ratio 0.5 --ft-epochs "$STRUCT_FT"
### [ONNX]    export";                      python src/export_onnx.py
### [Q1.5]    benchmark Q1";                python src/benchmark_q1.py | tee results/logs/q1_benchmark_console.txt
### [Q2.6]    benchmark Q2";                python src/benchmark_q2.py | tee results/logs/q2_benchmark_console.txt
### [ONNX]    CPU inference benchmark";     python src/onnx_inference.py | tee results/logs/onnx_benchmark_console.txt
### [report]  figures";                     python src/make_report_assets.py
### done - see results/"
