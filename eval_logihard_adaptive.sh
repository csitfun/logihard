#!/bin/bash
# LogiHard-2k Adaptive Evaluation Runner
# This script uses IRT-CAT to evaluate model ability with ~40 questions.

python3 evaluation/eval_logihard_adaptive.py \
  --input data/benchmark/logihard_2k_tiered.jsonl \
  --model deepseek-v4-pro \
  --api-key xxx \
  --max-items 60 \
  --hard-mode \
  --timeout 600 \
  --resume
