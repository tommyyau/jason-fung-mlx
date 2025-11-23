#!/bin/bash

# Script to run DPO Pipeline (Run 11)
# Model: Llama 3.2-3B-Instruct
# Goal: Fine-tune using DPO to prefer Insulin Model over CICO.

set -e

echo "========================================="
echo "DPO Training Run 11 - Llama 3.2-3B-Instruct"
echo "========================================="

echo ""
echo "========================================="
echo "Step 1: Generate DPO Data with Llama"
echo "========================================="
python3 scripts/phase3-prepare-data-mlx/06_generate_dpo_pairs.py

echo ""
echo "========================================="
echo "Step 2: Precompute Reference Logps"
echo "========================================="
python3 scripts/phase4-fine-tune-model/10_train_dpo.py \
  --config config/mlx_llama-3b_dpo_run11.yaml \
  --stage precompute

echo ""
echo "========================================="
echo "Step 3: Run DPO Training"
echo "========================================="
python3 scripts/phase4-fine-tune-model/10_train_dpo.py \
  --config config/mlx_llama-3b_dpo_run11.yaml \
  --stage train

echo ""
echo "========================================="
echo "Step 4: Fuse Adapters"
echo "========================================="
python -m mlx_lm fuse \
  --model mlx-community/Llama-3.2-3B-Instruct \
  --adapter-path models/llama-3b-dpo-run11 \
  --save-path models/llama-3b-dpo-run11-fused \
  --de-quantize

echo ""
echo "========================================="
echo "Step 5: Test Model"
echo "========================================="
echo "Prompt: 'Should I count calories to lose weight?'"
python -m mlx_lm.generate \
  --model models/llama-3b-dpo-run11-fused \
  --prompt "Should I count calories to lose weight?" \
  --max-tokens 200

echo ""
echo "DONE!"
