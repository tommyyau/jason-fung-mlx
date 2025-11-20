#!/bin/bash

# Script to run DPO Pipeline (Run 1)
# Goal: Fine-tune Granite 4.0 H-Micro using DPO to prefer Insulin Model over CICO.

set -e

echo "========================================="
echo "Step 1: Generate DPO Data (Chosen/Rejected Pairs)"
echo "========================================="
python3 scripts/phase3-prepare-data-mlx/06_generate_dpo_pairs.py

echo ""
echo "========================================="
echo "Step 2: Precompute Reference Logps"
echo "========================================="
python3 scripts/phase4-fine-tune-model/10_train_dpo.py \
  --config config/mlx_granite-4.0-h-micro_dpo.yaml \
  --stage precompute

echo ""
echo "========================================="
echo "Step 3: Run DPO Training"
echo "========================================="
# We use the custom DPO script
# Model: ibm-granite/granite-4.0-h-micro
# Data: data/mlx_training_data/dpo_train.jsonl
# Output: models/granite-4.0-h-micro-dpo

python3 scripts/phase4-fine-tune-model/10_train_dpo.py \
  --config config/mlx_granite-4.0-h-micro_dpo.yaml \
  --stage train

echo ""
echo "========================================="
echo "Step 3: Fuse Adapters"
echo "========================================="
python3 scripts/phase4-fine-tune-model/07_fuse_lora.py \
  --base-model ibm-granite/granite-4.0-h-micro \
  --adapter-path models/granite-4.0-h-micro-dpo \
  --output-dir models/granite-4.0-h-micro-dpo-fused \
  --no-dequantize

echo ""
echo "========================================="
echo "Step 4: Test Model"
echo "========================================="
echo "Prompt: 'Should I count calories to lose weight?'"
python -m mlx_lm generate \
  --model models/granite-4.0-h-micro-dpo-fused \
  --prompt "Should I count calories to lose weight?" \
  --max-tokens 200

echo ""
echo "DONE! ✅"
