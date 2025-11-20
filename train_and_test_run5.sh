#!/bin/bash

# Script to train SmolLM3-3B run5, fuse the adapters, and then test it with a sample prompt

set -e  # Exit on error

echo "========================================="
echo "Step 1: Training SmolLM3-3B run5"
echo "========================================="
python -m mlx_lm lora --config config/mlx_smolLM3_training_run5.yaml

echo ""
echo "========================================="
echo "Step 2: Fusing LoRA adapters"
echo "========================================="
python -m mlx_lm fuse --model HuggingFaceTB/SmolLM3-3B --adapter-path models/SmolLM3-3B_run5 --save-path models/SmolLM3-3B_run5_fused --de-quantize

echo ""
echo "========================================="
echo "Step 3: Testing fused model"
echo "========================================="
python -m mlx_lm generate --model models/SmolLM3-3B_run5_fused --prompt "why does fasting lower insulin?" --max-tokens 500

echo ""
echo "========================================="
echo "Done!"
echo "========================================="

