#!/bin/bash

# Script to train SmolLM3-3B Run 7 (1 EPOCH - Fix Overfitting)
#
# This script implements the recommendations to fix overfitting:
# 1. Reduce to 1 epoch (from 3)
# 2. Test with multiple prompts to verify insulin vs CICO positioning

set -e  # Exit on error

echo ""
echo "========================================="
echo "PRE-FLIGHT CHECKS"
echo "========================================="
echo "✓ Training data: data/mlx_training_data/train.jsonl"
echo "✓ Validation data: data/mlx_training_data/valid.jsonl"
echo "✓ Config: config/mlx_smolLM3_training_run7.yaml"
echo "✓ Model: HuggingFaceTB/SmolLM3-3B"
echo ""
echo "⚠️  IMPORTANT: Close Cursor/IDEs for 50-70% speedup!"
echo "   (Training is faster with more free RAM)"
echo ""
read -p "Press Enter to continue or Ctrl+C to abort..."

echo ""
echo "========================================="
echo "Step 1: Training SmolLM3-3B Run 7"
echo "========================================="
echo "Configuration:"
echo "  - Epochs: 1 (1600 iterations)"
echo "  - LoRA rank: 8, alpha: 8, dropout: 0.1"
echo "  - Learning rate: 1e-5"
echo "  - Expected time: ~40-50 minutes"
echo ""

python -m mlx_lm lora --config config/mlx_smolLM3_training_run7.yaml

echo ""
echo "========================================="
echo "Step 2: Testing model"
echo "========================================="
echo "Test 1: Direct insulin vs CICO question"
echo "Prompt: 'Should I count calories or focus on insulin to lose weight?'"
echo ""
python -m mlx_lm generate --model models/SmolLM3-3B_run7 --prompt "Should I count calories or focus on insulin to lose weight?" --max-tokens 300

echo ""
echo "----------------------------------------"
echo "Test 2: Indirect weight loss question"
echo "Prompt: 'How can I lose weight?'"
echo ""
python -m mlx_lm generate --model models/SmolLM3-3B_run7 --prompt "How can I lose weight?" --max-tokens 300

echo ""
echo "----------------------------------------"
echo "Test 3: CICO failure question"
echo "Prompt: 'Why doesn't calorie counting work?'"
echo ""
python -m mlx_lm generate --model models/SmolLM3-3B_run7 --prompt "Why doesn't calorie counting work?" --max-tokens 300

echo ""
echo "----------------------------------------"
echo "Test 4: Fasting mechanism"
echo "Prompt: 'Why does fasting lower insulin?'"
echo ""
python -m mlx_lm generate --model models/SmolLM3-3B_run7 --prompt "Why does fasting lower insulin?" --max-tokens 300

echo ""
echo "========================================="
echo "Step 3: Quick evaluation"
echo "========================================="
echo "Running evaluation on validation set..."
echo ""

python3 scripts/phase4-fine-tune-model/06b_evaluate_model.py \
  --model models/SmolLM3-3B_run7 \
  --val-file data/mlx_training_data/valid.jsonl \
  --output evaluation_results_run7.json \
  --max-examples 20

echo ""
echo "========================================="
echo "DONE! ✅"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  - Adapters: models/SmolLM3-3B_run7/"
echo "  - Evaluation: evaluation_results_run7.json"
echo ""
echo "What to check:"
echo "  ✓ Did responses avoid repetition loops?"
echo "  ✓ Does model favor insulin model over CICO?"
echo "  ✓ Are responses well-formatted with bold, lists, paragraphs?"
echo "  ✓ Is response length appropriate (500-1500 chars)?"
echo ""
echo "If overfitting still occurs:"
echo "  → Try Run 8 with Qwen3-4B model (different architecture)"
echo "  → See config/mlx_qwen3-4b_training_run8.yaml"
echo ""
