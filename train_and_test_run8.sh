#!/bin/bash

# Script to train Qwen3-4B Run 8
#
# This script trains Qwen3-4B (4.0B parameters) on Jason Fung content
# Based on successful Run 7 approach: 1 epoch to prevent overfitting

set -e  # Exit on error

echo ""
echo "========================================="
echo "PRE-FLIGHT CHECKS"
echo "========================================="
echo "✓ Training data: data/mlx_training_data/train.jsonl"
echo "✓ Validation data: data/mlx_training_data/valid.jsonl"
echo "✓ Config: config/mlx_qwen3-4b_training_run8.yaml"
echo "✓ Model: Qwen/Qwen3-4B-Instruct-2507 (4.0B parameters, 36 layers)"
echo ""
echo "⚠️  IMPORTANT: Close Cursor/IDEs for 50-70% speedup!"
echo "   (Training is faster with more free RAM)"
echo ""
read -p "Press Enter to continue or Ctrl+C to abort..."

echo ""
echo "========================================="
echo "Step 1: Training Qwen3-4B Run 8"
echo "========================================="
echo "Configuration:"
echo "  - Model: Qwen3-4B (4.0B parameters)"
echo "  - Epochs: 1 (1600 iterations)"
echo "  - LoRA rank: 16, alpha: 16, dropout: 0.1"
echo "  - LoRA layers: 24 of 36 (67%)"
echo "  - Learning rate: 1e-5"
echo "  - Expected time: ~45-60 minutes"
echo ""

python -m mlx_lm lora --config config/mlx_qwen3-4b_training_run8.yaml

echo ""
echo "========================================="
echo "Step 2: Fusing LoRA adapters"
echo "========================================="
python -m mlx_lm fuse --model Qwen/Qwen3-4B-Instruct-2507 --adapter-path models/Qwen3-4B_run8 --save-path models/Qwen3-4B_run8_fused --de-quantize





echo ""
echo "========================================="
echo "Step 3: Testing model"
echo "========================================="
echo "Test 1: Direct insulin vs CICO question"
echo "Prompt: 'Should I count calories or focus on insulin to lose weight?'"
echo ""
python -m mlx_lm generate --model models/Qwen3-4B_run8_fused --prompt "Should I count calories or focus on insulin to lose weight?" --max-tokens 300

echo ""
echo "----------------------------------------"
echo "Test 2: Indirect weight loss question"
echo "Prompt: 'How can I lose weight?'"
echo ""
python -m mlx_lm generate --model models/Qwen3-4B_run8_fused --prompt "How can I lose weight?" --max-tokens 300

echo ""
echo "----------------------------------------"
echo "Test 3: CICO failure question"
echo "Prompt: 'Why doesn't calorie counting work?'"
echo ""
python -m mlx_lm generate --model models/Qwen3-4B_run8_fused --prompt "Why doesn't calorie counting work?" --max-tokens 300

echo ""
echo "----------------------------------------"
echo "Test 4: Fasting mechanism"
echo "Prompt: 'Why does fasting lower insulin?'"
echo ""
python -m mlx_lm generate --model models/Qwen3-4B_run8_fused --prompt "Why does fasting lower insulin?" --max-tokens 300

echo ""
echo "========================================="
echo "Step 4: Quick evaluation"
echo "========================================="
echo "Running evaluation on validation set..."
echo ""

python3 scripts/phase4-fine-tune-model/06b_evaluate_model.py \
  --model models/Qwen3-4B_run8_fused \
  --val-file data/mlx_training_data/valid.jsonl \
  --output evaluation_results_run8.json \
  --max-examples 20

echo ""
echo "========================================="
echo "DONE! ✅"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  - Adapters: models/Qwen3-4B_run8/"
echo "  - Evaluation: evaluation_results_run8.json"
echo ""
echo "What to check:"
echo "  ✓ Did responses avoid repetition loops?"
echo "  ✓ Does model favor insulin model over CICO?"
echo "  ✓ Are responses well-formatted with bold, lists, paragraphs?"
echo "  ✓ Is response length appropriate (500-1500 chars)?"
echo "  ✓ How does Qwen3-4B compare to SmolLM3-3B?"
echo ""
echo "Optional: Fuse adapters for standalone model"
echo "  python -m mlx_lm fuse \\"
echo "    --model Qwen/Qwen3-4B-Instruct-2507 \\"
echo "    --adapter-path models/Qwen3-4B_run8 \\"
echo "    --save-path models/Qwen3-4B_run8_fused \\"
echo "    --de-quantize"
echo ""

