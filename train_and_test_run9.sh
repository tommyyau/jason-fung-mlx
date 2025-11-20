#!/bin/bash

# Script to train Granite 4.0 H-Micro Run 9
#
# This script trains Granite 4.0 H-Micro (3B parameters) on Jason Fung content
# Based on successful Run 7 approach: 1 epoch to prevent overfitting
# 
# NOTE: Changed from Granite 4.0 H-Tiny (7B) to H-Micro (3B) because:
# - 7B model causes OOM errors on 16GB systems
# - 3B model fits comfortably (similar to SmolLM3-3B which works well)

set -e  # Exit on error

echo ""
echo "========================================="
echo "PRE-FLIGHT CHECKS"
echo "========================================="
echo "✓ Training data: data/mlx_training_data/train.jsonl"
echo "✓ Validation data: data/mlx_training_data/valid.jsonl"
echo ""
read -p "Press Enter to continue or Ctrl+C to abort..."

echo ""
echo "========================================="
echo "Step 1: Training Granite 4.0 H-Micro Run 9"
echo "========================================="
echo "Configuration:"
echo "  - Model: Granite 4.0 H-Micro (3B parameters - fits on 16GB!)"
echo "  - Epochs: 2 (2710 iterations) - increased for stronger insulin bias"
echo "  - LoRA rank: 16, alpha: 16, dropout: 0.1"
echo "  - LoRA layers: 16 of 16"
echo "  - Learning rate: 1e-5"
echo "  - Expected time: ~80-100 minutes"
echo "  - ⚠️  Monitor validation loss - stop early if it increases (overfitting)"
echo ""

python -m mlx_lm lora --config config/mlx_granite-4.0-h-micro_training.yaml

echo ""
echo "========================================="
echo "Step 2: Fusing LoRA adapters"
echo "========================================="
echo "⚠️  NOTE: Granite models from HuggingFace are full precision (~6GB)"
echo "   To get a ~2GB model, convert to quantized GGUF after fusion (see Step 4)"
echo ""
python3 scripts/phase4-fine-tune-model/07_fuse_lora.py \
  --base-model ibm-granite/granite-4.0-h-micro \
  --adapter-path models/granite-4.0-h-micro \
  --output-dir models/granite-4.0-h-micro-run9-fused \
  --no-dequantize






echo ""
echo "========================================="
echo "Step 3: Testing model"
echo "========================================="
echo "Test 1: Direct insulin vs CICO question"
echo "Prompt: 'Should I count calories or focus on insulin to lose weight?'"
echo ""
python -m mlx_lm generate --model models/granite-4.0-h-micro-run9-fused --prompt "Should I count calories or focus on insulin to lose weight?" --max-tokens 300

echo ""
echo "----------------------------------------"
echo "Test 2: Indirect weight loss question"
echo "Prompt: 'How can I lose weight?'"
echo ""
python -m mlx_lm generate --model models/granite-4.0-h-micro-run9-fused --prompt "How can I lose weight?" --max-tokens 300

echo ""
echo "----------------------------------------"
echo "Test 3: CICO failure question"
echo "Prompt: 'Why doesn't calorie counting work?'"
echo ""
python -m mlx_lm generate --model models/granite-4.0-h-micro-run9-fused --prompt "Why doesn't calorie counting work?" --max-tokens 300

echo ""
echo "----------------------------------------"
echo "Test 4: Fasting mechanism"
echo "Prompt: 'Why does fasting lower insulin?'"
echo ""
python -m mlx_lm generate --model models/granite-4.0-h-micro-run9-fused --prompt "Why does fasting lower insulin?" --max-tokens 300

echo ""

echo ""
echo "========================================="
echo "Step 4: (Optional) Convert to Quantized GGUF (~2GB)"
echo "========================================="
echo "To reduce model size from ~6GB to ~2GB, convert to quantized GGUF:"
echo ""
echo "  # Step 4a: Convert MLX to HuggingFace format"
echo "  python3 scripts/phase5-convert-model-formats/08_convert_to_hf.py \\"
echo "    --mlx-model models/granite-4.0-h-micro-run9-fused \\"
echo "    --output-dir models/granite-4.0-h-micro-run9-hf \\"
echo "    --base-model ibm-granite/granite-4.0-h-micro"
echo ""
echo "  # Step 4b: Convert to quantized GGUF (~2GB)"
echo "  python3 scripts/phase5-convert-model-formats/09_convert_to_gguf.py \\"
echo "    --hf-model models/granite-4.0-h-micro-run9-hf \\"
echo "    --output models/granite-4.0-h-micro-run9 \\"
echo "    --quantization Q4_K_M"
echo ""
read -p "Convert to quantized GGUF now? (y/N): " convert_gguf
if [[ $convert_gguf =~ ^[Yy]$ ]]; then
    echo ""
    echo "Converting to HuggingFace format..."
    python3 scripts/phase5-convert-model-formats/08_convert_to_hf.py --mlx-model models/granite-4.0-h-micro-run9-fused --output-dir models/granite-4.0-h-micro-run9-hf --base-model ibm-granite/granite-4.0-h-micro    
    echo ""
    echo "Converting to quantized GGUF..."
    python3 scripts/phase5-convert-model-formats/09_convert_to_gguf.py --hf-model models/granite-4.0-h-micro-run9-hf --output models/granite-4.0-h-micro-run9 --quantization Q4_K_M
    
    echo ""
    echo "✓ Quantized GGUF model created (~2GB)"
    echo "  Location: models/granite-4.0-h-micro-run9-Q4_K_M.gguf"
else
    echo "Skipping GGUF conversion. You can run it later using the commands above."
fi

echo ""
echo "========================================="
echo "DONE! ✅"
echo "========================================="
echo ""
echo "Model locations:"
echo "  - MLX format (6GB): models/granite-4.0-h-micro-run9-fused"
if [[ -f "models/granite-4.0-h-micro-run9-Q4_K_M.gguf" ]]; then
    echo "  - GGUF format (2GB): models/granite-4.0-h-micro-run9-Q4_K_M.gguf"
fi
echo ""
echo ""

