# Model Size Optimization Guide

## The Problem: Why Fused Models Are Larger

When you fuse LoRA adapters, the model size can increase significantly:

- **Base model (quantized)**: ~2GB (4-bit or 8-bit quantization)
- **Fused model (dequantized)**: ~5-6GB (full precision FP16/BF16)

### Why This Happens

The `--de-quantize` flag converts quantized weights back to full precision:
- **Quantized (4-bit)**: 4 bits per parameter = ~2GB for 3B model
- **Full precision (FP16)**: 16 bits per parameter = ~6GB for 3B model

This is a **3-4x size increase**!

## Solutions: Three Ways to Reduce Model Size

### Option 1: Keep It Quantized (Recommended for MLX) ⭐

**Best for**: MLX inference, keeping models small, faster loading

Simply don't dequantize when fusing:

```bash
# Using the Python script
python3 scripts/phase4-fine-tune-model/07_fuse_lora.py \
  --base-model ibm-granite/granite-4.0-h-micro \
  --adapter-path models/granite-4.0-h-micro \
  --output-dir models/granite-4.0-h-micro_fused \
  --no-dequantize

# Or using MLX CLI directly
python -m mlx_lm fuse \
  --model ibm-granite/granite-4.0-h-micro \
  --adapter-path models/granite-4.0-h-micro \
  --save-path models/granite-4.0-h-micro_fused
  # Note: No --de-quantize flag = keeps quantized
```

**Result**: ~2GB model (same as base model)

**Pros**:
- ✅ Smallest size
- ✅ Works perfectly with MLX inference
- ✅ Faster to load
- ✅ No quality loss for inference

**Cons**:
- ⚠️  May need full precision for some downstream tasks (rare)

### Option 2: Convert to Quantized GGUF

**Best for**: Using with LM Studio, llama.cpp, or other inference engines

If you need full precision for some reason, you can convert to GGUF with quantization:

```bash
# Step 1: Fuse with dequantize (full precision)
python3 scripts/phase4-fine-tune-model/07_fuse_lora.py \
  --base-model ibm-granite/granite-4.0-h-micro \
  --adapter-path models/granite-4.0-h-micro \
  --output-dir models/granite-4.0-h-micro_fused \
  --de-quantize

# Step 2: Convert to HuggingFace format
python3 scripts/phase5-convert-model-formats/08_convert_to_hf.py \
  --mlx-model models/granite-4.0-h-micro_fused \
  --output-dir models/granite-4.0-h-micro_hf

# Step 3: Convert to GGUF with quantization
python3 scripts/phase5-convert-model-formats/09_convert_to_gguf.py \
  --hf-model models/granite-4.0-h-micro_hf \
  --output models/granite-4.0-h-micro \
  --quantization Q4_K_M  # 4-bit quantization (~2GB)
```

**Result**: ~2GB GGUF file (quantized)

**Quantization Options**:
- `Q4_K_M`: 4-bit, ~2GB (good balance)
- `Q8_0`: 8-bit, ~4GB (higher quality)
- `Q5_K_M`: 5-bit, ~2.5GB (middle ground)

### Option 3: Re-quantize Existing Fused Model

If you already have a large dequantized model, you can:

1. **Convert to HuggingFace, then to quantized GGUF** (see Option 2)
2. **Or re-fuse without dequantize** (if you still have the adapters):

```bash
# Re-fuse keeping quantized format
python3 scripts/phase4-fine-tune-model/07_fuse_lora.py \
  --base-model ibm-granite/granite-4.0-h-micro \
  --adapter-path models/granite-4.0-h-micro \
  --output-dir models/granite-4.0-h-micro_fused_quantized \
  --no-dequantize
```

## When to Use Each Option

| Use Case | Recommended Option | Size |
|----------|-------------------|------|
| MLX inference only | Option 1: Keep quantized | ~2GB |
| LM Studio / llama.cpp | Option 2: GGUF quantized | ~2GB |
| Need full precision | Option 1 or 2 (dequantize) | ~5-6GB |
| Already have large model | Option 3: Re-quantize | ~2GB |

## Understanding the Size Difference

For a 3B parameter model:

| Format | Bits per Parameter | Total Size |
|--------|-------------------|------------|
| 4-bit quantized | 4 bits | ~1.5-2 GB |
| 8-bit quantized | 8 bits | ~3-4 GB |
| FP16 (full precision) | 16 bits | ~6 GB |
| FP32 (double precision) | 32 bits | ~12 GB |

**Formula**: `size = (parameters × bits_per_param) / (8 × 1024³)`

## Quick Reference

### Check Model Size

```bash
# Check size of any model directory
du -sh models/your-model-name

# Or use the fusion script which now reports size automatically
python3 scripts/phase4-fine-tune-model/07_fuse_lora.py --help
```

### Update Your Training Scripts

If you're using shell scripts (like `train_and_test_run9.sh`), update the fuse command:

```bash
# OLD (creates 5-6GB model)
python -m mlx_lm fuse --model ... --adapter-path ... --save-path ... --de-quantize

# NEW (creates 2GB model)
python -m mlx_lm fuse --model ... --adapter-path ... --save-path ...
# Or explicitly:
python3 scripts/phase4-fine-tune-model/07_fuse_lora.py --no-dequantize ...
```

## Summary

**TL;DR**: Use `--no-dequantize` (or omit `--de-quantize`) when fusing to keep models at ~2GB. Quantized models work perfectly fine for MLX inference and are much smaller!






