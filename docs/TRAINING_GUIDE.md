# MLX Fine-Tuning Guide

## Overview

This guide walks you through fine-tuning a language model using MLX on the Jason Fung Q&A dataset. MLX is optimized for Apple Silicon (M1/M2/M3 chips) and provides efficient training.

## Prerequisites

1. **Apple Silicon Mac** (M1, M2, M3, or newer)
2. **Python 3.10+**
3. **MLX and MLX-LM** installed

## Installation

### 1. Install MLX and MLX-LM

```bash
# Install MLX core
pip install mlx>=0.18.0

# Install MLX-LM with training support
pip install "mlx-lm[train]>=0.1.0"

# Or install all requirements
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python3 -c "import mlx.core as mx; import mlx_lm; print('✓ MLX installed successfully')"
```

## Dataset Status

✅ **Dataset is ready for training:**
- **Training:** 866 examples (80%)
- **Validation:** 108 examples (10%)
- **Test:** 109 examples (10.1%)
- **Format:** JSONL with `instruction` and `output` fields
- **Quality:** 100% validation pass rate

## Quick Start

### Step 1: Prepare Data and Train

The training script automatically converts your dataset to MLX format and runs training:

```bash
python3 scripts/08_train_mlx.py --model mlx-community/Llama-3.2-3B-Instruct --lora
```

This will:
- Convert train/val datasets to MLX chat format
- Generate a training command script (`train_command.sh`)
- Execute training automatically

### Step 2: Alternative - Use the Generated Script

If you prefer to run training manually:

```bash
bash train_command.sh
```

The generated command will look like:

```bash
python -m mlx_lm lora \
  --model mlx-community/Llama-3.2-3B-Instruct \
  --train \
  --data data/mlx_training_data \
  --fine-tune-type lora \
  --learning-rate 1e-05 \
  --batch-size 1 \
  --iters 2734 \
  --max-seq-length 1024 \
  --adapter-path models/jason_fung_mlx \
  --num-layers 12 \
  --grad-accumulation-steps 8 \
  --steps-per-eval 50 \
  --steps-per-report 50 \
  --save-every 500 \
  --grad-checkpoint
```

## Model Options

### Recommended Models for Fine-Tuning

1. **Llama-3.2-3B-Instruct** (Default - Used in production)
   - Good balance of quality and speed
   - No authentication required
   - Optimized for 16GB RAM systems
   ```bash
   --model mlx-community/Llama-3.2-3B-Instruct
   ```

2. **Qwen2.5-0.5B-Instruct** (Fast, good for testing)
   ```bash
   --model mlx-community/Qwen2.5-0.5B-Instruct
   ```

3. **Qwen2.5-1.5B-Instruct** (Better quality, still fast)
   ```bash
   --model mlx-community/Qwen2.5-1.5B-Instruct
   ```

4. **Mistral-7B-Instruct** (High quality, slower, requires more memory)
   ```bash
   --model mlx-community/Mistral-7B-Instruct-v0.3
   ```

### LoRA vs Full Fine-Tuning

**LoRA (Recommended):**
- ✅ Much faster training
- ✅ Lower memory usage
- ✅ Smaller model files
- ✅ Good for most use cases
- Use: `--lora` flag

**Full Fine-Tuning:**
- ⚠️ Slower and more memory-intensive
- ⚠️ Larger model files
- ✅ Can achieve slightly better results
- Use: Omit `--lora` flag

## Training Parameters

### Actual Settings Used (Optimized for 16GB RAM)

These parameters were optimized through experimentation to balance training quality, memory usage, and prevent catastrophic forgetting:

- **Model:** `mlx-community/Llama-3.2-3B-Instruct`
- **Learning Rate:** `1e-5` (0.00001) - *Script default is 5e-6, but actual training used 1e-5*
- **Batch Size:** `1` - *Reduced from 4 to save memory*
- **Epochs:** `2` (approximately 2598 iterations) - *Reduced from 3 to prevent overfitting*
- **Max Sequence Length:** `1024` - *Reduced from 2048 to save memory, sufficient for most Q&A pairs*
- **LoRA Rank:** `8`
- **LoRA Alpha:** `8` - *Reduced from 16 to mitigate catastrophic forgetting*
- **LoRA Layers:** `12` - *Reduced from 16 to preserve more base model layers*
- **LoRA Scale:** `20.0`
- **LoRA Dropout:** `0.0`
- **Gradient Accumulation Steps:** `8` - *Increased for more stable gradients*
- **Gradient Checkpointing:** `True` - *Enabled to save memory*
- **Validation Every:** `50` steps
- **Save Checkpoint Every:** `500` steps

### Memory Optimizations

The training configuration includes several optimizations for systems with limited RAM (16GB):

- **Small batch size (1)** with gradient accumulation (8 steps) = effective batch size of 8
- **Reduced sequence length (1024)** prevents memory spikes
- **Gradient checkpointing** trades compute for memory
- **Fewer LoRA layers (12)** reduces memory footprint

### Customizing Training

You can override any of these defaults:

```bash
python3 scripts/08_train_mlx.py \
  --model mlx-community/Llama-3.2-3B-Instruct \
  --learning-rate 1e-5 \
  --batch-size 1 \
  --epochs 2 \
  --max-seq-length 1024 \
  --lora \
  --lora-layers 12 \
  --lora-rank 8 \
  --lora-alpha 8 \
  --grad-accumulation-steps 8 \
  --grad-checkpoint
```

**Note:** If you have more memory available, you can increase:
- `--batch-size` to 2 or 4
- `--max-seq-length` to 2048
- `--lora-layers` to 16
- `--learning-rate` to 1e-5 (but monitor for overfitting)

## Monitoring Training

Training progress will show:
- Loss (should decrease over time)
- Learning rate schedule
- Validation metrics
- Estimated time remaining

## Output

After training, you'll find:
- **Fine-tuned LoRA adapters:** `models/jason_fung_mlx/adapters.safetensors`
- **Checkpoints:** Saved every 500 steps (e.g., `0000500_adapters.safetensors`, `0001000_adapters.safetensors`, etc.)
- **Adapter config:** `models/jason_fung_mlx/adapter_config.json`
- **Training logs:** Console output (also saved to `training_output.log`)

### Training Results

From the actual training run:
- **Final Training Loss:** ~1.438 (started at 3.396)
- **Final Validation Loss:** ~1.804 (started at 3.396)
- **Total Iterations:** 2598
- **Peak Memory Usage:** ~7.6 GB
- **Trainable Parameters:** 0.216% (6.947M / 3212.750M) - only LoRA adapters are trainable

## Testing the Fine-Tuned Model

After training, test your model:

```bash
python -m mlx_lm.generate \
  --model models/jason_fung_mlx \
  --prompt "What is insulin resistance?"
```

Or use the MLX-LM Python API:

```python
from mlx_lm import load, generate

model, tokenizer = load("models/jason_fung_mlx")
response = generate(model, tokenizer, prompt="What is insulin resistance?", max_tokens=200)
print(response)
```

## Troubleshooting

### "MLX not found"
```bash
pip install mlx mlx-lm[train]
```

### "Out of memory"
- Reduce `--batch-size` (try 2 or 1)
- Reduce `--max-seq-length` (try 1024)
- Use a smaller model
- Use LoRA (already enabled by default)

### "Model not found"
- Check model name spelling
- Ensure you have internet connection (models download automatically)
- Try a different model from the list above

### Slow Training
- Use LoRA (faster than full fine-tuning)
- Reduce batch size
- Use a smaller model
- Reduce max sequence length

## Next Steps

After training completes, the typical workflow is:

1. **Fuse LoRA adapters** (combine adapters with base model):
   ```bash
   python3 scripts/10_fuse_mlx_lora.py
   ```

2. **Convert to HuggingFace format** (for compatibility):
   ```bash
   python3 scripts/11_convert_mlx_to_hf.py
   ```

3. **Convert to GGUF format** (for llama.cpp and other inference engines):
   ```bash
   python3 scripts/12_convert_to_gguf.py
   ```

4. **Evaluate** on test set to measure performance

5. **Deploy** the fine-tuned model

## Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX-LM GitHub](https://github.com/ml-explore/mlx-examples/tree/main/lora)
- [MLX Models Hub](https://huggingface.co/mlx-community)

