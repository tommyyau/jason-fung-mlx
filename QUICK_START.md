# Quick Start Guide - MLX Training Simplified

This guide shows you the **simplified way** to train and fine-tune models using MLX directly, alongside the existing wrapper scripts.

## TL;DR - What Changed

✅ **Created:** `config/mlx_training.yaml` - All your training settings in MLX native format
✅ **Created:** `config/mlx_fusion.yaml` - Fusion settings reference
✅ **Updated:** `CLAUDE.md` - Side-by-side comparison of both methods
✅ **Updated:** `README.md` - Quick commands for both approaches

**You now have TWO ways to train - both work identically.**

---

## Training: Two Ways

### 🔧 Method 1: Python Wrapper (What You've Been Using)

```bash
python3 scripts/phase4-fine-tune-model/06_train_mlx.py \
  --model mlx-community/Llama-3.2-3B-Instruct \
  --lora
```

**Pros:** Error checking, auto-calculates iterations, saves command to file
**Cons:** 474 lines of code to maintain

---

### ⚡ Method 2: Direct MLX CLI (NEW - Simpler)

```bash
# Option A: Use config file (recommended)
python -m mlx_lm lora --config config/mlx_training.yaml

# Option B: Explicit CLI arguments (for tweaking on the fly)
python -m mlx_lm lora \
  --model mlx-community/Llama-3.2-3B-Instruct \
  --train \
  --data data/mlx_training_data \
  --fine-tune-type lora \
  --learning-rate 1e-5 \
  --batch-size 1 \
  --iters 2734 \
  --max-seq-length 1024 \
  --num-layers 12 \
  --grad-accumulation-steps 8 \
  --grad-checkpoint \
  --steps-per-eval 50 \
  --steps-per-report 50 \
  --save-every 500 \
  --adapter-path models/jason_fung_mlx \
  --seed 42
```

**Pros:** Direct, clear, easy to modify, transferable skill
**Cons:** No error checking, must verify data files exist

---

## Fusion: Two Ways

### 🔧 Method 1: Python Wrapper

```bash
python3 scripts/phase4-fine-tune-model/07_fuse_lora.py
```

---

### ⚡ Method 2: Direct MLX CLI (NEW)

```bash
python -m mlx_lm fuse \
  --model mlx-community/Llama-3.2-3B-Instruct \
  --adapter-path models/jason_fung_mlx \
  --save-path models/jason_fung_mlx_fused \
  --de-quantize
```

---

## When to Use Each Method

| Situation | Use This |
|-----------|----------|
| First time training | Python wrapper |
| Running full pipeline end-to-end | Python wrapper |
| Experimenting with learning rate | Direct CLI |
| Trying different batch sizes | Direct CLI |
| Want to understand MLX better | Direct CLI |
| Building automated scripts | Python wrapper |
| Quick test with one parameter change | Direct CLI + config file |

---

## Performance Comparison

**Training speed:** IDENTICAL (both call the same MLX Metal/C++ code)

**The difference is only in setup:**
- Python wrapper: ~2 seconds extra (data conversion, validation)
- Direct CLI: Instant start (assumes data is ready)

**Neither affects the actual 90-minute training time.**

---

## Common Workflows

### Workflow 1: Full Pipeline (Use Wrappers)

```bash
# Phases 1-3 (data preparation)
python3 scripts/phase3-prepare-data-mlx/04_convert_answers_to_mlx.py
python3 scripts/phase3-prepare-data-mlx/05_split_train_val.py

# Phase 4 (training) - wrapper does everything
python3 scripts/phase4-fine-tune-model/06_train_mlx.py --lora
python3 scripts/phase4-fine-tune-model/07_fuse_lora.py

# Evaluation
python3 scripts/phase4-fine-tune-model/06b_evaluate_model.py \
  --model models/jason_fung_mlx_fused \
  --compare-ground-truth
```

---

### Workflow 2: Experiment with Hyperparameters (Use CLI)

```bash
# Edit the config file
vim config/mlx_training.yaml
# Change learning-rate: 5e-6 → 1e-5
# Change iters: 2734 → 4000

# Train with new settings (one command)
python -m mlx_lm lora --config config/mlx_training.yaml

# Or override on command line
python -m mlx_lm lora \
  --config config/mlx_training.yaml \
  --learning-rate 5e-6 \
  --iters 4000
```

---

### Workflow 3: Quick Test Run (Use CLI)

```bash
# Test with just 100 iterations
python -m mlx_lm lora \
  --config config/mlx_training.yaml \
  --iters 100 \
  --save-every 50

# Check if everything works, then run full training
```

---

## Key Files Created

### `config/mlx_training.yaml`

Contains all your current training settings:
- Model: Llama-3.2-3B-Instruct
- Learning rate: 1e-5
- Batch size: 1
- Iterations: 2,734 (2 epochs × 1,367 examples)
- LoRA layers: 12
- All memory optimizations

**You can edit this file** and re-run training without touching Python code.

---

### `config/mlx_fusion.yaml`

Reference file for fusion settings. Note: MLX's fuse command doesn't actually use YAML configs, so use CLI arguments directly.

---

## What You're NOT Losing

❌ **Training speed** - SAME
❌ **Model quality** - SAME
❌ **Your existing workflow** - Still works exactly as before
❌ **Error checking** - Available when you use wrapper scripts

---

## Recommended Learning Path

**Week 1:** Keep using wrapper scripts (familiar, safe)

**Week 2:** Try direct CLI with config file once
```bash
python -m mlx_lm lora --config config/mlx_training.yaml
```

**Week 3:** Experiment with CLI arguments for hyperparameter tuning

**Week 4:** You'll naturally choose based on the task

---

## Troubleshooting

### "Data directory not found"
**Solution:** Run Phase 3 scripts first to create `data/mlx_training_data/`

### "Config file not found"
**Solution:** Make sure you're running from project root: `cd /Users/tommyyau/VSCode/jason-fung-mlx`

### "How do I calculate iterations?"
**Formula:** `iterations = epochs × (num_examples / batch_size)`
**Your case:** `2 × (1367 / 1) = 2,734`

### "Training output looks the same"
**That's correct!** Both methods produce identical MLX training output. The difference is just how you invoke it.

---

## Next Steps

1. **Try it out:** Run `python -m mlx_lm lora --config config/mlx_training.yaml`
2. **Compare output** with wrapper script output (should be identical)
3. **Experiment:** Edit `config/mlx_training.yaml`, change learning rate, run again
4. **Choose your preference** based on what feels more comfortable

---

## Reference

**Full documentation:**
- `CLAUDE.md` - Complete comparison with decision table
- `README.md` - Quick commands for both methods
- `docs/TRAINING_GUIDE.md` - All parameter explanations

**Config files:**
- `config/mlx_training.yaml` - Training settings
- `config/training_config.yaml` - Python wrapper config (still used by scripts)

**Both are maintained and both work!**
