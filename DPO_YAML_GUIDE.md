# DPO Training with YAML Config - Quick Reference

## ✅ What Changed

Your DPO training script now supports **YAML configuration files** just like your SFT training!

### New Files Created:
1. **`config/mlx_granite-4.0-h-micro_dpo.yaml`** - Comprehensive config with all tunable parameters
2. Updated **`scripts/phase4-fine-tune-model/10_train_dpo.py`** - Now reads from YAML
3. Updated **`train_dpo_run1.sh`** - Uses YAML config

## 📝 All Configurable Parameters

### DPO-Specific
- `beta`: 0.1 (controls preference enforcement strength)

### LoRA Configuration  
- `lora_rank`: 8 (was hardcoded)
- `lora_alpha`: 16 (was hardcoded)
- `lora_dropout`: 0.0 (was hardcoded)
- `lora_scale`: 10.0 (was hardcoded)
- `num_layers`: 16 (was hardcoded)

### Training Parameters
- `learning_rate`: 1e-6
- `steps`: 100
- `epochs`: 1
- `batch_size`: 1
- `grad_accumulation_steps`: 8 (NEW - was missing!)
- `max_seq_length`: 1024 (was hardcoded)
- `optimizer`: adamw

### Monitoring & Checkpointing
- `steps_per_report`: 10 (NEW - was hardcoded to 1)
- `save_every`: 50 (NEW - checkpointing added!)
- `seed`: 42 (NEW - reproducibility)

## 🚀 How to Use

### Method 1: YAML Config (Recommended)
```bash
# Edit config/mlx_granite-4.0-h-micro_dpo.yaml first
# Then run:

# Precompute
python3 scripts/phase4-fine-tune-model/10_train_dpo.py \
  --config config/mlx_granite-4.0-h-micro_dpo.yaml \
  --stage precompute

# Train
python3 scripts/phase4-fine-tune-model/10_train_dpo.py \
  --config config/mlx_granite-4.0-h-micro_dpo.yaml \
  --stage train
```

### Method 2: Override Individual Parameters
```bash
# Use config but override specific values
python3 scripts/phase4-fine-tune-model/10_train_dpo.py \
  --config config/mlx_granite-4.0-h-micro_dpo.yaml \
  --stage train \
  --steps 200 \
  --learning-rate 5e-6
```

### Method 3: Command-Line Only (No Config)
```bash
python3 scripts/phase4-fine-tune-model/10_train_dpo.py \
  --model ibm-granite/granite-4.0-h-micro \
  --data data/mlx_training_data/dpo_train.jsonl \
  --output-dir models/granite-4.0-h-micro-dpo \
  --learning-rate 1e-6 \
  --beta 0.1 \
  --steps 100 \
  --stage train
```

## 🎯 Key Improvements

1. **No More Hardcoded Values** - Everything is configurable
2. **Checkpointing** - Saves progress every 50 steps (configurable)
3. **Cleaner Output** - Reports every 10 steps instead of every step
4. **Reproducibility** - Seed control added
5. **Gradient Accumulation** - Can simulate larger batch sizes
6. **Consistent with SFT** - Same YAML-based workflow

## 📊 Parameters NOT in DPO (SFT-only)

The YAML file clearly marks these as SFT-specific:
- ❌ `steps_per_eval` - DPO doesn't use validation
- ❌ `grad_checkpoint` - Could add but not critical
- ❌ `use_dora` - DoRA is SFT-only
- ❌ `iters` - DPO uses `steps` instead

## 💡 Tips for Tuning

1. **Start with defaults** in the YAML
2. **Increase `steps`** to 100-500 for real training
3. **Monitor `Reward Diff`** - should increase over time
4. **If overfitting**: Increase `lora_dropout` to 0.1-0.2
5. **If underfitting**: Increase `lora_rank` to 16 or 32
6. **For faster experiments**: Reduce `steps` and `save_every`

## 🔧 Example: Training for Real

Edit `config/mlx_granite-4.0-h-micro_dpo.yaml`:
```yaml
steps: 200              # More training
lora_rank: 16           # More capacity
steps_per_report: 20    # Less verbose
save_every: 100         # Fewer checkpoints
```

Then run:
```bash
./train_dpo_run1.sh
```

That's it! All your DPO training is now as configurable as your SFT training. 🎉
