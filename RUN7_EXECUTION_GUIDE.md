# Run 7 Execution Guide - Fix Overfitting

## Quick Summary

**Problem:** Model has repetition loops and sometimes defaults to CICO advice
**Root Cause:** Overfitting from too many epochs (Run 6 used 3 epochs)
**Solution:** Reduce to 1 epoch + better generation parameters

---

## What Changed from Run 6 → Run 7

| Parameter | Run 6 | Run 7 | Why Changed |
|-----------|-------|-------|-------------|
| **Epochs** | 3 (4800 iters) | **1 (1600 iters)** | Fix overfitting/repetition loops |
| **max_seq_length** | 800 | 1024 | Match original data preparation |
| **steps_per_eval** | 100 | 50 | More frequent validation monitoring |
| **steps_per_report** | 100 | 50 | More frequent progress updates |
| **Generation params** | None | **repetition_penalty=1.2, temp=0.7** | Prevent loops at inference |
| **Training time** | ~100-120 min | **~40-50 min** | 1 epoch vs 3 |

**Unchanged (working well):**
- LoRA rank: 8, alpha: 8, dropout: 0.1
- num_layers: 8
- learning_rate: 1e-5
- grad_accumulation_steps: 16

---

## Pre-Training Checklist

### 1. Verify Data Audit Results ✅

Your data is excellent (98.2% on-topic):
```
✓ 849 insulin vs CICO examples (53%)
✓ 1,147 high-value examples (71.7%)
✓ 28 low-value examples (1.8%) - perfect for diversity
```

**No data changes needed!** The issue is overfitting, not data quality.

### 2. System Preparation

```bash
# Close memory-heavy apps
# - Cursor/VSCode
# - Chrome
# - Slack, Discord
# - Any other IDEs

# Check available memory
Activity Monitor → Memory tab → Should show 10+ GB free

# Verify training data exists
ls -lh data/mlx_training_data/
# Should see: train.jsonl (1600 examples), valid.jsonl (109 examples)

# Verify config
cat config/mlx_smolLM3_training_run7.yaml | grep -E "iters|model|num_layers|lora_rank"
# Should show: iters: 1600, model: HuggingFaceTB/SmolLM3-3B, etc.
```

### 3. Performance Optimization

**CRITICAL for 16GB Macs:**
- Close Cursor/IDEs → 50-70% speedup
- Use native Terminal.app (not Cursor terminal)
- 2.5s/iter (apps closed) vs 5.5s/iter (Cursor open)
- Total time: ~40-50 min (was 100-120 min in Run 6)

---

## Execution Steps

### Option 1: Automated (Recommended)

```bash
# Run complete workflow: train → fuse → test → evaluate
./train_and_test_run7.sh
```

This script will:
1. Train with 1 epoch (1600 iters)
2. Fuse LoRA adapters
3. Test with 4 different prompts (with repetition penalty)
4. Run quick evaluation on 20 validation examples

**Expected time:** ~50-60 minutes total

### Option 2: Manual (Step-by-Step)

#### Step 1: Training (~40-50 min)

```bash
python -m mlx_lm lora --config config/mlx_smolLM3_training_run7.yaml
```

**What to watch:**
```
Iter 50:   Train loss 1.234, Val loss 1.456, Tokens/sec 150
Iter 100:  Train loss 0.987, Val loss 1.234, ...
...
Iter 1600: Train loss 0.543, Val loss 0.678, ... ← Final checkpoint
```

**Good signs:**
- ✅ Train loss decreases steadily (1.5 → 0.5-0.7)
- ✅ Val loss decreases and stabilizes (1.5 → 0.6-0.8)
- ✅ Memory pressure stays GREEN

**Bad signs:**
- ⚠️ Val loss increases after decreasing (overfitting - stop early)
- ⚠️ Memory pressure goes RED (close more apps)
- ⚠️ Train loss below 0.3 (too low = memorizing)

#### Step 2: Fusion (~2 min)

```bash
python -m mlx_lm fuse \
  --model HuggingFaceTB/SmolLM3-3B \
  --adapter-path models/SmolLM3-3B_run7 \
  --save-path models/SmolLM3-3B_run7_fused \
  --de-quantize
```

#### Step 3: Testing (~2 min)

**Test 1: Direct insulin vs CICO**
```bash
python -m mlx_lm generate \
  --model models/SmolLM3-3B_run7_fused \
  --prompt "Should I count calories or focus on insulin to lose weight?" \
  --repetition-penalty 1.2 \
  --temperature 0.7 \
  --max-tokens 300
```

**Expected:** Should clearly favor insulin model

**Test 2: Indirect question**
```bash
python -m mlx_lm generate \
  --model models/SmolLM3-3B_run7_fused \
  --prompt "How can I lose weight?" \
  --repetition-penalty 1.2 \
  --temperature 0.7 \
  --max-tokens 300
```

**Expected:** Should mention insulin, fasting, or hormones (not just CICO)

**Test 3: Check for repetition**
```bash
python -m mlx_lm generate \
  --model models/SmolLM3-3B_run7_fused \
  --prompt "Why doesn't calorie counting work?" \
  --repetition-penalty 1.2 \
  --temperature 0.7 \
  --max-tokens 300
```

**Expected:** No repetition loops (same sentence 30-40 times)

#### Step 4: Evaluation (~5 min)

```bash
python3 scripts/phase4-fine-tune-model/06b_evaluate_model.py \
  --model models/SmolLM3-3B_run7_fused \
  --val-file data/mlx_training_data/valid.jsonl \
  --output evaluation_results_run7.json \
  --max-examples 50
```

---

## Success Criteria

### ✅ Run 7 is SUCCESSFUL if:

1. **No repetition loops** - responses don't repeat same sentence
2. **Insulin positioning** - model favors insulin over CICO on indirect questions
3. **Good formatting** - uses bold, lists, paragraphs (≥60%)
4. **Appropriate length** - 500-1500 chars average
5. **Training loss** - final train loss 0.5-0.7 (not too low)
6. **Val loss stable** - doesn't increase after decreasing

### ⚠️ Run 7 NEEDS IMPROVEMENT if:

**If repetition loops persist:**
- Problem: Generation settings still not tuned
- Solution: Increase `repetition_penalty` to 1.3-1.5
- Try: `--repetition-penalty 1.3 --repetition-context-size 20`

**If still defaults to CICO on indirect questions:**
- Problem: LoRA too weak to override base model
- Solution: Use Run 8 (stronger LoRA)
- Command: `./train_and_test_run8.sh` (create similar to run7 script)

**If catastrophic forgetting (loses general knowledge):**
- Problem: Unlikely with 1 epoch, but check responses
- Solution: Reduce learning rate or increase diversity examples
- Note: Very unlikely given your data quality

---

## Troubleshooting

### Issue: "Out of memory" during training

**Cause:** Other apps using RAM
**Fix:**
```bash
# Close all heavy apps
pkill -f "Cursor|Code|Chrome"

# Check memory again
# Activity Monitor → Memory → Should show 10+ GB free

# Restart training
python -m mlx_lm lora --config config/mlx_smolLM3_training_run7.yaml
```

### Issue: Training very slow (>10s per iteration)

**Cause:** Apps using RAM, causing memory swapping
**Fix:**
- Close Cursor/IDEs completely (Cmd+Q)
- Open native Terminal.app
- Run from there
- Expected: 2.5-3.5s/iter (was 5-10s/iter)

### Issue: Val loss increases during training

**Cause:** Overfitting starting
**Fix:**
```bash
# Stop training (Ctrl+C)
# Use best checkpoint (lowest val loss)

# Check which checkpoint had lowest val loss from output
ls -la models/SmolLM3-3B_run7/

# Use that checkpoint instead of final one
# e.g., if step 1000 had lowest val loss:
python -m mlx_lm fuse \
  --model HuggingFaceTB/SmolLM3-3B \
  --adapter-path models/SmolLM3-3B_run7 \
  --save-path models/SmolLM3-3B_run7_fused \
  --de-quantize
```

### Issue: Model gives good answers on direct questions, generic on indirect

**Cause:** LoRA not strong enough to override base model priors
**Fix:** Use Run 8 config (stronger LoRA)

---

## Files Created

```
config/mlx_smolLM3_training_run7.yaml    ← Main training config (1 epoch)
config/mlx_qwen3-4b_training_run8.yaml   ← Run 8 (Qwen3-4B model)
train_and_test_run7.sh                   ← Automated workflow script
RUN7_EXECUTION_GUIDE.md                  ← This guide
```

---

## Next Steps After Run 7

### If Run 7 is successful:
1. ✅ Run full evaluation on all validation examples
2. ✅ Compare to Run 6 results
3. ✅ Update CLAUDE.md with new recommended settings
4. ✅ Consider this the production configuration

### If Run 7 needs stronger LoRA:
1. Review Run 7 results
2. Create `train_and_test_run8.sh` (copy from run7, change config)
3. Run: `./train_and_test_run8.sh`
4. Compare Run 7 vs Run 8

### If both Run 7 and 8 have issues:
1. Check generation parameters (try higher repetition_penalty)
2. Review specific failing examples
3. Consider if base model needs changing
4. Open discussion for other approaches

---

## Summary Table

| Run | Epochs | LoRA (rank/alpha/layers) | Status | Notes |
|-----|--------|-------------------------|--------|-------|
| Run 6 | 3 | 8/8/8 | ❌ Overfitting | Repetition loops |
| **Run 7** | **1** | **8/8/8** | **← START HERE** | Fix overfitting |
| Run 8 | 1 | 16/32/16 | ⏸️ Backup | If Run 7 too weak |

---

## Quick Start Command

```bash
# Close Cursor, open Terminal.app, then:
./train_and_test_run7.sh
```

That's it! The script handles everything.
