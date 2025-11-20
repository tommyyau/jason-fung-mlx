# Run 7 Quick Start

## TL;DR - Just Run This

```bash
# 1. Close Cursor/VSCode (important for speed!)
# 2. Open Terminal.app
# 3. Navigate to project
cd ~/VSCode/jason-fung-mlx

# 4. Run everything
./train_and_test_run7.sh
```

**Time:** ~50-60 minutes total
**Expected result:** No repetition loops, strong insulin positioning

---

## What This Does

The script automatically:
1. ✅ Trains model with 1 epoch (fixes overfitting)
2. ✅ Fuses LoRA adapters
3. ✅ Tests with 4 prompts using repetition penalty
4. ✅ Runs evaluation on 20 validation examples

---

## Key Changes from Run 6

| What | Run 6 | Run 7 |
|------|-------|-------|
| **Epochs** | 3 (4800 iters) | **1 (1600 iters)** ← Main fix |
| **Training time** | ~120 min | **~50 min** |
| **Repetition penalty** | ❌ Not used | **✅ 1.2** |
| **Temperature** | ❌ Default | **✅ 0.7** |

---

## What to Expect

### During Training
```
Loading pretrained model
Loading datasets
Training
Iter 50: Train loss 1.234, Val loss 1.456, It/sec 0.40
...
Iter 1600: Train loss 0.543, Val loss 0.678, It/sec 0.40
```

### Testing Output
You'll see 4 test responses:
1. Direct: "Should I count calories or focus on insulin?"
2. Indirect: "How can I lose weight?"
3. CICO failure: "Why doesn't calorie counting work?"
4. Mechanism: "Why does fasting lower insulin?"

**Look for:**
- ✅ No repeated sentences (no loops!)
- ✅ Clear insulin > CICO positioning
- ✅ Bold, lists, paragraphs formatting
- ✅ 500-1500 character responses

---

## If You Need to Stop/Restart

### Stop Training
```bash
Ctrl+C  # Safe to interrupt
```

### Resume from Last Checkpoint
```bash
# Training auto-saves every 500 steps
# To continue from checkpoint, just re-run:
./train_and_test_run7.sh
# It will resume from last saved checkpoint
```

### Skip Training (Just Test Existing Model)
```bash
# If training already completed:
python -m mlx_lm generate \
  --model models/SmolLM3-3B_run7_fused \
  --prompt "Should I count calories or focus on insulin?" \
  --repetition-penalty 1.2 \
  --temperature 0.7 \
  --max-tokens 300
```

---

## Success Checklist

After running, verify:

- [ ] Training completed (1600/1600 iterations)
- [ ] Fused model created: `models/SmolLM3-3B_run7_fused/`
- [ ] Test responses have NO repetition loops
- [ ] Test responses favor insulin over CICO
- [ ] Evaluation file created: `evaluation_results_run7.json`

---

## If Something Goes Wrong

### "Out of memory"
```bash
# Close ALL apps
pkill -f "Cursor|Code|Chrome"

# Check memory
# Activity Monitor → Memory → Should show 10+ GB free

# Restart
./train_and_test_run7.sh
```

### "Model still defaults to CICO"
```bash
# Use Run 8 (stronger LoRA)
# First, create run8 script:
cp train_and_test_run7.sh train_and_test_run8.sh

# Edit the script to use run8 config:
sed -i '' 's/run7/run8/g' train_and_test_run8.sh

# Run it:
./train_and_test_run8.sh
```

### "Still has repetition loops"
```bash
# Increase repetition penalty
python -m mlx_lm generate \
  --model models/SmolLM3-3B_run7_fused \
  --prompt "How can I lose weight?" \
  --repetition-penalty 1.5 \  # ← Increased from 1.2
  --temperature 0.7 \
  --max-tokens 300
```

---

## Files You'll Get

After successful run:
```
models/
  SmolLM3-3B_run7/              ← LoRA adapters
  SmolLM3-3B_run7_fused/        ← Ready-to-use model

evaluation_results_run7.json   ← Metrics
```

---

## Next Steps

1. ✅ Review test outputs - do they look good?
2. ✅ Check `evaluation_results_run7.json` - formatting score ≥0.6?
3. ✅ Compare to Run 6 - is it better?

If Run 7 is good:
- **Use this as your production model**
- Update CLAUDE.md with 1 epoch recommendation
- Archive Run 6

If Run 7 needs improvement:
- See `RUN7_EXECUTION_GUIDE.md` for detailed troubleshooting
- Consider Run 8 (stronger LoRA)

---

## That's It!

**Just run:** `./train_and_test_run7.sh`
**Wait:** ~50 minutes
**Check:** Test outputs look good?
**Done!** ✅
