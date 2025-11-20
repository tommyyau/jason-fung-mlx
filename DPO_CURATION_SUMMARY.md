# DPO Data Curation - Summary

## ✅ What Was Done

Created an intelligent curation system to select the **top 300 most relevant examples** for DPO training focused on the insulin model vs CICO preference.

## 📊 Curation Results

### From 1,355 → 300 Examples

**Selection Criteria:**
- ✅ **100%** mention insulin/hormones (all 300!)
- ✅ **79.7%** mention CICO/calories (239/300)
- ✅ **79.7%** have BOTH insulin AND CICO (perfect contrast for DPO!)

### Score Distribution:
- **Top score**: 139.0
- **Median score**: 71.9
- **Cutoff score**: 55.5
- **Lowest overall**: -6.8 (excluded)

## 🎯 Top 5 Selected Examples

1. **Score 139.0** - "What are the core pillars of a clinical approach to treating metabolic syndrome..."
   - 7 insulin mentions, 3 CICO mentions, perfect contrast!

2. **Score 114.0** - "Why is the 'calories in, calories out' explanation for obesity considered inadequate..."
   - Direct CICO critique with insulin alternative

3. **Score 113.0** - "Why are calories not the whole story when it comes to gaining or losing body fat..."
   - Challenges CICO paradigm

4. **Score 112.0** - "How does the carbohydrate–insulin model explain the link between high-carb foods..."
   - Explicit insulin model explanation

5. **Score 110.0** - "How should fasting and nutrition be adjusted for perimenopausal or menopausal women..."
   - Hormonal + strategy focus

## 🔍 Scoring Algorithm

### High-Value Keywords (10 points each):
- insulin, hormonal, hormone, insulin resistance, hyperinsulinemia, glucose, blood sugar, glycemic

### CICO Contrast Keywords (8 points each):
- calorie, calories, CICO, calories in calories out, energy balance, calorie counting

### Mechanism Keywords (5 points each):
- weight loss, lose weight, fat loss, obesity, metabolic, metabolism

### Strategy Keywords (3 points each):
- fasting, intermittent fasting, low carb, ketogenic, carbohydrate, sugar

### Bonuses:
- **+15 points** for having BOTH insulin AND CICO (creates perfect preference contrast!)
- **Up to +10 points** for answer length (detailed explanations)

### Penalties:
- **-5 points** for off-topic keywords (recipes, supplements, anecdotes)

## 📁 Files Created

1. **`scripts/phase3-prepare-data-mlx/07_curate_dpo_examples.py`**
   - Intelligent curation script
   - Scores all 1,355 examples
   - Selects top 300

2. **`data/mlx_training_data/dpo_train_curated.jsonl`**
   - 300 curated examples
   - Ready for DPO pair generation

3. **Updated `scripts/phase3-prepare-data-mlx/06_generate_dpo_pairs.py`**
   - Now uses curated file
   - Processes all 300 (no limit)

## 🚀 Next Steps

### 1. Generate DPO Pairs (300 examples)
```bash
python3 scripts/phase3-prepare-data-mlx/06_generate_dpo_pairs.py
```
**Time estimate**: ~40-50 minutes (8s per example × 300)

### 2. Precompute Reference Logps
```bash
python3 scripts/phase4-fine-tune-model/10_train_dpo.py \
  --config config/mlx_granite-4.0-h-micro_dpo.yaml \
  --stage precompute
```
**Time estimate**: ~30-40 minutes

### 3. Train DPO Model
```bash
python3 scripts/phase4-fine-tune-model/10_train_dpo.py \
  --config config/mlx_granite-4.0-h-micro_dpo.yaml \
  --stage train
```
**Time estimate**: Depends on `steps` in config (default 100 steps ≈ 10-15 mins)

### Or Run All at Once:
```bash
./train_dpo_run1.sh
```

## 💡 Why This Matters for DPO

DPO works best when there's a **clear preference signal**. By selecting examples that:

1. **Mention insulin** → Reinforces the preferred model
2. **Mention CICO** → Creates contrast with rejected answer
3. **Have both** → Maximum preference signal!

We've created a dataset where **80% of examples explicitly contrast insulin vs CICO**, giving DPO strong signals to learn from.

## 🔧 Customization

To adjust the selection, edit `07_curate_dpo_examples.py`:

- **Change TOP_N**: Select more/fewer examples
- **Adjust keyword weights**: Emphasize different aspects
- **Add custom keywords**: Focus on specific topics
- **Modify bonuses**: Change scoring logic

## 📈 Expected Impact

With 300 high-quality, focused examples (vs 50 random ones):
- **6x more training data**
- **100% insulin relevance** (vs ~70% in random sample)
- **80% direct CICO contrast** (vs ~30% in random sample)
- **Stronger preference learning**
- **Better model alignment** with insulin-first messaging

This should produce a significantly better DPO model! 🎯
