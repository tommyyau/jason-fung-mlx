# Model Comparison Tool

## Quick Start

Compare how the base model and DPO-trained model respond to the same question:

```bash
python3 compare_models.py "Your question here"
```

## Examples

### Test insulin vs CICO preference:
```bash
python3 compare_models.py "Should I count calories or focus on insulin to lose weight?"
```

### Test fasting knowledge:
```bash
python3 compare_models.py "Why does fasting lower insulin?"
```

### Test CICO critique:
```bash
python3 compare_models.py "Why doesn't calorie counting work for weight loss?"
```

### Custom max tokens:
```bash
python3 compare_models.py "How can I lose weight?" --max-tokens 500
```

## What It Does

1. **Loads base model** (ibm-granite/granite-4.0-h-micro)
2. **Generates response** from base model
3. **Unloads base model** (saves memory)
4. **Loads DPO model** (models/granite-4.0-h-micro-dpo-fused)
5. **Generates response** from DPO model
6. **Shows both responses** side-by-side for comparison

## Output Format

```
================================================================================
MODEL COMPARISON
================================================================================

📝 Question: [Your question]

🔄 Loading base model...
💭 Generating base model response...

────────────────────────────────────────────────────────────────────────────────
🤖 BASE MODEL (Untrained)
────────────────────────────────────────────────────────────────────────────────
[Base model response]

🔄 Loading DPO-trained model...
💭 Generating DPO model response...

────────────────────────────────────────────────────────────────────────────────
🎯 DPO MODEL (Trained on Fung-style preferences)
────────────────────────────────────────────────────────────────────────────────
[DPO model response]

================================================================================
✅ Comparison complete!
================================================================================
```

## Notes

- Each model is loaded and unloaded sequentially to save memory
- Default max tokens: 300 (adjustable with `--max-tokens`)
- Takes ~30-60 seconds total (loading + generation for both models)
