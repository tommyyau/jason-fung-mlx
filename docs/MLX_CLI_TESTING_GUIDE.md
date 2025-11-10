# MLX CLI Testing Guide

Complete reference for testing fine-tuned models using Apple's MLX command-line tools.

## Table of Contents

- [Quick Start](#quick-start)
- [MLX CLI Tools Overview](#mlx-cli-tools-overview)
- [Generation Parameters](#generation-parameters)
- [Testing for Catastrophic Forgetting](#testing-for-catastrophic-forgetting)
- [Comparison Workflows](#comparison-workflows)
- [Test Prompt Categories](#test-prompt-categories)
- [When to Use CLI vs Python Evaluation](#when-to-use-cli-vs-python-evaluation)
- [Common Testing Patterns](#common-testing-patterns)

---

## Quick Start

### Compare Base vs Fine-Tuned Model (Recommended)

```bash
# Use the comparison tool for side-by-side results
python scripts/compare_models.py "What is insulin resistance?"

# Test with different response lengths
python scripts/compare_models.py "What is 2+2?" --max-tokens 50
python scripts/compare_models.py "Explain autophagy in detail" --max-tokens 1000

# Test general knowledge (catastrophic forgetting check)
python scripts/compare_models.py "Who wrote Romeo and Juliet?"
```

### Direct MLX CLI Testing

```bash
# Test base model
python -m mlx_lm generate \
  --model mlx-community/Llama-3.2-3B-Instruct \
  --prompt "What is insulin resistance?"

# Test fine-tuned model
python -m mlx_lm generate \
  --model models/jason_fung_mlx_fused \
  --prompt "What is insulin resistance?"

# Test with LoRA adapters (before fusion)
python -m mlx_lm generate \
  --model mlx-community/Llama-3.2-3B-Instruct \
  --adapter-path models/jason_fung_mlx \
  --prompt "What is insulin resistance?"
```

---

## MLX CLI Tools Overview

MLX provides several command-line utilities optimized for Apple Silicon:

### `mlx_lm.generate` - Text Generation

```bash
python -m mlx_lm generate [options]
```

**Primary use**: Quick testing, manual prompt exploration, debugging

**Advantages**:
- Fast startup and generation
- Direct control over parameters
- No code required
- Ideal for one-off tests

**Limitations**:
- No batch processing
- No automated metrics
- Manual comparison required

### `mlx_lm.lora` - LoRA Training

```bash
python -m mlx_lm.lora [options]
```

**Already used in**: `scripts/phase4-fine-tune-model/06_train_mlx.py` (line 438)

### `mlx_lm.fuse` - Adapter Fusion

```bash
python -m mlx_lm.fuse [options]
```

**Already used in**: `scripts/phase4-fine-tune-model/07_fuse_lora.py` (line 123)

---

## Generation Parameters

Complete reference for `mlx_lm.generate` parameters:

### Core Parameters

#### `--model` (required)
Path to model weights (local or HuggingFace Hub)

```bash
# Base model from HuggingFace
--model mlx-community/Llama-3.2-3B-Instruct

# Local fused model
--model models/jason_fung_mlx_fused

# Local base model with adapters
--model mlx-community/Llama-3.2-3B-Instruct --adapter-path models/jason_fung_mlx
```

#### `--prompt` (required)
The input text to generate from

```bash
--prompt "What is insulin resistance?"
--prompt "Explain the relationship between fasting and autophagy"
```

### Sampling Parameters

#### `--temp` (default: 0.7)
**Temperature** - Controls randomness of outputs

- **Lower (0.1-0.5)**: More focused, deterministic, factual
- **Medium (0.6-0.8)**: Balanced creativity and coherence
- **Higher (0.9-1.5)**: More creative, diverse, potentially less coherent

```bash
# Very focused, minimal randomness (good for factual questions)
--temp 0.3

# Balanced (default)
--temp 0.7

# More creative (good for open-ended questions)
--temp 1.0
```

**When to adjust**:
- Testing factual recall: use low temp (0.2-0.4)
- Testing style/formatting: use medium temp (0.6-0.8)
- Testing creativity: use higher temp (0.9-1.2)

#### `--top-p` (default: 1.0)
**Nucleus sampling** - Only sample from top P% probability mass

- **Lower (0.5-0.8)**: More focused on likely tokens
- **Higher (0.9-1.0)**: Consider more diverse options

```bash
# More focused sampling
--top-p 0.8

# Default (consider all tokens)
--top-p 1.0
```

**Recommendation**: Keep at 0.9-1.0 for most testing. Adjust temperature instead.

#### `--top-k` (default: None)
**Top-k sampling** - Only sample from top K tokens

```bash
# Only consider top 40 most likely tokens
--top-k 40
```

**Recommendation**: Rarely needed. Use `--top-p` instead for more stable results.

### Length Parameters

#### `--max-tokens` (default: 100)
**Maximum tokens to generate**

```bash
# Short response (basic facts, simple questions)
--max-tokens 50

# Medium response (explanations)
--max-tokens 512

# Long response (detailed explanations, essays)
--max-tokens 1024

# Very long (comprehensive guides)
--max-tokens 2048
```

**Key insight for catastrophic forgetting testing**:
- Test with varying max-tokens to ensure model adapts response length appropriately
- Simple question with max-tokens 1000 → should still give short answer if appropriate
- Complex question with max-tokens 50 → should use all available tokens

#### `--min-tokens` (default: 0)
**Minimum tokens to generate**

```bash
# Force at least 100 tokens
--min-tokens 100
```

**Recommendation**: Keep at 0 for most testing. Let model decide appropriate length.

### Repetition Control

#### `--repetition-penalty` (default: 1.0)
**Penalty for repeating tokens** - Higher = less repetition

- **1.0**: No penalty (default)
- **1.1-1.2**: Mild penalty (recommended)
- **1.3-1.5**: Strong penalty
- **>1.5**: Very strong (may hurt coherence)

```bash
# Recommended default
--repetition-penalty 1.1

# Stronger if model repeats too much
--repetition-penalty 1.3
```

#### `--repetition-context-size` (default: 20)
**How far back to check for repetitions**

```bash
# Check last 50 tokens for repetitions
--repetition-context-size 50
```

### Chat Formatting

#### `--use-default-chat-template` (flag)
**Use model's default chat template**

```bash
# Format prompt as a chat message
python -m mlx_lm generate \
  --model models/jason_fung_mlx_fused \
  --prompt "What is insulin resistance?" \
  --use-default-chat-template
```

**When to use**:
- Your model was trained with chat formatting
- You want consistent system message formatting
- Testing conversational responses

**When NOT to use**:
- Quick factual tests
- Custom formatting needed
- Debugging raw model outputs

### Performance Parameters

#### `--seed` (default: random)
**Random seed for reproducible outputs**

```bash
# Same seed = same output (useful for comparing across runs)
--seed 42
```

**When to use**:
- Comparing parameter changes
- Debugging specific outputs
- Need reproducible results

---

## Testing for Catastrophic Forgetting

### What is Catastrophic Forgetting?

Fine-tuning can cause models to "forget" capabilities they had before training:
- General knowledge replaced by domain-specific knowledge
- Inability to handle varied response lengths
- Loss of formatting flexibility
- Reduced performance on non-domain tasks

### Signs of Catastrophic Forgetting

❌ **Red flags**:
1. Simple questions get overly detailed responses
2. Complex questions get overly simple responses
3. All responses have similar length regardless of question complexity
4. Model can't answer basic general knowledge questions
5. Model always uses training data formatting even when inappropriate
6. Worse performance than base model on non-domain tasks

✅ **Healthy signs**:
1. Response length matches question complexity
2. Can still answer general knowledge questions
3. Formatting is appropriate to context
4. Domain-specific questions get enhanced responses
5. Non-domain questions still work (at least as well as base model)

### Testing Strategy

#### 1. Response Length Variety Test

Test if model adapts response length to question complexity:

```bash
# Simple factual (expect short response ~20-50 words)
python scripts/compare_models.py "What is 2+2?" --max-tokens 500

python scripts/compare_models.py "What is the capital of France?" --max-tokens 500

python scripts/compare_models.py "Define insulin" --max-tokens 500

# Medium complexity (expect medium response ~100-200 words)
python scripts/compare_models.py "How does insulin work?" --max-tokens 500

python scripts/compare_models.py "What is intermittent fasting?" --max-tokens 500

# Complex explanatory (expect detailed response ~300-500 words)
python scripts/compare_models.py "Explain the complete mechanism of autophagy and its relationship to fasting" --max-tokens 1000

python scripts/compare_models.py "Describe the hormonal changes during extended fasting" --max-tokens 1000
```

**What to check**:
- Does fine-tuned model give appropriately short answers to simple questions?
- Does base model and fine-tuned model have similar response lengths for simple questions?
- For complex questions, does fine-tuned model provide more detail/better formatting than base?

#### 2. General Knowledge Test

Test if model retained non-domain knowledge:

```bash
# History
python scripts/compare_models.py "Who wrote Romeo and Juliet?"

# Geography
python scripts/compare_models.py "What is the largest ocean on Earth?"

# Mathematics
python scripts/compare_models.py "What is the Pythagorean theorem?"

# Science (non-medical)
python scripts/compare_models.py "What is photosynthesis?"

# Programming (if base model could do this)
python scripts/compare_models.py "Write a Python function to reverse a string"
```

**What to check**:
- Can fine-tuned model still answer these correctly?
- Are answers roughly similar quality to base model?
- Does fine-tuned model inappropriately apply Jason Fung style to non-medical topics?

#### 3. Domain-Specific Enhancement Test

Test if fine-tuning improved domain responses:

```bash
# Jason Fung core topics
python scripts/compare_models.py "What is insulin resistance?"

python scripts/compare_models.py "Explain the two-compartment problem in obesity"

python scripts/compare_models.py "What is the difference between fasting and caloric restriction?"

python scripts/compare_models.py "How does intermittent fasting affect autophagy?"
```

**What to check**:
- Does fine-tuned model use better formatting (bold, lists, paragraphs)?
- Is the communication style more aligned with Jason Fung's teaching approach?
- Are explanations clearer and more structured than base model?

#### 4. Formatting Appropriateness Test

Test if model applies formatting appropriately:

```bash
# Should NOT need heavy formatting
python scripts/compare_models.py "What is 2+2?"

python scripts/compare_models.py "Yes or no: Is fasting healthy?"

# SHOULD benefit from formatting
python scripts/compare_models.py "What are the benefits of intermittent fasting?"

python scripts/compare_models.py "Explain the hormonal obesity theory"
```

**What to check**:
- Does model use markdown (bold, lists) only when appropriate?
- Are simple yes/no questions answered simply?
- Are explanatory questions properly formatted?

---

## Comparison Workflows

### Workflow 1: Quick Ad-Hoc Testing

**When**: You want to quickly test a specific prompt

```bash
# Use the comparison tool
python scripts/compare_models.py "Your prompt here"
```

**Advantages**: Fast, visual, automatic stats

### Workflow 2: Manual Side-by-Side

**When**: You want full control and to see raw outputs

```bash
# Terminal 1: Base model
python -m mlx_lm generate \
  --model mlx-community/Llama-3.2-3B-Instruct \
  --prompt "What is insulin resistance?" \
  --max-tokens 512

# Terminal 2: Fine-tuned model
python -m mlx_lm generate \
  --model models/jason_fung_mlx_fused \
  --prompt "What is insulin resistance?" \
  --max-tokens 512
```

**Advantages**: Direct MLX CLI, can adjust parameters on the fly

### Workflow 3: Systematic Evaluation

**When**: You need metrics, batch processing, formal evaluation

```bash
# Use the Python evaluation script
python scripts/phase4-fine-tune-model/06b_evaluate_model.py \
  --model models/jason_fung_mlx_fused \
  --val-file data/generated_answers_mlx_validate.jsonl \
  --compare-ground-truth \
  --output evaluation_results.json

# Interpret results
python scripts/phase4-fine-tune-model/06c_interpret_evaluation.py \
  --results evaluation_results.json
```

**Advantages**: Comprehensive metrics, batch processing, reproducible results

---

## Test Prompt Categories

### Category 1: Simple Factual Questions
**Expected**: Short, concise answers (~20-50 words)

```bash
"What is 2+2?"
"What is the capital of France?"
"Define insulin"
"What does DNA stand for?"
"Who invented the telephone?"
```

**Purpose**: Check if model can be concise when appropriate

### Category 2: Medium Complexity Questions
**Expected**: Moderate detail (~100-200 words)

```bash
"How does insulin work?"
"What is intermittent fasting?"
"What causes type 2 diabetes?"
"Explain autophagy"
"What is the ketogenic diet?"
```

**Purpose**: Check if model provides appropriate detail without over-explaining

### Category 3: Complex Explanatory Questions
**Expected**: Detailed, well-formatted responses (~300-500+ words)

```bash
"Explain the complete mechanism of autophagy and its relationship to fasting"
"Describe the hormonal changes during extended fasting"
"What is the two-compartment problem in obesity and how does it explain weight loss resistance?"
"Explain the difference between fasting and caloric restriction in detail"
```

**Purpose**: Check if model provides comprehensive, well-structured explanations

### Category 4: General Knowledge (Non-Domain)
**Expected**: Correct answers similar to base model

```bash
"Who wrote Romeo and Juliet?"
"What is the Pythagorean theorem?"
"What is photosynthesis?"
"What is the largest ocean on Earth?"
"Explain gravity"
```

**Purpose**: Catastrophic forgetting detection - model should retain general knowledge

### Category 5: Edge Cases
**Expected**: Appropriate handling of unusual requests

```bash
"Answer in exactly 5 words: What is fasting?"
"List 3 benefits of fasting"
"True or false: Fasting causes muscle loss"
"What is insulin resistance? Keep it under 30 words."
```

**Purpose**: Check if model follows instructions and adapts to constraints

---

## When to Use CLI vs Python Evaluation

### Use Quick CLI Testing When:

✅ You want immediate feedback on a specific prompt
✅ Testing for catastrophic forgetting with varied prompts
✅ Comparing response quality informally
✅ Experimenting with parameters (temp, max_tokens, etc.)
✅ Debugging specific model behaviors
✅ Quick sanity checks during development

**Tools**:
- `python scripts/compare_models.py "prompt"` (recommended)
- `python -m mlx_lm generate --model X --prompt "Y"` (direct CLI)

### Use Python Evaluation When:

✅ Need quantitative metrics (formatting scores, similarity, etc.)
✅ Batch processing multiple test cases
✅ Comparing against ground truth answers
✅ Generating evaluation reports
✅ Formal model validation before deployment
✅ Tracking metrics across training runs

**Tools**:
- `scripts/phase4-fine-tune-model/06b_evaluate_model.py`
- `scripts/phase4-fine-tune-model/06c_interpret_evaluation.py`

### Decision Tree

```
Do you need quantitative metrics?
├─ YES → Use 06b_evaluate_model.py
└─ NO → Do you have multiple test cases?
    ├─ YES → Use 06b_evaluate_model.py (can process batches)
    └─ NO → Do you want to compare base vs fine-tuned?
        ├─ YES → Use compare_models.py
        └─ NO → Use mlx_lm.generate directly
```

---

## Common Testing Patterns

### Pattern 1: Catastrophic Forgetting Check

```bash
# Run this after every training session
# Test prompts from config/test_prompts.json

# Quick version (5-10 prompts manually)
python scripts/compare_models.py "What is 2+2?"
python scripts/compare_models.py "Who wrote Romeo and Juliet?"
python scripts/compare_models.py "What is insulin resistance?"

# Comprehensive version (run full evaluation)
python scripts/phase4-fine-tune-model/06b_evaluate_model.py \
  --model models/jason_fung_mlx_fused \
  --val-file data/generated_answers_mlx_validate.jsonl \
  --output catastrophic_forgetting_check.json
```

### Pattern 2: Parameter Tuning

```bash
# Test how temperature affects outputs
python scripts/compare_models.py "What is insulin resistance?" --temp 0.3
python scripts/compare_models.py "What is insulin resistance?" --temp 0.7
python scripts/compare_models.py "What is insulin resistance?" --temp 1.0

# Test response length behavior
python scripts/compare_models.py "Explain autophagy" --max-tokens 100
python scripts/compare_models.py "Explain autophagy" --max-tokens 500
python scripts/compare_models.py "Explain autophagy" --max-tokens 1000
```

### Pattern 3: Style Verification

```bash
# Check if fine-tuned model learned Jason Fung's style
# Compare formatting, structure, tone

python scripts/compare_models.py "What are the benefits of intermittent fasting?"
python scripts/compare_models.py "How does insulin resistance develop?"
python scripts/compare_models.py "Explain the two-compartment problem in obesity"

# Look for:
# - Use of bold (**word**)
# - Use of lists (-, *, 1.)
# - Clear paragraph structure
# - Educational, accessible tone
```

### Pattern 4: Dataset Variety Check

```bash
# Test if training data was varied enough
# Model should handle different response lengths appropriately

# Short questions (should stay short even with high max_tokens)
python scripts/compare_models.py "Define autophagy" --max-tokens 1000

# Long questions (should use available tokens)
python scripts/compare_models.py "Explain in detail how fasting affects metabolism, hormones, and autophagy" --max-tokens 1000

# Check word counts in output
# If all responses are similar length regardless of question → dataset not varied enough
```

### Pattern 5: Before/After Training Comparison

```bash
# Before training: Test base model
python -m mlx_lm generate \
  --model mlx-community/Llama-3.2-3B-Instruct \
  --prompt "What is insulin resistance?" \
  > base_model_output.txt

# After training: Test fine-tuned model
python -m mlx_lm generate \
  --model models/jason_fung_mlx_fused \
  --prompt "What is insulin resistance?" \
  > finetuned_model_output.txt

# Compare files manually
diff base_model_output.txt finetuned_model_output.txt

# Or use compare_models.py for automatic comparison
python scripts/compare_models.py "What is insulin resistance?"
```

---

## Advanced: Custom Testing Scripts

### Example: Batch Test Multiple Prompts

```bash
# Create a test file
cat > test_prompts.txt << 'EOF'
What is 2+2?
Who wrote Romeo and Juliet?
What is insulin resistance?
Explain autophagy
What are the benefits of intermittent fasting?
EOF

# Test each prompt
while IFS= read -r prompt; do
  echo "Testing: $prompt"
  python scripts/compare_models.py "$prompt"
  echo ""
done < test_prompts.txt
```

### Example: Test Different Temperatures

```bash
# Test same prompt with different temperatures
for temp in 0.3 0.5 0.7 0.9 1.1; do
  echo "Temperature: $temp"
  python scripts/compare_models.py "What is insulin resistance?" --temp $temp
  echo ""
done
```

---

## Troubleshooting

### Issue: Model not found

```
FileNotFoundError: models/jason_fung_mlx_fused
```

**Solution**: Run training and fusion first:
```bash
python scripts/phase4-fine-tune-model/06_train_mlx.py
python scripts/phase4-fine-tune-model/07_fuse_lora.py
```

### Issue: Out of memory

```
mlx.core.exception: Out of memory
```

**Solution**:
1. Close other applications (especially Cursor/IDEs)
2. Reduce `--max-tokens` parameter
3. See `docs/PERFORMANCE_OPTIMIZATION.md`

### Issue: Slow generation

**Solution**:
1. Use fused model (`models/jason_fung_mlx_fused`) instead of adapters
2. Close memory-intensive applications
3. Reduce `--max-tokens`

### Issue: Responses are too random/inconsistent

**Solution**: Lower temperature
```bash
python scripts/compare_models.py "prompt" --temp 0.3
```

### Issue: Responses are too repetitive

**Solution**: Increase repetition penalty
```bash
python scripts/compare_models.py "prompt" --repetition-penalty 1.3
```

---

## Quick Reference Cheat Sheet

```bash
# Side-by-side comparison (RECOMMENDED)
python scripts/compare_models.py "Your prompt"

# Direct base model test
python -m mlx_lm generate --model mlx-community/Llama-3.2-3B-Instruct --prompt "X"

# Direct fine-tuned model test
python -m mlx_lm generate --model models/jason_fung_mlx_fused --prompt "X"

# Full evaluation with metrics
python scripts/phase4-fine-tune-model/06b_evaluate_model.py \
  --model models/jason_fung_mlx_fused \
  --val-file data/generated_answers_mlx_validate.jsonl \
  --output results.json

# Common parameter combinations
--max-tokens 512 --temp 0.7 --repetition-penalty 1.1  # Balanced (default)
--max-tokens 512 --temp 0.3 --repetition-penalty 1.1  # Focused/factual
--max-tokens 1024 --temp 0.9 --repetition-penalty 1.0 # Creative/exploratory
--max-tokens 100 --temp 0.5 --repetition-penalty 1.2  # Short/concise
```

---

## See Also

- `config/test_prompts.json` - Categorized test prompt library
- `docs/FINE_TUNING_SAGA.md` - Lessons learned about catastrophic forgetting
- `docs/TRAINING_GUIDE.md` - Training parameters and their effects
- `docs/PERFORMANCE_OPTIMIZATION.md` - Speed and memory optimization
- `scripts/phase4-fine-tune-model/06b_evaluate_model.py` - Systematic evaluation script
