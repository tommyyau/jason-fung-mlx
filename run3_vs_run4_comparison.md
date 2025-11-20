# Run 3 vs Run 4: Detailed Comparison

## Executive Summary

**Run 3** focused on maximum adaptation with 16 LoRA layers and 3 epochs (1800 iterations).  
**Run 4** emphasizes better regularization to prevent overfitting with 12 LoRA layers, dropout, and 2 epochs (1600 iterations).

---

## Key Differences

### 1. **LoRA Configuration**

| Parameter | Run 3 | Run 4 | Impact |
|-----------|-------|-------|--------|
| **num_layers** | 16 | 12 | Run 4 preserves more base model layers (25% fewer adapted layers) |
| **lora_rank** | 8 | 8 | Same rank (no change) |
| **lora_alpha** | Not set (default) | 8 | Run 4 explicitly reduces LoRA influence |
| **lora_dropout** | 0.0 | 0.1 | Run 4 adds dropout for regularization |

**Analysis:**
- Run 4 reduces the number of adapted layers from 16 to 12, preserving 25% more of the base model's original capabilities
- The addition of `lora_alpha: 8` and `lora_dropout: 0.1` in Run 4 provides explicit regularization to prevent overfitting
- Run 3 adapts more layers, which could lead to better style learning but higher risk of catastrophic forgetting

---

### 2. **Training Duration & Epochs**

| Parameter | Run 3 | Run 4 | Impact |
|-----------|-------|-------|--------|
| **iters** | 1800 | 1600 | Run 4 trains for 11% fewer iterations |
| **Effective epochs** | ~3 epochs | ~2 epochs | Run 4 stops earlier to prevent overfitting |

**Analysis:**
- Run 3: 1800 iterations = ~3 epochs (1367 examples / batch_size 2 = 683.5 iters/epoch)
- Run 4: 1600 iterations = ~2 epochs (1600 examples / batch_size 2 = 800 iters/epoch)
- Run 4's shorter training duration is intentional to reduce overfitting risk

---

### 3. **Sequence Length**

| Parameter | Run 3 | Run 4 | Impact |
|-----------|-------|-------|--------|
| **max_seq_length** | 512 | 700 | Run 4 handles longer sequences |

**Analysis:**
- Run 3: 512 tokens (optimized for memory, max entry was 511 tokens)
- Run 4: 700 tokens (config says 800, but adapter_config shows 700 - provides buffer for longer examples)
- Run 4 can process longer context, which may be important for maintaining full question-answer context

---

### 4. **Gradient Accumulation**

| Parameter | Run 3 | Run 4 | Impact |
|-----------|-------|-------|--------|
| **grad_accumulation_steps** | 4 | 16 | Run 4 uses 4x more accumulation |
| **Effective batch size** | 2 × 4 = 8 | 2 × 16 = 32 | Run 4 has much larger effective batch |

**Analysis:**
- Run 4's 16 gradient accumulation steps creates a much larger effective batch size (32 vs 8)
- This provides more stable gradients and smoother learning
- However, it also means updates happen less frequently (every 16 steps vs every 4 steps)

---

### 5. **Validation Monitoring**

| Parameter | Run 3 | Run 4 | Impact |
|-----------|-------|-------|--------|
| **steps_per_eval** | 200 | 50 | Run 4 validates 4x more frequently |

**Analysis:**
- Run 3: Validates every 200 steps (4 times during training)
- Run 4: Validates every 50 steps (32 times during training)
- Run 4's more frequent validation allows earlier detection of overfitting

---

### 6. **Training Metrics (Run 3 Only)**

Run 3 completed training with the following metrics:

| Metric | Value |
|--------|-------|
| Final iteration | 1800 |
| Final train loss | 1.443 |
| Final validation loss | 1.629 |
| Train-val gap | 0.186 (12.9% higher validation loss) |
| Tokens/sec | 180.2 |
| Iterations/sec | 0.417 |
| Total trained tokens | 751,673 |
| Peak memory | 8.08 GB |
| Validation time | 28.9 seconds |

**Analysis:**
- The train-val gap of 0.186 suggests some overfitting (validation loss is 12.9% higher than training loss)
- This gap validates the need for Run 4's improved regularization approach

---

## Design Philosophy Comparison

### Run 3: Maximum Adaptation
- **Goal**: Learn Jason Fung's style as strongly as possible
- **Strategy**: 
  - Adapt more layers (16 vs 12)
  - Train longer (3 epochs vs 2)
  - No explicit regularization (no dropout, no alpha control)
- **Risk**: Higher chance of overfitting and catastrophic forgetting

### Run 4: Balanced Regularization
- **Goal**: Learn style while preserving base model capabilities
- **Strategy**:
  - Adapt fewer layers (12 vs 16) - preserve more base model
  - Train shorter (2 epochs vs 3) - prevent overfitting
  - Explicit regularization (dropout 0.1, alpha 8)
  - More frequent validation monitoring
  - Larger effective batch size for stability
- **Risk**: May learn style less strongly, but more generalizable

---

## Expected Outcomes

### Run 3 Advantages:
- ✅ Potentially stronger style learning (more adapted layers, longer training)
- ✅ Better at mimicking Jason Fung's specific patterns

### Run 3 Disadvantages:
- ❌ Higher risk of overfitting (evidenced by 12.9% train-val gap)
- ❌ May forget base model capabilities (catastrophic forgetting)
- ❌ Less generalizable to new questions

### Run 4 Advantages:
- ✅ Better generalization (regularization prevents memorization)
- ✅ Preserves more base model capabilities
- ✅ Earlier overfitting detection (frequent validation)
- ✅ More stable training (larger effective batch size)
- ✅ Can handle longer sequences (700 vs 512 tokens)

### Run 4 Disadvantages:
- ❌ May learn style less strongly (fewer adapted layers, shorter training)
- ❌ Slower gradient updates (16 vs 4 accumulation steps)

---

## Recommendations

1. **If Run 4 has completed training**, compare validation losses:
   - If Run 4's validation loss is lower than Run 3's (1.629), it's likely better
   - If Run 4's train-val gap is smaller, it's more generalizable

2. **For production use**, Run 4's approach is likely better because:
   - Better generalization means it will work on new questions
   - Preserved base model capabilities mean it won't forget basic language understanding
   - Regularization prevents memorization of training data

3. **For style learning**, Run 3 might be better if:
   - You only care about mimicking style on similar questions
   - You're okay with potential overfitting
   - You want maximum adaptation

4. **Best approach**: Evaluate both models on a held-out test set to see which performs better on:
   - Style mimicry (subjective evaluation)
   - Generalization (performance on new questions)
   - Factual accuracy (if applicable)

---

## Next Steps

1. **Extract Run 4 metrics** (if training completed):
   ```bash
   # If you have training logs, extract metrics
   python3 scripts/phase4-fine-tune-model/06e_compare_training_runs.py \
     --runs SmolLM3-3B_run4 \
     --training-output path/to/run4_training.log \
     --run-metrics run4_metrics.json
   ```

2. **Evaluate both models** on validation set:
   ```bash
   python3 scripts/phase4-fine-tune-model/06b_evaluate_model.py \
     --model models/SmolLM3-3B_run3 \
     --compare-ground-truth
   
   python3 scripts/phase4-fine-tune-model/06b_evaluate_model.py \
     --model models/SmolLM3-3B_run4 \
     --compare-ground-truth
   ```

3. **Compare outputs** side-by-side on same prompts to see which better captures Jason Fung's style

---

## Configuration Files Reference

- Run 3: `config/mlx_smolLM3_training_run3.yaml`
- Run 4: `config/mlx_smolLM3_training_run4.yaml`

