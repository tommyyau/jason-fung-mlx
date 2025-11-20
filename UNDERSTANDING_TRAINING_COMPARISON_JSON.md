# Understanding `training_runs_comparison.json`

This document explains the structure and meaning of the training runs comparison JSON file.

---

## File Structure Overview

The JSON file contains comparison data for one or more training runs. Here's the top-level structure:

```json
{
  "timestamp": "2025-11-11T18:05:11.313398",  // When the comparison was generated
  "runs": [                                    // Array of training run data
    { /* Run 1 data */ },
    { /* Run 2 data */ }
  ],
  "comparison_table": "..."                   // Formatted text table for quick viewing
}
```

---

## Run Data Structure

Each run in the `runs` array contains:

### 1. **Basic Information**

```json
{
  "run_name": "SmolLM3-3B_run4",              // Name of the training run
  "run_path": "/path/to/models/SmolLM3-3B_run4",  // Where the model is stored
  "timestamp": "2025-11-11T18:05:11.312599"  // When this run data was collected
}
```

### 2. **Hyperparameters** (Training Configuration)

These are the settings used during training, loaded from `adapter_config.json`:

| Field | What It Means | Example Value | Why It Matters |
|-------|---------------|---------------|----------------|
| `model` | Base model being fine-tuned | `"HuggingFaceTB/SmolLM3-3B"` | Determines starting capabilities |
| `learning_rate` | How fast the model learns | `1e-05` (0.00001) | Too high = unstable, too low = slow |
| `batch_size` | Examples processed at once | `2` | Memory vs speed tradeoff |
| `iters` | Total training iterations | `1600` | How long to train |
| `max_seq_length` | Maximum tokens per example | `700` | Longer = more context, more memory |
| `num_layers` | How many LoRA layers to adapt | `12` | More = stronger adaptation, risk of forgetting |
| `lora_rank` | LoRA matrix rank | `8` | Controls adaptation capacity |
| `grad_accumulation_steps` | Batch size multiplier | `16` | Effective batch = batch_size × this |
| `grad_checkpoint` | Memory optimization | `true` | Saves memory, slightly slower |
| `steps_per_eval` | How often to validate | `50` | More frequent = catch overfitting earlier |
| `steps_per_report` | How often to print progress | `50` | More frequent = better monitoring |
| `save_every` | How often to save checkpoints | `500` | More frequent = more recovery points |

**Key Calculations:**
- **Effective batch size** = `batch_size × grad_accumulation_steps`
  - Example: `2 × 16 = 32` (Run 4)
- **Epochs** ≈ `iters / (dataset_size / batch_size)`
  - Example: `1600 / (1600 / 2) = 2 epochs` (Run 4)

### 3. **Training Metrics** (Empty in Your File)

```json
"training_metrics": {}  // Should contain per-iteration metrics
```

**What Should Be Here:**
- Arrays tracking metrics at each evaluation step
- Used to plot loss curves and analyze training progress

**Why It's Empty:**
- Training metrics are extracted from training log output
- If no log file was provided, this stays empty
- The comparison script needs either:
  - A log file: `--training-output path/to/training.log`
  - Manual input: `--manual` flag
  - Pre-extracted metrics: `--run-metrics metrics.json`

### 4. **Final Metrics** (Missing for Run 4)

```json
"final_metrics": {
  "final_iteration": 1800,           // Last iteration completed
  "final_train_loss": 1.443,         // Training loss at end
  "final_val_loss": 1.629,           // Validation loss at end
  "final_learning_rate": 1e-05,      // Learning rate at end
  "final_tokens_per_sec": 180.235,   // Training speed
  "final_iterations_per_sec": 0.417, // Iteration speed
  "total_trained_tokens": 751673,    // Total tokens processed
  "peak_memory_gb": 8.082,           // Peak RAM usage
  "final_val_time_sec": 28.928       // Time for validation
}
```

**What Each Metric Means:**

| Metric | What It Tells You | Good Values |
|--------|-------------------|-------------|
| `final_train_loss` | How well model fits training data | Lower is better (but too low = overfitting) |
| `final_val_loss` | How well model generalizes | Should be close to train loss (gap < 10%) |
| `train_val_gap` | Overfitting indicator | `val_loss - train_loss` (smaller is better) |
| `tokens_per_sec` | Training throughput | Higher = faster training |
| `iterations_per_sec` | Training speed | Higher = faster |
| `peak_memory_gb` | Memory usage | Should stay under available RAM |
| `total_trained_tokens` | Total data processed | `iters × batch_size × avg_seq_length` |

**Interpreting Loss:**
- **Train loss < Val loss**: Normal (some overfitting expected)
- **Gap > 15%**: Significant overfitting (model memorizing training data)
- **Gap < 5%**: Good generalization
- **Val loss increasing**: Model is overfitting (stop training)

---

## Understanding Your Current File

Looking at `training_runs_comparison.json`:

### What You Have:
✅ **Run 4 hyperparameters** - Complete configuration
✅ **Run path** - Know where the model is stored
✅ **Comparison table** - Formatted summary

### What's Missing:
❌ **Run 4 training metrics** - No loss curves or progress data
❌ **Run 4 final metrics** - Don't know final loss, speed, memory usage
❌ **Run 3 data** - Only Run 4 is in this file

---

## How to Get Complete Data

### Option 1: Extract from Training Logs

If you have the training output saved:

```bash
python3 scripts/phase4-fine-tune-model/06e_compare_training_runs.py \
  --runs SmolLM3-3B_run4 \
  --training-output path/to/run4_training.log \
  --output training_runs_comparison.json
```

The script will parse lines like:
```
Iter 1600: Train loss 1.234, Learning Rate 1.000e-05, It/sec 0.450, Tokens/sec 200.123, Trained Tokens 800000, Peak mem 8.5 GB
Iter 1600: Val loss 1.456, Val took 30.5s
```

### Option 2: Manual Input

If you remember the final metrics:

```bash
python3 scripts/phase4-fine-tune-model/06e_compare_training_runs.py \
  --runs SmolLM3-3B_run4 \
  --manual \
  --output training_runs_comparison.json
```

Then enter metrics when prompted.

### Option 3: Create Metrics JSON

Create a file `run4_metrics.json`:

```json
{
  "SmolLM3-3B_run4": {
    "final_iteration": 1600,
    "final_train_loss": 1.234,
    "final_val_loss": 1.456,
    "final_learning_rate": 1e-05,
    "final_tokens_per_sec": 200.0,
    "final_iterations_per_sec": 0.450,
    "total_trained_tokens": 800000,
    "peak_memory_gb": 8.5,
    "final_val_time_sec": 30.5
  }
}
```

Then run:

```bash
python3 scripts/phase4-fine-tune-model/06e_compare_training_runs.py \
  --runs SmolLM3-3B_run4 \
  --run-metrics run4_metrics.json \
  --output training_runs_comparison.json
```

---

## Comparing Multiple Runs

To compare Run 3 and Run 4 together:

```bash
python3 scripts/phase4-fine-tune-model/06e_compare_training_runs.py \
  --runs SmolLM3-3B_run3 SmolLM3-3B_run4 \
  --run-metrics run3_metrics.json \
  --training-output run4_training.log \
  --output training_runs_comparison.json \
  --report comparison_report.md
```

This will:
1. Load Run 3 metrics from `run3_metrics.json`
2. Extract Run 4 metrics from `run4_training.log`
3. Compare hyperparameters side-by-side
4. Generate a markdown report

---

## Key Insights from Your Data

### Run 4 Configuration Analysis:

**Strengths:**
- ✅ **Regularization**: `lora_dropout: 0.1` prevents overfitting
- ✅ **Frequent validation**: `steps_per_eval: 50` catches issues early
- ✅ **Stable gradients**: `grad_accumulation_steps: 16` = effective batch of 32
- ✅ **Longer sequences**: `max_seq_length: 700` handles full context

**Trade-offs:**
- ⚖️ **Fewer adapted layers**: `num_layers: 12` vs Run 3's 16 (less adaptation, more preservation)
- ⚖️ **Shorter training**: `iters: 1600` vs Run 3's 1800 (2 epochs vs 3)

**What to Check:**
1. **Final validation loss** - Should be lower than Run 3's 1.629 if regularization worked
2. **Train-val gap** - Should be smaller than Run 3's 12.9% gap
3. **Training speed** - Compare tokens/sec to see impact of longer sequences

---

## Next Steps

1. **Extract Run 4 metrics** using one of the methods above
2. **Compare with Run 3** to see if regularization improved generalization
3. **Evaluate both models** on validation set:
   ```bash
   python3 scripts/phase4-fine-tune-model/06b_evaluate_model.py \
     --model models/SmolLM3-3B_run3 \
     --compare-ground-truth
   
   python3 scripts/phase4-fine-tune-model/06b_evaluate_model.py \
     --model models/SmolLM3-3B_run4 \
     --compare-ground-truth
   ```

4. **Choose the better model** based on:
   - Lower validation loss = better generalization
   - Smaller train-val gap = less overfitting
   - Better style mimicry (subjective evaluation)

---

## Quick Reference: Field Meanings

| Field | Category | Meaning |
|-------|----------|---------|
| `run_name` | Info | Name identifier |
| `run_path` | Info | File system location |
| `hyperparameters.*` | Config | Training settings |
| `training_metrics.*` | Progress | Per-iteration data |
| `final_metrics.*` | Results | End-of-training values |
| `comparison_table` | Summary | Formatted text table |

---

## Example: Complete Run Data

Here's what a complete run entry looks like:

```json
{
  "run_name": "SmolLM3-3B_run3",
  "run_path": "/path/to/models/SmolLM3-3B_run3",
  "timestamp": "2025-11-11T18:01:08.776893",
  "hyperparameters": {
    "model": "HuggingFaceTB/SmolLM3-3B",
    "learning_rate": 1e-05,
    "batch_size": 2,
    "iters": 1800,
    "max_seq_length": 512,
    "num_layers": 16,
    "lora_rank": 8,
    "grad_accumulation_steps": 4,
    "grad_checkpoint": true,
    "steps_per_eval": 200,
    "steps_per_report": 50,
    "save_every": 500
  },
  "training_metrics": {
    "iterations": [50, 100, 150, ..., 1800],
    "train_loss": [2.1, 1.9, 1.7, ..., 1.443],
    "val_loss": [2.2, 2.0, 1.8, ..., 1.629],
    "tokens_per_sec": [150, 160, 170, ..., 180.235]
  },
  "final_metrics": {
    "final_iteration": 1800,
    "final_train_loss": 1.443,
    "final_val_loss": 1.629,
    "final_learning_rate": 1e-05,
    "final_tokens_per_sec": 180.235,
    "final_iterations_per_sec": 0.417,
    "total_trained_tokens": 751673,
    "peak_memory_gb": 8.082,
    "final_val_time_sec": 28.928
  }
}
```

This shows:
- **Configuration**: 16 layers, 1800 iterations, 512 seq length
- **Progress**: Loss decreasing from 2.1 → 1.443
- **Results**: Final train loss 1.443, val loss 1.629 (12.9% gap = overfitting)

