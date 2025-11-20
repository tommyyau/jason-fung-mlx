# How to Compare Validation Loss During Training

The `06e_compare_training_runs.py` script now includes plotting functionality to visualize validation loss curves!

## Quick Start

### Step 1: Get Training Logs

You need the training output logs that contain loss information. These logs should have lines like:

```
Iter 50: Train loss 2.345, Learning Rate 1.000e-05, It/sec 0.417, Tokens/sec 180.235
Iter 50: Val loss 2.567, Val took 28.928s
Iter 100: Train loss 2.123, Learning Rate 1.000e-05, It/sec 0.420, Tokens/sec 182.100
Iter 100: Val loss 2.345, Val took 29.100s
...
```

**Where to find logs:**
- If you saved training output: `training_output.log` or similar
- If you ran training in terminal: Copy/paste the output to a file
- If you used `tee`: Check the log file you specified

### Step 2: Compare Runs with Plotting

```bash
# Compare run3 and run4, extract metrics from logs, and plot loss curves
# The script will automatically match log files to runs by name
python3 scripts/phase4-fine-tune-model/06e_compare_training_runs.py \
  --runs SmolLM3-3B_run3 SmolLM3-3B_run4 \
  --training-output run3_training.log run4_training.log \
  --plot run3_vs_run4_loss_curves.png \
  --output run3_run4_comparison.json
```

**Note:** The script will try to match log files to runs by name. If filenames contain the run name (e.g., `run3_training.log` matches `SmolLM3-3B_run3`), it will use the correct file for each run.

### Step 3: Alternative - Use Existing Metrics

If you already have metrics extracted (like `run3_metrics.json`):

```bash
python3 scripts/phase4-fine-tune-model/06e_compare_training_runs.py \
  --runs SmolLM3-3B_run3 SmolLM3-3B_run4 \
  --run-metrics run3_metrics.json \
  --training-output run4_training.log \
  --plot run3_vs_run4_loss_curves.png
```

## What the Plot Shows

The script generates a plot with **two subplots**:

1. **Top: Training Loss** - Shows how training loss decreases over iterations
2. **Bottom: Validation Loss** - Shows validation loss (this is what you want to compare!)

**Key things to look for:**
- **Lower validation loss** = Better generalization
- **Gap between train/val** = Overfitting indicator (smaller gap is better)
- **Validation loss increasing** = Model is overfitting (should stop training)
- **Smooth curves** = Stable training

## Example: Comparing Run 3 vs Run 4

```bash
# If you have run3 metrics already
python3 scripts/phase4-fine-tune-model/06e_compare_training_runs.py \
  --runs SmolLM3-3B_run3 SmolLM3-3B_run4 \
  --run-metrics run3_metrics.json \
  --training-output run4_training.log \
  --plot run3_vs_run4_comparison.png \
  --output run3_run4_comparison.json \
  --report run3_run4_report.md
```

This will:
1. ✅ Extract Run 3 metrics from `run3_metrics.json`
2. ✅ Extract Run 4 metrics from `run4_training.log`
3. ✅ Generate comparison table
4. ✅ **Plot loss curves** showing validation loss side-by-side
5. ✅ Save JSON and markdown report

## If You Don't Have Logs

If you don't have training logs saved, you have a few options:

### Option 1: Re-run Training with Log Capture

```bash
# Capture training output to a file
python -m mlx_lm lora --config config/mlx_smolLM3_training_run4.yaml 2>&1 | tee run4_training.log
```

### Option 2: Extract from Terminal History

If you still have the terminal window open:
1. Scroll up to find the training output
2. Copy the relevant lines (Iter X: Train loss..., Iter X: Val loss...)
3. Paste into a file: `run4_training.log`

### Option 3: Use Final Metrics Only

If you only remember the final values, you can still compare:

```bash
python3 scripts/phase4-fine-tune-model/06e_compare_training_runs.py \
  --runs SmolLM3-3B_run3 SmolLM3-3B_run4 \
  --run-metrics run3_metrics.json \
  --manual
```

Then enter Run 4's final metrics when prompted.

## Understanding the Output

### Plot File (`*.png`)
- **Training Loss Plot**: Shows how well model fits training data
- **Validation Loss Plot**: Shows generalization (what you care about!)
- **Log scale**: Makes it easier to see differences
- **Different markers**: Train (circles) vs Val (squares)

### JSON File (`*.json`)
Contains all metrics in structured format:
- `training_metrics`: Per-iteration data (for plotting)
- `final_metrics`: Final values only
- `hyperparameters`: Training configuration

### Markdown Report (`*.md`)
Human-readable summary with tables and formatted data

## Troubleshooting

### "No training metrics available for plotting"
- **Cause**: No loss data found in logs or metrics
- **Fix**: Make sure training logs contain lines with "Train loss" and "Val loss"

### "matplotlib not available"
- **Fix**: Install matplotlib: `pip install matplotlib`

### Plot shows only one run
- **Cause**: Only one run has training metrics
- **Fix**: Provide training logs for both runs

### Validation loss not showing
- **Cause**: Logs might only have training loss
- **Fix**: Check that logs contain "Val loss" lines (validation happens at `steps_per_eval` intervals)

## Quick Reference

```bash
# Basic comparison with plotting
python3 scripts/phase4-fine-tune-model/06e_compare_training_runs.py \
  --runs RUN1 RUN2 \
  --training-output log1.log \
  --training-output log2.log \
  --plot output.png

# With existing metrics
python3 scripts/phase4-fine-tune-model/06e_compare_training_runs.py \
  --runs RUN1 RUN2 \
  --run-metrics metrics.json \
  --training-output log2.log \
  --plot output.png

# Auto-plot (if metrics available)
python3 scripts/phase4-fine-tune-model/06e_compare_training_runs.py \
  --runs RUN1 RUN2 \
  --training-output log1.log \
  --training-output log2.log
# Plot will be auto-generated as training_loss_comparison.png
```

## Next Steps

After comparing validation loss:
1. **Identify the better run** (lower validation loss = better)
2. **Check for overfitting** (train-val gap should be small)
3. **Evaluate both models** on test set to confirm
4. **Use the better model** for inference

