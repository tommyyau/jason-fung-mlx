# MLX Training Configuration Parameters Reference

Complete list of all parameters that can be tweaked in MLX training YAML config files (e.g., `config/mlx_smolLM3_training_run4.yaml`).

## Core Model & Data Parameters

| Parameter | Type | Description | Example | Notes |
|-----------|------|-------------|---------|-------|
| `model` | string | Path to local model directory or Hugging Face repo | `HuggingFaceTB/SmolLM3-3B` | Required |
| `train` | boolean | Enable training mode | `true` | Required for training |
| `data` | string | Directory with `{train, valid, test}.jsonl` files or Hugging Face dataset name | `data/mlx_training_data` | Required |

## Fine-Tuning Method

| Parameter | Type | Description | Options | Default |
|-----------|------|-------------|---------|---------|
| `fine_tune_type` | string | Type of fine-tuning to perform | `lora`, `dora`, `full` | `lora` |

## Optimizer Configuration

| Parameter | Type | Description | Options | Default |
|-----------|------|-------------|---------|---------|
| `optimizer` | string | Optimizer to use for training | `adam`, `adamw`, `muon`, `sgd`, `adafactor` | `adamw` |

## LoRA-Specific Parameters

| Parameter | Type | Description | Typical Range | Notes |
|-----------|------|-------------|---------------|-------|
| `num_layers` | integer | Number of layers to fine-tune | `-1` (all) to `16` | `-1` = all layers, `12` recommended for style learning |
| `lora_rank` | integer | LoRA rank (dimensionality of adaptation) | `4` to `32` | Lower = less capacity, less forgetting. `8` is balanced |
| `lora_alpha` | integer | LoRA alpha (scaling factor) | `4` to `32` | Usually set to `lora_rank` or `2×lora_rank`. `8` is conservative |
| `lora_dropout` | float | LoRA dropout rate for regularization | `0.0` to `0.3` | `0.1` recommended to prevent overfitting |
| `use_dora` | boolean | Enable DoRA (Weight-Decomposed Low-Rank Adaptation) | `true`, `false` | `false` saves memory, `true` can improve quality |

## Training Hyperparameters

| Parameter | Type | Description | Typical Range | Notes |
|-----------|------|-------------|---------------|-------|
| `learning_rate` | float | Adam learning rate | `1e-6` to `1e-4` | `1e-5` is conservative, prevents catastrophic forgetting |
| `batch_size` | integer | Minibatch size | `1` to `8` | `1` for 16GB RAM, `2-4` if you have more memory |
| `iters` | integer | Total number of training iterations | Calculated: `epochs × dataset_size / batch_size` | Must calculate manually |
| `max_seq_length` | integer | Maximum sequence length | `512` to `2048` | `1024` is good balance, `512` saves memory |
| `grad_accumulation_steps` | integer | Steps to accumulate before optimizer update | `1` to `32` | Effective batch size = `batch_size × grad_accumulation_steps` |
| `grad_checkpoint` | boolean | Use gradient checkpointing to reduce memory | `true`, `false` | `true` saves memory, slightly slower |

## Training Monitoring & Checkpointing

| Parameter | Type | Description | Typical Range | Notes |
|-----------|------|-------------|---------------|-------|
| `steps_per_report` | integer | Number of steps between loss reporting | `10` to `100` | `50` is good balance |
| `steps_per_eval` | integer | Number of steps between validations | `50` to `500` | `50` recommended to catch overfitting early |
| `save_every` | integer | Save checkpoint every N iterations | `100` to `1000` | `500` is reasonable |
| `adapter_path` | string | Save/load path for fine-tuned weights | Path string | Directory where adapters are saved |
| `resume_adapter_file` | string | Path to resume training from checkpoint | Path string | Optional, for resuming training |

## Evaluation & Testing

| Parameter | Type | Description | Options | Notes |
|-----------|------|-------------|---------|-------|
| `test` | boolean | Evaluate on test set after training | `true`, `false` | Optional |
| `test_batches` | integer | Number of test set batches | `-1` (all) or positive integer | `-1` uses entire test set |
| `val_batches` | integer | Number of validation batches | `-1` (all) or positive integer | `-1` uses entire validation set |

## Advanced Options

| Parameter | Type | Description | Options | Notes |
|-----------|------|-------------|---------|-------|
| `mask_prompt` | boolean | Mask the prompt in the loss when training | `true`, `false` | Only compute loss on assistant response |
| `seed` | integer | PRNG seed for reproducibility | Any integer | `42` is common default |
| `report_to` | string | Services to report logs to | `wandb`, `swanlab`, or comma-separated | Optional, for experiment tracking |
| `project_name` | string | Project name for logging | String | Defaults to root directory name |

## Parameter Relationships & Recommendations

### Memory-Constrained Systems (16GB RAM)

```yaml
batch_size: 1
max_seq_length: 512-1024
grad_checkpoint: true
num_layers: 12  # Fewer layers = less memory
grad_accumulation_steps: 8-16  # Compensate for small batch size
```

### Preventing Catastrophic Forgetting

```yaml
learning_rate: 1e-5  # Conservative
lora_rank: 8  # Lower rank = less adaptation
lora_alpha: 8  # Match rank or lower
num_layers: 12  # Fewer layers = more preserved base model
lora_dropout: 0.1  # Regularization
```

### Style Learning (Your Use Case)

```yaml
learning_rate: 1e-5
lora_rank: 8
lora_alpha: 8
num_layers: 12
lora_dropout: 0.1
grad_accumulation_steps: 8-16
```

### Maximum Quality (More Memory Available)

```yaml
batch_size: 2-4
max_seq_length: 1024-2048
num_layers: 16  # More layers
lora_rank: 16  # Higher rank
lora_alpha: 16-32
grad_checkpoint: false  # If you have memory
```

## Parameter Calculation Examples

### Calculating `iters` (Iterations)

```
iters = (epochs × dataset_size) / batch_size

Example:
- 2 epochs
- 1600 training examples
- batch_size = 2
- iters = (2 × 1600) / 2 = 1600
```

### Effective Batch Size

```
effective_batch_size = batch_size × grad_accumulation_steps

Example:
- batch_size = 1
- grad_accumulation_steps = 16
- effective_batch_size = 1 × 16 = 16
```

## Complete Example Configuration

```yaml
# Base model
model: HuggingFaceTB/SmolLM3-3B

# Enable training
train: true

# Data directory
data: data/mlx_training_data

# Fine-tuning method
fine_tune_type: lora

# Optimizer
optimizer: adamw

# LoRA configuration
num_layers: 12
lora_rank: 8
lora_alpha: 8
lora_dropout: 0.1
use_dora: false

# Training hyperparameters
learning_rate: 1e-5
batch_size: 1
iters: 1600
max_seq_length: 800

# Gradient handling
grad_accumulation_steps: 16
grad_checkpoint: true

# Training monitoring
steps_per_eval: 50
steps_per_report: 50
save_every: 500

# Output directory
adapter_path: models/SmolLM3-3B_run5

# Reproducibility
seed: 42

# Optional: Experiment tracking
# report_to: wandb
# project_name: jason-fung-finetuning
```

## Notes

- **YAML format**: Use underscores (`lora_rank`) not hyphens (`lora-rank`) in YAML files
- **CLI format**: Use hyphens (`--lora-rank`) in command-line arguments
- **Defaults**: If a parameter is not specified, MLX uses its internal defaults
- **Validation**: MLX will error if required parameters are missing or invalid
- **Memory**: Monitor memory usage with Activity Monitor (macOS) - should stay GREEN

## See Also

- `docs/TRAINING_GUIDE.md` - Detailed training guide with parameter explanations
- `docs/FINE_TUNING_SAGA.md` - Lessons learned about parameter tuning
- `docs/PERFORMANCE_OPTIMIZATION.md` - Memory optimization tips

