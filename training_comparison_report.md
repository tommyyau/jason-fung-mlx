# Training Runs Comparison

Generated: 2025-11-11T13:38:00.608223

```
========================================================================================================================
TRAINING RUNS COMPARISON
========================================================================================================================

HYPERPARAMETERS
------------------------------------------------------------------------------------------------------------------------
         Run |           LR |        Batch |        Iters |      Seq Len |  LoRA Layers |    LoRA Rank |     Grad Acc |   Grad Check
------------------------------------------------------------------------------------------------------------------------
  SmolLM3-3B |     1.00e-05 |            1 |         1367 |         1024 |           12 |           16 |            8 |            ✓
SmolLM3-3B_run2 |     1.00e-05 |            2 |         2051 |          512 |           16 |           16 |            4 |            ✓
SmolLM3-3B_run3 |     1.00e-05 |            2 |         1800 |          512 |           16 |            8 |            4 |            ✓


TRAINING METRICS
------------------------------------------------------------------------------------------------------------------------
            Run |      Final Iter |      Train Loss |        Val Loss |      Tokens/sec |          It/sec |   Peak Mem (GB)
------------------------------------------------------------------------------------------------------------------------
     SmolLM3-3B |             N/A |             N/A |             N/A |             N/A |             N/A |             N/A
SmolLM3-3B_run2 |             N/A |             N/A |             N/A |             N/A |             N/A |             N/A
SmolLM3-3B_run3 |            1800 |          1.4430 |          1.6290 |           180.2 |           0.417 |            8.08

========================================================================================================================
```

## Detailed Data

```json
[
  {
    "run_name": "SmolLM3-3B",
    "run_path": "/Users/tommyyau/VSCode/jason-fung-mlx/models/SmolLM3-3B",
    "timestamp": "2025-11-11T13:38:00.606886",
    "hyperparameters": {
      "model": "HuggingFaceTB/SmolLM3-3B",
      "learning_rate": 1e-05,
      "batch_size": 1,
      "iters": 1367,
      "max_seq_length": 1024,
      "num_layers": 12,
      "lora_rank": 16,
      "grad_accumulation_steps": 8,
      "grad_checkpoint": true,
      "steps_per_eval": 200,
      "steps_per_report": 50,
      "save_every": 500
    },
    "training_metrics": {},
    "final_metrics": {}
  },
  {
    "run_name": "SmolLM3-3B_run2",
    "run_path": "/Users/tommyyau/VSCode/jason-fung-mlx/models/SmolLM3-3B_run2",
    "timestamp": "2025-11-11T13:38:00.607314",
    "hyperparameters": {
      "model": "HuggingFaceTB/SmolLM3-3B",
      "learning_rate": 1e-05,
      "batch_size": 2,
      "iters": 2051,
      "max_seq_length": 512,
      "num_layers": 16,
      "lora_rank": 16,
      "grad_accumulation_steps": 4,
      "grad_checkpoint": true,
      "steps_per_eval": 200,
      "steps_per_report": 50,
      "save_every": 500
    },
    "training_metrics": {},
    "final_metrics": {}
  },
  {
    "run_name": "SmolLM3-3B_run3",
    "run_path": "/Users/tommyyau/VSCode/jason-fung-mlx/models/SmolLM3-3B_run3",
    "timestamp": "2025-11-11T13:38:00.607596",
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
    "training_metrics": {},
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
]
```
