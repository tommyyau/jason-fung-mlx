# DPO Implementation Plan

## Goal
Implement Direct Preference Optimization (DPO) to fine-tune the Granite 4.0 H-Micro model. This involves creating a dataset of (prompt, chosen, rejected) pairs and running DPO training.

## 1. Data Preparation
We need to convert our existing Q&A pairs into DPO format.
- **Chosen**: The existing "Answer" from our high-quality dataset (Dr. Fung's style).
- **Rejected**: We will generate a new answer using the *base* model (Granite 3B). This represents the "generic" or "default" style that we want to move away from.

### Script: `scripts/phase3-prepare-data-mlx/06_generate_dpo_pairs.py`
- **Input**: `data/mlx_training_data/train.jsonl` (Granite format) or `data/generated_answers.jsonl`.
- **Process**:
    1. Load the base model (`ibm-granite/granite-4.0-h-micro`).
    2. For each Q&A pair:
        - Extract the Question (Prompt).
        - Keep the existing Answer as **Chosen**.
        - Generate a new Answer using the base model as **Rejected**.
- **Output**: `data/mlx_training_data/dpo_train.jsonl`
- **Format**: `{"prompt": "...", "chosen": "...", "rejected": "..."}`

## 2. DPO Training Script
Since `mlx-lm` CLI doesn't support DPO out-of-the-box in this version, we will create a custom training script.

### Script: `scripts/phase4-fine-tune-model/10_train_dpo.py`
- **Dependencies**: `mlx`, `mlx_lm`.
- **Features**:
    - Load Model (Policy) and Reference Model (Frozen).
    - LoRA support (we only train adapters).
    - DPO Loss calculation.
    - Gradient Checkpointing (crucial for 16GB RAM).
    - Batch size 1 with accumulation.

## 3. Configuration
### File: `config/mlx_granite-4.0-h-micro_dpo.yaml`
- **Beta**: 0.1 (DPO parameter).
- **Learning Rate**: 1e-6 (typically lower than SFT).
- **LoRA Rank**: 8 or 16.

## 4. Execution
### Script: `train_dpo_run1.sh`
- Run data generation.
- Run DPO training.
- Fuse and test.

## User Review Required
- **Data Generation Time**: Generating "rejected" answers for the entire dataset might take time (approx 10-20s per example). For ~1300 examples, that's ~4-7 hours.
- **Proposal**: We can start with a smaller subset (e.g., 100 examples) to test the pipeline.
