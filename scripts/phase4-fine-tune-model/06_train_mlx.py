#!/usr/bin/env python3
"""
Step 06 – Train MLX Model
─────────────────────────
Fine-tunes an instruction model with MLX using the prepared Q&A dataset
(`generated_answers_mlx_train.jsonl` / `generated_answers_mlx_validate.jsonl`)
and writes LoRA adapters to `models/jason_fung_mlx`.
"""

import json
import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import Dict, List

# ─────────────────────────────
# Configuration
# ─────────────────────────────
TRAIN_FILE = "data/generated_answers_mlx_train.jsonl"
VAL_FILE = "data/generated_answers_mlx_validate.jsonl"
TEST_FILE = "data/jason_fung_qna_mlx_test.jsonl"

# Default model (can be overridden via command line)
# Using Llama 3.2 3B Instruct - no authentication required, pre-converted by MLX
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct"  # Llama 3.2 3B Instruct (no auth needed)

# Training hyperparameters
# Optimized for 16GB RAM M1 MacBook Pro
LEARNING_RATE = 5e-6  # Reduced from 1e-5 to mitigate catastrophic forgetting
BATCH_SIZE = 1  # Reduced from 4 to save memory
NUM_EPOCHS = 2  # Reduced from 3 to mitigate catastrophic forgetting
GRADIENT_ACCUMULATION_STEPS = 8  # Increased for more stable gradients and reduced catastrophic forgetting
MAX_SEQ_LENGTH = 1024  # Increased to prevent truncation of longer answers
SAVE_EVERY_N_STEPS = 500
EVAL_STEPS = 50

# Output directory
OUTPUT_DIR = "models/jason_fung_mlx"


def check_mlx_installed():
    """Check if mlx and mlx-lm are installed."""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        print("✓ MLX core found")
    except ImportError:
        print("❌ Error: MLX not found. Install with: pip install mlx")
        sys.exit(1)

    try:
        # Try to import mlx-lm components
        import mlx_lm
        print("✓ mlx-lm found")
        return True
    except ImportError:
        print("❌ Error: mlx-lm not found. Install with: pip install mlx-lm")
        print("   Or: pip install 'mlx-lm[train]' for training support")
        sys.exit(1)


def convert_to_chat_format(instruction: str, output: str) -> List[Dict]:
    """
    Convert instruction/output format to chat format for instruction-tuned models.

    Returns messages in chat format:
    [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output}
    ]
    """
    return [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output},
    ]


def prepare_dataset_for_mlx(input_file: str, output_file: str) -> int:
    """
    Convert JSONL dataset to MLX training format.

    MLX expects either:
    1. Chat format: {"messages": [{"role": "user", "content": "..."}, ...]}
    2. Text format: {"text": "..."}

    We'll use chat format for instruction-tuned models.
    """
    print(f"\nConverting dataset: {input_file} -> {output_file}")

    examples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data = json.loads(line)
                    instruction = data.get("instruction", "").strip()
                    output = data.get("output", "").strip()

                    if not instruction or not output:
                        print(f"  ⚠️  Skipping line {line_num}: missing instruction or output")
                        continue

                    # Convert to chat format
                    messages = convert_to_chat_format(instruction, output)
                    examples.append({"messages": messages})

                except json.JSONDecodeError as e:
                    print(f"  ⚠️  JSON error on line {line_num}: {str(e)[:100]}")

    # Save in JSONL format
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"  ✓ Converted {len(examples)} examples")
    return len(examples)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Fine-tune a model with MLX on Jason Fung Q&A dataset")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to fine-tune (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default=TRAIN_FILE,
        help=f"Training dataset (default: {TRAIN_FILE})",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default=VAL_FILE,
        help=f"Validation dataset (default: {VAL_FILE})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory for fine-tuned model (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Number of epochs (default: {NUM_EPOCHS})",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=MAX_SEQ_LENGTH,
        help=f"Maximum sequence length (default: {MAX_SEQ_LENGTH})",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Use LoRA (Low-Rank Adaptation) for efficient fine-tuning",
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=12,  # Reduced from 16 to preserve more base model layers
        help="Number of LoRA layers (default: 12)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=6,  # Balanced: lower than 8 to reduce forgetting, higher than 4 for style learning
        help="LoRA rank (default: 6)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=8,  # Reduced from 16 to mitigate catastrophic forgetting
        help="LoRA alpha (default: 8)",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=16.0,
        help="LoRA scale (default: 16.0)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,  # Increased from 0.05 for better regularization
        help="LoRA dropout (default: 0.1)",
    )
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        default=True,
        help="Use gradient checkpointing to save memory (default: True)",
    )
    parser.add_argument(
        "--no-grad-checkpoint",
        dest="grad_checkpoint",
        action="store_false",
        help="Disable gradient checkpointing (uses more memory)",
    )
    parser.add_argument(
        "--grad-accumulation-steps",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
        help=f"Gradient accumulation steps (default: {GRADIENT_ACCUMULATION_STEPS})",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=True,
        help="Execute training (default: True). Use --no-execute to only prepare data.",
    )
    parser.add_argument(
        "--no-execute",
        dest="execute",
        action="store_false",
        help="Only prepare data, don't execute training",
    )
    parser.add_argument(
        "--steps-per-eval",
        type=int,
        default=EVAL_STEPS,
        help=f"Run validation every N steps (default: {EVAL_STEPS})",
    )
    parser.add_argument(
        "--steps-per-report",
        type=int,
        default=EVAL_STEPS,
        help=f"Report training progress every N steps (default: {EVAL_STEPS})",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=SAVE_EVERY_N_STEPS,
        help=f"Save LoRA adapters every N steps (default: {SAVE_EVERY_N_STEPS})",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("MLX Fine-Tuning: Jason Fung Q&A Dataset")
    print("=" * 80)

    # Check MLX installation
    check_mlx_installed()

    # Check input files
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    train_path = project_root / args.train_file
    val_path = project_root / args.val_file

    if not train_path.exists():
        print(f"\n❌ Error: Training file not found: {train_path}")
        print(f"   Please run newscripts/05_split_train_val.py first")
        sys.exit(1)

    if not val_path.exists():
        print(f"\n❌ Error: Validation file not found: {val_path}")
        print(f"   Please run newscripts/05_split_train_val.py first")
        sys.exit(1)

    # Prepare datasets for MLX
    print("\n" + "=" * 80)
    print("Preparing Datasets")
    print("=" * 80)

    train_mlx_file = project_root / "data" / "jason_fung_qna_mlx_train_mlx.jsonl"
    val_mlx_file = project_root / "data" / "jason_fung_qna_mlx_val_mlx.jsonl"

    train_count = prepare_dataset_for_mlx(str(train_path), str(train_mlx_file))
    val_count = prepare_dataset_for_mlx(str(val_path), str(val_mlx_file))

    if train_count == 0:
        print("\n❌ Error: No training examples found")
        sys.exit(1)

    # Build MLX training command
    print("\n" + "=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Training examples: {train_count}")
    print(f"Validation examples: {val_count}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"Output directory: {args.output_dir}")
    print(f"LoRA: {args.lora}")
    if args.lora:
        print(f"  LoRA layers: {args.lora_layers}")
        print(f"  LoRA rank: {args.lora_rank}")
        print(f"  LoRA alpha: {args.lora_alpha}")
        print(f"  LoRA scale: {args.lora_scale}")
        print(f"  LoRA dropout: {args.lora_dropout}")
    print(f"Gradient checkpointing: {args.grad_checkpoint}")
    print(f"Gradient accumulation steps: {args.grad_accumulation_steps}")
    print(f"Validation every: {args.steps_per_eval} steps")
    print(f"Progress report every: {args.steps_per_report} steps")
    print(f"Save checkpoint every: {args.save_every} steps")
    print(f"\n💡 Memory optimizations enabled for 16GB RAM:")
    print(f"   - Batch size: {args.batch_size} (smaller = less memory)")
    print(f"   - Max sequence length: {args.max_seq_length} (shorter = less memory)")
    print(f"   - Gradient checkpointing: {'ON' if args.grad_checkpoint else 'OFF'}")
    print(f"   - Gradient accumulation: {args.grad_accumulation_steps} steps")

    # Build the mlx-lm lora train command
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # MLX-LM expects data in a directory with train.jsonl, valid.jsonl, test.jsonl
    # or individual files. We'll create a data directory structure.
    data_dir = project_root / "data" / "mlx_training_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Copy train and val files to the expected locations
    train_dest = data_dir / "train.jsonl"
    val_dest = data_dir / "valid.jsonl"
    shutil.copy(train_mlx_file, train_dest)
    shutil.copy(val_mlx_file, val_dest)

    # Calculate iterations: epochs * (num_examples / batch_size)
    num_iterations = args.epochs * (train_count // args.batch_size)
    if num_iterations == 0:
        num_iterations = args.epochs * 100  # Fallback

    cmd_parts = [
        "python",
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        args.model,
        "--train",
        "--data",
        str(data_dir),
        "--fine-tune-type",
        "lora",
        "--learning-rate",
        str(args.learning_rate),
        "--batch-size",
        str(args.batch_size),
        "--iters",
        str(num_iterations),
        "--max-seq-length",
        str(args.max_seq_length),
        "--adapter-path",
        str(output_dir),
        "--num-layers",
        str(args.lora_layers),
        "--grad-accumulation-steps",
        str(args.grad_accumulation_steps),
        "--steps-per-eval",
        str(args.steps_per_eval),
        "--steps-per-report",
        str(args.steps_per_report),
        "--save-every",
        str(args.save_every),
    ]

    # Add gradient checkpointing for memory efficiency
    if args.grad_checkpoint:
        cmd_parts.append("--grad-checkpoint")

    # Save the command to a file for easy execution
    cmd_file = project_root / "train_command.sh"
    with open(cmd_file, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# MLX Training Command\n")
        f.write("# Generated by newscripts/06_train_mlx.py\n\n")
        f.write(" ".join(cmd_parts) + "\n")

    os.chmod(cmd_file, 0o755)

    if not args.execute:
        print("\n" + "=" * 80)
        print("Training Command Prepared")
        print("=" * 80)
        print("\nTo run training, execute:")
        print("  " + " ".join(cmd_parts))
        print(f"\nOr run: bash {cmd_file}")
        print("=" * 80)
        return

    # Execute training
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    print(f"\nTraining will begin now. This may take several hours.")
    print(f"Progress will be displayed below.\n")

    try:
        # mlx_lm uses command-line interface, so we use subprocess
        import subprocess

        print(f"Executing: {' '.join(cmd_parts)}\n")
        result = subprocess.run(cmd_parts, check=False)

        if result.returncode == 0:
            print("\n" + "=" * 80)
            print("Training Complete!")
            print("=" * 80)
            print(f"\n✓ Fine-tuned model saved to: {output_dir}")
            print(f"\nNext steps:")
            print(f"  1. Fuse LoRA adapters: python3 newscripts/07_fuse_lora.py")
            print(f"  2. Convert to HuggingFace: python3 newscripts/08_convert_to_hf.py")
            print(f"  3. Convert to GGUF: python3 newscripts/09_convert_to_gguf.py")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("Training Failed")
            print("=" * 80)
            print(f"\nTraining exited with code {result.returncode}")
            print(f"You can try running manually: bash {cmd_file}")
            sys.exit(result.returncode)

    except ImportError as e:
        print(f"\n❌ Error: Could not import mlx_lm.train: {e}")
        print(f"   Make sure mlx-lm is installed: pip install 'mlx-lm[train]'")
        print(f"\n   Falling back to command generation.")
        print(f"   Run training manually: bash {cmd_file}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print(f"   You can try running manually: bash {cmd_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()


