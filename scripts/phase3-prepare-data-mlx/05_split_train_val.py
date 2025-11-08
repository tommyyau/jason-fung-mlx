#!/usr/bin/env python3
"""
Step 05 – Split MLX Dataset
───────────────────────────
Shuffles `data/generated_answers_mlx.jsonl` with a fixed seed and produces
train/validation splits (`generated_answers_mlx_train.jsonl`, `generated_answers_mlx_validate.jsonl`)
for MLX fine-tuning.
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Dict

# ─────────────────────────────
# Config
# ─────────────────────────────
INPUT_FILE = "data/generated_answers_mlx.jsonl"
TRAIN_FILE = "data/generated_answers_mlx_train.jsonl"
VAL_FILE = "data/generated_answers_mlx_validate.jsonl"

TRAIN_SPLIT = 0.8  # 80% for training
VAL_SPLIT = 0.2    # 20% for validation
SEED = 42          # For reproducibility

# Get project root
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent


def load_dataset(input_path: Path) -> List[Dict]:
    """Load MLX format dataset from JSONL file."""
    examples = []

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return examples

    print(f"→ Loading dataset from: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)

                # Validate required fields
                if "instruction" not in data or "output" not in data:
                    print(f"  ⚠️  Skipping line {line_num}: missing 'instruction' or 'output' field")
                    continue

                instruction = data.get("instruction", "").strip()
                output = data.get("output", "").strip()

                if not instruction or not output:
                    print(f"  ⚠️  Skipping line {line_num}: empty 'instruction' or 'output'")
                    continue

                examples.append(data)

            except json.JSONDecodeError as e:
                print(f"  ⚠️  JSON error on line {line_num}: {str(e)[:100]}")
                continue
            except Exception as e:
                print(f"  ⚠️  Error processing line {line_num}: {type(e).__name__}: {str(e)[:100]}")
                continue

    print(f"  ✓ Loaded {len(examples)} examples")
    return examples


def split_dataset(
    examples: List[Dict],
    train_split: float,
    val_split: float,
    seed: int = 42
) -> tuple[List[Dict], List[Dict]]:
    """
    Split dataset into train and validation sets.

    Args:
        examples: List of example dictionaries
        train_split: Proportion for training (e.g., 0.8 for 80%)
        val_split: Proportion for validation (e.g., 0.2 for 20%)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_examples, val_examples)
    """
    # Validate splits sum to 1.0
    total = train_split + val_split
    if abs(total - 1.0) > 0.01:
        print(f"⚠️  Warning: Splits don't sum to 1.0 (sum={total}), normalizing...")
        train_split /= total
        val_split /= total

    # Shuffle with seed for reproducibility
    random.seed(seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)

    # Calculate split point
    n_total = len(shuffled)
    n_train = int(n_total * train_split)

    # Split
    train_examples = shuffled[:n_train]
    val_examples = shuffled[n_train:]

    return train_examples, val_examples


def save_jsonl(examples: List[Dict], output_path: Path) -> None:
    """Save examples to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            # Preserve all formatting (ensure_ascii=False keeps unicode and formatting)
            json_line = json.dumps(example, ensure_ascii=False) + "\n"
            f.write(json_line)


def main():
    """Main entry point."""
    input_path = project_root / INPUT_FILE
    train_path = project_root / TRAIN_FILE
    val_path = project_root / VAL_FILE

    print(f"{'='*70}")
    print(f"Splitting MLX Dataset into Train and Validation")
    print(f"{'='*70}\n")

    # Load dataset
    examples = load_dataset(input_path)

    if len(examples) == 0:
        print(f"❌ No examples loaded. Check input file: {input_path}")
        sys.exit(1)

    # Split dataset
    print(f"\n→ Splitting dataset:")
    print(f"  Train: {TRAIN_SPLIT*100:.0f}%")
    print(f"  Validation: {VAL_SPLIT*100:.0f}%")
    print(f"  Seed: {SEED} (for reproducibility)")

    train_examples, val_examples = split_dataset(examples, TRAIN_SPLIT, VAL_SPLIT, SEED)

    print(f"\n→ Split results:")
    print(f"  Train: {len(train_examples)} examples ({len(train_examples)/len(examples)*100:.1f}%)")
    print(f"  Validation: {len(val_examples)} examples ({len(val_examples)/len(examples)*100:.1f}%)")

    # Save splits
    print(f"\n→ Saving splits...")
    save_jsonl(train_examples, train_path)
    print(f"  ✓ Train: {train_path}")

    save_jsonl(val_examples, val_path)
    print(f"  ✓ Validation: {val_path}")

    # Verify formatting is preserved
    if train_examples:
        sample = train_examples[0]
        has_formatting = any(marker in sample.get("output", "") for marker in ["**", "\n\n", "- ", "* "])
        if has_formatting:
            print(f"\n  ✓ Formatting preserved in output (bold, lists, paragraphs)")

    # Final summary
    print(f"\n{'='*70}")
    print(f"✅ SPLIT COMPLETE")
    print(f"{'='*70}")
    print(f"   Total examples: {len(examples)}")
    print(f"   Train: {len(train_examples)} ({len(train_examples)/len(examples)*100:.1f}%)")
    print(f"   Validation: {len(val_examples)} ({len(val_examples)/len(examples)*100:.1f}%)")
    print(f"\n   Files ready for MLX training:")
    print(f"   - {train_path}")
    print(f"   - {val_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()


