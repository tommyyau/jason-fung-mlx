#!/usr/bin/env python3
"""
Step 04c – Convert Answers to Granite Format
────────────────────────────────────────────
Converts `data/generated_answers.jsonl` to Granite's simple text format.
Granite models use a simple text format: {"text": "Question: ...\nAnswer: ..."}

This is different from chat format - it's a plain text concatenation.
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Dict

# Get project root
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

# ─────────────────────────────
# Config
# ─────────────────────────────
INPUT_FILE = "data/generated_answers.jsonl"
OUTPUT_DIR = "data/mlx_training_data"
TRAIN_FILE = "train.jsonl"
VALID_FILE = "valid.jsonl"

TRAIN_SPLIT = 0.8  # 80% for training
VAL_SPLIT = 0.2    # 20% for validation
SEED = 42          # For reproducibility


def convert_to_granite_format(input_path: Path) -> List[Dict]:
    """
    Convert generated answers JSONL to Granite's simple text format.
    
    Granite format: {"text": "Question: ...\nAnswer: ..."}
    
    Args:
        input_path: Path to generated_answers.jsonl
        
    Returns:
        List of examples in Granite format
    """
    print(f"→ Loading answers from: {input_path}")
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return []
    
    examples = []
    skipped = 0
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                
                # Extract question and answer
                question = data.get("question", "").strip()
                answer = data.get("answer", "").strip()
                
                # Validate required fields
                if not question:
                    print(f"  ⚠️  Skipping line {line_num}: missing 'question' field")
                    skipped += 1
                    continue
                
                if not answer:
                    print(f"  ⚠️  Skipping line {line_num}: missing 'answer' field")
                    skipped += 1
                    continue
                
                # Create Granite format: simple text concatenation
                # Format: "Question: ...\nAnswer: ..."
                text_content = f"Question: {question}\nAnswer: {answer}"
                
                granite_entry = {
                    "text": text_content
                }
                
                examples.append(granite_entry)
                
            except json.JSONDecodeError as e:
                print(f"  ⚠️  JSON error on line {line_num}: {str(e)[:100]}")
                skipped += 1
                continue
            except Exception as e:
                print(f"  ⚠️  Error processing line {line_num}: {type(e).__name__}: {str(e)[:100]}")
                skipped += 1
                continue
    
    print(f"  ✓ Converted {len(examples)} examples to Granite format")
    if skipped > 0:
        print(f"  ⚠️  Skipped {skipped} invalid lines")
    
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
    output_dir = project_root / OUTPUT_DIR
    train_path = output_dir / TRAIN_FILE
    valid_path = output_dir / VALID_FILE
    
    print(f"{'='*70}")
    print(f"Converting Answers to Granite Format")
    print(f"{'='*70}\n")
    
    # Convert to Granite format
    examples = convert_to_granite_format(input_path)
    
    if len(examples) == 0:
        print(f"❌ No examples converted. Check input file and errors above.")
        sys.exit(1)
    
    # Show sample
    if examples:
        print(f"\n→ Sample entry (first example):")
        sample = examples[0]
        text_preview = sample['text'][:200] + "..." if len(sample['text']) > 200 else sample['text']
        print(f"  {text_preview}")
    
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
    
    save_jsonl(val_examples, valid_path)
    print(f"  ✓ Validation: {valid_path}")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"✅ CONVERSION COMPLETE")
    print(f"{'='*70}")
    print(f"   Total examples: {len(examples)}")
    print(f"   Train: {len(train_examples)} ({len(train_examples)/len(examples)*100:.1f}%)")
    print(f"   Validation: {len(val_examples)} ({len(val_examples)/len(examples)*100:.1f}%)")
    print(f"\n   Files ready for Granite training:")
    print(f"   - {train_path}")
    print(f"   - {valid_path}")
    print(f"\n   Format: Granite simple text format")
    print(f"   Structure: {{\"text\": \"Question: ...\\nAnswer: ...\"}}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

