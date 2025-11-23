#!/usr/bin/env python3
"""
Step 05c – Create 100 Validation Examples in Granite Format
───────────────────────────────────────────────────────────
Creates exactly 100 validation examples in the same Granite format as training data.
Format: {"text": "Question: ...\nAnswer: ..."}
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
VALID_FILE = "valid.jsonl"

TARGET_VAL_COUNT = 100  # Exactly 100 validation examples
SEED = 42  # For reproducibility


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


def select_validation_examples(
    examples: List[Dict],
    target_count: int,
    seed: int = 42
) -> List[Dict]:
    """
    Select exactly target_count examples for validation.
    
    Args:
        examples: List of all example dictionaries
        target_count: Number of validation examples to select
        seed: Random seed for reproducibility
        
    Returns:
        List of validation examples
    """
    if len(examples) < target_count:
        print(f"⚠️  Warning: Only {len(examples)} examples available, but {target_count} requested.")
        print(f"   Using all {len(examples)} examples for validation.")
        return examples
    
    # Shuffle with seed for reproducibility
    random.seed(seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)
    
    # Select exactly target_count examples
    val_examples = shuffled[:target_count]
    
    return val_examples


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
    valid_path = output_dir / VALID_FILE
    
    print(f"{'='*70}")
    print(f"Creating 100 Validation Examples in Granite Format")
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
    
    # Select validation examples
    print(f"\n→ Selecting validation examples:")
    print(f"  Target: {TARGET_VAL_COUNT} examples")
    print(f"  Seed: {SEED} (for reproducibility)")
    
    val_examples = select_validation_examples(examples, TARGET_VAL_COUNT, SEED)
    
    print(f"\n→ Selected {len(val_examples)} validation examples")
    
    # Save validation file
    print(f"\n→ Saving validation file...")
    save_jsonl(val_examples, valid_path)
    print(f"  ✓ Validation: {valid_path}")
    
    # Verify format matches training format
    if val_examples:
        sample = val_examples[0]
        if "text" in sample and "Question:" in sample["text"] and "Answer:" in sample["text"]:
            print(f"\n  ✓ Format verified: Granite text format")
            print(f"  ✓ Structure: {{\"text\": \"Question: ...\\nAnswer: ...\"}}")
        else:
            print(f"\n  ⚠️  Warning: Format may not match training format")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"✅ VALIDATION FILE CREATED")
    print(f"{'='*70}")
    print(f"   Total examples available: {len(examples)}")
    print(f"   Validation examples: {len(val_examples)}")
    print(f"\n   File ready for Granite training:")
    print(f"   - {valid_path}")
    print(f"\n   Format: Granite simple text format (same as train.jsonl)")
    print(f"   Structure: {{\"text\": \"Question: ...\\nAnswer: ...\"}}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()






