#!/usr/bin/env python3
"""
Step 04b – Convert MLX Training Data to Gemma Format
────────────────────────────────────────────────────
Converts MLX format JSONL files (with "messages" structure) to Gemma format
with <start_of_turn> and <end_of_turn> tags for Google's Gemma training.
"""

import json
import os
import sys
from pathlib import Path

# ─────────────────────────────
# Config
# ─────────────────────────────
INPUT_TRAIN = "data/mlx_training_data/train.jsonl"
INPUT_VALID = "data/mlx_training_data/valid.jsonl"
OUTPUT_TRAIN = "data/mlx_training_data/train_gemma.jsonl"
OUTPUT_VALID = "data/mlx_training_data/valid_gemma.jsonl"

# Get project root
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent


def convert_to_gemma_format(input_path: Path, output_path: Path) -> int:
    """
    Convert MLX format JSONL to Gemma format.

    Args:
        input_path: Path to MLX format JSONL file (with "messages" structure)
        output_path: Path to output Gemma format JSONL file

    Returns:
        Number of examples converted
    """
    print(f"→ Loading MLX data from: {input_path}")

    if not input_path.exists():
        print(f"  ⚠️  Input file not found: {input_path}")
        return 0

    converted_data = []
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)

                # Validate structure
                if "messages" not in item:
                    print(f"  ⚠️  Skipping line {line_num}: missing 'messages' field")
                    skipped += 1
                    continue

                if not isinstance(item["messages"], list):
                    print(f"  ⚠️  Skipping line {line_num}: 'messages' must be a list")
                    skipped += 1
                    continue

                # Convert to Gemma format
                text = ""
                for message in item["messages"]:
                    role = message.get("role", "")
                    content = message.get("content", "")

                    if role == "user":
                        text += f"<start_of_turn>user\n{content}<end_of_turn>\n"
                    elif role == "assistant":
                        text += f"<start_of_turn>model\n{content}<end_of_turn>\n"
                    else:
                        print(f"  ⚠️  Skipping unknown role '{role}' on line {line_num}")
                        # Continue processing other messages

                if not text.strip():
                    print(f"  ⚠️  Skipping line {line_num}: no valid messages found")
                    skipped += 1
                    continue

                converted_data.append({"text": text})

            except json.JSONDecodeError as e:
                print(f"  ⚠️  JSON error on line {line_num}: {str(e)[:100]}")
                skipped += 1
                continue
            except Exception as e:
                print(f"  ⚠️  Error processing line {line_num}: {type(e).__name__}: {str(e)[:100]}")
                skipped += 1
                continue

    if not converted_data:
        print(f"  ❌ No valid examples found in input file")
        return 0

    # Write output file
    print(f"→ Writing Gemma format to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in converted_data:
            json_line = json.dumps(item, ensure_ascii=False) + "\n"
            f.write(json_line)

    print(f"✓ Converted {len(converted_data)} examples to Gemma format")
    if skipped > 0:
        print(f"  ⚠️  Skipped {skipped} invalid lines")

    # Show a sample entry
    if converted_data:
        print(f"\n→ Sample entry (first example):")
        sample_text = converted_data[0]["text"]
        preview = sample_text[:200].replace("\n", "\\n")
        print(f"  {preview}...")
        has_turn_tags = "<start_of_turn>" in sample_text and "<end_of_turn>" in sample_text
        if has_turn_tags:
            print(f"  ✓ Gemma turn tags detected")
        else:
            print(f"  ⚠️  No Gemma turn tags detected in sample")

    return len(converted_data)


def main():
    """Main entry point."""
    train_input = project_root / INPUT_TRAIN
    train_output = project_root / OUTPUT_TRAIN
    valid_input = project_root / INPUT_VALID
    valid_output = project_root / OUTPUT_VALID

    print(f"{'='*70}")
    print(f"Converting MLX Training Data to Gemma Format")
    print(f"{'='*70}\n")

    # Convert training data
    train_count = convert_to_gemma_format(train_input, train_output)

    # Convert validation data if it exists
    valid_count = 0
    if valid_input.exists():
        print()
        valid_count = convert_to_gemma_format(valid_input, valid_output)
    else:
        print(f"\n→ Validation file not found: {valid_input}")
        print(f"  ℹ️  Skipping validation data conversion")

    # Summary
    print(f"\n{'='*70}")
    if train_count > 0:
        print(f"✅ SUCCESS")
        print(f"{'='*70}")
        print(f"   Training data: {train_count} examples")
        print(f"   Output: {train_output}")
        if valid_count > 0:
            print(f"   Validation data: {valid_count} examples")
            print(f"   Output: {valid_output}")
        print(f"   Format: Gemma JSONL (text with <start_of_turn> tags)")
        print(f"{'='*70}")
    else:
        print(f"❌ No examples converted. Check input files and errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

