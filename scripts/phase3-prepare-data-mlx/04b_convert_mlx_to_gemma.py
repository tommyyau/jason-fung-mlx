#!/usr/bin/env python3
"""
Step 04b – Convert MLX Training Data to Gemma Format
────────────────────────────────────────────────────
Converts MLX format JSONL files (with "messages" structure) to Gemma format
with <start_of_turn> and <end_of_turn> tags for Google's Gemma training.
Splits data: 1600 examples to train.jsonl, rest to valid.jsonl
"""

import json
import os
import sys
from pathlib import Path

# ─────────────────────────────
# Config
# ─────────────────────────────
INPUT_TRAIN = "data/mlx_training_data/train_original.jsonl"
INPUT_VALID = "data/mlx_training_data/valid_original.jsonl"
OUTPUT_TRAIN = "data/mlx_training_data/train.jsonl"
OUTPUT_VALID = "data/mlx_training_data/valid.jsonl"
TRAIN_SIZE = 1600  # Number of examples to put in train.jsonl

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
    valid_input = project_root / INPUT_VALID
    train_output = project_root / OUTPUT_TRAIN
    valid_output = project_root / OUTPUT_VALID

    print(f"{'='*70}")
    print(f"Converting MLX Training Data to Gemma Format")
    print(f"Split: {TRAIN_SIZE} examples → train.jsonl, rest → valid.jsonl")
    print(f"{'='*70}\n")

    # Load and convert all data
    all_converted = []
    
    # Process training file
    if train_input.exists():
        print(f"→ Loading from: {train_input}")
        with open(train_input, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if "messages" not in item or not isinstance(item["messages"], list):
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
                    
                    if text.strip():
                        all_converted.append({"text": text})
                except Exception as e:
                    print(f"  ⚠️  Error on line {line_num}: {str(e)[:100]}")
                    continue
    else:
        print(f"  ❌ Input file not found: {train_input}")
        sys.exit(1)
    
    # Process validation file if it exists
    if valid_input.exists():
        print(f"→ Loading from: {valid_input}")
        with open(valid_input, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if "messages" not in item or not isinstance(item["messages"], list):
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
                    
                    if text.strip():
                        all_converted.append({"text": text})
                except Exception as e:
                    print(f"  ⚠️  Error on line {line_num}: {str(e)[:100]}")
                    continue
    
    total_examples = len(all_converted)
    print(f"\n→ Total examples converted: {total_examples}")
    
    if total_examples == 0:
        print(f"❌ No examples converted. Check input files and errors above.")
        sys.exit(1)
    
    # Split data: first TRAIN_SIZE to train, rest to valid
    train_data = all_converted[:TRAIN_SIZE]
    valid_data = all_converted[TRAIN_SIZE:]
    
    print(f"→ Splitting: {len(train_data)} → train.jsonl, {len(valid_data)} → valid.jsonl")
    
    # Write train.jsonl
    train_output.parent.mkdir(parents=True, exist_ok=True)
    with open(train_output, "w", encoding="utf-8") as f:
        for item in train_data:
            json_line = json.dumps(item, ensure_ascii=False) + "\n"
            f.write(json_line)
    print(f"✓ Written {len(train_data)} examples to: {train_output}")
    
    # Write valid.jsonl
    if valid_data:
        with open(valid_output, "w", encoding="utf-8") as f:
            for item in valid_data:
                json_line = json.dumps(item, ensure_ascii=False) + "\n"
                f.write(json_line)
        print(f"✓ Written {len(valid_data)} examples to: {valid_output}")
    
    # Show sample
    if train_data:
        print(f"\n→ Sample entry (first example):")
        sample_text = train_data[0]["text"]
        preview = sample_text[:200].replace("\n", "\\n")
        print(f"  {preview}...")
        has_turn_tags = "<start_of_turn>" in sample_text and "<end_of_turn>" in sample_text
        if has_turn_tags:
            print(f"  ✓ Gemma turn tags detected")
        else:
            print(f"  ⚠️  No Gemma turn tags detected in sample")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ SUCCESS")
    print(f"{'='*70}")
    print(f"   Total examples: {total_examples}")
    print(f"   Training data: {len(train_data)} examples → {train_output}")
    print(f"   Validation data: {len(valid_data)} examples → {valid_output}")
    print(f"   Format: Gemma JSONL (text with <start_of_turn> tags)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

