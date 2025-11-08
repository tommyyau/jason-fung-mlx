#!/usr/bin/env python3
"""
Step 04 – Convert Answers to MLX Format
──────────────────────────────────────
Transforms `data/generated_answers.jsonl` into MLX instruction-tuning JSONL entries
(`instruction`, `output`) while preserving markdown formatting for training.
"""

import json
import sys
from pathlib import Path

# ─────────────────────────────
# Config
# ─────────────────────────────
INPUT_FILE = "data/generated_answers.jsonl"
OUTPUT_FILE = "data/generated_answers_mlx.jsonl"

# Get project root
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent


def convert_to_mlx_format(input_path: Path, output_path: Path) -> int:
    """
    Convert generated answers JSONL to MLX training format.

    Args:
        input_path: Path to generated_answers.jsonl
        output_path: Path to output MLX format JSONL

    Returns:
        Number of examples converted
    """
    print(f"→ Loading answers from: {input_path}")

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return 0

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

                # Create MLX format entry
                # Preserve all formatting in the answer (markdown, newlines, etc.)
                mlx_entry = {
                    "instruction": question,
                    "output": answer,  # All formatting preserved as-is
                }

                examples.append(mlx_entry)

            except json.JSONDecodeError as e:
                print(f"  ⚠️  JSON error on line {line_num}: {str(e)[:100]}")
                skipped += 1
                continue
            except Exception as e:
                print(f"  ⚠️  Error processing line {line_num}: {type(e).__name__}: {str(e)[:100]}")
                skipped += 1
                continue

    if not examples:
        print(f"❌ No valid examples found in input file")
        return 0

    # Write output file
    print(f"→ Writing MLX format to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            # Use ensure_ascii=False to preserve unicode characters
            # This also preserves all formatting (newlines, markdown, etc.)
            json_line = json.dumps(example, ensure_ascii=False) + "\n"
            f.write(json_line)

    print(f"✓ Converted {len(examples)} examples to MLX format")
    if skipped > 0:
        print(f"  ⚠️  Skipped {skipped} invalid lines")

    # Show a sample entry to verify formatting is preserved
    if examples:
        print(f"\n→ Sample entry (first example):")
        sample = examples[0]
        print(f"  Instruction: {sample['instruction'][:80]}...")
        print(f"  Output preview: {sample['output'][:150]}...")
        has_formatting = any(marker in sample["output"] for marker in ["**", "\n\n", "- ", "* "])
        if has_formatting:
            print(f"  ✓ Formatting detected (bold, lists, paragraphs)")
        else:
            print(f"  ℹ️  No markdown formatting detected in sample")

    return len(examples)


def main():
    """Main entry point."""
    input_path = project_root / INPUT_FILE
    output_path = project_root / OUTPUT_FILE

    print(f"{'='*70}")
    print(f"Converting Generated Answers to MLX Training Format")
    print(f"{'='*70}\n")

    count = convert_to_mlx_format(input_path, output_path)

    if count > 0:
        print(f"\n{'='*70}")
        print(f"✅ SUCCESS")
        print(f"{'='*70}")
        print(f"   Converted: {count} examples")
        print(f"   Output: {output_path}")
        print(f"   Format: MLX JSONL (instruction, output)")
        print(f"   All formatting preserved in answers")
        print(f"{'='*70}")
    else:
        print(f"\n❌ No examples converted. Check input file and errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()


