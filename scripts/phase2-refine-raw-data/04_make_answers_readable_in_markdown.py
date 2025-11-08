#!/usr/bin/env python3
"""
Step 10 – Format Answers to Markdown (Optional)
──────────────────────────────────────────────
Turns `data/generated_answers.jsonl` into a readable Markdown review document,
preserving formatting and appending metadata for each Q&A pair.
"""

import json
from pathlib import Path
from typing import List, Dict

# ─────────────────────────────
# Config
# ─────────────────────────────
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
INPUT_FILE = project_root / "data" / "generated_answers.jsonl"
OUTPUT_FILE = project_root / "data" / "formatted_answers.md"


def load_answers(input_file: Path) -> List[Dict]:
    """Load answers from JSONL file."""
    answers = []

    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        return answers

    print(f"→ Reading answers from: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
                answers.append(entry)
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Warning: Skipping invalid JSON on line {line_num}: {str(e)[:100]}")
                continue

    print(f"✓ Loaded {len(answers)} Q&A pairs")
    return answers


def format_answer_to_markdown(entry: Dict, index: int) -> str:
    """Format a single Q&A entry to Markdown."""
    question = entry.get("question", "")
    answer = entry.get("answer", "")
    video_title = entry.get("video_title", "Unknown")
    video_id = entry.get("video_id", "")
    tags = entry.get("tags", [])

    # Build Markdown content
    md_lines = []

    # Question as heading
    md_lines.append(f"## {index}. {question}\n")

    # Answer (preserve all formatting - it's already in Markdown format)
    formatted_answer = answer.replace("\\n", "\n")
    md_lines.append(formatted_answer)
    md_lines.append("")  # Empty line after answer

    # Metadata section
    md_lines.append("**Metadata:**")
    md_lines.append(f"- **Video Title:** {video_title}")
    md_lines.append(f"- **Video ID:** {video_id}")
    if tags:
        tags_str = ", ".join([f"`{tag}`" for tag in tags])
        md_lines.append(f"- **Tags:** {tags_str}")
    md_lines.append("")  # Empty line
    md_lines.append("---")  # Separator
    md_lines.append("")  # Empty line

    return "\n".join(md_lines)


def generate_markdown(answers: List[Dict], output_file: Path) -> None:
    """Generate Markdown file from answers."""
    print(f"→ Generating Markdown file: {output_file}")

    md_content = []

    # Header
    md_content.append("# Generated Answers - Formatted Review")
    md_content.append("")
    md_content.append(f"This document contains {len(answers)} question-answer pairs with all formatting preserved.")
    md_content.append("")
    md_content.append("---")
    md_content.append("")

    # Process each answer
    for i, entry in enumerate(answers, 1):
        md_entry = format_answer_to_markdown(entry, i)
        md_content.append(md_entry)

    # Write file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))

    print(f"✓ Generated Markdown file with {len(answers)} Q&A pairs")
    print(f"  Output: {output_file}")


def main():
    """Main function."""
    print("=" * 60)
    print("Format Answers to Markdown")
    print("=" * 60)
    print()

    # Load answers
    answers = load_answers(INPUT_FILE)

    if not answers:
        print("❌ No answers to process. Exiting.")
        return

    # Generate Markdown
    generate_markdown(answers, OUTPUT_FILE)

    print()
    print("✓ Done!")


if __name__ == "__main__":
    main()


