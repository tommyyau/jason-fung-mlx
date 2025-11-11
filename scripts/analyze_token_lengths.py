#!/usr/bin/env python3
"""
Analyze Token Lengths in Training Data

This script analyzes your training data to check if any Q&A pairs exceed
the max_seq_length (default 1024 tokens), which would cause truncation.
"""

import json
import sys
from pathlib import Path
from collections import Counter

def count_tokens_approximate(text: str) -> int:
    """
    Approximate token count using simple heuristic.
    Real tokenizers vary, but this gives a reasonable estimate.
    Rule of thumb: ~4 characters per token for English text.
    """
    return len(text) // 4

def count_tokens_accurate(text: str, model_name: str = "gpt2") -> int:
    """
    Accurate token count using tiktoken (if available).
    Falls back to approximate if tiktoken not installed.
    """
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4/Llama-like
        return len(encoding.encode(text))
    except ImportError:
        print("ℹ️  tiktoken not installed, using approximate count (install: pip install tiktoken)")
        return count_tokens_approximate(text)

def analyze_jsonl_file(file_path: Path, max_seq_length: int = 1024):
    """Analyze a JSONL file for token lengths."""

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return None

    token_lengths = []
    exceeds_max = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)

                # Handle MLX format: {"messages": [...]}
                if "messages" in item:
                    full_text = ""
                    for msg in item["messages"]:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        full_text += f"{role}: {content}\n"

                    token_count = count_tokens_accurate(full_text)
                    token_lengths.append(token_count)

                    if token_count > max_seq_length:
                        exceeds_max.append({
                            "line": line_num,
                            "tokens": token_count,
                            "preview": full_text[:100].replace('\n', ' ')
                        })

                # Handle Gemma format: {"text": "..."}
                elif "text" in item:
                    text = item["text"]
                    token_count = count_tokens_accurate(text)
                    token_lengths.append(token_count)

                    if token_count > max_seq_length:
                        exceeds_max.append({
                            "line": line_num,
                            "tokens": token_count,
                            "preview": text[:100].replace('\n', ' ')
                        })

            except json.JSONDecodeError:
                print(f"⚠️  Skipping invalid JSON on line {line_num}")
                continue

    return {
        "token_lengths": token_lengths,
        "exceeds_max": exceeds_max
    }

def print_statistics(data, max_seq_length: int, file_name: str):
    """Print statistics about token lengths."""

    token_lengths = data["token_lengths"]
    exceeds_max = data["exceeds_max"]

    if not token_lengths:
        print(f"❌ No data found in {file_name}")
        return

    # Calculate statistics
    min_tokens = min(token_lengths)
    max_tokens = max(token_lengths)
    avg_tokens = sum(token_lengths) / len(token_lengths)
    median_tokens = sorted(token_lengths)[len(token_lengths) // 2]

    # Percentiles
    sorted_lengths = sorted(token_lengths)
    p90 = sorted_lengths[int(len(sorted_lengths) * 0.90)]
    p95 = sorted_lengths[int(len(sorted_lengths) * 0.95)]
    p99 = sorted_lengths[int(len(sorted_lengths) * 0.99)]

    # Count by range
    ranges = {
        "0-256": sum(1 for t in token_lengths if t <= 256),
        "257-512": sum(1 for t in token_lengths if 256 < t <= 512),
        "513-1024": sum(1 for t in token_lengths if 512 < t <= 1024),
        "1025-2048": sum(1 for t in token_lengths if 1024 < t <= 2048),
        "2049+": sum(1 for t in token_lengths if t > 2048),
    }

    print(f"\n{'='*70}")
    print(f"📊 Token Length Analysis: {file_name}")
    print(f"{'='*70}")
    print(f"Total examples: {len(token_lengths)}")
    print(f"Current max_seq_length: {max_seq_length}")
    print()

    print("📈 Statistics:")
    print(f"  Min tokens:     {min_tokens:,}")
    print(f"  Max tokens:     {max_tokens:,}")
    print(f"  Average:        {avg_tokens:,.1f}")
    print(f"  Median:         {median_tokens:,}")
    print(f"  90th percentile: {p90:,}")
    print(f"  95th percentile: {p95:,}")
    print(f"  99th percentile: {p99:,}")
    print()

    print("📊 Distribution:")
    for range_name, count in ranges.items():
        pct = (count / len(token_lengths)) * 100
        bar = "█" * int(pct / 2)
        print(f"  {range_name:12} : {count:4} ({pct:5.1f}%) {bar}")
    print()

    # Truncation warning
    if exceeds_max:
        pct_truncated = (len(exceeds_max) / len(token_lengths)) * 100
        print(f"⚠️  WARNING: {len(exceeds_max)} examples ({pct_truncated:.1f}%) exceed max_seq_length!")
        print()
        print("🔴 Examples that will be truncated:")
        for item in exceeds_max[:5]:  # Show first 5
            print(f"  Line {item['line']:4}: {item['tokens']:,} tokens")
            print(f"    Preview: {item['preview']}...")

        if len(exceeds_max) > 5:
            print(f"    ... and {len(exceeds_max) - 5} more")
        print()

        # Recommendation
        print("💡 RECOMMENDATION:")
        if p95 <= max_seq_length:
            print(f"  ✅ Current max_seq_length ({max_seq_length}) is fine for 95% of your data.")
            print(f"  📝 {len(exceeds_max)} long examples will be truncated (acceptable loss).")
        elif p95 <= 2048:
            recommended = 2048
            print(f"  ⚠️  Consider increasing max_seq_length to {recommended}")
            print(f"     This will prevent truncation for 95% of examples.")
        else:
            recommended = p95
            print(f"  ⚠️  Consider increasing max_seq_length to {recommended}")
            print(f"     This will prevent truncation for 95% of examples.")

        if max_tokens > 4096:
            print(f"  ⚠️  Some examples are very long ({max_tokens:,} tokens).")
            print(f"     Consider splitting long Q&A pairs or filtering them out.")
    else:
        print(f"✅ All examples fit within max_seq_length ({max_seq_length})")
        print()
        print("💡 RECOMMENDATION:")
        if p95 < max_seq_length * 0.5:
            recommended = max(512, p95)
            print(f"  You could potentially DECREASE max_seq_length to {recommended}")
            print(f"  This would save memory and speed up training.")
        else:
            print(f"  Current max_seq_length ({max_seq_length}) is optimal.")

    print(f"{'='*70}\n")

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze token lengths in training data"
    )
    parser.add_argument(
        "--train",
        default="data/mlx_training_data/train.jsonl",
        help="Path to training data (default: data/mlx_training_data/train.jsonl)"
    )
    parser.add_argument(
        "--valid",
        default="data/mlx_training_data/valid.jsonl",
        help="Path to validation data (default: data/mlx_training_data/valid.jsonl)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Current max_seq_length setting (default: 1024)"
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    train_file = project_root / args.train
    valid_file = project_root / args.valid

    print(f"\n🔍 Analyzing Token Lengths")
    print(f"{'='*70}\n")

    # Analyze training data
    if train_file.exists():
        train_data = analyze_jsonl_file(train_file, args.max_seq_length)
        if train_data:
            print_statistics(train_data, args.max_seq_length, "train.jsonl")
    else:
        print(f"❌ Training file not found: {train_file}")
        print(f"   Make sure you've run the data preparation scripts first.")
        sys.exit(1)

    # Analyze validation data
    if valid_file.exists():
        valid_data = analyze_jsonl_file(valid_file, args.max_seq_length)
        if valid_data:
            print_statistics(valid_data, args.max_seq_length, "valid.jsonl")
    else:
        print(f"ℹ️  Validation file not found: {valid_file}")

if __name__ == "__main__":
    main()
