#!/usr/bin/env python3
"""
Count tokens in MLX training data to determine max_seq_length.

Usage:
    python3 scripts/phase4-fine-tune-model/count_tokens.py [--file path/to/train.jsonl] [--model HuggingFaceTB/SmolLM3-3B]
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import argparse

try:
    from transformers import AutoTokenizer
except ImportError:
    print("❌ Error: transformers library not installed")
    print("   Install with: pip install transformers")
    sys.exit(1)


def count_tokens_for_entry(tokenizer, entry: Dict) -> int:
    """
    Count tokens for a single training entry in MLX format.
    
    MLX format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    if "messages" not in entry:
        return 0
    
    # Use the tokenizer's chat template to format the messages
    # This matches how MLX will tokenize during training
    try:
        # Apply chat template to format messages
        formatted = tokenizer.apply_chat_template(
            entry["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        # Tokenize the formatted string
        tokens = tokenizer.encode(formatted, add_special_tokens=True)
        return len(tokens)
    except Exception as e:
        # Fallback: manually tokenize messages
        total_tokens = 0
        for message in entry["messages"]:
            content = message.get("content", "")
            if content:
                tokens = tokenizer.encode(content, add_special_tokens=False)
                total_tokens += len(tokens)
        # Add special tokens for chat format (approximate)
        return total_tokens + 10  # Rough estimate for chat formatting tokens


def analyze_training_data(file_path: Path, model_name: str = "HuggingFaceTB/SmolLM3-3B"):
    """Analyze token counts in training data."""
    print(f"→ Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        sys.exit(1)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    print(f"→ Analyzing tokens in: {file_path}")
    
    token_counts = []
    max_tokens = 0
    max_entry_idx = 0
    max_entry = None
    
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                entry = json.loads(line)
                token_count = count_tokens_for_entry(tokenizer, entry)
                token_counts.append(token_count)
                
                if token_count > max_tokens:
                    max_tokens = token_count
                    max_entry_idx = idx
                    max_entry = entry
                    
            except json.JSONDecodeError as e:
                print(f"  ⚠️  JSON error on line {idx}: {e}")
                continue
            except Exception as e:
                print(f"  ⚠️  Error on line {idx}: {e}")
                continue
    
    if not token_counts:
        print("❌ No valid entries found")
        return
    
    # Statistics
    total_entries = len(token_counts)
    avg_tokens = sum(token_counts) / total_entries
    sorted_counts = sorted(token_counts)
    median_tokens = sorted_counts[total_entries // 2]
    p95_tokens = sorted_counts[int(total_entries * 0.95)]
    p99_tokens = sorted_counts[int(total_entries * 0.99)]
    
    print(f"\n📊 Token Count Statistics:")
    print(f"   Total entries: {total_entries:,}")
    print(f"   Min tokens:    {min(token_counts):,}")
    print(f"   Max tokens:    {max_tokens:,} ← Line {max_entry_idx}")
    print(f"   Avg tokens:    {avg_tokens:,.1f}")
    print(f"   Median tokens: {median_tokens:,}")
    print(f"   95th percentile: {p95_tokens:,}")
    print(f"   99th percentile: {p99_tokens:,}")
    
    print(f"\n💡 Recommended max_seq_length:")
    print(f"   Minimum: {max_tokens} (to fit all entries)")
    print(f"   Recommended: {max_tokens + 50} (add buffer for safety)")
    print(f"   For 95% coverage: {p95_tokens}")
    print(f"   For 99% coverage: {p99_tokens}")
    
    if max_entry:
        print(f"\n📝 Longest entry (line {max_entry_idx}, {max_tokens} tokens):")
        messages = max_entry.get("messages", [])
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"   [{role}]: {preview}")
    
    # Show distribution
    print(f"\n📈 Distribution:")
    bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, float('inf')]
    for i in range(len(bins) - 1):
        count = sum(1 for tc in token_counts if bins[i] <= tc < bins[i+1])
        pct = (count / total_entries) * 100
        if bins[i+1] == float('inf'):
            print(f"   {bins[i]:4.0f}+ tokens: {count:4d} entries ({pct:5.1f}%)")
        else:
            print(f"   {bins[i]:4.0f}-{bins[i+1]:4.0f} tokens: {count:4d} entries ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Count tokens in MLX training data")
    parser.add_argument(
        "--file",
        type=str,
        default="data/mlx_training_data/train.jsonl",
        help="Path to training JSONL file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolLM3-3B",
        help="Model name for tokenizer"
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    file_path = project_root / args.file
    
    analyze_training_data(file_path, args.model)


if __name__ == "__main__":
    main()
























