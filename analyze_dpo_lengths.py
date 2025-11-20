#!/usr/bin/env python3
"""
Analyze DPO data sequence lengths
"""
import json
from pathlib import Path
from mlx_lm import load

# Load tokenizer
print("Loading tokenizer...")
_, tokenizer = load("ibm-granite/granite-4.0-h-micro")

# Load DPO data
data_path = "data/mlx_training_data/dpo_train.jsonl"
print(f"Analyzing {data_path}...\n")

chosen_lengths = []
rejected_lengths = []

with open(data_path, 'r') as f:
    for line in f:
        if line.strip():
            item = json.loads(line)
            
            # Tokenize chosen and rejected (including prompt)
            chosen_text = item['prompt'] + item['chosen']
            rejected_text = item['prompt'] + item['rejected']
            
            chosen_tokens = tokenizer.encode(chosen_text)
            rejected_tokens = tokenizer.encode(rejected_text)
            
            chosen_lengths.append(len(chosen_tokens))
            rejected_lengths.append(len(rejected_tokens))

# Statistics
print(f"{'='*70}")
print(f"SEQUENCE LENGTH ANALYSIS")
print(f"{'='*70}")
print(f"Total examples: {len(chosen_lengths)}")
print()

print(f"CHOSEN (prompt + chosen answer):")
print(f"  Min: {min(chosen_lengths)} tokens")
print(f"  Max: {max(chosen_lengths)} tokens")
print(f"  Mean: {sum(chosen_lengths)/len(chosen_lengths):.1f} tokens")
print(f"  Median: {sorted(chosen_lengths)[len(chosen_lengths)//2]} tokens")
print()

print(f"REJECTED (prompt + rejected answer):")
print(f"  Min: {min(rejected_lengths)} tokens")
print(f"  Max: {max(rejected_lengths)} tokens")
print(f"  Mean: {sum(rejected_lengths)/len(rejected_lengths):.1f} tokens")
print(f"  Median: {sorted(rejected_lengths)[len(rejected_lengths)//2]} tokens")
print()

# Check how many exceed 1024
chosen_over_1024 = sum(1 for l in chosen_lengths if l > 1024)
rejected_over_1024 = sum(1 for l in rejected_lengths if l > 1024)

print(f"TRUNCATION ANALYSIS (max_seq_length=1024):")
print(f"  Chosen examples > 1024 tokens: {chosen_over_1024} ({chosen_over_1024/len(chosen_lengths)*100:.1f}%)")
print(f"  Rejected examples > 1024 tokens: {rejected_over_1024} ({rejected_over_1024/len(rejected_lengths)*100:.1f}%)")
print()

if chosen_over_1024 > 0 or rejected_over_1024 > 0:
    print(f"⚠️  WARNING: Some examples will be truncated at 1024 tokens!")
    print(f"   Consider increasing max_seq_length in config to 2048 or 4096")
else:
    print(f"✅ All examples fit within 1024 tokens - no truncation needed!")

print(f"{'='*70}")

# Show longest examples
print(f"\nLONGEST EXAMPLES:")
max_chosen_idx = chosen_lengths.index(max(chosen_lengths))
max_rejected_idx = rejected_lengths.index(max(rejected_lengths))

with open(data_path, 'r') as f:
    for i, line in enumerate(f):
        if i == max_chosen_idx:
            item = json.loads(line)
            print(f"\nLongest CHOSEN ({max(chosen_lengths)} tokens):")
            print(f"  Prompt: {item['prompt'][:100]}...")
            print(f"  Chosen: {item['chosen'][:100]}...")
        if i == max_rejected_idx:
            item = json.loads(line)
            print(f"\nLongest REJECTED ({max(rejected_lengths)} tokens):")
            print(f"  Prompt: {item['prompt'][:100]}...")
            print(f"  Rejected: {item['rejected'][:100]}...")
