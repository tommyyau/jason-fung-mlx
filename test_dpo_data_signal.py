#!/usr/bin/env python3
"""
Test if DPO data actually has preference signal
Check if chosen responses have higher probability than rejected
"""
import json
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

print("="*70)
print("TESTING DPO DATA SIGNAL")
print("="*70)

# Load model and data
model_path = "ibm-granite/granite-4.0-h-micro"
data_path = "data/mlx_training_data/dpo_train.jsonl"

print(f"\nLoading model: {model_path}")
model, tokenizer = load(model_path)

print(f"Loading data: {data_path}")
data = []
with open(data_path, 'r') as f:
    for i, line in enumerate(f):
        if i >= 5:  # Only check first 5 examples
            break
        data.append(json.loads(line))

print(f"Loaded {len(data)} examples")

# Function to compute log probability
def compute_logp(text):
    tokens = tokenizer.encode(text)
    input_ids = mx.array(tokens)[None, :]
    logits = model(input_ids)

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    ce = nn.losses.cross_entropy(shift_logits, shift_labels, reduction='none')
    log_probs = -ce

    return log_probs.sum().item()

# Function to compute log probability with response masking
def compute_logp_response_only(prompt, response):
    # Tokenize separately
    prompt_tokens = tokenizer.encode(prompt)
    prompt_len = len(prompt_tokens)

    # Tokenize full sequence
    full_text = prompt + response
    full_tokens = tokenizer.encode(full_text)

    # Create response mask
    response_mask = mx.concatenate([
        mx.zeros(min(prompt_len, len(full_tokens))),
        mx.ones(max(0, len(full_tokens) - prompt_len))
    ])

    # Compute log probs
    input_ids = mx.array(full_tokens)[None, :]
    logits = model(input_ids)

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    ce = nn.losses.cross_entropy(shift_logits, shift_labels, reduction='none')
    log_probs = -ce

    # Apply response mask
    shift_response_mask = response_mask[1:]
    masked_log_probs = log_probs[0] * shift_response_mask

    return masked_log_probs.sum().item()

# Check each example
print("\n" + "="*70)
print("CHECKING PREFERENCE SIGNAL")
print("="*70)

chosen_wins = 0
rejected_wins = 0

for i, item in enumerate(data):
    prompt = item['prompt']
    chosen = item['chosen']
    rejected = item['rejected']

    print(f"\nExample {i+1}:")
    print(f"  Prompt: {prompt[:60]}...")
    print(f"  Chosen: {chosen[:80]}...")
    print(f"  Rejected: {rejected[:80]}...")

    # Method 1: Full sequence log prob
    chosen_logp_full = compute_logp(prompt + chosen)
    rejected_logp_full = compute_logp(prompt + rejected)

    print(f"\n  Full sequence log probs:")
    print(f"    Chosen:   {chosen_logp_full:.2f}")
    print(f"    Rejected: {rejected_logp_full:.2f}")
    print(f"    Difference: {chosen_logp_full - rejected_logp_full:.2f}")

    # Method 2: Response-only log prob (correct DPO)
    chosen_logp_resp = compute_logp_response_only(prompt, chosen)
    rejected_logp_resp = compute_logp_response_only(prompt, rejected)

    print(f"\n  Response-only log probs (CORRECT):")
    print(f"    Chosen:   {chosen_logp_resp:.2f}")
    print(f"    Rejected: {rejected_logp_resp:.2f}")
    print(f"    Difference: {chosen_logp_resp - rejected_logp_resp:.2f}")

    if chosen_logp_resp > rejected_logp_resp:
        print(f"    ✅ Chosen has HIGHER log prob (good signal)")
        chosen_wins += 1
    elif chosen_logp_resp < rejected_logp_resp:
        print(f"    ❌ Rejected has HIGHER log prob (WRONG signal!)")
        rejected_wins += 1
    else:
        print(f"    ⚠️  Same log prob (no signal)")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Chosen wins: {chosen_wins}/{len(data)} ({chosen_wins/len(data)*100:.1f}%)")
print(f"Rejected wins: {rejected_wins}/{len(data)} ({rejected_wins/len(data)*100:.1f}%)")

if chosen_wins > rejected_wins:
    print("\n✅ Data has CORRECT preference signal (chosen > rejected)")
    print("   DPO should learn to prefer chosen responses")
elif rejected_wins > chosen_wins:
    print("\n❌ Data has INVERTED signal (rejected > chosen)!")
    print("   DPO will learn the OPPOSITE of what we want!")
else:
    print("\n⚠️  No clear signal in data")
    print("   DPO won't learn much")

# Check if signal is STRONG enough
if chosen_wins > 0:
    avg_diff = sum(
        compute_logp_response_only(item['prompt'], item['chosen']) -
        compute_logp_response_only(item['prompt'], item['rejected'])
        for item in data
    ) / len(data)

    print(f"\nAverage log prob difference: {avg_diff:.4f}")
    if abs(avg_diff) < 0.5:
        print("⚠️  Signal is VERY WEAK (< 0.5)")
        print("   May need more training steps or higher learning rate")
    elif abs(avg_diff) < 2.0:
        print("⚠️  Signal is weak (< 2.0)")
    else:
        print("✅ Signal is strong enough for DPO")

print("\n" + "="*70)
