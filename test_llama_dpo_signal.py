#!/usr/bin/env python3
"""
Test if Llama 3B has trainable DPO signal
Check if chosen responses have reasonable log prob difference from rejected
"""
import json
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

print("="*70)
print("TESTING DPO SIGNAL FOR LLAMA 3.2-3B-Instruct")
print("="*70)

# Load model and data
model_path = "mlx-community/Llama-3.2-3B-Instruct"
data_path = "data/mlx_training_data/dpo_train.jsonl"

print(f"\nLoading model: {model_path}")
model, tokenizer = load(model_path)

print(f"Loading data: {data_path}")
data = []
with open(data_path, 'r') as f:
    for i, line in enumerate(f):
        if i >= 10:  # Check first 10 examples
            break
        data.append(json.loads(line))

print(f"Loaded {len(data)} examples for testing")

def compute_logp_response_only(prompt, response, normalize=True):
    """Compute log probability for response only (not prompt)"""
    prompt_tokens = tokenizer.encode(prompt)
    prompt_len = len(prompt_tokens)

    full_text = prompt + response
    full_tokens = tokenizer.encode(full_text)

    # Create response mask
    response_mask = mx.concatenate([
        mx.zeros(min(prompt_len, len(full_tokens))),
        mx.ones(max(0, len(full_tokens) - prompt_len))
    ])

    input_ids = mx.array(full_tokens)[None, :]
    logits = model(input_ids)

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    ce = nn.losses.cross_entropy(shift_logits, shift_labels, reduction='none')
    log_probs = -ce

    shift_response_mask = response_mask[1:]
    masked_log_probs = log_probs[0] * shift_response_mask

    if normalize:
        num_tokens = shift_response_mask.sum()
        return (masked_log_probs.sum() / mx.maximum(num_tokens, mx.array(1.0))).item()
    else:
        return masked_log_probs.sum().item()

# Check each example
print("\n" + "="*70)
print("CHECKING PREFERENCE SIGNAL (normalized log probs)")
print("="*70)

chosen_wins = 0
rejected_wins = 0
differences = []

for i, item in enumerate(data):
    prompt = item['prompt']
    chosen = item['chosen']
    rejected = item['rejected']

    chosen_logp = compute_logp_response_only(prompt, chosen)
    rejected_logp = compute_logp_response_only(prompt, rejected)
    diff = chosen_logp - rejected_logp
    differences.append(diff)

    print(f"\nExample {i+1}:")
    print(f"  Chosen logp:   {chosen_logp:.4f}")
    print(f"  Rejected logp: {rejected_logp:.4f}")
    print(f"  Difference:    {diff:.4f}", end="")

    if diff > 0:
        print(" ✅ (chosen preferred)")
        chosen_wins += 1
    else:
        print(" ❌ (rejected preferred)")
        rejected_wins += 1

# Summary
avg_diff = sum(differences) / len(differences)
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Chosen wins: {chosen_wins}/{len(data)} ({chosen_wins/len(data)*100:.1f}%)")
print(f"Rejected wins: {rejected_wins}/{len(data)} ({rejected_wins/len(data)*100:.1f}%)")
print(f"Average difference: {avg_diff:.4f}")

print("\n" + "="*70)
print("TRAINABILITY ASSESSMENT")
print("="*70)

if abs(avg_diff) > 2.0:
    print("⚠️  LARGE BIAS DETECTED")
    print(f"   Average difference of {avg_diff:.4f} is quite large")
    print("   The model already has a strong preference")
    print("   DPO may struggle to shift this")
elif abs(avg_diff) > 0.5:
    print("✅ MODERATE SIGNAL - TRAINABLE")
    print(f"   Average difference of {avg_diff:.4f} is in good range")
    print("   DPO should be able to shift preferences")
elif abs(avg_diff) > 0.1:
    print("✅ WEAK SIGNAL - TRAINABLE")
    print(f"   Average difference of {avg_diff:.4f} is weak but workable")
    print("   May need more epochs or higher learning rate")
else:
    print("⚠️  VERY WEAK SIGNAL")
    print(f"   Average difference of {avg_diff:.4f} is very small")
    print("   Model treats chosen/rejected almost equally")

if rejected_wins > chosen_wins:
    print("\n⚠️  Model prefers REJECTED over CHOSEN!")
    print("   This is actually GOOD for DPO - clear room for improvement")
else:
    print("\n✅ Model already prefers CHOSEN over REJECTED")
    print("   DPO will reinforce this preference")

print("="*70)
