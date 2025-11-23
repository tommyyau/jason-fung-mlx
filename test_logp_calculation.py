#!/usr/bin/env python3
"""
Test if log probability calculation is correct
"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

print("="*70)
print("TESTING LOG PROBABILITY CALCULATION")
print("="*70)

# Load model
model_path = "ibm-granite/granite-4.0-h-micro"
print(f"\nLoading model: {model_path}")
model, tokenizer = load(model_path)

# Simple test case
text = "Question: What is insulin?\nAnswer: Insulin is a hormone."
tokens = tokenizer.encode(text)

print(f"\nText: {repr(text)}")
print(f"Tokens: {len(tokens)} tokens")
print(f"Token IDs: {tokens[:10]}...")

# Convert to MLX array
input_ids = mx.array(tokens)[None, :]  # Add batch dimension
print(f"Input shape: {input_ids.shape}")

# Get model logits
logits = model(input_ids)
print(f"Logits shape: {logits.shape}")  # (batch, seq_len, vocab_size)

# Manual log probability calculation
print("\n" + "="*70)
print("MANUAL LOG PROB CALCULATION")
print("="*70)

# Shift for next-token prediction
shift_logits = logits[:, :-1, :]  # (batch, seq_len-1, vocab_size)
shift_labels = input_ids[:, 1:]    # (batch, seq_len-1)

print(f"Shift logits shape: {shift_logits.shape}")
print(f"Shift labels shape: {shift_labels.shape}")

# Method 1: Using cross_entropy (current implementation)
print("\nMethod 1: Using cross_entropy")
ce_per_token = nn.losses.cross_entropy(shift_logits, shift_labels, reduction='none')
print(f"  Cross entropy shape: {ce_per_token.shape}")
print(f"  CE per token (first 5): {ce_per_token[0, :5]}")

log_probs_method1 = -ce_per_token
print(f"  Log probs (first 5): {log_probs_method1[0, :5]}")

sum_log_prob_method1 = log_probs_method1.sum()
print(f"  Sum log prob: {sum_log_prob_method1.item():.4f}")

# Method 2: Manual calculation to verify
print("\nMethod 2: Manual calculation")
# Apply softmax to get probabilities
probs = mx.softmax(shift_logits, axis=-1)  # (batch, seq_len-1, vocab_size)

# Get probability of the actual next token
# For each position, get prob of the label token
batch_idx = 0
token_probs = []
for i in range(shift_labels.shape[1]):
    label_token = shift_labels[batch_idx, i].item()
    token_prob = probs[batch_idx, i, label_token].item()
    token_log_prob = mx.log(probs[batch_idx, i, label_token]).item()
    token_probs.append((i, label_token, token_prob, token_log_prob))

print(f"  First 5 token probabilities:")
for i, token_id, prob, log_prob in token_probs[:5]:
    print(f"    Position {i}, Token {token_id}: P={prob:.6f}, log(P)={log_prob:.4f}")

manual_sum = sum(log_prob for _, _, _, log_prob in token_probs)
print(f"  Manual sum log prob: {manual_sum:.4f}")

# Compare
print("\n" + "="*70)
print("COMPARISON")
print("="*70)
diff = abs(sum_log_prob_method1.item() - manual_sum)
print(f"Method 1 (CE-based): {sum_log_prob_method1.item():.4f}")
print(f"Method 2 (Manual):   {manual_sum:.4f}")
print(f"Difference:          {diff:.6f}")

if diff < 0.01:
    print("✅ Methods match! Log prob calculation is correct.")
else:
    print("❌ Methods DON'T match! Bug in log prob calculation!")

# Test with masking (the actual DPO case)
print("\n" + "="*70)
print("TESTING WITH RESPONSE MASK")
print("="*70)

# Say first 5 tokens are prompt, rest are response
prompt_len = 5
response_mask = mx.concatenate([
    mx.zeros(prompt_len),
    mx.ones(len(tokens) - prompt_len)
])

print(f"Response mask: {response_mask[:10]}")

# Apply mask (shifted)
shift_response_mask = response_mask[1:]  # Shift by 1
masked_log_probs = log_probs_method1[0] * shift_response_mask

print(f"Masked log probs (first 10): {masked_log_probs[:10]}")

sum_masked = masked_log_probs.sum().item()
print(f"Sum of masked log probs: {sum_masked:.4f}")

# Manual verification
expected_sum = sum(log_prob for i, (_, _, _, log_prob) in enumerate(token_probs) if i >= prompt_len - 1)
print(f"Expected (manual): {expected_sum:.4f}")

diff2 = abs(sum_masked - expected_sum)
if diff2 < 0.01:
    print("✅ Masking works correctly!")
else:
    print(f"❌ Masking bug! Difference: {diff2:.6f}")

print("\n" + "="*70)
