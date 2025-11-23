#!/usr/bin/env python3
"""
Test if BPE tokenization causes boundary alignment issues
"""
from mlx_lm import load

print("="*70)
print("TESTING TOKENIZATION BOUNDARY BUG")
print("="*70)

model_path = "ibm-granite/granite-4.0-h-micro"
print(f"\nLoading tokenizer from {model_path}...")
_, tokenizer = load(model_path)

# Example from DPO data
prompt = "Question: Should I count calories to lose weight?\nAnswer:"
chosen = " No, focus on reducing insulin through fasting and low-carb eating."
rejected = " Yes, create a calorie deficit by eating less than you burn."

print(f"\nPrompt: {repr(prompt)}")
print(f"Chosen: {repr(chosen)}")
print(f"Rejected: {repr(rejected)}")

# Method 1: Tokenize separately (what the code THINKS it's doing)
print("\n" + "="*70)
print("METHOD 1: Tokenize prompt separately")
print("="*70)

prompt_tokens = tokenizer.encode(prompt)
print(f"Prompt tokens: {len(prompt_tokens)} tokens")
print(f"  First 5: {prompt_tokens[:5]}")
print(f"  Last 5: {prompt_tokens[-5:]}")

# Method 2: Tokenize together (what ACTUALLY happens)
print("\n" + "="*70)
print("METHOD 2: Tokenize prompt+chosen together")
print("="*70)

chosen_text = prompt + chosen
chosen_tokens_together = tokenizer.encode(chosen_text)
print(f"Full sequence: {len(chosen_tokens_together)} tokens")
print(f"  First 5: {chosen_tokens_together[:5]}")

# The critical question: Are the first N tokens the same?
prompt_len = len(prompt_tokens)
print(f"\n" + "="*70)
print("COMPARISON")
print("="*70)

# Check if first prompt_len tokens match
if chosen_tokens_together[:prompt_len] == prompt_tokens:
    print(f"✅ First {prompt_len} tokens MATCH")
    print("   Separate tokenization gives same result as combined.")
else:
    print(f"❌ BOUNDARY MISMATCH!")
    print(f"   Expected first {prompt_len} tokens to match prompt,")
    print("   but BPE tokenization is different!")

    # Find where they diverge
    for i in range(min(len(prompt_tokens), len(chosen_tokens_together))):
        if prompt_tokens[i] != chosen_tokens_together[i]:
            print(f"\n   Divergence at position {i}:")
            print(f"     Prompt alone: {prompt_tokens[i-2:i+3]}")
            print(f"     Full sequence: {chosen_tokens_together[i-2:i+3]}")
            break

    # This means the response mask is MISALIGNED!
    print(f"\n   ⚠️  CRITICAL: Response mask will be wrong!")
    print(f"      Current code creates mask with {prompt_len} zeros,")
    print(f"      but actual prompt in full sequence is different length!")

# Test if we can fix it by tokenizing separately and concatenating
print("\n" + "="*70)
print("FIX: Tokenize separately and concatenate")
print("="*70)

try:
    # Try without special tokens
    chosen_response_tokens = tokenizer.encode(chosen, add_special_tokens=False)
    print(f"✅ tokenizer.encode() supports add_special_tokens=False")
    print(f"   Prompt: {len(prompt_tokens)} tokens")
    print(f"   Response: {len(chosen_response_tokens)} tokens")

    # Concatenate
    chosen_tokens_concat = prompt_tokens + chosen_response_tokens
    print(f"   Concatenated: {len(chosen_tokens_concat)} tokens")

    if chosen_tokens_concat == list(chosen_tokens_together):
        print(f"✅ Concatenation matches joint tokenization")
    else:
        print(f"⚠️  Concatenation DOESN'T match joint tokenization")
        print(f"   Joint: {len(chosen_tokens_together)} tokens")
        print(f"   Concat: {len(chosen_tokens_concat)} tokens")
        print(f"   This is expected due to BPE boundary effects")
        print(f"\n   SOLUTION: Use concatenated tokens + aligned mask!")

except TypeError as e:
    print(f"❌ add_special_tokens not supported: {e}")
    print(f"   Will need different approach to fix tokenization bug")

print("\n" + "="*70)
