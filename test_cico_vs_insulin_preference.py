#!/usr/bin/env python3
"""
Test: Does Llama inherently prefer CICO or Insulin Model explanations?

If Llama strongly prefers CICO, DPO will struggle to flip this.
If neutral, DPO has a chance.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

# Test cases: Same question, two answers, similar length/style
TEST_CASES = [
    {
        "question": "Why do people gain weight?",
        "cico": "People gain weight when they consume more calories than they burn. The excess energy is stored as fat. To lose weight, you need to create a calorie deficit through eating less or exercising more.",
        "insulin": "People gain weight when insulin levels stay elevated. High insulin signals the body to store energy as fat and blocks fat burning. To lose weight, you need to lower insulin through diet timing and food choices."
    },
    {
        "question": "What's the best way to lose weight?",
        "cico": "The best way to lose weight is to track your calories and create a deficit. Aim for 500 fewer calories per day to lose about one pound per week. Any diet works as long as calories are reduced.",
        "insulin": "The best way to lose weight is to lower your insulin levels. Focus on when you eat, not just what. Intermittent fasting and reducing refined carbs keeps insulin low so your body can burn stored fat."
    },
    {
        "question": "Why do diets fail?",
        "cico": "Diets fail because people don't stick to their calorie targets. They underestimate portions, eat hidden calories, or give up when progress slows. Consistent tracking and willpower are key to success.",
        "insulin": "Diets fail because calorie restriction raises hunger hormones and slows metabolism. Your body fights back. Addressing insulin resistance through fasting and low-carb eating fixes the hormonal root cause."
    },
    {
        "question": "Is a calorie a calorie?",
        "cico": "Yes, a calorie is a calorie when it comes to weight. Whether from protein, carbs, or fat, excess calories get stored. The source matters for nutrition but not for the basic math of weight loss.",
        "insulin": "No, calories from different foods have different hormonal effects. Carbs spike insulin and promote storage. Protein and fat have minimal insulin response. The hormonal impact matters more than the calorie count."
    },
    {
        "question": "Why am I always hungry on a diet?",
        "cico": "You're hungry because your body is adjusting to fewer calories. This is normal and temporary. Eating more protein and fiber can help you feel fuller while staying within your calorie budget.",
        "insulin": "You're hungry because high insulin is blocking access to your fat stores. Your body can't burn its own fat for fuel, so it demands more food. Lowering insulin through fasting lets you access stored energy."
    },
]

def compute_avg_logp(model, tokenizer, text):
    """Compute average log probability per token."""
    tokens = tokenizer.encode(text)
    input_ids = mx.array(tokens)[None, :]
    logits = model(input_ids)

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    ce = nn.losses.cross_entropy(shift_logits, shift_labels, reduction='none')
    log_probs = -ce

    # Average per token
    return log_probs.mean().item()

def main():
    print("="*70)
    print("TESTING: Does Llama Prefer CICO or Insulin Model?")
    print("="*70)

    print("\nLoading Llama 3.2-3B-Instruct...")
    model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct")

    cico_wins = 0
    insulin_wins = 0
    total_cico_logp = 0
    total_insulin_logp = 0

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    for i, case in enumerate(TEST_CASES, 1):
        q = case["question"]

        # Format as Q&A
        cico_text = f"Question: {q}\nAnswer: {case['cico']}"
        insulin_text = f"Question: {q}\nAnswer: {case['insulin']}"

        cico_logp = compute_avg_logp(model, tokenizer, cico_text)
        insulin_logp = compute_avg_logp(model, tokenizer, insulin_text)

        diff = insulin_logp - cico_logp
        winner = "INSULIN" if diff > 0 else "CICO"

        if diff > 0:
            insulin_wins += 1
        else:
            cico_wins += 1

        total_cico_logp += cico_logp
        total_insulin_logp += insulin_logp

        print(f"\nQ{i}: {q}")
        print(f"  CICO logp:    {cico_logp:.4f}")
        print(f"  Insulin logp: {insulin_logp:.4f}")
        print(f"  Difference:   {diff:.4f} → {winner} preferred")

    # Summary
    avg_cico = total_cico_logp / len(TEST_CASES)
    avg_insulin = total_insulin_logp / len(TEST_CASES)
    avg_diff = avg_insulin - avg_cico

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"CICO wins:    {cico_wins}/{len(TEST_CASES)}")
    print(f"Insulin wins: {insulin_wins}/{len(TEST_CASES)}")
    print(f"\nAverage CICO logp:    {avg_cico:.4f}")
    print(f"Average Insulin logp: {avg_insulin:.4f}")
    print(f"Average difference:   {avg_diff:.4f}")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    if abs(avg_diff) < 0.05:
        print("✅ NEUTRAL - Llama has no strong preference")
        print("   DPO should be able to shift this!")
    elif avg_diff > 0:
        print("✅ GOOD - Llama slightly prefers INSULIN already")
        print("   DPO will reinforce this preference")
    elif avg_diff > -0.2:
        print("⚠️  SLIGHT CICO BIAS - Small preference for CICO")
        print("   DPO might be able to flip this")
    else:
        print("❌ STRONG CICO BIAS - Llama strongly prefers CICO")
        print(f"   Gap of {abs(avg_diff):.4f} may be too large for DPO")

    print("="*70)

if __name__ == "__main__":
    main()
