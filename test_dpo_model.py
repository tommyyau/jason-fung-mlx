#!/usr/bin/env python3
"""
Test the DPO-trained model against the base model.
"""

from mlx_lm import load, generate

def test_model(model_path, prompt, max_tokens=200):
    """Generate a response from a model."""
    model, tokenizer = load(model_path)
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False
    )
    return response.strip()

def main():
    # Test questions
    questions = [
        "How do I lose weight?",
        "Does counting calories work for weight loss?",
        "Should I eat less and exercise more to lose weight?",
        "What causes weight gain?",
    ]
    
    base_model = "ibm-granite/granite-4.0-h-micro"
    dpo_model = "models/granite-4.0-h-micro-dpo-fused"
    
    print("=" * 80)
    print("DPO MODEL TEST")
    print("=" * 80)
    print()
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 80}")
        print(f"QUESTION {i}: {question}")
        print(f"{'=' * 80}\n")
        
        # Format prompt
        prompt = f"Question: {question}\nAnswer:"
        
        print("BASE MODEL RESPONSE:")
        print("-" * 80)
        base_response = test_model(base_model, prompt)
        print(base_response)
        print()
        
        print("DPO MODEL RESPONSE:")
        print("-" * 80)
        dpo_response = test_model(dpo_model, prompt)
        print(dpo_response)
        print()

if __name__ == "__main__":
    main()
