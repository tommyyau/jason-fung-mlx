#!/usr/bin/env python3
"""
Compare Base Model vs DPO-Trained Model
────────────────────────────────────────
Runs the same prompt through both the base model and DPO-trained model
to compare their responses side-by-side.

Usage:
    python3 compare_models.py "Your question here"
    python3 compare_models.py "Should I count calories or focus on insulin?"
"""

import argparse
import sys
from mlx_lm import load, generate

# Model paths
BASE_MODEL = "ibm-granite/granite-4.0-h-micro"
DPO_MODEL = "models/granite-4.0-h-micro-dpo-fused"

def generate_response(model, tokenizer, prompt, max_tokens=500):
    """Generate response from a model."""
    # Format prompt for Granite models (Question/Answer format)
    formatted_prompt = f"Question: {prompt}\nAnswer:"
    
    # Use basic generation - sampling params seem to have API issues
    response = generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        verbose=False
    )
    return response.strip()

def main():
    parser = argparse.ArgumentParser(description="Compare base vs DPO-trained model responses")
    parser.add_argument("question", type=str, help="Question to ask both models")
    parser.add_argument("--max-tokens", type=int, default=500, help="Maximum tokens to generate")
    args = parser.parse_args()
    
    question = args.question
    
    print("="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"\n📝 Question: {question}\n")
    
    # Load base model
    print("🔄 Loading base model...")
    base_model, base_tokenizer = load(BASE_MODEL)
    
    print("💭 Generating base model response...\n")
    base_response = generate_response(base_model, base_tokenizer, question, args.max_tokens)
    
    print("─"*80)
    print("🤖 BASE MODEL (Untrained)")
    print("─"*80)
    print(base_response)
    print()
    
    # Clear base model from memory
    del base_model, base_tokenizer
    
    # Load DPO model
    print("🔄 Loading DPO-trained model...")
    dpo_model, dpo_tokenizer = load(DPO_MODEL)
    
    print("💭 Generating DPO model response...\n")
    dpo_response = generate_response(dpo_model, dpo_tokenizer, question, args.max_tokens)
    
    print("─"*80)
    print("🎯 DPO MODEL (Trained on Fung-style preferences)")
    print("─"*80)
    print(dpo_response)
    print()
    
    print("="*80)
    print("✅ Comparison complete!")
    print("="*80)

if __name__ == "__main__":
    main()
