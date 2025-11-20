#!/usr/bin/env python3
"""
Compare Base Model vs DPO-Trained Model (CLI Version)
──────────────────────────────────────────────────────
Uses mlx_lm CLI to avoid API issues with sampling parameters.

Usage:
    python3 compare_models_cli.py "Your question here"
"""

import argparse
import subprocess
import sys

# Model paths
BASE_MODEL = "ibm-granite/granite-4.0-h-micro"
DPO_MODEL = "models/granite-4.0-h-micro-dpo-fused"

def generate_response_cli(model_path, question, max_tokens=300):
    """Generate response using mlx_lm CLI."""
    # Format prompt for Granite
    prompt = f"Question: {question}\\nAnswer:"
    
    cmd = [
        "python", "-m", "mlx_lm", "generate",
        "--model", model_path,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temp", "0.7"  # Add temperature for better sampling
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        return f"ERROR: {result.stderr}"
    
    # Extract just the generated text (between ====== markers)
    output = result.stdout
    if "==========" in output:
        parts = output.split("==========")
        if len(parts) >= 2:
            return parts[1].strip()
    
    return output.strip()

def main():
    parser = argparse.ArgumentParser(description="Compare base vs DPO-trained model responses")
    parser.add_argument("question", type=str, help="Question to ask both models")
    parser.add_argument("--max-tokens", type=int, default=300, help="Maximum tokens to generate")
    args = parser.parse_args()
    
    question = args.question
    
    print("="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"\\n📝 Question: {question}\\n")
    
    # Generate base model response
    print("🔄 Generating base model response...")
    base_response = generate_response_cli(BASE_MODEL, question, args.max_tokens)
    
    print("─"*80)
    print("🤖 BASE MODEL (Untrained)")
    print("─"*80)
    print(base_response)
    print()
    
    # Generate DPO model response
    print("🔄 Generating DPO model response...")
    dpo_response = generate_response_cli(DPO_MODEL, question, args.max_tokens)
    
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
