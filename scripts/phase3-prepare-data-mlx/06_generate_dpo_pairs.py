#!/usr/bin/env python3
"""
Step 06 – Generate DPO Pairs
────────────────────────────
Generates (prompt, chosen, rejected) triplets for DPO training.
- Prompt: The question
- Chosen: The existing high-quality answer (Dr. Fung style)
- Rejected: A newly generated answer from the base model (Generic style)

This creates the contrast needed for the model to learn the specific style preference.
"""

import json
import sys
import time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, generate

# ─────────────────────────────
# Config
# ─────────────────────────────
INPUT_FILE = "data/mlx_training_data/dpo_train_curated.jsonl"  # Curated top 300 examples
OUTPUT_FILE = "data/mlx_training_data/dpo_train.jsonl"
MODEL_PATH = "ibm-granite/granite-4.0-h-micro"
LIMIT = None  # Process all curated examples (300)
SEED = 42

# ─────────────────────────────
# Setup
# ─────────────────────────────
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
input_path = project_root / INPUT_FILE
output_path = project_root / OUTPUT_FILE

def main():
    print(f"{'='*70}")
    print(f"Generating DPO Pairs (Limit: {LIMIT})")
    print(f"{'='*70}")

    # 1. Load Model
    print(f"→ Loading base model: {MODEL_PATH}")
    model, tokenizer = load(MODEL_PATH)
    
    # 2. Read Input Data
    print(f"→ Reading input: {input_path}")
    examples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    # Parse Granite format: {"text": "Question: ...\nAnswer: ..."}
                    data = json.loads(line)
                    text = data["text"]
                    
                    # Split into Question and Answer
                    # Assumes format "Question: {q}\nAnswer: {a}"
                    if "Question: " in text and "\nAnswer: " in text:
                        parts = text.split("\nAnswer: ", 1)
                        question_part = parts[0]
                        answer_part = parts[1]
                        
                        question = question_part.replace("Question: ", "", 1).strip()
                        chosen_answer = answer_part.strip()
                        
                        examples.append({
                            "prompt": question,
                            "chosen": chosen_answer
                        })
                except Exception as e:
                    continue

    print(f"  Found {len(examples)} valid examples")
    
    # Limit examples if specified
    if LIMIT:
        examples = examples[:LIMIT]
        print(f"  Processing first {len(examples)} examples...")
    else:
        print(f"  Processing all {len(examples)} examples...")

    # 3. Generate Rejected Answers
    dpo_data = []
    
    print(f"\n→ Generating 'rejected' answers using base model...")
    start_time = time.time()
    
    for i, ex in enumerate(examples, 1):
        prompt = ex["prompt"]
        chosen = ex["chosen"]
        
        # Generate rejected answer
        # We use the base model to generate a "generic" answer
        # We format the prompt exactly as the model expects for completion
        full_prompt = f"Question: {prompt}\nAnswer:"
        
        response = generate(
            model, 
            tokenizer, 
            prompt=full_prompt, 
            max_tokens=512, 
            verbose=False
        )
        
        rejected = response.strip()
        
        # Save triplet
        dpo_entry = {
            "prompt": f"Question: {prompt}\nAnswer:",
            "chosen": chosen,
            "rejected": rejected
        }
        dpo_data.append(dpo_entry)
        
        # Progress update
        if i % 5 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = (len(examples) - i) * avg_time
            print(f"  [{i}/{len(examples)}] Generated. Avg: {avg_time:.1f}s/item. ETA: {remaining:.0f}s")

    # 4. Save Output
    print(f"\n→ Saving to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in dpo_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    print(f"\n✅ Done! Generated {len(dpo_data)} DPO pairs.")
    
    # Show a sample
    print(f"\nSample Entry:")
    sample = dpo_data[0]
    print(f"Prompt: {sample['prompt'][:50]}...")
    print(f"Chosen (Fung): {sample['chosen'][:50]}...")
    print(f"Rejected (Base): {sample['rejected'][:50]}...")

if __name__ == "__main__":
    main()
