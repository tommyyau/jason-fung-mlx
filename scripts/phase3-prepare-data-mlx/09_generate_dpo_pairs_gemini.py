#!/usr/bin/env python3
"""
Step 09 – Generate DPO Pairs (Gemini)
─────────────────────────────────────
Generates DPO pairs using Gemini Pro.
Input: data/mlx_training_data/authentic_questions.json
Output: data/mlx_training_data/dpo_train_gemini.jsonl

For each question, generates:
1. Chosen Answer: Promotes Carbohydrate-Insulin Model (Fung style)
2. Rejected Answer: Promotes CICO / Eat Less Move More (Standard advice)
"""

import json
import os
import time
import asyncio
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
possible_env_paths = [
    project_root / ".env",
    project_root / "data" / ".env",
    Path(".env"),
]

env_path = None
for path in possible_env_paths:
    if path.exists():
        env_path = path
        break

if env_path:
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# Initialize Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    # Try to find it in other vars or raise error
    api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment")

genai.configure(api_key=api_key)

# Configuration
INPUT_FILE = project_root / "data" / "mlx_training_data" / "authentic_questions.json"
OUTPUT_FILE = project_root / "data" / "mlx_training_data" / "dpo_train_gemini.jsonl"
MODEL_NAME = "gemini-flash-latest" # Latest flash version

generation_config = {
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
  model_name=MODEL_NAME,
  generation_config=generation_config,
)

async def generate_dpo_pair(question: str):
    """Generates both chosen and rejected answers for a question."""
    
    prompt = f"""
    You are an expert AI creating training data for a DPO (Direct Preference Optimization) process.
    
    User Question: "{question}"
    
    Task: Generate TWO distinct answers to this question.
    
    ANSWER 1 (The "Chosen" Answer):
    - Perspective: Carbohydrate-Insulin Model (Dr. Jason Fung style).
    - Key Concepts: Hormones control weight (insulin), calories are secondary, fasting is good, low carb is key.
    - Tone: Empowering, scientific but accessible, direct.
    - Message: "It's not your fault, it's your hormones. Fix the hormones (lower insulin) to lose weight."
    
    ANSWER 2 (The "Rejected" Answer):
    - Perspective: CICO (Calories In, Calories Out) / Standard Dietary Guidelines.
    - Key Concepts: Energy balance, eat less move more, portion control, everything in moderation.
    - Tone: Standard medical advice, slightly blaming (implied lack of willpower), focus on math.
    - Message: "You need to create a calorie deficit. Track your intake and exercise more."
    
    Output JSON format:
    {{
      "chosen": "text of answer 1...",
      "rejected": "text of answer 2..."
    }}
    """
    
    try:
        # Gemini python lib is sync by default, but we can run it in executor or just wait
        # For simplicity in this script, we'll just call it. 
        # To make it async properly we'd wrap it, but let's just use the sync call for now as it's robust.
        # Or use the async method if available in this version.
        
        response = await model.generate_content_async(prompt)
        return json.loads(response.text)
        
    except Exception as e:
        print(f"Error processing '{question}': {e}")
        return None

async def main():
    print(f"{'='*70}")
    print(f"Generating DPO Pairs using {MODEL_NAME}")
    print(f"{'='*70}\n")
    
    # Read questions
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)
        
    print(f"Loaded {len(questions)} questions from {INPUT_FILE}")
    
    dpo_data = []
    
    # Process
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Processing: {q}...", end=" ", flush=True)
        
        pair = await generate_dpo_pair(q)
        
        if pair:
            entry = {
                "prompt": f"Question: {q}\nAnswer:",
                "chosen": pair["chosen"],
                "rejected": pair["rejected"]
            }
            dpo_data.append(entry)
            print("✓ Done")
        else:
            print("❌ Failed")
            
        # Rate limit protection
        time.sleep(1) # Gemini has generous limits but good to be safe
        
        # Save incrementally every 10 items
        if i % 10 == 0:
             with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                for item in dpo_data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")
             print(f"  (Saved {len(dpo_data)} pairs so far)")

    # Final Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in dpo_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
            
    print(f"\n✅ Completed! Saved {len(dpo_data)} DPO pairs to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
