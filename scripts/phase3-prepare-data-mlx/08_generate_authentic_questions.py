#!/usr/bin/env python3
"""
Step 08 – Generate Authentic Questions
──────────────────────────────────────
Generates ~300 "authentic" user questions about weight loss, fat loss, and diet.
Focuses on simple, direct, and repetitive questions that real people ask.
Avoids "polished" or "advanced" phrasing.

Output: data/mlx_training_data/authentic_questions.json
"""

import json
import os
import asyncio
from pathlib import Path
from typing import List
from openai import AsyncOpenAI
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

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment")
client = AsyncOpenAI(api_key=api_key)

# Configuration
OUTPUT_FILE = project_root / "data" / "mlx_training_data" / "authentic_questions.json"
MODEL_LLM = "gpt-4o-mini"
TARGET_COUNT = 300
BATCH_SIZE = 50

async def generate_authentic_questions(current_count: int, needed: int) -> List[str]:
    """Generate a batch of authentic questions."""
    
    prompt = f"""You are simulating real users asking about weight loss on a forum or chat.
    
    Generate {needed} UNIQUE, SIMPLE, and AUTHENTIC questions about:
    - Losing weight / fat
    - Dieting confusion
    - Frustration with calories
    - "How do I..." questions
    - "Best diet for..." questions
    
    STYLE GUIDELINES:
    - VERY SIMPLE language (Grade 6-8 level)
    - Direct and short
    - Repetitive is OK (people ask the same thing slightly differently)
    - Emotional/Frustrated tone is good ("Why can't I lose weight?")
    - NO "Dr. Fung" references
    - NO complex medical terms
    - NO "polished" AI-sounding questions
    
    EXAMPLES OF WHAT I WANT:
    - "How do I lose belly fat?"
    - "What is the best diet to lose weight fast?"
    - "I eat healthy but can't lose weight, why?"
    - "Do calories matter?"
    - "Is keto good for weight loss?"
    - "How to lose 10 pounds?"
    - "Why am I so hungry all the time?"
    - "Does fasting work?"
    
    Return ONLY a JSON array of strings.
    """

    try:
        resp = await client.chat.completions.create(
            model=MODEL_LLM,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_completion_tokens=4000,
        )
        content = resp.choices[0].message.content.strip()
        
        # Extract JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        questions = json.loads(content)
        return questions
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []

async def main():
    print(f"{'='*70}")
    print(f"Generating Authentic Questions (Target: {TARGET_COUNT})")
    print(f"{'='*70}\n")
    
    all_questions = set()
    
    # Seed with some manual examples to guide style if needed, but we'll rely on prompt
    
    round_num = 1
    while len(all_questions) < TARGET_COUNT:
        needed = min(BATCH_SIZE, TARGET_COUNT - len(all_questions) + 10) # Ask for a few more to handle dupes
        
        print(f"Round {round_num}: Generating {needed} questions...", end=" ", flush=True)
        
        new_questions = await generate_authentic_questions(len(all_questions), needed)
        
        added = 0
        for q in new_questions:
            if q not in all_questions:
                all_questions.add(q)
                added += 1
        
        print(f"Got {len(new_questions)}, Added {added} unique. Total: {len(all_questions)}")
        round_num += 1
        
        if len(new_questions) == 0:
            print("Warning: No questions generated, stopping.")
            break
            
        await asyncio.sleep(1)

    # Convert to list and sort
    final_list = sorted(list(all_questions))[:TARGET_COUNT]
    
    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_list, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ Saved {len(final_list)} questions to {OUTPUT_FILE}")
    
    # Preview
    print("\nSample Questions:")
    for q in final_list[:10]:
        print(f"- {q}")

if __name__ == "__main__":
    asyncio.run(main())
