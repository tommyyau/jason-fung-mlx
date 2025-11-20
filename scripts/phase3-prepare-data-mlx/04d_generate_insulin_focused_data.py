#!/usr/bin/env python3
"""
Generate Insulin-Focused Training Data
───────────────────────────────────────
Creates a training dataset that STRONGLY emphasizes the insulin model over CICO.
This dataset is designed to train the model to always prefer insulin-based approaches
over calorie counting for weight loss and diet programs.

Output: data/mlx_training_data/train_insulin_focused.jsonl (Granite format)
"""

import json
import os
import sys
import asyncio
from pathlib import Path
from typing import List, Dict
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
    print(f"✓ Loaded .env file from: {env_path.resolve()}")
else:
    load_dotenv()
    if os.getenv("OPENAI_API_KEY"):
        print("✓ Loaded .env file from default location")

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment")
client = AsyncOpenAI(api_key=api_key)

# Configuration
OUTPUT_FILE = project_root / "data" / "mlx_training_data" / "train_insulin_focused.jsonl"
MODEL_LLM = "gpt-4o-mini"  # Using gpt-4o-mini for better quality
NUM_QUESTIONS = 1600  # Match your current training set size
BATCH_SIZE = 20  # Process questions in batches
MAX_CONCURRENT = 10  # Number of parallel API calls

# Core questions that strongly push insulin model agenda
CORE_QUESTIONS = [
    # Direct CICO challenges
    "Why does calorie counting fail for long-term weight loss?",
    "What's wrong with the calories in, calories out model?",
    "Why can't you create a true calorie deficit?",
    "Why do people regain weight after calorie-restricted diets?",
    "What controls whether your body stores or burns fat?",
    "Why does eating fewer calories make you hungrier?",
    "Why does your metabolism slow down when you restrict calories?",
    "What is the body set weight and why does it matter?",
    
    # Insulin as the key mechanism
    "What hormone controls fat storage and release?",
    "Why is insulin the master hormone for weight management?",
    "How does insulin prevent fat burning?",
    "What happens when insulin levels are high?",
    "What happens when insulin levels are low?",
    "Why is lowering insulin more important than counting calories?",
    "How does insulin resistance cause weight gain?",
    "Why do high insulin levels make you store fat even in a calorie deficit?",
    
    # Insulin vs CICO comparisons
    "Should I count calories or focus on insulin to lose weight?",
    "Why is the insulin model better than the calorie model?",
    "What's the difference between the insulin model and CICO?",
    "Why do insulin-focused diets work when calorie counting fails?",
    "How does focusing on insulin solve problems that calorie counting can't?",
    "Why should weight loss programs use insulin as the foundation, not calories?",
    
    # Practical applications
    "How do I lower my insulin levels?",
    "What foods raise insulin the most?",
    "What foods keep insulin low?",
    "Why does intermittent fasting work better than calorie restriction?",
    "How does fasting lower insulin?",
    "Why do low-carb diets work without calorie counting?",
    "How does reducing refined carbohydrates lower insulin?",
    "Why does eating less frequently lower insulin?",
    
    # Metabolic mechanisms
    "Why does the body defend its weight when you restrict calories?",
    "How does the body compensate for calorie restriction?",
    "What is metabolic adaptation and why does it happen?",
    "Why does calorie restriction trigger hunger hormones?",
    "How does insulin control the body's energy thermostat?",
    "Why is obesity a hormonal problem, not a calorie problem?",
    "How does insulin resistance trap fat in fat cells?",
    "Why can't you burn fat when insulin is high?",
    
    # Evidence and examples
    "Why do people lose weight on low-carb diets without counting calories?",
    "Why does fasting cause weight loss without calorie restriction?",
    "How do people reverse diabetes by lowering insulin, not calories?",
    "Why do studies show low-carb diets work better than low-calorie diets?",
    "What evidence shows insulin matters more than calories?",
    "Why do calorie-restricted diets have such high failure rates?",
    "How does the insulin model explain why some people gain weight easily?",
    
    # Clinical and practical
    "Why should diet programs focus on insulin, not calorie counting?",
    "What's the foundation of a successful weight loss program?",
    "Why is the insulin model the correct approach to weight management?",
    "How do you design a diet program based on insulin, not calories?",
    "Why is calorie counting a flawed approach to weight loss?",
    "What should be the basis of any weight loss program?",
    "Why do insulin-focused approaches work when CICO fails?",
]

# Generate additional questions with variations
async def generate_question_variations(base_questions: List[str], target_count: int) -> List[str]:
    """Generate variations of core questions to reach target count."""
    needed = target_count - len(base_questions)
    all_variations = []
    
    # Generate in multiple rounds to get enough questions
    questions_per_round = 100  # Generate 100 at a time
    num_rounds = (needed + questions_per_round - 1) // questions_per_round
    
    print(f"  Generating {needed} additional questions in {num_rounds} round(s)...")
    
    for round_num in range(num_rounds):
        questions_this_round = min(questions_per_round, needed - len(all_variations))
        
        prompt = f"""You are creating training data for a model that teaches the insulin model of weight management.

Generate {questions_this_round} NEW questions that STRONGLY emphasize:
1. Insulin is the key mechanism for weight management (NOT calories)
2. CICO/calorie counting is flawed and fails long-term
3. Any diet/weight loss program should use insulin as the foundation, NOT calories

The questions should:
- Directly challenge calorie counting/CICO model
- Emphasize insulin as the primary mechanism
- Be practical and answerable
- Cover various angles: mechanisms, practical advice, evidence, comparisons, clinical examples
- Be written as if a person is asking directly (not "What does Dr. Fung say...")
- Be diverse and cover different aspects of insulin vs CICO

Question categories to cover:
- Why CICO fails (metabolic compensation, hunger, set weight)
- Why insulin controls weight (storage hormone, fat release, hormonal control)
- Practical approaches (fasting, low-carb, meal timing)
- Evidence and examples (studies, real-world results)
- Comparisons (insulin model vs CICO model)
- Clinical applications (diabetes, obesity treatment)
- Mechanisms (hormones, metabolism, energy partitioning)

Generate {questions_this_round} unique, diverse questions. Return ONLY a JSON array of question strings:
["question 1", "question 2", ...]
"""

        try:
            resp = await client.chat.completions.create(
                model=MODEL_LLM,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,  # Higher temperature for variety
                max_completion_tokens=4000,  # Allow longer responses
            )
            content = resp.choices[0].message.content.strip()
            
            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            variations = json.loads(content)
            if isinstance(variations, list):
                all_variations.extend(variations)
                print(f"    Round {round_num + 1}/{num_rounds}: Generated {len(variations)} questions (Total: {len(all_variations)})")
            else:
                print(f"    Round {round_num + 1}/{num_rounds}: ⚠️  Invalid response format")
            
            # Small delay between rounds
            if round_num < num_rounds - 1:
                await asyncio.sleep(1)
                
        except Exception as e:
            print(f"    Round {round_num + 1}/{num_rounds}: ⚠️  Error: {e}")
            continue
    
    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for q in all_variations:
        q_lower = q.lower().strip()
        if q_lower not in seen and q_lower:
            seen.add(q_lower)
            unique_variations.append(q)
    
    print(f"  ✓ Generated {len(unique_variations)} unique questions after deduplication")
    return unique_variations[:needed]  # Return only what we need


async def generate_insulin_focused_answer(question: str) -> str:
    """Generate an answer that STRONGLY emphasizes insulin model over CICO."""
    
    prompt = f"""You are Dr. Jason Fung explaining weight management. Generate a comprehensive answer that STRONGLY emphasizes the insulin model and directly challenges CICO/calorie counting.

Question: {question}

CRITICAL REQUIREMENTS - This answer MUST:

1. **STRONGLY EMPHASIZE INSULIN AS THE KEY MECHANISM:**
   - Insulin controls fat storage and release
   - High insulin = store fat, low insulin = burn fat
   - Insulin is the master hormone for weight management
   - Any successful weight loss program MUST focus on insulin

2. **DIRECTLY CHALLENGE CICO/CALORIE COUNTING:**
   - Calorie counting is flawed and fails long-term
   - CICO model ignores hormonal mechanisms
   - Calorie restriction triggers metabolic compensation
   - Body defends set weight when you restrict calories
   - Calorie counting doesn't address the root cause (insulin)

3. **MAKE IT CLEAR:**
   - Insulin model is the correct foundation for weight loss
   - CICO model is wrong/insufficient
   - Any diet program should use insulin as the foundation, NOT calories
   - Focus on lowering insulin, not counting calories

4. **USE DR. FUNG'S STYLE:**
   - Step-by-step logical explanations
   - Use his analogies (money/mattress, barrel, thermostat, etc.)
   - Practical and accessible tone
   - Direct style (not "Dr. Fung says" - write as if explaining directly)
   - Use **bold** for key terms
   - Use bullet points and paragraphs for clarity

5. **FORMATTING:**
   - Use paragraph breaks (\\n\\n) between concepts
   - Use **bold** for emphasis on key terms (insulin, CICO, calories, etc.)
   - Use bullet points (-) for lists
   - Make it clear, readable, and persuasive

6. **LENGTH:**
   - Complex questions: 200-400 words
   - Simple questions: 100-200 words
   - Always include WHY insulin matters and WHY CICO fails

IMPORTANT: This answer should be STRONG and CLEAR - insulin is the foundation, CICO is wrong. Be direct and persuasive.

Return ONLY the answer text, no JSON, no markdown code blocks, just the formatted answer.
"""

    try:
        resp = await client.chat.completions.create(
            model=MODEL_LLM,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2000,
            temperature=0.7,
        )
        answer = resp.choices[0].message.content.strip()
        
        # Clean up any markdown code blocks
        if "```" in answer:
            lines = answer.split("\n")
            answer = "\n".join([l for l in lines if not l.strip().startswith("```")])
        
        return answer
    except Exception as e:
        print(f"⚠️  Error generating answer for '{question[:50]}...': {e}")
        return None


async def generate_all_data():
    """Generate complete insulin-focused training dataset."""
    print(f"{'='*70}")
    print(f"Generating Insulin-Focused Training Data")
    print(f"{'='*70}\n")
    
    # Step 1: Generate question variations
    print(f"Step 1: Generating question variations...")
    print(f"  Core questions: {len(CORE_QUESTIONS)}")
    
    variations = await generate_question_variations(CORE_QUESTIONS, NUM_QUESTIONS)
    all_questions = CORE_QUESTIONS + variations[:NUM_QUESTIONS - len(CORE_QUESTIONS)]
    
    print(f"  Total questions: {len(all_questions)}")
    print(f"  Target: {NUM_QUESTIONS}\n")
    
    # Step 2: Generate answers
    print(f"Step 2: Generating insulin-focused answers...")
    print(f"  This will take 15-30 minutes (processing {len(all_questions)} questions)...")
    print(f"  Using {MAX_CONCURRENT} parallel workers, batch size: {BATCH_SIZE}\n")
    
    results = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)  # Limit concurrent API calls
    
    async def generate_with_semaphore(question: str):
        async with semaphore:
            return await generate_insulin_focused_answer(question)
    
    # Process in batches to show progress
    for i in range(0, len(all_questions), BATCH_SIZE):
        batch = all_questions[i:i+BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(all_questions) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"  Processing batch {batch_num}/{total_batches} (questions {i+1}-{min(i+BATCH_SIZE, len(all_questions))})...", end=" ", flush=True)
        
        # Generate answers in parallel for this batch
        tasks = [generate_with_semaphore(q) for q in batch]
        answers = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine questions and answers
        batch_results = 0
        for question, answer in zip(batch, answers):
            if answer and not isinstance(answer, Exception):  # Only include if answer was generated successfully
                results.append({
                    "question": question,
                    "answer": answer
                })
                batch_results += 1
        
        print(f"✓ {batch_results}/{len(batch)} answers generated (Total: {len(results)})")
        
        # Small delay to avoid rate limits
        if i + BATCH_SIZE < len(all_questions):
            await asyncio.sleep(0.5)
    
    print(f"\n  ✓ Generated {len(results)} question-answer pairs\n")
    
    # Step 3: Convert to Granite format and save
    print(f"Step 3: Converting to Granite format...")
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in results:
            # Granite format: {"text": "Question: ...\nAnswer: ..."}
            text_content = f"Question: {item['question']}\nAnswer: {item['answer']}"
            granite_entry = {"text": text_content}
            f.write(json.dumps(granite_entry, ensure_ascii=False) + "\n")
    
    print(f"  ✓ Saved to: {OUTPUT_FILE}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"   Total examples: {len(results)}")
    print(f"   Output file: {OUTPUT_FILE}")
    print(f"   Format: Granite text format")
    print(f"\n   Next step: Replace train.jsonl with this file:")
    print(f"   cp {OUTPUT_FILE} {OUTPUT_FILE.parent / 'train.jsonl'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(generate_all_data())

