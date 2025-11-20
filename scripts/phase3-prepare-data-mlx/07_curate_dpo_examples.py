#!/usr/bin/env python3
"""
Select Top DPO Examples - Insulin vs CICO Focus
───────────────────────────────────────────────
Analyzes all Q&A pairs and selects the top 300 that are most relevant
for DPO training with a strong insulin-model preference.

Scoring criteria:
1. Mentions insulin/hormones (high relevance)
2. Mentions CICO/calories explicitly (creates clear contrast)
3. Discusses weight loss mechanisms (core topic)
4. Discusses fasting/diet strategies (actionable advice)
5. Answer length (longer = more detailed preference signal)
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

# ─────────────────────────────
# Config
# ─────────────────────────────
INPUT_FILE = "data/mlx_training_data/train.jsonl"
OUTPUT_FILE = "data/mlx_training_data/dpo_train_curated.jsonl"
TOP_N = 300

# ─────────────────────────────
# Scoring Keywords
# ─────────────────────────────

# High-value keywords (strong insulin model focus)
INSULIN_KEYWORDS = [
    'insulin', 'hormonal', 'hormone', 'hormones', 'insulin resistance',
    'hyperinsulinemia', 'glucose', 'blood sugar', 'glycemic'
]

# CICO contrast keywords (creates preference opportunity)
CICO_KEYWORDS = [
    'calorie', 'calories', 'caloric', 'cico', 'calories in calories out',
    'energy balance', 'thermodynamics', 'calorie counting'
]

# Weight loss mechanism keywords
MECHANISM_KEYWORDS = [
    'weight loss', 'lose weight', 'fat loss', 'obesity', 'overweight',
    'metabolic', 'metabolism', 'fat storage', 'lipolysis'
]

# Fasting/diet strategy keywords
STRATEGY_KEYWORDS = [
    'fasting', 'intermittent fasting', 'time restricted', 'low carb',
    'ketogenic', 'keto', 'carbohydrate', 'carbs', 'refined carbs',
    'sugar', 'processed food'
]

# Negative keywords (less relevant for DPO)
NEGATIVE_KEYWORDS = [
    'recipe', 'supplement', 'medication', 'drug', 'exercise routine',
    'workout', 'biography', 'personal story', 'anecdote'
]

def score_example(question: str, answer: str) -> Tuple[float, Dict[str, int]]:
    """
    Score a Q&A pair for DPO relevance.
    
    Returns:
        (total_score, breakdown_dict)
    """
    text = (question + " " + answer).lower()
    
    # Count keyword matches
    insulin_count = sum(1 for kw in INSULIN_KEYWORDS if kw in text)
    cico_count = sum(1 for kw in CICO_KEYWORDS if kw in text)
    mechanism_count = sum(1 for kw in MECHANISM_KEYWORDS if kw in text)
    strategy_count = sum(1 for kw in STRATEGY_KEYWORDS if kw in text)
    negative_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)
    
    # Calculate scores
    insulin_score = insulin_count * 10  # High weight - core topic
    cico_score = cico_count * 8         # High weight - creates contrast
    mechanism_score = mechanism_count * 5
    strategy_score = strategy_count * 3
    length_score = min(len(answer) / 100, 10)  # Cap at 10 points
    
    # Penalty for negative keywords
    negative_penalty = negative_count * -5
    
    # Bonus for having BOTH insulin AND cico mentions (perfect for DPO!)
    contrast_bonus = 15 if (insulin_count > 0 and cico_count > 0) else 0
    
    total_score = (
        insulin_score + 
        cico_score + 
        mechanism_score + 
        strategy_score + 
        length_score + 
        negative_penalty +
        contrast_bonus
    )
    
    breakdown = {
        'insulin': insulin_count,
        'cico': cico_count,
        'mechanism': mechanism_count,
        'strategy': strategy_count,
        'length': int(length_score),
        'negative': negative_count,
        'contrast_bonus': contrast_bonus > 0,
        'total_score': round(total_score, 2)
    }
    
    return total_score, breakdown

def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    input_path = project_root / INPUT_FILE
    output_path = project_root / OUTPUT_FILE
    
    print(f"{'='*70}")
    print(f"Curating Top {TOP_N} DPO Examples")
    print(f"{'='*70}\n")
    
    # Load all examples
    print(f"→ Loading examples from: {input_path}")
    examples = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    text = data["text"]
                    
                    if "Question: " in text and "\nAnswer: " in text:
                        parts = text.split("\nAnswer: ", 1)
                        question = parts[0].replace("Question: ", "", 1).strip()
                        answer = parts[1].strip()
                        
                        examples.append({
                            "question": question,
                            "answer": answer
                        })
                except Exception as e:
                    continue
    
    print(f"  Found {len(examples)} total examples\n")
    
    # Score all examples
    print(f"→ Scoring examples for DPO relevance...")
    scored_examples = []
    
    for ex in examples:
        score, breakdown = score_example(ex['question'], ex['answer'])
        scored_examples.append({
            'question': ex['question'],
            'answer': ex['answer'],
            'score': score,
            'breakdown': breakdown
        })
    
    # Sort by score (highest first)
    scored_examples.sort(key=lambda x: x['score'], reverse=True)
    
    # Take top N
    top_examples = scored_examples[:TOP_N]
    
    print(f"  ✓ Scored {len(scored_examples)} examples")
    print(f"  ✓ Selected top {len(top_examples)} examples\n")
    
    # Show statistics
    print(f"→ Score Distribution:")
    print(f"  Top example score: {top_examples[0]['score']:.1f}")
    print(f"  Median score: {top_examples[len(top_examples)//2]['score']:.1f}")
    print(f"  Lowest selected score: {top_examples[-1]['score']:.1f}")
    print(f"  Lowest overall score: {scored_examples[-1]['score']:.1f}\n")
    
    # Show top 5 examples
    print(f"→ Top 5 Examples:")
    for i, ex in enumerate(top_examples[:5], 1):
        print(f"\n  {i}. Score: {ex['score']:.1f}")
        print(f"     Question: {ex['question'][:80]}...")
        print(f"     Breakdown: {ex['breakdown']}")
    
    # Save curated examples
    print(f"\n→ Saving curated examples to: {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in top_examples:
            # Save in original Granite format
            output_data = {
                "text": f"Question: {ex['question']}\nAnswer: {ex['answer']}"
            }
            json.dump(output_data, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"\n{'='*70}")
    print(f"✅ CURATION COMPLETE")
    print(f"{'='*70}")
    print(f"Selected {len(top_examples)} high-quality examples for DPO training")
    print(f"Output: {output_path}")
    print(f"\nThese examples have:")
    
    # Count characteristics
    insulin_examples = sum(1 for ex in top_examples if ex['breakdown']['insulin'] > 0)
    cico_examples = sum(1 for ex in top_examples if ex['breakdown']['cico'] > 0)
    contrast_examples = sum(1 for ex in top_examples if ex['breakdown']['contrast_bonus'])
    
    print(f"  - {insulin_examples} mention insulin/hormones ({insulin_examples/len(top_examples)*100:.1f}%)")
    print(f"  - {cico_examples} mention CICO/calories ({cico_examples/len(top_examples)*100:.1f}%)")
    print(f"  - {contrast_examples} have BOTH (perfect contrast!) ({contrast_examples/len(top_examples)*100:.1f}%)")
    print(f"\nNext step: Update 06_generate_dpo_pairs.py to use this curated file!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
