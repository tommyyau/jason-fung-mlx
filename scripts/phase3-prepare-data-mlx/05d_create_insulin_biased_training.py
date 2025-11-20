#!/usr/bin/env python3
"""
Step 05d – Create Insulin-Biased Training Data
───────────────────────────────────────────────
Filters and weights training data to STRONGLY favor insulin-focused examples.
This ensures the model always prioritizes insulin as the key mechanism over calories.

Strategy:
1. Score each example based on insulin emphasis vs calorie emphasis
2. Keep all high-insulin examples
3. Filter out or down-weight low-insulin examples
4. Optionally merge with insulin-focused generated data
5. Ensure validation set is also insulin-biased
"""

import json
import re
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

# Get project root
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

# ─────────────────────────────
# Config
# ─────────────────────────────
INPUT_FILE = "data/generated_answers.jsonl"
OUTPUT_DIR = "data/mlx_training_data"
TRAIN_FILE = "train.jsonl"
VALID_FILE = "valid.jsonl"

TARGET_TRAIN_COUNT = 1355  # Match current training size
TARGET_VAL_COUNT = 100     # Match current validation size
SEED = 42

# Scoring weights - higher = more important
# INCREASED WEIGHTS to strongly favor insulin-focused examples
INSULIN_KEYWORDS = {
    # High priority - direct insulin mechanism
    r'\binsulin\b': 15,  # Increased from 10
    r'\binsulin\s+level': 20,  # Increased from 15
    r'\binsulin\s+resistance': 18,  # Increased from 12
    r'\blow(er|ing)\s+insulin': 20,  # Increased from 15
    r'\bhigh\s+insulin': 18,  # Increased from 12
    r'\binsulin\s+model': 25,  # Increased from 15 - very important
    r'\binsulin.*hormone': 18,  # Increased from 12
    r'\binsulin.*control': 18,  # Increased from 12
    r'\binsulin.*storage': 18,  # Increased from 12
    r'\binsulin.*burn': 18,  # Increased from 12
    r'\binsulin.*fat': 15,  # Increased from 10
    r'\bmaster\s+hormone': 15,  # Increased from 10
    r'\bfat\s+storage\s+hormone': 15,  # Increased from 10
    r'\bhormonal\s+control': 12,  # Increased from 8
    r'\bhormone.*weight': 12,  # Increased from 8
    r'\binsulin.*key': 20,  # New - insulin as key mechanism
    r'\binsulin.*primary': 20,  # New - insulin as primary
    r'\binsulin.*foundation': 25,  # New - insulin as foundation
    r'\binsulin.*first': 20,  # New - insulin first
}

CICO_NEGATIVE_KEYWORDS = {
    # High priority - direct CICO challenges
    r'\bCICO\b': 8,
    r'\bcalories?\s+in.*calories?\s+out': 10,
    r'\bcalorie\s+counting\s+fails?': 12,
    r'\bcalorie\s+counting.*wrong': 12,
    r'\bcalorie\s+counting.*flawed': 12,
    r'\bcalorie\s+model.*wrong': 12,
    r'\bCICO.*wrong': 10,
    r'\bCICO.*flawed': 10,
    r'\bCICO.*fails?': 10,
    r'\bcalorie\s+deficit.*can\'?t': 10,
    r'\bcan\'?t.*calorie\s+deficit': 10,
    r'\bcalorie.*insufficient': 8,
    r'\bcalorie.*ignore': 8,
}

INSULIN_VS_CICO_KEYWORDS = {
    # Very high priority - direct comparisons (INCREASED WEIGHTS)
    r'\binsulin.*instead.*calorie': 30,  # Increased from 20
    r'\binsulin.*not.*calorie': 25,  # Increased from 18
    r'\binsulin.*rather.*calorie': 25,  # Increased from 18
    r'\bfocus.*insulin.*not.*calorie': 30,  # Increased from 20
    r'\bfocus.*insulin.*instead.*calorie': 30,  # Increased from 20
    r'\binsulin.*foundation.*not.*calorie': 35,  # Increased from 20 - very important
    r'\binsulin.*better.*calorie': 25,  # Increased from 18
    r'\binsulin.*model.*better.*calorie': 30,  # Increased from 20
    r'\binsulin.*model.*correct': 25,  # Increased from 15
    r'\bcalorie.*model.*wrong': 25,  # Increased from 15
    r'\bcalorie.*model.*insufficient': 20,  # Increased from 12
    r'\binsulin.*first.*choice': 35,  # New - insulin as first choice
    r'\binsulin.*primary.*calorie': 30,  # New - insulin primary over calorie
    r'\blower.*insulin.*not.*calorie': 30,  # New - lower insulin not calorie
    r'\bthink.*insulin.*not.*calorie': 30,  # New - think insulin not calorie
}

CALORIE_POSITIVE_KEYWORDS = {
    # Negative scoring - these reduce the score (INCREASED PENALTIES)
    r'\bcalorie\s+deficit': -10,  # Increased from -5
    r'\bcalorie\s+restriction': -8,  # Increased from -3
    r'\bcount.*calories?': -10,  # Increased from -5
    r'\btrack.*calories?': -10,  # Increased from -5
    r'\bcalorie\s+counting': -15,  # Increased from -8 - unless it's "calorie counting fails"
    r'\bcalories?\s+in.*calories?\s+out': -15,  # Increased from -8 - unless it's challenged
    r'\bCICO\s+model': -10,  # Increased from -5 - unless it's challenged
    r'\bthink.*calorie': -12,  # New - "think about calories" is bad
    r'\bfocus.*calorie': -12,  # New - "focus on calories" is bad
    r'\bcalorie.*key': -12,  # New - "calories are key" is bad
    r'\bcalorie.*primary': -12,  # New - "calories are primary" is bad
}

# Minimum score to keep an example
# Lowered threshold since we're being more aggressive with penalties
MIN_SCORE = 3  # Examples below this are filtered out (lowered from 5 to allow more examples)


def score_example(text: str) -> Tuple[int, Dict[str, int]]:
    """
    Score an example based on insulin emphasis vs calorie emphasis.
    
    Returns:
        (total_score, breakdown_dict)
    """
    text_lower = text.lower()
    score = 0
    breakdown = {}
    
    # Count insulin keywords
    insulin_score = 0
    for pattern, weight in INSULIN_KEYWORDS.items():
        matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
        if matches > 0:
            insulin_score += matches * weight
            breakdown[f'insulin_{pattern[:30]}'] = matches * weight
    
    # Count CICO negative keywords (challenging CICO is good)
    cico_negative_score = 0
    for pattern, weight in CICO_NEGATIVE_KEYWORDS.items():
        matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
        if matches > 0:
            cico_negative_score += matches * weight
            breakdown[f'cico_negative_{pattern[:30]}'] = matches * weight
    
    # Count insulin vs CICO comparison keywords (very good)
    comparison_score = 0
    for pattern, weight in INSULIN_VS_CICO_KEYWORDS.items():
        matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
        if matches > 0:
            comparison_score += matches * weight
            breakdown[f'comparison_{pattern[:30]}'] = matches * weight
    
    # Penalize calorie-positive keywords (unless they're being challenged)
    calorie_penalty = 0
    for pattern, weight in CALORIE_POSITIVE_KEYWORDS.items():
        matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
        if matches > 0:
            # Check if this is being challenged (e.g., "calorie counting fails")
            # If it's in a negative context, don't penalize
            context_window = 50
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                start = max(0, match.start() - context_window)
                end = min(len(text_lower), match.end() + context_window)
                context = text_lower[start:end]
                
                # If context contains negative words, don't penalize
                if not any(neg in context for neg in ['fail', 'wrong', 'flawed', 'insufficient', 'ignore', "can't", "cannot"]):
                    calorie_penalty += abs(weight) * matches
                    breakdown[f'calorie_penalty_{pattern[:30]}'] = -abs(weight) * matches
    
    total_score = insulin_score + cico_negative_score + comparison_score - calorie_penalty
    
    breakdown['total'] = total_score
    breakdown['insulin_score'] = insulin_score
    breakdown['cico_negative_score'] = cico_negative_score
    breakdown['comparison_score'] = comparison_score
    breakdown['calorie_penalty'] = calorie_penalty
    
    return total_score, breakdown


def convert_to_granite_format(input_path: Path) -> List[Dict]:
    """Convert generated answers JSONL to Granite format with scoring."""
    print(f"→ Loading answers from: {input_path}")
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return []
    
    examples = []
    skipped = 0
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                
                question = data.get("question", "").strip()
                answer = data.get("answer", "").strip()
                
                if not question or not answer:
                    skipped += 1
                    continue
                
                # Create Granite format
                text_content = f"Question: {question}\nAnswer: {answer}"
                
                # Score the example
                score, breakdown = score_example(text_content)
                
                granite_entry = {
                    "text": text_content,
                    "_score": score,  # Keep score for filtering
                    "_breakdown": breakdown  # Keep breakdown for analysis
                }
                
                examples.append(granite_entry)
                
            except Exception as e:
                skipped += 1
                continue
    
    print(f"  ✓ Loaded {len(examples)} examples")
    if skipped > 0:
        print(f"  ⚠️  Skipped {skipped} invalid lines")
    
    return examples


def filter_and_weight_examples(
    examples: List[Dict],
    min_score: int = MIN_SCORE
) -> List[Dict]:
    """
    Filter examples by score and sort by insulin emphasis.
    
    Returns examples sorted by score (highest first).
    """
    # Filter by minimum score
    filtered = [ex for ex in examples if ex.get("_score", 0) >= min_score]
    
    print(f"\n→ Filtering examples:")
    print(f"  Total examples: {len(examples)}")
    print(f"  Examples with score >= {min_score}: {len(filtered)}")
    
    if len(filtered) < len(examples):
        print(f"  Filtered out: {len(examples) - len(filtered)} low-insulin examples")
    
    # Sort by score (highest first)
    filtered.sort(key=lambda x: x.get("_score", 0), reverse=True)
    
    # Show score distribution
    scores = [ex.get("_score", 0) for ex in filtered]
    if scores:
        print(f"\n→ Score distribution:")
        print(f"  Min: {min(scores)}")
        print(f"  Max: {max(scores)}")
        print(f"  Mean: {sum(scores) / len(scores):.1f}")
        print(f"  Median: {sorted(scores)[len(scores)//2]}")
    
    # Show top examples
    print(f"\n→ Top 5 highest-scoring examples:")
    for i, ex in enumerate(filtered[:5], 1):
        score = ex.get("_score", 0)
        text_preview = ex['text'][:100].replace('\n', ' ')
        print(f"  {i}. Score: {score} - {text_preview}...")
    
    return filtered


def select_training_examples(
    examples: List[Dict],
    target_count: int,
    seed: int = 42
) -> List[Dict]:
    """
    Select examples for training, prioritizing high-scoring ones.
    
    Strategy:
    - Take top 80% from highest-scoring examples
    - Take remaining 20% randomly from rest (to maintain diversity)
    """
    if len(examples) <= target_count:
        return examples
    
    random.seed(seed)
    
    # Take top examples (80% of target)
    top_count = int(target_count * 0.8)
    top_examples = examples[:top_count]
    
    # Take remaining from rest (20% of target, for diversity)
    remaining_count = target_count - top_count
    remaining_examples = examples[top_count:]
    
    if len(remaining_examples) >= remaining_count:
        random.shuffle(remaining_examples)
        selected_remaining = remaining_examples[:remaining_count]
    else:
        selected_remaining = remaining_examples
    
    selected = top_examples + selected_remaining
    
    # Shuffle final selection (but keep high-scoring bias)
    random.shuffle(selected)
    
    return selected


def save_jsonl(examples: List[Dict], output_path: Path, remove_metadata: bool = True) -> None:
    """Save examples to JSONL file, optionally removing scoring metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            # Create clean entry without metadata
            if remove_metadata:
                clean_example = {"text": example["text"]}
            else:
                clean_example = example
            
            json_line = json.dumps(clean_example, ensure_ascii=False) + "\n"
            f.write(json_line)


def main():
    """Main entry point."""
    input_path = project_root / INPUT_FILE
    output_dir = project_root / OUTPUT_DIR
    train_path = output_dir / TRAIN_FILE
    valid_path = output_dir / VALID_FILE
    
    print(f"{'='*70}")
    print(f"Creating Insulin-Biased Training Data")
    print(f"{'='*70}\n")
    
    # Load and score examples
    examples = convert_to_granite_format(input_path)
    
    if len(examples) == 0:
        print(f"❌ No examples loaded. Check input file.")
        return
    
    # Filter and weight by insulin emphasis
    filtered_examples = filter_and_weight_examples(examples, MIN_SCORE)
    
    if len(filtered_examples) < TARGET_TRAIN_COUNT:
        print(f"\n⚠️  Warning: Only {len(filtered_examples)} examples pass filter, but {TARGET_TRAIN_COUNT} requested.")
        print(f"   Lowering minimum score threshold...")
        # Lower threshold and try again
        filtered_examples = filter_and_weight_examples(examples, min_score=0)
    
    # Select training examples (prioritize high-scoring)
    print(f"\n→ Selecting training examples:")
    print(f"  Target: {TARGET_TRAIN_COUNT} examples")
    print(f"  Strategy: Top 90% highest-scoring + 10% random for diversity")
    print(f"  (Increased from 80/20 to 90/10 for stronger insulin bias)")
    
    # Override the function's default 80/20 split with 90/10
    if len(filtered_examples) <= TARGET_TRAIN_COUNT:
        train_examples = filtered_examples
    else:
        random.seed(SEED)
        top_count = int(TARGET_TRAIN_COUNT * 0.9)  # 90% top-scoring
        top_examples = filtered_examples[:top_count]
        remaining_count = TARGET_TRAIN_COUNT - top_count
        remaining_examples = filtered_examples[top_count:]
        random.shuffle(remaining_examples)
        selected_remaining = remaining_examples[:remaining_count]
        train_examples = top_examples + selected_remaining
        random.shuffle(train_examples)
    
    # Select validation examples (also insulin-biased)
    print(f"\n→ Selecting validation examples:")
    print(f"  Target: {TARGET_VAL_COUNT} examples")
    
    # Remove training examples from pool
    train_texts = {ex["text"] for ex in train_examples}
    remaining_examples = [ex for ex in filtered_examples if ex["text"] not in train_texts]
    
    val_examples = select_training_examples(remaining_examples, TARGET_VAL_COUNT, SEED + 1)
    
    # Save files
    print(f"\n→ Saving files...")
    save_jsonl(train_examples, train_path, remove_metadata=True)
    print(f"  ✓ Train: {train_path} ({len(train_examples)} examples)")
    
    save_jsonl(val_examples, valid_path, remove_metadata=True)
    print(f"  ✓ Validation: {valid_path} ({len(val_examples)} examples)")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"✅ INSULIN-BIASED DATA CREATED")
    print(f"{'='*70}")
    print(f"   Training examples: {len(train_examples)}")
    print(f"   Validation examples: {len(val_examples)}")
    print(f"   Average training score: {sum(ex.get('_score', 0) for ex in train_examples) / len(train_examples):.1f}")
    print(f"   Average validation score: {sum(ex.get('_score', 0) for ex in val_examples) / len(val_examples):.1f}")
    print(f"\n   Files ready for Granite training:")
    print(f"   - {train_path}")
    print(f"   - {valid_path}")
    print(f"\n   Format: Granite text format")
    print(f"   Bias: Strongly favors insulin-focused examples")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

