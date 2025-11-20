#!/usr/bin/env python3
"""
Split Questions into Train/Val for MLX Training
───────────────────────────────────────────────
Loads questions from generated_questions.json, matches them with answers from
generated_answers.jsonl, formats as MLX messages format, and splits into
1600 training examples and the rest for validation.
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Get project root
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

# ─────────────────────────────
# Config
# ─────────────────────────────
QUESTIONS_FILE = project_root / "data/generated_questions.json"
ANSWERS_FILE = project_root / "data/generated_answers.jsonl"
TRAIN_FILE = project_root / "data/mlx_training_data/train.jsonl"
VAL_FILE = project_root / "data/mlx_training_data/valid.jsonl"

TRAIN_COUNT = 1600  # Exact number of training examples
SEED = 42  # For reproducibility


def load_questions(questions_path: Path) -> List[Dict]:
    """Load questions from JSON file."""
    print(f"→ Loading questions from: {questions_path}")
    
    if not questions_path.exists():
        print(f"❌ Questions file not found: {questions_path}")
        return []
    
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    print(f"  ✓ Loaded {len(questions)} questions")
    return questions


def load_answers(answers_path: Path) -> Dict[str, Dict]:
    """
    Load answers from JSONL file and create a lookup dictionary.
    Key: question text (normalized), Value: full answer entry
    """
    print(f"→ Loading answers from: {answers_path}")
    
    if not answers_path.exists():
        print(f"❌ Answers file not found: {answers_path}")
        return {}
    
    answers_lookup = {}
    skipped = 0
    
    with open(answers_path, "r", encoding="utf-8") as f:
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
                
                # Use normalized question as key (lowercase, strip)
                key = question.lower().strip()
                answers_lookup[key] = data
                
            except json.JSONDecodeError as e:
                print(f"  ⚠️  JSON error on line {line_num}: {str(e)[:100]}")
                skipped += 1
                continue
    
    print(f"  ✓ Loaded {len(answers_lookup)} answers")
    if skipped > 0:
        print(f"  ⚠️  Skipped {skipped} invalid lines")
    
    return answers_lookup


def match_questions_with_answers(
    questions: List[Dict],
    answers_lookup: Dict[str, Dict]
) -> List[Dict]:
    """
    Match questions with their answers and format as MLX messages format.
    
    Returns list of entries in MLX format:
    {"messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]}
    """
    matched = []
    unmatched = []
    
    for q_entry in questions:
        question_text = q_entry.get("question", "").strip()
        
        if not question_text:
            continue
        
        # Try to find matching answer
        key = question_text.lower().strip()
        answer_entry = answers_lookup.get(key)
        
        if answer_entry:
            answer_text = answer_entry.get("answer", "").strip()
            
            if answer_text:
                # Format as MLX messages format
                mlx_entry = {
                    "messages": [
                        {"role": "user", "content": question_text},
                        {"role": "assistant", "content": answer_text}
                    ]
                }
                matched.append(mlx_entry)
            else:
                unmatched.append(question_text)
        else:
            unmatched.append(question_text)
    
    print(f"\n→ Matching results:")
    print(f"  ✓ Matched: {len(matched)} questions with answers")
    if unmatched:
        print(f"  ⚠️  Unmatched: {len(unmatched)} questions without answers")
        if len(unmatched) <= 5:
            print(f"     Examples: {unmatched[:3]}")
    
    return matched


def split_dataset(
    examples: List[Dict],
    train_count: int,
    seed: int = 42
) -> tuple[List[Dict], List[Dict]]:
    """
    Split dataset into train and validation sets.
    
    Args:
        examples: List of example dictionaries
        train_count: Exact number of examples for training
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_examples, val_examples)
    """
    if len(examples) < train_count:
        print(f"⚠️  Warning: Only {len(examples)} examples available, but {train_count} requested for training")
        train_count = len(examples)
    
    # Shuffle with seed for reproducibility
    random.seed(seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)
    
    # Split
    train_examples = shuffled[:train_count]
    val_examples = shuffled[train_count:]
    
    return train_examples, val_examples


def save_jsonl(examples: List[Dict], output_path: Path) -> None:
    """Save examples to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            # Preserve all formatting (ensure_ascii=False keeps unicode and formatting)
            json_line = json.dumps(example, ensure_ascii=False) + "\n"
            f.write(json_line)


def main():
    """Main entry point."""
    print(f"{'='*70}")
    print(f"Splitting Questions into Train/Val for MLX Training")
    print(f"{'='*70}\n")
    
    # Load questions
    questions = load_questions(QUESTIONS_FILE)
    if not questions:
        print(f"❌ No questions loaded. Exiting.")
        sys.exit(1)
    
    # Load answers
    answers_lookup = load_answers(ANSWERS_FILE)
    if not answers_lookup:
        print(f"❌ No answers loaded. Exiting.")
        sys.exit(1)
    
    # Match questions with answers
    matched_examples = match_questions_with_answers(questions, answers_lookup)
    
    if not matched_examples:
        print(f"❌ No matched examples. Exiting.")
        sys.exit(1)
    
    # Split dataset
    print(f"\n→ Splitting dataset:")
    print(f"  Train: {TRAIN_COUNT} examples")
    print(f"  Validation: {len(matched_examples) - TRAIN_COUNT} examples (remaining)")
    print(f"  Seed: {SEED} (for reproducibility)")
    
    train_examples, val_examples = split_dataset(matched_examples, TRAIN_COUNT, SEED)
    
    print(f"\n→ Split results:")
    print(f"  Train: {len(train_examples)} examples")
    print(f"  Validation: {len(val_examples)} examples")
    
    # Save splits
    print(f"\n→ Saving splits...")
    save_jsonl(train_examples, TRAIN_FILE)
    print(f"  ✓ Train: {TRAIN_FILE}")
    
    save_jsonl(val_examples, VAL_FILE)
    print(f"  ✓ Validation: {VAL_FILE}")
    
    # Verify formatting
    if train_examples:
        sample = train_examples[0]
        if "messages" in sample:
            has_formatting = any(
                marker in sample["messages"][1].get("content", "")
                for marker in ["**", "\n\n", "- ", "* "]
            )
            if has_formatting:
                print(f"\n  ✓ Formatting preserved in output (bold, lists, paragraphs)")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"✅ SPLIT COMPLETE")
    print(f"{'='*70}")
    print(f"   Total matched examples: {len(matched_examples)}")
    print(f"   Train: {len(train_examples)} examples")
    print(f"   Validation: {len(val_examples)} examples")
    print(f"\n   Files ready for MLX training:")
    print(f"   - {TRAIN_FILE}")
    print(f"   - {VAL_FILE}")
    print(f"   Format: MLX messages format (chat format)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()




















