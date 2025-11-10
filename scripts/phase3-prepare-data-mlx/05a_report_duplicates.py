#!/usr/bin/env python3
"""
Step 05a – Report Duplicate Questions in Training Data
───────────────────────────────────────────────────────
Analyzes the training dataset to find near-duplicate questions and reports
on their variations. Uses SequenceMatcher to detect questions with >85% similarity.

This script only REPORTS duplicates - it does not modify or remove them.
"""

import json
import sys
from pathlib import Path
from difflib import SequenceMatcher, unified_diff
from typing import List, Dict, Tuple, Optional

# Load configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root / "config"))
from load_config import load_config, get_data_config

# ─────────────────────────────
# Config
# ─────────────────────────────
config = load_config()
data_config = get_data_config()

TRAIN_FILE = data_config.get("train_file", "data/generated_answers_mlx_train.jsonl")
SIMILARITY_THRESHOLD = 0.85  # 85% similarity threshold (as per audit recommendation)


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip whitespace."""
    return text.lower().strip()


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity ratio between two texts using SequenceMatcher.
    
    Returns a value between 0.0 (completely different) and 1.0 (identical).
    """
    return SequenceMatcher(None, normalize_text(text1), normalize_text(text2)).ratio()


def find_differences(text1: str, text2: str) -> List[str]:
    """
    Find and return the specific differences between two texts.
    Returns a list of difference descriptions.
    """
    diff_lines = list(unified_diff(
        text1.splitlines(keepends=True),
        text2.splitlines(keepends=True),
        lineterm='',
        fromfile='Question 1',
        tofile='Question 2'
    ))
    
    if not diff_lines:
        return ["No differences found (texts are identical)"]
    
    # Extract meaningful differences
    differences = []
    for line in diff_lines[2:]:  # Skip header lines
        if line.startswith('---') or line.startswith('+++'):
            continue
        if line.startswith('@@'):
            continue
        if line.strip():
            differences.append(line.rstrip())
    
    return differences if differences else ["Texts differ but differences are subtle"]


def find_duplicate_questions(
    examples: List[Dict],
    threshold: float = 0.85
) -> List[Tuple[int, int, float, str, str, str, str]]:
    """
    Find near-duplicate questions in the dataset.
    
    Args:
        examples: List of example dictionaries with 'instruction' and 'output' fields
        threshold: Similarity threshold (default 0.85 = 85%)
    
    Returns:
        List of tuples: (index1, index2, similarity, question1, question2, answer1_preview, answer2_preview)
    """
    duplicates = []
    
    print(f"→ Comparing {len(examples)} questions for duplicates (threshold: {threshold*100:.0f}%)...")
    
    for i in range(len(examples)):
        if i % 100 == 0 and i > 0:
            print(f"  Progress: {i}/{len(examples)} comparisons...")
        
        q1 = examples[i].get("instruction", "")
        if not q1:
            continue
        
        for j in range(i + 1, len(examples)):
            q2 = examples[j].get("instruction", "")
            if not q2:
                continue
            
            similarity = calculate_similarity(q1, q2)
            
            if similarity >= threshold:
                # Get answer previews (first 100 chars)
                a1 = examples[i].get("output", "")[:100] + "..." if len(examples[i].get("output", "")) > 100 else examples[i].get("output", "")
                a2 = examples[j].get("output", "")[:100] + "..." if len(examples[j].get("output", "")) > 100 else examples[j].get("output", "")
                
                duplicates.append((i, j, similarity, q1, q2, a1, a2))
    
    return duplicates


def analyze_duplicate_pair(
    idx1: int,
    idx2: int,
    similarity: float,
    q1: str,
    q2: str,
    a1: str,
    a2: str
) -> Dict:
    """
    Analyze a duplicate pair and return detailed information about the variation.
    """
    # Character-level differences
    q1_len = len(q1)
    q2_len = len(q2)
    len_diff = abs(q1_len - q2_len)
    len_diff_pct = (len_diff / max(q1_len, q2_len)) * 100 if max(q1_len, q2_len) > 0 else 0
    
    # Word-level differences
    q1_words = q1.split()
    q2_words = q2.split()
    word_diff = abs(len(q1_words) - len(q2_words))
    
    # Find specific differences
    differences = find_differences(q1, q2)
    
    # Answer length comparison
    a1_full_len = len(a1)
    a2_full_len = len(a2)
    answer_len_diff = abs(a1_full_len - a2_full_len)
    
    return {
        "indices": (idx1, idx2),
        "similarity": similarity,
        "question1": q1,
        "question2": q2,
        "question1_length": q1_len,
        "question2_length": q2_len,
        "length_difference": len_diff,
        "length_difference_pct": len_diff_pct,
        "word_count1": len(q1_words),
        "word_count2": len(q2_words),
        "word_difference": word_diff,
        "differences": differences,
        "answer1_preview": a1,
        "answer2_preview": a2,
        "answer1_length": a1_full_len,
        "answer2_length": a2_full_len,
        "answer_length_difference": answer_len_diff,
    }


def load_dataset(input_path: Path) -> List[Dict]:
    """Load MLX format dataset from JSONL file."""
    examples = []
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return examples
    
    print(f"→ Loading dataset from: {input_path}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                
                if "instruction" not in data:
                    print(f"  ⚠️  Skipping line {line_num}: missing 'instruction' field")
                    continue
                
                examples.append(data)
                
            except json.JSONDecodeError as e:
                print(f"  ⚠️  JSON error on line {line_num}: {str(e)[:100]}")
                continue
            except Exception as e:
                print(f"  ⚠️  Error processing line {line_num}: {type(e).__name__}: {str(e)[:100]}")
                continue
    
    print(f"  ✓ Loaded {len(examples)} examples")
    return examples


def print_duplicate_report(duplicates: List[Tuple], examples: List[Dict]) -> None:
    """Print a detailed report of all duplicates found."""
    
    if not duplicates:
        print(f"\n{'='*70}")
        print(f"✅ NO DUPLICATES FOUND")
        print(f"{'='*70}")
        print(f"\nNo questions with similarity >= {SIMILARITY_THRESHOLD*100:.0f}% were found.")
        print(f"This is good - your dataset appears to have unique questions.")
        return
    
    # Separate exact duplicates (100%) from near-duplicates
    exact_duplicates = [d for d in duplicates if d[2] >= 1.0]
    near_duplicates = [d for d in duplicates if d[2] < 1.0]
    
    print(f"\n{'='*70}")
    print(f"DUPLICATE REPORT")
    print(f"{'='*70}")
    print(f"\nFound {len(duplicates)} duplicate pair(s) with similarity >= {SIMILARITY_THRESHOLD*100:.0f}%")
    print(f"  - Exact duplicates (100% identical): {len(exact_duplicates)}")
    print(f"  - Near-duplicates ({SIMILARITY_THRESHOLD*100:.0f}%-99%): {len(near_duplicates)}")
    
    print(f"\nWhat 'duplicate' means:")
    print(f"  - Exact duplicates: Character-for-character identical questions")
    print(f"  - Near-duplicates: Questions that are {SIMILARITY_THRESHOLD*100:.0f}%+ similar")
    print(f"    (May have minor variations: word order, punctuation, phrasing)")
    print(f"    (Same core question asked slightly differently)")
    
    # Show exact duplicates first if they exist
    if exact_duplicates:
        print(f"\n{'='*70}")
        print(f"EXACT DUPLICATES (100% Identical)")
        print(f"{'='*70}\n")
        
        for pair_num, (idx1, idx2, similarity, q1, q2, a1, a2) in enumerate(exact_duplicates, 1):
            analysis = analyze_duplicate_pair(idx1, idx2, similarity, q1, q2, a1, a2)
            
            print(f"\n{'─'*70}")
            print(f"EXACT DUPLICATE #{pair_num}")
            print(f"{'─'*70}")
            print(f"\n📍 Indices: {idx1} and {idx2} (0-based)")
            print(f"📊 Similarity: {similarity*100:.2f}% (IDENTICAL)")
            
            print(f"\n📝 QUESTION (appears twice):")
            print(f"   {q1}")
            print(f"   Length: {analysis['question1_length']} chars, {analysis['word_count1']} words")
            
            print(f"\n💬 ANSWER COMPARISON:")
            print(f"   Answer 1 length: {analysis['answer1_length']} chars")
            print(f"   Answer 2 length: {analysis['answer2_length']} chars")
            print(f"   Length difference: {analysis['answer_length_difference']} chars")
            
            if analysis['answer_length_difference'] > 50:
                print(f"   ⚠️  Note: Same question but answers differ significantly in length")
            
            print(f"\n   Answer 1 (first 200 chars):")
            print(f"   {a1[:200]}...")
            print(f"\n   Answer 2 (first 200 chars):")
            print(f"   {a2[:200]}...")
    
    # Show near-duplicates if they exist
    if near_duplicates:
        print(f"\n{'='*70}")
        print(f"NEAR-DUPLICATES ({SIMILARITY_THRESHOLD*100:.0f}%-99% Similar)")
        print(f"{'='*70}\n")
        
        # Limit to showing first 10 near-duplicates to avoid overwhelming output
        show_near = near_duplicates[:10] if len(near_duplicates) > 10 else near_duplicates
        
        for pair_num, (idx1, idx2, similarity, q1, q2, a1, a2) in enumerate(show_near, 1):
            analysis = analyze_duplicate_pair(idx1, idx2, similarity, q1, q2, a1, a2)
            
            print(f"\n{'─'*70}")
            print(f"NEAR-DUPLICATE #{pair_num}")
            print(f"{'─'*70}")
            print(f"\n📍 Indices: {idx1} and {idx2} (0-based)")
            print(f"📊 Similarity: {similarity*100:.2f}%")
            
            print(f"\n📝 QUESTION 1 (Index {idx1}):")
            print(f"   {q1}")
            print(f"   Length: {analysis['question1_length']} chars, {analysis['word_count1']} words")
            
            print(f"\n📝 QUESTION 2 (Index {idx2}):")
            print(f"   {q2}")
            print(f"   Length: {analysis['question2_length']} chars, {analysis['word_count2']} words")
            
            print(f"\n🔍 VARIATIONS:")
            print(f"   Length difference: {analysis['length_difference']} chars ({analysis['length_difference_pct']:.1f}%)")
            print(f"   Word count difference: {analysis['word_difference']} words")
            
            # Show word differences
            q1_words = set(q1.lower().split())
            q2_words = set(q2.lower().split())
            only_in_q1 = q1_words - q2_words
            only_in_q2 = q2_words - q1_words
            
            if only_in_q1:
                print(f"\n   Words only in Question 1: {', '.join(sorted(only_in_q1))}")
            if only_in_q2:
                print(f"   Words only in Question 2: {', '.join(sorted(only_in_q2))}")
        
        if len(near_duplicates) > 10:
            print(f"\n   ... and {len(near_duplicates) - 10} more near-duplicate pairs (not shown)")
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total duplicate pairs: {len(duplicates)}")
    print(f"  - Exact duplicates: {len(exact_duplicates)}")
    print(f"  - Near-duplicates: {len(near_duplicates)}")
    print(f"Total unique questions affected: {len(set([d[0] for d in duplicates] + [d[1] for d in duplicates]))}")
    print(f"Effective dataset size reduction: {len(duplicates)} examples")
    print(f"\n💡 Recommendation:")
    if len(exact_duplicates) > 0:
        print(f"   Found {len(exact_duplicates)} exact duplicate(s) - these should be removed.")
    if len(near_duplicates) > 0:
        if len(near_duplicates) <= 5:
            print(f"   Found {len(near_duplicates)} near-duplicate(s) - consider manual review.")
        else:
            print(f"   Found {len(near_duplicates)} near-duplicates - consider automated deduplication.")
    print(f"{'='*70}\n")


def main():
    """Main entry point."""
    train_path = project_root / TRAIN_FILE
    
    print(f"{'='*70}")
    print(f"Duplicate Question Detection Report")
    print(f"{'='*70}\n")
    print(f"Analyzing: {train_path}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD*100:.0f}%")
    print(f"(Questions with {SIMILARITY_THRESHOLD*100:.0f}%+ similarity are considered duplicates)\n")
    
    # Load dataset
    examples = load_dataset(train_path)
    
    if len(examples) == 0:
        print(f"❌ No examples loaded. Check input file: {train_path}")
        sys.exit(1)
    
    # Find duplicates
    duplicates = find_duplicate_questions(examples, threshold=SIMILARITY_THRESHOLD)
    
    # Print report
    print_duplicate_report(duplicates, examples)


if __name__ == "__main__":
    main()

