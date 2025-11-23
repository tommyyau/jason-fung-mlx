#!/usr/bin/env python3
"""
Step 06b2 – Compare Model Responses
────────────────────────────────────
Compares saved model responses against ground truth answers using heuristic metrics.

This script is fast because it only does comparisons, no generation.
"""

import json
import sys
import argparse
import re
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import statistics

# ─────────────────────────────
# Configuration
# ─────────────────────────────
DEFAULT_RESPONSES_FILE = "data/model_responses.jsonl"
DEFAULT_VAL_FILE = "data/generated_answers_mlx_validate.jsonl"


def check_formatting_compliance(text: str) -> Dict[str, bool]:
    """Check if text has proper markdown formatting."""
    return {
        "has_bold": "**" in text,
        "has_lists": any(marker in text for marker in ["- ", "* ", "1. ", "2. "]),
        "has_paragraphs": "\n\n" in text,
        "has_emphasis": any(marker in text for marker in ["**", "*", "_"]),
    }


def calculate_formatting_score(text: str) -> float:
    """Calculate formatting score (0-1) based on markdown elements."""
    checks = check_formatting_compliance(text)
    score = 0.0
    
    if checks["has_bold"]:
        score += 0.3
    if checks["has_lists"]:
        score += 0.3
    if checks["has_paragraphs"]:
        score += 0.2
    if checks["has_emphasis"]:
        score += 0.2
    
    return min(score, 1.0)


def load_responses(responses_file: Path) -> List[Dict]:
    """Load generated responses from JSONL file."""
    responses = []
    
    if not responses_file.exists():
        print(f"❌ Responses file not found: {responses_file}")
        return responses
    
    print(f"→ Loading responses from: {responses_file}")
    
    with open(responses_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                responses.append(data)
            except json.JSONDecodeError as e:
                print(f"  ⚠️  JSON error on line {line_num}: {str(e)[:100]}")
                continue
    
    print(f"  ✓ Loaded {len(responses)} responses")
    return responses


def compare_responses(responses: List[Dict]) -> List[Dict]:
    """Compare responses against ground truth."""
    results = []
    
    for resp_data in responses:
        instruction = resp_data.get("instruction", "")
        response = resp_data.get("response", "")
        ground_truth = resp_data.get("ground_truth", "")
        
        if not response:
            results.append({
                "instruction": instruction,
                "error": "No response generated",
                "response_length": 0,
                "formatting_score": 0.0,
            })
            continue
        
        # Calculate metrics
        response_length = len(response)
        formatting_checks = check_formatting_compliance(response)
        formatting_score = calculate_formatting_score(response)
        
        result = {
            "instruction": instruction,
            "response": response,
            "response_length": response_length,
            "formatting_score": formatting_score,
            "formatting_checks": formatting_checks,
        }
        
        # Add ground truth comparison if available
        if ground_truth:
            result["ground_truth_length"] = len(ground_truth)
            result["ground_truth_formatting_score"] = calculate_formatting_score(ground_truth)
            
            # Length similarity
            if len(ground_truth) > 0:
                length_ratio = min(response_length, len(ground_truth)) / max(response_length, len(ground_truth))
                result["length_similarity"] = length_ratio
            else:
                result["length_similarity"] = 0.0
        
        results.append(result)
    
    return results


def calculate_statistics(results: List[Dict]) -> Dict:
    """Calculate aggregate statistics from comparison results."""
    if not results:
        return {}
    
    valid_results = [r for r in results if "error" not in r]
    
    if not valid_results:
        return {"error": "No valid results to analyze"}
    
    stats = {
        "total_examples": len(results),
        "valid_examples": len(valid_results),
        "error_count": len(results) - len(valid_results),
    }
    
    # Response length statistics
    lengths = [r["response_length"] for r in valid_results]
    stats["response_length"] = {
        "mean": statistics.mean(lengths),
        "median": statistics.median(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "std": statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
    }
    
    # Formatting statistics
    formatting_scores = [r["formatting_score"] for r in valid_results]
    stats["formatting_score"] = {
        "mean": statistics.mean(formatting_scores),
        "median": statistics.median(formatting_scores),
        "min": min(formatting_scores),
        "max": max(formatting_scores),
    }
    
    # Formatting compliance rates
    formatting_checks = defaultdict(int)
    for r in valid_results:
        for key, value in r.get("formatting_checks", {}).items():
            if value:
                formatting_checks[key] += 1
    
    stats["formatting_compliance"] = {
        key: (count / len(valid_results)) * 100
        for key, count in formatting_checks.items()
    }
    
    # Ground truth comparison (if available)
    if any("ground_truth_length" in r for r in valid_results):
        length_similarities = [
            r.get("length_similarity", 0.0)
            for r in valid_results
            if "length_similarity" in r
        ]
        if length_similarities:
            stats["ground_truth_comparison"] = {
                "mean_length_similarity": statistics.mean(length_similarities),
                "median_length_similarity": statistics.median(length_similarities),
            }
    
    return stats


def print_summary(stats: Dict):
    """Print evaluation summary."""
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total examples: {stats.get('total_examples', 0)}")
    print(f"  Valid examples: {stats.get('valid_examples', 0)}")
    print(f"  Errors: {stats.get('error_count', 0)}")
    
    if "response_length" in stats:
        rl = stats["response_length"]
        print(f"\n📏 Response Length:")
        print(f"  Mean: {rl['mean']:.0f} characters")
        print(f"  Median: {rl['median']:.0f} characters")
        print(f"  Range: {rl['min']} - {rl['max']} characters")
        print(f"  Std Dev: {rl['std']:.0f} characters")
    
    if "formatting_score" in stats:
        fs = stats["formatting_score"]
        print(f"\n✨ Formatting Score (0-1):")
        print(f"  Mean: {fs['mean']:.3f}")
        print(f"  Median: {fs['median']:.3f}")
        print(f"  Range: {fs['min']:.3f} - {fs['max']:.3f}")
    
    if "formatting_compliance" in stats:
        fc = stats["formatting_compliance"]
        print(f"\n📝 Formatting Compliance:")
        print(f"  Has bold (**): {fc.get('has_bold', 0):.1f}%")
        print(f"  Has lists (-, *, 1.): {fc.get('has_lists', 0):.1f}%")
        print(f"  Has paragraphs (\\n\\n): {fc.get('has_paragraphs', 0):.1f}%")
        print(f"  Has emphasis: {fc.get('has_emphasis', 0):.1f}%")
    
    if "ground_truth_comparison" in stats:
        gt = stats["ground_truth_comparison"]
        print(f"\n🎯 Ground Truth Comparison:")
        print(f"  Mean length similarity: {gt.get('mean_length_similarity', 0):.3f}")
        print(f"  Median length similarity: {gt.get('median_length_similarity', 0):.3f}")
    
    print("\n" + "=" * 80)


def find_next_run_number(project_root: Path) -> int:
    """Find the next run number for output file."""
    pattern = re.compile(r"comparison_results_run(\d+)\.json")
    max_run = 0
    
    for file_path in project_root.glob("comparison_results_run*.json"):
        match = pattern.match(file_path.name)
        if match:
            run_num = int(match.group(1))
            max_run = max(max_run, run_num)
    
    return max_run + 1


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(
        description="Compare saved model responses against ground truth"
    )
    parser.add_argument(
        "--responses",
        type=str,
        default=DEFAULT_RESPONSES_FILE,
        help=f"File containing generated responses (default: {DEFAULT_RESPONSES_FILE})",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default=DEFAULT_VAL_FILE,
        help=f"Validation file (for reference, default: {DEFAULT_VAL_FILE})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for detailed results (JSON format). Auto-generates if not provided",
    )

    args = parser.parse_args()
    
    print("=" * 80)
    print("Response Comparison")
    print("=" * 80)
    
    # Get paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    responses_file = project_root / args.responses
    val_file = project_root / args.val_file
    
    # Validate paths
    if not responses_file.exists():
        print(f"\n❌ Error: Responses file not found: {responses_file}")
        print(f"   Run 06b1_generate_responses.py first to generate responses")
        sys.exit(1)
    
    # Auto-generate output filename if not provided
    if args.output is None:
        next_run = find_next_run_number(project_root)
        args.output = f"comparison_results_run{next_run}.json"
        print(f"→ Auto-detected next run number: {next_run}")
        print(f"→ Output will be saved to: {args.output}")
    
    # Load responses
    responses = load_responses(responses_file)
    
    if len(responses) == 0:
        print(f"\n❌ No responses loaded")
        sys.exit(1)
    
    # Compare responses
    print(f"\n→ Comparing {len(responses)} responses against ground truth...")
    results = compare_responses(responses)
    
    # Calculate statistics
    print(f"\n→ Calculating statistics...")
    stats = calculate_statistics(results)
    
    # Print summary
    print_summary(stats)
    
    # Save detailed results
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "responses_file": str(responses_file),
        "validation_file": str(val_file),
        "total_examples": len(responses),
        "statistics": stats,
        "results": results,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Detailed results saved to: {output_path}")
    print("\n✅ Comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()







































