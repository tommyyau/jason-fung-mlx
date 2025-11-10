#!/usr/bin/env python3
"""
Step 06c – Interpret Evaluation Results
───────────────────────────────────────
Analyzes evaluation results and provides clear interpretation with benchmarks
to help understand if the fine-tuning was successful.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import statistics

# ─────────────────────────────
# Benchmarks (what's considered "good")
# ─────────────────────────────
BENCHMARKS = {
    "response_length": {
        "min_good": 500,      # Minimum for a good response
        "target_mean": 800,    # Target average (based on training data)
        "max_reasonable": 3000, # Maximum before it's too long
    },
    "formatting_score": {
        "min_good": 0.5,      # At least 50% formatting elements
        "target": 0.7,        # Target formatting score
        "excellent": 0.9,     # Excellent formatting
    },
    "formatting_compliance": {
        "has_bold": {"min": 40, "target": 60},      # % with bold text
        "has_lists": {"min": 50, "target": 70},    # % with lists
        "has_paragraphs": {"min": 70, "target": 85}, # % with paragraphs
    },
    "length_similarity": {
        "min_good": 0.4,      # At least 40% similarity to ground truth length
        "target": 0.6,        # Target similarity
    },
}


def load_evaluation_results(results_file: Path) -> Dict:
    """Load evaluation results from JSON file."""
    with open(results_file, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_response_length(stats: Dict) -> Dict:
    """Analyze response length metrics."""
    rl = stats.get("response_length", {})
    mean_len = rl.get("mean", 0)
    median_len = rl.get("median", 0)
    min_len = rl.get("min", 0)
    max_len = rl.get("max", 0)
    
    bench = BENCHMARKS["response_length"]
    
    analysis = {
        "mean_length": mean_len,
        "median_length": median_len,
        "min_length": min_len,
        "max_length": max_len,
        "assessment": "unknown",
        "score": 0.0,
        "notes": [],
    }
    
    # Score based on mean length
    if mean_len >= bench["target_mean"]:
        analysis["score"] = 1.0
        analysis["assessment"] = "excellent"
        analysis["notes"].append(f"Mean length ({mean_len:.0f} chars) exceeds target ({bench['target_mean']} chars)")
    elif mean_len >= bench["min_good"]:
        ratio = (mean_len - bench["min_good"]) / (bench["target_mean"] - bench["min_good"])
        analysis["score"] = 0.5 + (ratio * 0.5)  # 0.5 to 1.0
        analysis["assessment"] = "good"
        analysis["notes"].append(f"Mean length ({mean_len:.0f} chars) is good, approaching target")
    else:
        ratio = mean_len / bench["min_good"]
        analysis["score"] = ratio * 0.5  # 0.0 to 0.5
        analysis["assessment"] = "needs_improvement"
        analysis["notes"].append(f"Mean length ({mean_len:.0f} chars) is below minimum ({bench['min_good']} chars)")
    
    # Check for outliers
    if min_len < 100:
        analysis["notes"].append(f"⚠️  Some responses are very short (min: {min_len} chars)")
    if max_len > bench["max_reasonable"]:
        analysis["notes"].append(f"⚠️  Some responses are very long (max: {max_len} chars)")
    
    return analysis


def analyze_formatting(stats: Dict) -> Dict:
    """Analyze formatting compliance metrics."""
    fs = stats.get("formatting_score", {})
    fc = stats.get("formatting_compliance", {})
    
    mean_score = fs.get("mean", 0)
    median_score = fs.get("median", 0)
    
    bench = BENCHMARKS["formatting_score"]
    fc_bench = BENCHMARKS["formatting_compliance"]
    
    analysis = {
        "mean_score": mean_score,
        "median_score": median_score,
        "assessment": "unknown",
        "score": 0.0,
        "notes": [],
        "compliance": {},
    }
    
    # Score based on mean formatting
    if mean_score >= bench["excellent"]:
        analysis["score"] = 1.0
        analysis["assessment"] = "excellent"
        analysis["notes"].append(f"Formatting score ({mean_score:.2f}) is excellent")
    elif mean_score >= bench["target"]:
        ratio = (mean_score - bench["target"]) / (bench["excellent"] - bench["target"])
        analysis["score"] = 0.7 + (ratio * 0.3)  # 0.7 to 1.0
        analysis["assessment"] = "good"
        analysis["notes"].append(f"Formatting score ({mean_score:.2f}) meets target")
    elif mean_score >= bench["min_good"]:
        ratio = (mean_score - bench["min_good"]) / (bench["target"] - bench["min_good"])
        analysis["score"] = 0.4 + (ratio * 0.3)  # 0.4 to 0.7
        analysis["assessment"] = "acceptable"
        analysis["notes"].append(f"Formatting score ({mean_score:.2f}) is acceptable but below target")
    else:
        ratio = mean_score / bench["min_good"]
        analysis["score"] = ratio * 0.4  # 0.0 to 0.4
        analysis["assessment"] = "needs_improvement"
        analysis["notes"].append(f"Formatting score ({mean_score:.2f}) needs improvement")
    
    # Check individual formatting elements
    for element, value in fc.items():
        element_bench = fc_bench.get(element, {})
        target = element_bench.get("target", 50)
        min_val = element_bench.get("min", 30)
        
        if value >= target:
            status = "✓"
        elif value >= min_val:
            status = "~"
        else:
            status = "✗"
        
        analysis["compliance"][element] = {
            "percentage": value,
            "status": status,
            "meets_target": value >= target,
        }
    
    return analysis


def analyze_ground_truth_comparison(stats: Dict) -> Dict:
    """Analyze comparison with ground truth answers."""
    gt = stats.get("ground_truth_comparison", {})
    
    if not gt:
        return {
            "available": False,
            "note": "Ground truth comparison not available",
        }
    
    mean_sim = gt.get("mean_length_similarity", 0)
    median_sim = gt.get("median_length_similarity", 0)
    
    bench = BENCHMARKS["length_similarity"]
    
    analysis = {
        "available": True,
        "mean_similarity": mean_sim,
        "median_similarity": median_sim,
        "assessment": "unknown",
        "score": 0.0,
        "notes": [],
    }
    
    # Score based on similarity
    if mean_sim >= bench["target"]:
        analysis["score"] = 1.0
        analysis["assessment"] = "excellent"
        analysis["notes"].append(f"Length similarity ({mean_sim:.2f}) matches ground truth well")
    elif mean_sim >= bench["min_good"]:
        ratio = (mean_sim - bench["min_good"]) / (bench["target"] - bench["min_good"])
        analysis["score"] = 0.5 + (ratio * 0.5)  # 0.5 to 1.0
        analysis["assessment"] = "good"
        analysis["notes"].append(f"Length similarity ({mean_sim:.2f}) is reasonable")
    else:
        ratio = mean_sim / bench["min_good"]
        analysis["score"] = ratio * 0.5  # 0.0 to 0.5
        analysis["assessment"] = "needs_improvement"
        analysis["notes"].append(f"Length similarity ({mean_sim:.2f}) is low - responses differ significantly from ground truth")
    
    # Interpretation
    if mean_sim < 0.3:
        analysis["notes"].append("⚠️  Model responses are much longer/shorter than training data")
    elif mean_sim > 0.8:
        analysis["notes"].append("ℹ️  Model responses closely match training data length")
    
    return analysis


def calculate_overall_score(analyses: Dict) -> Dict:
    """Calculate overall assessment score."""
    scores = []
    weights = {
        "response_length": 0.3,
        "formatting": 0.4,
        "ground_truth": 0.3,
    }
    
    if "response_length" in analyses:
        scores.append(("response_length", analyses["response_length"]["score"], weights["response_length"]))
    
    if "formatting" in analyses:
        scores.append(("formatting", analyses["formatting"]["score"], weights["formatting"]))
    
    if "ground_truth" in analyses and analyses["ground_truth"].get("available"):
        scores.append(("ground_truth", analyses["ground_truth"]["score"], weights["ground_truth"]))
    else:
        # Adjust weights if ground truth not available
        total_weight = sum(w for _, _, w in scores)
        if total_weight > 0:
            scores = [(name, score, weight / total_weight) for name, score, weight in scores]
    
    weighted_score = sum(score * weight for _, score, weight in scores)
    
    # Determine overall grade
    if weighted_score >= 0.8:
        grade = "A (Excellent)"
        assessment = "The fine-tuning was very successful. The model learned the style well."
    elif weighted_score >= 0.7:
        grade = "B (Good)"
        assessment = "The fine-tuning was successful. The model shows good style learning."
    elif weighted_score >= 0.6:
        grade = "C (Acceptable)"
        assessment = "The fine-tuning was partially successful. Some improvement needed."
    elif weighted_score >= 0.5:
        grade = "D (Needs Improvement)"
        assessment = "The fine-tuning needs work. The model hasn't learned the style well enough."
    else:
        grade = "F (Poor)"
        assessment = "The fine-tuning was not successful. Consider retraining with different parameters."
    
    return {
        "overall_score": weighted_score,
        "grade": grade,
        "assessment": assessment,
        "component_scores": {name: score for name, score, _ in scores},
    }


def print_interpretation(data: Dict, analyses: Dict, overall: Dict):
    """Print human-readable interpretation of results."""
    stats = data.get("statistics", {})
    
    print("=" * 80)
    print("EVALUATION INTERPRETATION")
    print("=" * 80)
    
    print(f"\n📊 OVERALL ASSESSMENT")
    print(f"   Grade: {overall['grade']}")
    print(f"   Score: {overall['overall_score']:.1%}")
    print(f"   {overall['assessment']}")
    
    print(f"\n{'='*80}")
    print(f"📏 RESPONSE LENGTH ANALYSIS")
    print(f"{'='*80}")
    rl_analysis = analyses.get("response_length", {})
    print(f"   Assessment: {rl_analysis.get('assessment', 'unknown').upper().replace('_', ' ')}")
    print(f"   Score: {rl_analysis.get('score', 0):.1%}")
    print(f"   Mean length: {rl_analysis.get('mean_length', 0):.0f} characters")
    print(f"   Median length: {rl_analysis.get('median_length', 0):.0f} characters")
    print(f"   Range: {rl_analysis.get('min_length', 0)} - {rl_analysis.get('max_length', 0)} characters")
    for note in rl_analysis.get("notes", []):
        print(f"   {note}")
    
    print(f"\n{'='*80}")
    print(f"✨ FORMATTING ANALYSIS")
    print(f"{'='*80}")
    fmt_analysis = analyses.get("formatting", {})
    print(f"   Assessment: {fmt_analysis.get('assessment', 'unknown').upper().replace('_', ' ')}")
    print(f"   Score: {fmt_analysis.get('score', 0):.1%}")
    print(f"   Mean formatting score: {fmt_analysis.get('mean_score', 0):.2f} / 1.0")
    print(f"   Median formatting score: {fmt_analysis.get('median_score', 0):.2f} / 1.0")
    
    print(f"\n   Formatting Element Compliance:")
    compliance = fmt_analysis.get("compliance", {})
    for element, info in compliance.items():
        status = info.get("status", "?")
        pct = info.get("percentage", 0)
        element_name = element.replace("_", " ").title()
        print(f"   {status} {element_name}: {pct:.1f}%")
    
    for note in fmt_analysis.get("notes", []):
        print(f"   {note}")
    
    if "ground_truth" in analyses and analyses["ground_truth"].get("available"):
        print(f"\n{'='*80}")
        print(f"🎯 GROUND TRUTH COMPARISON")
        print(f"{'='*80}")
        gt_analysis = analyses.get("ground_truth", {})
        print(f"   Assessment: {gt_analysis.get('assessment', 'unknown').upper().replace('_', ' ')}")
        print(f"   Score: {gt_analysis.get('score', 0):.1%}")
        print(f"   Mean length similarity: {gt_analysis.get('mean_similarity', 0):.2f}")
        print(f"   Median length similarity: {gt_analysis.get('median_similarity', 0):.2f}")
        for note in gt_analysis.get("notes", []):
            print(f"   {note}")
    
    print(f"\n{'='*80}")
    print(f"💡 RECOMMENDATIONS")
    print(f"{'='*80}")
    
    recommendations = []
    
    # Response length recommendations
    if rl_analysis.get("score", 0) < 0.6:
        recommendations.append("• Consider adjusting max_tokens or training to improve response length")
    
    # Formatting recommendations
    if fmt_analysis.get("score", 0) < 0.6:
        recommendations.append("• Model needs better formatting training - check training data quality")
        if compliance.get("has_bold", {}).get("percentage", 0) < 50:
            recommendations.append("• Increase examples with bold text in training data")
        if compliance.get("has_lists", {}).get("percentage", 0) < 50:
            recommendations.append("• Increase examples with lists in training data")
    
    # Ground truth recommendations
    if "ground_truth" in analyses and analyses["ground_truth"].get("available"):
        if analyses["ground_truth"].get("score", 0) < 0.5:
            recommendations.append("• Model responses differ significantly from training style - may need more epochs or better data")
    
    if not recommendations:
        recommendations.append("• Model is performing well! Consider testing on new questions to verify generalization.")
    
    for rec in recommendations:
        print(f"   {rec}")
    
    print("\n" + "=" * 80)


def main():
    """Main interpretation function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Interpret evaluation results with benchmarks and grades"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="evaluation_results.json",
        help="Path to evaluation results JSON file (default: evaluation_results.json)",
    )
    
    args = parser.parse_args()
    
    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"❌ Error: Results file not found: {results_path}")
        sys.exit(1)
    
    print(f"→ Loading evaluation results from: {results_path}")
    data = load_evaluation_results(results_path)
    
    stats = data.get("statistics", {})
    if not stats or "error" in stats:
        print(f"❌ Error: Invalid results file or no statistics available")
        sys.exit(1)
    
    # Analyze each component
    print(f"→ Analyzing results...")
    analyses = {
        "response_length": analyze_response_length(stats),
        "formatting": analyze_formatting(stats),
    }
    
    # Add ground truth comparison if available
    if "ground_truth_comparison" in stats:
        analyses["ground_truth"] = analyze_ground_truth_comparison(stats)
    
    # Calculate overall score
    overall = calculate_overall_score(analyses)
    
    # Print interpretation
    print_interpretation(data, analyses, overall)
    
    # Save interpretation
    interpretation = {
        "overall": overall,
        "analyses": analyses,
        "benchmarks": BENCHMARKS,
    }
    
    output_path = results_path.parent / f"{results_path.stem}_interpretation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(interpretation, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Detailed interpretation saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

















