#!/usr/bin/env python3
"""
Step 06d – Generate Markdown Report from Comparison Results
───────────────────────────────────────────────────────────
Converts comparison_results JSON files into readable Markdown reports
with statistics, analysis, and detailed results.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# ─────────────────────────────
# Configuration
# ─────────────────────────────


def load_json_file(file_path: Path) -> Dict:
    """Load JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_statistics_section(stats: Dict) -> List[str]:
    """Format statistics section for markdown."""
    lines = []
    
    lines.append("## 📊 Overall Statistics")
    lines.append("")
    lines.append(f"- **Total Examples:** {stats.get('total_examples', 0)}")
    lines.append(f"- **Valid Examples:** {stats.get('valid_examples', 0)}")
    lines.append(f"- **Errors:** {stats.get('error_count', 0)}")
    lines.append("")
    
    # Response length
    if "response_length" in stats:
        rl = stats["response_length"]
        lines.append("### Response Length")
        lines.append("")
        lines.append(f"- **Mean:** {rl.get('mean', 0):.0f} characters")
        lines.append(f"- **Median:** {rl.get('median', 0):.0f} characters")
        lines.append(f"- **Range:** {rl.get('min', 0)} - {rl.get('max', 0)} characters")
        lines.append(f"- **Std Dev:** {rl.get('std', 0):.0f} characters")
        lines.append("")
    
    # Formatting score
    if "formatting_score" in stats:
        fs = stats["formatting_score"]
        lines.append("### Formatting Score (0-1)")
        lines.append("")
        lines.append(f"- **Mean:** {fs.get('mean', 0):.3f}")
        lines.append(f"- **Median:** {fs.get('median', 0):.3f}")
        lines.append(f"- **Range:** {fs.get('min', 0):.3f} - {fs.get('max', 0):.3f}")
        lines.append("")
    
    # Formatting compliance
    if "formatting_compliance" in stats:
        fc = stats["formatting_compliance"]
        lines.append("### Formatting Compliance")
        lines.append("")
        lines.append(f"- **Has Bold (`**`):** {fc.get('has_bold', 0):.1f}%")
        lines.append(f"- **Has Lists (`-`, `*`, `1.`):** {fc.get('has_lists', 0):.1f}%")
        lines.append(f"- **Has Paragraphs (`\\n\\n`):** {fc.get('has_paragraphs', 0):.1f}%")
        lines.append(f"- **Has Emphasis:** {fc.get('has_emphasis', 0):.1f}%")
        lines.append("")
    
    # Ground truth comparison
    if "ground_truth_comparison" in stats:
        gt = stats["ground_truth_comparison"]
        lines.append("### Ground Truth Comparison")
        lines.append("")
        lines.append(f"- **Mean Length Similarity:** {gt.get('mean_length_similarity', 0):.3f}")
        lines.append(f"- **Median Length Similarity:** {gt.get('median_length_similarity', 0):.3f}")
        lines.append("")
    
    return lines


def format_interpretation_section(interpretation: Dict) -> List[str]:
    """Format interpretation section for markdown."""
    lines = []
    
    overall = interpretation.get("overall", {})
    analyses = interpretation.get("analyses", {})
    
    lines.append("## 📈 Evaluation Interpretation")
    lines.append("")
    
    # Overall assessment
    lines.append("### Overall Assessment")
    lines.append("")
    lines.append(f"- **Grade:** {overall.get('grade', 'N/A')}")
    lines.append(f"- **Score:** {overall.get('overall_score', 0):.1%}")
    lines.append(f"- **Assessment:** {overall.get('assessment', 'N/A')}")
    lines.append("")
    
    # Component scores
    component_scores = overall.get("component_scores", {})
    if component_scores:
        lines.append("#### Component Scores")
        lines.append("")
        for component, score in component_scores.items():
            component_name = component.replace("_", " ").title()
            lines.append(f"- **{component_name}:** {score:.1%}")
        lines.append("")
    
    # Response length analysis
    if "response_length" in analyses:
        rl = analyses["response_length"]
        lines.append("### Response Length Analysis")
        lines.append("")
        lines.append(f"- **Assessment:** {rl.get('assessment', 'unknown').upper().replace('_', ' ')}")
        lines.append(f"- **Score:** {rl.get('score', 0):.1%}")
        lines.append(f"- **Mean Length:** {rl.get('mean_length', 0):.0f} characters")
        lines.append(f"- **Median Length:** {rl.get('median_length', 0):.0f} characters")
        lines.append(f"- **Range:** {rl.get('min_length', 0)} - {rl.get('max_length', 0)} characters")
        notes = rl.get("notes", [])
        if notes:
            lines.append("")
            lines.append("**Notes:**")
            for note in notes:
                lines.append(f"- {note}")
        lines.append("")
    
    # Formatting analysis
    if "formatting" in analyses:
        fmt = analyses["formatting"]
        lines.append("### Formatting Analysis")
        lines.append("")
        lines.append(f"- **Assessment:** {fmt.get('assessment', 'unknown').upper().replace('_', ' ')}")
        lines.append(f"- **Score:** {fmt.get('score', 0):.1%}")
        lines.append(f"- **Mean Formatting Score:** {fmt.get('mean_score', 0):.2f} / 1.0")
        lines.append(f"- **Median Formatting Score:** {fmt.get('median_score', 0):.2f} / 1.0")
        lines.append("")
        
        compliance = fmt.get("compliance", {})
        if compliance:
            lines.append("#### Formatting Element Compliance")
            lines.append("")
            for element, info in compliance.items():
                status = info.get("status", "?")
                pct = info.get("percentage", 0)
                element_name = element.replace("_", " ").title()
                meets_target = info.get("meets_target", False)
                target_indicator = "✓" if meets_target else "~"
                lines.append(f"- {target_indicator} **{element_name}:** {pct:.1f}%")
            lines.append("")
        
        notes = fmt.get("notes", [])
        if notes:
            lines.append("**Notes:**")
            for note in notes:
                lines.append(f"- {note}")
            lines.append("")
    
    # Ground truth comparison
    if "ground_truth" in analyses and analyses["ground_truth"].get("available"):
        gt = analyses["ground_truth"]
        lines.append("### Ground Truth Comparison")
        lines.append("")
        lines.append(f"- **Assessment:** {gt.get('assessment', 'unknown').upper().replace('_', ' ')}")
        lines.append(f"- **Score:** {gt.get('score', 0):.1%}")
        lines.append(f"- **Mean Length Similarity:** {gt.get('mean_similarity', 0):.2f}")
        lines.append(f"- **Median Length Similarity:** {gt.get('median_similarity', 0):.2f}")
        notes = gt.get("notes", [])
        if notes:
            lines.append("")
            lines.append("**Notes:**")
            for note in notes:
                lines.append(f"- {note}")
        lines.append("")
    
    return lines


def format_sample_results(results: List[Dict], max_samples: int = 10) -> List[str]:
    """Format sample results for markdown."""
    lines = []
    
    lines.append("## 📋 Sample Results")
    lines.append("")
    lines.append(f"Showing {min(max_samples, len(results))} sample results from the evaluation.")
    lines.append("")
    
    valid_results = [r for r in results if "error" not in r]
    samples = valid_results[:max_samples]
    
    for i, result in enumerate(samples, 1):
        lines.append(f"### Example {i}")
        lines.append("")
        lines.append(f"**Question:** {result.get('instruction', 'N/A')}")
        lines.append("")
        
        response = result.get("response", "")
        if len(response) > 500:
            response = response[:500] + "..."
        lines.append("**Response:**")
        lines.append("")
        lines.append("```")
        lines.append(response)
        lines.append("```")
        lines.append("")
        
        lines.append("**Metrics:**")
        lines.append(f"- Response Length: {result.get('response_length', 0)} characters")
        lines.append(f"- Formatting Score: {result.get('formatting_score', 0):.2f}")
        
        if "ground_truth_length" in result:
            lines.append(f"- Ground Truth Length: {result.get('ground_truth_length', 0)} characters")
            lines.append(f"- Length Similarity: {result.get('length_similarity', 0):.3f}")
        
        formatting_checks = result.get("formatting_checks", {})
        if formatting_checks:
            checks_str = ", ".join([k for k, v in formatting_checks.items() if v])
            if checks_str:
                lines.append(f"- Formatting Elements: {checks_str}")
        
        lines.append("")
        lines.append("---")
        lines.append("")
    
    return lines


def generate_markdown_report(
    comparison_data: Dict,
    interpretation_data: Optional[Dict] = None,
    max_samples: int = 10
) -> str:
    """Generate markdown report from comparison and interpretation data."""
    lines = []
    
    # Header
    lines.append("# Model Evaluation Report")
    lines.append("")
    lines.append(f"Generated from comparison results analysis.")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # File information
    lines.append("## 📁 File Information")
    lines.append("")
    if "responses_file" in comparison_data:
        lines.append(f"- **Responses File:** `{comparison_data['responses_file']}`")
    if "validation_file" in comparison_data:
        lines.append(f"- **Validation File:** `{comparison_data['validation_file']}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Statistics
    stats = comparison_data.get("statistics", {})
    if stats:
        lines.extend(format_statistics_section(stats))
        lines.append("---")
        lines.append("")
    
    # Interpretation (if available)
    if interpretation_data:
        lines.extend(format_interpretation_section(interpretation_data))
        lines.append("---")
        lines.append("")
    
    # Sample results
    results = comparison_data.get("results", [])
    if results:
        lines.extend(format_sample_results(results, max_samples))
    
    return "\n".join(lines)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate Markdown report from comparison results"
    )
    parser.add_argument(
        "--comparison-results",
        type=str,
        required=True,
        help="Path to comparison_results JSON file",
    )
    parser.add_argument(
        "--interpretation",
        type=str,
        default=None,
        help="Path to interpretation JSON file (optional, auto-detected if not provided)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output markdown file path (default: <comparison_results>_report.md)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum number of sample results to include (default: 10)",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Markdown Report Generator")
    print("=" * 80)
    
    # Get paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    comparison_path = project_root / args.comparison_results
    if not comparison_path.exists():
        comparison_path = Path(args.comparison_results)
        if not comparison_path.exists():
            print(f"❌ Error: Comparison results file not found: {args.comparison_results}")
            sys.exit(1)
    
    print(f"→ Loading comparison results from: {comparison_path}")
    comparison_data = load_json_file(comparison_path)
    
    # Try to find interpretation file
    interpretation_data = None
    if args.interpretation:
        interpretation_path = project_root / args.interpretation
        if not interpretation_path.exists():
            interpretation_path = Path(args.interpretation)
        if interpretation_path.exists():
            print(f"→ Loading interpretation from: {interpretation_path}")
            interpretation_data = load_json_file(interpretation_path)
        else:
            print(f"⚠️  Warning: Interpretation file not found: {args.interpretation}")
    else:
        # Auto-detect interpretation file
        interpretation_path = comparison_path.parent / f"{comparison_path.stem}_interpretation.json"
        if interpretation_path.exists():
            print(f"→ Auto-detected interpretation file: {interpretation_path}")
            interpretation_data = load_json_file(interpretation_path)
    
    # Generate markdown
    print(f"→ Generating markdown report...")
    markdown_content = generate_markdown_report(
        comparison_data,
        interpretation_data,
        max_samples=args.max_samples
    )
    
    # Determine output path
    if args.output:
        output_path = project_root / args.output
    else:
        output_path = comparison_path.parent / f"{comparison_path.stem}_report.md"
    
    # Write markdown file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    print(f"✓ Markdown report generated: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()






































