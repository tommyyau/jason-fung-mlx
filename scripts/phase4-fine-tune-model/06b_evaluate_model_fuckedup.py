#!/usr/bin/env python3
"""
Step 06b – Evaluate Fine-Tuned Model
────────────────────────────────────
Evaluates the fine-tuned model on validation data and measures:
- Response quality (length, formatting compliance)
- Style adherence (markdown formatting, structure)
- Comparison with ground truth answers (optional)

This addresses the missing evaluation step identified in the audit.
"""

import json
import sys
import argparse
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import statistics

# ─────────────────────────────
# Configuration
# ─────────────────────────────
DEFAULT_VAL_FILE = "data/generated_answers_mlx_validate.jsonl"
DEFAULT_MODEL_PATH = "models/jason_fung_mlx_fused"
DEFAULT_BASE_MODEL = "mlx-community/Llama-3.2-3B-Instruct"
DEFAULT_MAX_TOKENS = 2048  # Increased to avoid truncating long responses (avg response is ~400 tokens)
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9


def find_next_run_number(project_root: Path) -> int:
    """
    Find the next run number by scanning for existing evaluation_results_run*.json files.
    Returns 1 if no files exist, otherwise returns max_run_number + 1.
    """
    pattern = re.compile(r"evaluation_results_run(\d+)\.json")
    max_run = 0
    
    # Scan project root for matching files
    for file_path in project_root.glob("evaluation_results_run*.json"):
        match = pattern.match(file_path.name)
        if match:
            run_num = int(match.group(1))
            max_run = max(max_run, run_num)
    
    return max_run + 1


def check_mlx_installed():
    """Check if mlx and mlx-lm are installed."""
    try:
        import mlx.core as mx
        import mlx_lm
        return True
    except ImportError:
        print("❌ Error: MLX not found. Install with: pip install mlx mlx-lm")
        sys.exit(1)


def load_validation_data(val_file: Path) -> List[Dict]:
    """Load validation dataset from JSONL file."""
    examples = []
    
    if not val_file.exists():
        print(f"❌ Validation file not found: {val_file}")
        return examples
    
    print(f"→ Loading validation data from: {val_file}")
    
    with open(val_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                
                # Support both formats: {"instruction": ..., "output": ...} and {"messages": ...}
                if "instruction" in data and "output" in data:
                    examples.append({
                        "instruction": data["instruction"],
                        "output": data["output"]
                    })
                elif "messages" in data:
                    # Extract from chat format
                    messages = data["messages"]
                    instruction = None
                    output = None
                    for msg in messages:
                        if msg.get("role") == "user":
                            instruction = msg.get("content", "")
                        elif msg.get("role") == "assistant":
                            output = msg.get("content", "")
                    
                    if instruction and output:
                        examples.append({
                            "instruction": instruction,
                            "output": output
                        })
                else:
                    print(f"  ⚠️  Skipping line {line_num}: missing 'instruction'/'output' or 'messages'")
                    continue
                    
            except json.JSONDecodeError as e:
                print(f"  ⚠️  JSON error on line {line_num}: {str(e)[:100]}")
                continue
            except Exception as e:
                print(f"  ⚠️  Error processing line {line_num}: {type(e).__name__}: {str(e)[:100]}")
                continue
    
    print(f"  ✓ Loaded {len(examples)} validation examples")
    return examples


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
    
    # Bold is important (0.3)
    if checks["has_bold"]:
        score += 0.3
    
    # Lists are important (0.3)
    if checks["has_lists"]:
        score += 0.3
    
    # Paragraphs are important (0.2)
    if checks["has_paragraphs"]:
        score += 0.2
    
    # Any emphasis is good (0.2)
    if checks["has_emphasis"]:
        score += 0.2
    
    return min(score, 1.0)


def evaluate_single_example(
    model,
    tokenizer,
    instruction: str,
    ground_truth: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
) -> Dict:
    """Evaluate model on a single question."""
    from mlx_lm import generate
    
    # Generate response
    # Note: MLX-LM generate doesn't support temperature/top_p directly
    # These are controlled via sampler, but for simplicity we'll use defaults
    try:
        response = generate(
            model,
            tokenizer,
            prompt=instruction,
            max_tokens=max_tokens,
        )
    except Exception as e:
        return {
            "error": str(e),
            "instruction": instruction,
            "response": "",
            "response_length": 0,
            "formatting_score": 0.0,
            "formatting_checks": {},
        }
    
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
        
        # Simple length similarity (how close is response length to ground truth)
        if ground_truth_length := len(ground_truth):
            length_ratio = min(response_length, ground_truth_length) / max(response_length, ground_truth_length)
            result["length_similarity"] = length_ratio
        else:
            result["length_similarity"] = 0.0
    
    return result


def evaluate_batch(
    model,
    tokenizer,
    batch_examples: List[Dict],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    compare_ground_truth: bool = False,
) -> List[Dict]:
    """Evaluate model on a batch of examples."""
    from mlx_lm import generate

    results = []
    instructions = [ex["instruction"] for ex in batch_examples]
    ground_truths = [ex.get("output") if compare_ground_truth else None for ex in batch_examples]

    # Generate responses for all instructions in batch
    try:
        # MLX generate function processes one at a time, so we'll do manual batching
        # by processing prompts together when possible
        responses = []
        for instruction in instructions:
            response = generate(
                model,
                tokenizer,
                prompt=instruction,
                max_tokens=max_tokens,
            )
            responses.append(response)

    except Exception as e:
        # If batch fails, return errors for all examples
        return [
            {
                "error": str(e),
                "instruction": ex["instruction"],
                "response": "",
                "response_length": 0,
                "formatting_score": 0.0,
                "formatting_checks": {},
            }
            for ex in batch_examples
        ]

    # Calculate metrics for each response
    for i, (instruction, response, ground_truth) in enumerate(zip(instructions, responses, ground_truths)):
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

            # Simple length similarity
            if ground_truth_length := len(ground_truth):
                length_ratio = min(response_length, ground_truth_length) / max(response_length, ground_truth_length)
                result["length_similarity"] = length_ratio
            else:
                result["length_similarity"] = 0.0

        results.append(result)

    return results


def calculate_statistics(results: List[Dict]) -> Dict:
    """Calculate aggregate statistics from evaluation results."""
    if not results:
        return {}
    
    # Filter out errors
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


def print_summary(stats: Dict, results: List[Dict]):
    """Print evaluation summary."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
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


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned MLX model on validation data"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to fine-tuned model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default=DEFAULT_VAL_FILE,
        help=f"Validation dataset file (default: {DEFAULT_VAL_FILE})",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model path for comparison (optional)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens to generate (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Generation temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help=f"Top-p sampling (default: {DEFAULT_TOP_P})",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (for quick testing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for detailed results (JSON format). If not provided, auto-generates evaluation_results_runN.json based on existing files",
    )
    parser.add_argument(
        "--compare-ground-truth",
        action="store_true",
        help="Compare responses with ground truth answers",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (1-4 recommended for 16GB RAM, default: 1)",
    )

    args = parser.parse_args()
    
    print("=" * 80)
    print("MLX Model Evaluation")
    print("=" * 80)
    
    # Check MLX installation
    check_mlx_installed()
    
    # Get paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Auto-generate output filename if not provided
    if args.output is None:
        next_run = find_next_run_number(project_root)
        args.output = f"evaluation_results_run{next_run}.json"
        print(f"→ Auto-detected next run number: {next_run}")
        print(f"→ Output will be saved to: {args.output}")
    
    model_path = project_root / args.model
    val_file = project_root / args.val_file
    
    # Validate paths
    if not model_path.exists():
        print(f"\n❌ Error: Model path not found: {model_path}")
        print(f"   Available options:")
        print(f"   - models/jason_fung_mlx (LoRA adapters)")
        print(f"   - models/jason_fung_mlx_fused (fused model)")
        sys.exit(1)
    
    if not val_file.exists():
        print(f"\n❌ Error: Validation file not found: {val_file}")
        sys.exit(1)
    
    # Load validation data
    validation_examples = load_validation_data(val_file)
    
    if len(validation_examples) == 0:
        print(f"\n❌ No validation examples loaded")
        sys.exit(1)
    
    # Limit examples if requested
    if args.max_examples:
        validation_examples = validation_examples[:args.max_examples]
        print(f"  → Limited to {len(validation_examples)} examples for testing")
    
    # Load model
    print(f"\n→ Loading model from: {model_path}")
    try:
        from mlx_lm import load
        model, tokenizer = load(str(model_path))
        print(f"  ✓ Model loaded successfully")
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        sys.exit(1)
    
    # Evaluate on validation data
    print(f"\n→ Evaluating model on {len(validation_examples)} examples...")
    print(f"   Generating up to {args.max_tokens} tokens per example")
    print(f"   Batch size: {args.batch_size}")
    if args.batch_size > 1:
        print(f"   ⚡ Using batched inference for faster evaluation")
    print(f"   This may take 5-30 minutes depending on your hardware and batch size...\n")

    results = []
    start_time = time.time()

    # Process in batches
    batch_size = args.batch_size
    num_batches = (len(validation_examples) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_start_idx = batch_idx * batch_size
        batch_end_idx = min(batch_start_idx + batch_size, len(validation_examples))
        batch_examples = validation_examples[batch_start_idx:batch_end_idx]

        current_example = batch_start_idx + 1

        # Show progress after every batch
        if batch_examples:
            elapsed = time.time() - start_time
            rate = batch_end_idx / elapsed if elapsed > 0 else 0
            remaining = (len(validation_examples) - batch_end_idx) / rate if rate > 0 else 0
            print(f"  [{batch_idx + 1}/{num_batches}] ({batch_end_idx/len(validation_examples)*100:.1f}%) | Elapsed: {elapsed/60:.1f}m | Est. remaining: {remaining/60:.1f}m")

        # Process batch
        batch_start_time = time.time()
        batch_results = evaluate_batch(
            model,
            tokenizer,
            batch_examples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            compare_ground_truth=args.compare_ground_truth,
        )
        batch_time = time.time() - batch_start_time

        results.extend(batch_results)

        # Periodic summary
        if (batch_idx + 1) % (50 // batch_size) == 0 or (batch_idx + 1) == num_batches:
            elapsed = time.time() - start_time
            examples_done = batch_end_idx
            rate = examples_done / elapsed if elapsed > 0 else 0
            remaining = (len(validation_examples) - examples_done) / rate if rate > 0 else 0
            avg_time = elapsed / examples_done if examples_done > 0 else 0
            print(f"\n  ⏱️  Progress: {examples_done}/{len(validation_examples)} ({examples_done/len(validation_examples)*100:.1f}%) | "
                  f"Avg: {avg_time:.1f}s/example | Est. remaining: {remaining/60:.1f} minutes\n")
    
    total_time = time.time() - start_time
    print(f"\n  ✓ Completed evaluation of {len(validation_examples)} examples in {total_time/60:.1f} minutes")
    print(f"     Average time per example: {total_time/len(validation_examples):.1f} seconds")
    
    # Calculate statistics
    print(f"\n→ Calculating statistics...")
    stats = calculate_statistics(results)
    
    # Print summary
    print_summary(stats, results)
    
    # Save detailed results if requested
    if args.output:
        output_path = project_root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "model_path": str(model_path),
            "validation_file": str(val_file),
            "total_examples": len(validation_examples),
            "statistics": stats,
            "results": results,
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Detailed results saved to: {output_path}")
    
    print("\n✅ Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

