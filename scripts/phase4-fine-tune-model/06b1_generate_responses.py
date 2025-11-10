#!/usr/bin/env python3
"""
Step 06b1 – Generate Model Responses
────────────────────────────────────
Generates responses from the fine-tuned model for all validation questions
and saves them to a JSONL file for later comparison.

This allows you to:
- Generate responses once, compare many times
- Resume if generation fails partway
- Compare multiple models against the same ground truth
"""

import json
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

# ─────────────────────────────
# Configuration
# ─────────────────────────────
DEFAULT_VAL_FILE = "data/generated_answers_mlx_validate.jsonl"
DEFAULT_MODEL_PATH = "models/jason_fung_mlx_fused"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_OUTPUT_FILE = "data/model_responses.jsonl"
DEFAULT_BATCH_SIZE = 1


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
                        "output": data["output"]  # Keep ground truth for reference
                    })
                elif "messages" in data:
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


def generate_response(model, tokenizer, instruction: str, max_tokens: int) -> Optional[str]:
    """Generate a single response from the model."""
    from mlx_lm import generate
    
    try:
        response = generate(
            model,
            tokenizer,
            prompt=instruction,
            max_tokens=max_tokens,
        )
        return response
    except Exception as e:
        print(f"  ❌ Error generating response: {e}")
        return None


def save_responses(responses: List[Dict], output_file: Path, append: bool = False):
    """Save responses to JSONL file."""
    mode = "a" if append else "w"
    
    with open(output_file, mode, encoding="utf-8") as f:
        for response in responses:
            f.write(json.dumps(response, ensure_ascii=False) + "\n")


def load_existing_responses(output_file: Path) -> Dict[str, Dict]:
    """Load existing responses to avoid regenerating."""
    existing = {}
    
    if not output_file.exists():
        return existing
    
    print(f"→ Loading existing responses from: {output_file}")
    
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                instruction = data.get("instruction", "")
                if instruction:
                    existing[instruction] = data
            except json.JSONDecodeError:
                continue
    
    print(f"  ✓ Found {len(existing)} existing responses")
    return existing


def main():
    """Main generation function."""
    parser = argparse.ArgumentParser(
        description="Generate model responses for validation questions"
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
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output file for responses (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens to generate (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to generate (for testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file (skip already generated responses)",
    )

    args = parser.parse_args()
    
    print("=" * 80)
    print("Model Response Generation")
    print("=" * 80)
    
    # Check MLX installation
    check_mlx_installed()
    
    # Get paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    model_path = project_root / args.model
    val_file = project_root / args.val_file
    output_file = project_root / args.output
    
    # Validate paths
    if not model_path.exists():
        print(f"\n❌ Error: Model path not found: {model_path}")
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
    
    # Load existing responses if resuming
    existing_responses = {}
    if args.resume and output_file.exists():
        existing_responses = load_existing_responses(output_file)
    
    # Load model
    print(f"\n→ Loading model from: {model_path}")
    try:
        from mlx_lm import load
        model, tokenizer = load(str(model_path))
        print(f"  ✓ Model loaded successfully")
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        sys.exit(1)
    
    # Filter out already generated if resuming
    if args.resume:
        remaining_examples = [
            ex for ex in validation_examples
            if ex["instruction"] not in existing_responses
        ]
        print(f"  → {len(remaining_examples)} examples remaining to generate (out of {len(validation_examples)} total)")
    else:
        remaining_examples = validation_examples
        # Clear output file if not resuming
        if output_file.exists():
            output_file.unlink()
    
    if len(remaining_examples) == 0:
        print("  ✓ All responses already generated!")
        sys.exit(0)
    
    # Generate responses
    total_examples = len(validation_examples)
    print(f"\n→ Generating responses for {len(remaining_examples)} examples...")
    print(f"   Total questions in dataset: {total_examples}")
    print(f"   Max tokens per response: {args.max_tokens}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Output file: {output_file}")
    print(f"   This may take 30 minutes to 2+ hours depending on your hardware...\n")
    
    start_time = time.time()
    all_responses = []
    
    # Process examples one by one with progress indicator
    for idx, example in enumerate(remaining_examples, 1):
        instruction = example["instruction"]
        ground_truth = example.get("output", "")
        
        # Show progress before generating
        question_num = len(existing_responses) + idx
        print(f"[{question_num}/{total_examples}] Generating response {idx}/{len(remaining_examples)}...")
        print(f"  Question: {instruction[:80]}{'...' if len(instruction) > 80 else ''}")
        
        # Generate response
        gen_start = time.time()
        response = generate_response(model, tokenizer, instruction, args.max_tokens)
        gen_time = time.time() - gen_start
        
        if response is None:
            print(f"  ❌ Failed to generate response\n")
            continue
        
        # Save response with metadata
        response_data = {
            "instruction": instruction,
            "response": response,
            "ground_truth": ground_truth,  # Keep for reference
            "response_length": len(response),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        all_responses.append(response_data)
        
        # Save immediately after each response (incremental saves)
        save_responses([response_data], output_file, append=True)
        
        # Show progress after generating
        elapsed_total = time.time() - start_time
        avg_time = elapsed_total / idx if idx > 0 else 0
        remaining_count = len(remaining_examples) - idx
        estimated_remaining = avg_time * remaining_count if avg_time > 0 else 0
        
        print(f"  ✓ Generated ({gen_time:.1f}s) | Response length: {len(response)} chars")
        print(f"  Progress: {question_num}/{total_examples} ({question_num/total_examples*100:.1f}%) | "
              f"Elapsed: {elapsed_total/60:.1f}m | Remaining: ~{estimated_remaining/60:.1f}m | "
              f"Avg: {avg_time:.1f}s/response")
        print()  # Blank line for readability
    
    total_time = time.time() - start_time
    print("=" * 80)
    print(f"✓ Generated {len(all_responses)} responses in {total_time/60:.1f} minutes")
    print(f"  Average time per response: {total_time/len(all_responses):.1f} seconds")
    print(f"  Saved to: {output_file}")
    print("=" * 80)
    
    print(f"\nNext step: Run comparison script:")
    print(f"  python3 scripts/phase4-fine-tune-model/06b2_compare_responses.py \\")
    print(f"    --responses {args.output} \\")
    print(f"    --val-file {args.val_file}")


if __name__ == "__main__":
    main()














