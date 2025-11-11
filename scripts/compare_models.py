#!/usr/bin/env python3
"""
Quick Model Comparison Tool

Compare base model vs fine-tuned model responses side-by-side.
Designed for quick catastrophic forgetting checks without running full evaluation.

Usage:
    python scripts/compare_models.py "What is insulin resistance?"
    python scripts/compare_models.py "Explain autophagy" --max-tokens 500 --temp 0.7
    python scripts/compare_models.py "What is 2+2?" --max-tokens 50

    # With correctness validation (requires OPENAI_API_KEY)
    python scripts/compare_models.py "Who wrote Romeo and Juliet?" --verify
    python scripts/compare_models.py "What is the capital of France?" --verify
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Default model paths
BASE_MODEL = "mlx-community/Llama-3.2-3B-Instruct"
FINETUNED_MODEL = "models/jason_fung_mlx_fused"

# ANSI color codes for terminal output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Load environment variables for OpenAI API (needed for --verify)
env_path = None
for path in [Path.cwd() / ".env", Path("data/.env"), Path(__file__).parent.parent / ".env"]:
    if path.exists():
        env_path = path
        break

if env_path:
    load_dotenv(dotenv_path=env_path)

# OpenAI client (initialized lazily when --verify is used)
openai_client = None


def get_openai_client():
    """Get or create OpenAI client."""
    global openai_client
    if openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(f"{RED}Error: OPENAI_API_KEY not found in environment.{RESET}")
            print("Verification requires OpenAI API key in .env file.")
            print("Add: OPENAI_API_KEY=your_key_here to .env")
            sys.exit(1)
        openai_client = OpenAI(api_key=api_key)
    return openai_client


def validate_answer(question, answer, model_name="unknown"):
    """
    Use OpenAI as a judge to validate if an answer is correct.

    Args:
        question: The question that was asked
        answer: The model's answer
        model_name: Name of the model (for context)

    Returns:
        dict: {
            "is_correct": bool,
            "correctness": "correct" | "incorrect" | "partial",
            "explanation": str,
            "correct_answer": str (if incorrect)
        }
    """
    client = get_openai_client()

    validation_prompt = f"""You are evaluating whether a language model's answer to a question is correct.

Question: {question}

Model's Answer: {answer}

Please evaluate this answer and respond with a JSON object containing:
1. "correctness": "correct", "incorrect", or "partial"
2. "explanation": A brief explanation of why the answer is correct/incorrect/partial
3. "correct_answer": The correct answer (only if the model's answer is incorrect or partial)

For factual questions (e.g., "What is 2+2?", "Who wrote Romeo and Juliet?"):
- "correct" if the answer contains the right fact, even with extra context
- "incorrect" if the answer is factually wrong
- "partial" if the answer is incomplete but not wrong

For explanatory questions (e.g., "Explain photosynthesis"):
- "correct" if the explanation is factually accurate and covers the main points
- "incorrect" if there are significant factual errors
- "partial" if the explanation is incomplete or has minor issues

Be strict but fair. Minor wording differences are fine as long as the core facts are correct.

Respond ONLY with valid JSON, no other text:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a factual accuracy evaluator. Respond only with valid JSON."},
                {"role": "user", "content": validation_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        # Normalize to boolean
        result["is_correct"] = result["correctness"] in ["correct", "partial"]

        return result

    except Exception as e:
        print(f"{YELLOW}Warning: Validation failed for {model_name}: {e}{RESET}")
        return {
            "is_correct": None,
            "correctness": "unknown",
            "explanation": f"Validation error: {e}",
            "correct_answer": None
        }


def generate_response(model_path, prompt, max_tokens=512, temp=0.1, top_p=0.9, top_k=None, seed=None):
    """
    Generate response using MLX CLI.

    Args:
        model_path: Path to the model
        prompt: The prompt to test
        max_tokens: Maximum tokens to generate
        temp: Temperature for sampling
        top_p: Top-p (nucleus) sampling parameter
        top_k: Top-k sampling (optional)
        seed: Random seed for reproducibility (optional)

    Returns:
        tuple: (response_text, generation_time)
    """
    cmd = [
        "python", "-m", "mlx_lm", "generate",
        "--model", model_path,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temp", str(temp),
        "--top-p", str(top_p),
    ]

    # Add optional parameters
    if top_k is not None:
        cmd.extend(["--top-k", str(top_k)])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        generation_time = time.time() - start_time

        # MLX output format is: "==========\n{prompt}\n{response}"
        # We want just the response part
        output = result.stdout

        # Split on the prompt to get just the response
        if prompt in output:
            response = output.split(prompt, 1)[1].strip()
        else:
            response = output.strip()

        # Remove any leading "==========" separators
        response = response.lstrip("=").strip()

        return response, generation_time

    except subprocess.CalledProcessError as e:
        print(f"Error generating response: {e}")
        print(f"stderr: {e.stderr}")
        return None, None


def count_words(text):
    """Count words in text."""
    return len(text.split())


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_response(model_name, response, gen_time, validation_result=None):
    """
    Print formatted response with stats and optional validation.

    Args:
        model_name: Name of the model
        response: The generated response
        gen_time: Generation time in seconds
        validation_result: Optional dict from validate_answer()
    """
    word_count = count_words(response)
    char_count = len(response)

    print(f"\n{model_name}")
    print_separator("-")

    # Show validation status if available
    if validation_result:
        correctness = validation_result.get("correctness", "unknown")
        if correctness == "correct":
            print(f"{GREEN}{BOLD}✓ CORRECT{RESET}")
        elif correctness == "incorrect":
            print(f"{RED}{BOLD}✗ INCORRECT{RESET}")
        elif correctness == "partial":
            print(f"{YELLOW}{BOLD}⚠ PARTIAL{RESET}")
        else:
            print(f"{YELLOW}{BOLD}? UNKNOWN{RESET}")

    print(f"Response ({word_count} words, {char_count} chars, {gen_time:.2f}s):")
    print_separator("-")
    print(response)
    print_separator("-")

    # Show validation details if available
    if validation_result and validation_result.get("correctness") != "correct":
        print(f"\n{BOLD}Validation Details:{RESET}")
        print(f"  Explanation: {validation_result.get('explanation', 'N/A')}")
        if validation_result.get("correct_answer"):
            print(f"  {GREEN}Correct answer: {validation_result['correct_answer']}{RESET}")
        print_separator("-")


def compare_models(prompt, max_tokens=512, temp=0.1, top_p=0.9, top_k=None, seed=None,
                   base_model=BASE_MODEL, finetuned_model=FINETUNED_MODEL, verify=False):
    """
    Compare base model and fine-tuned model on the same prompt.

    Args:
        prompt: The prompt to test
        max_tokens: Maximum tokens to generate
        temp: Temperature for sampling
        top_p: Top-p sampling parameter
        top_k: Top-k sampling (optional)
        seed: Random seed for reproducibility (optional)
        base_model: Path to base model
        finetuned_model: Path to fine-tuned model
        verify: Whether to validate answers for correctness (requires OpenAI API)
    """
    print_separator("=")
    print(f"PROMPT: {prompt}")
    print_separator("=")

    params_str = f"\nParameters: max_tokens={max_tokens}, temp={temp}, top_p={top_p}"
    if top_k is not None:
        params_str += f", top_k={top_k}"
    if seed is not None:
        params_str += f", seed={seed}"
    print(params_str)

    # Check if fine-tuned model exists
    finetuned_path = Path(finetuned_model)
    if not finetuned_path.exists():
        print(f"\nWarning: Fine-tuned model not found at {finetuned_model}")
        print("Have you run the training and fusion steps?")
        print("Run: python scripts/phase4-fine-tune-model/06_train_mlx.py")
        print("Then: python scripts/phase4-fine-tune-model/07_fuse_lora.py")
        return

    # Generate from base model
    print(f"\n\nGenerating from BASE MODEL ({base_model})...")
    base_response, base_time = generate_response(
        base_model, prompt, max_tokens, temp, top_p, top_k, seed
    )

    if base_response is None:
        print("Failed to generate from base model.")
        return

    # Generate from fine-tuned model
    print(f"\nGenerating from FINE-TUNED MODEL ({finetuned_model})...")
    ft_response, ft_time = generate_response(
        finetuned_model, prompt, max_tokens, temp, top_p, top_k, seed
    )

    if ft_response is None:
        print("Failed to generate from fine-tuned model.")
        return

    # Validate responses if requested
    base_validation = None
    ft_validation = None
    if verify:
        print(f"\n{BLUE}Validating responses with OpenAI...{RESET}")

        print("  Validating base model response...")
        base_validation = validate_answer(prompt, base_response, "base model")

        print("  Validating fine-tuned model response...")
        ft_validation = validate_answer(prompt, ft_response, "fine-tuned model")

    # Display results side-by-side
    print("\n\n")
    print_separator("=", 100)
    print("COMPARISON RESULTS")
    if verify:
        print(f"{BLUE}(with correctness validation){RESET}")
    print_separator("=", 100)

    print_response("BASE MODEL", base_response, base_time, base_validation)
    print_response("FINE-TUNED MODEL", ft_response, ft_time, ft_validation)

    # Print comparison stats
    print("\n")
    print_separator("=", 100)
    print("STATISTICS")
    print_separator("=", 100)

    base_words = count_words(base_response)
    ft_words = count_words(ft_response)
    base_chars = len(base_response)
    ft_chars = len(ft_response)

    print(f"{'Metric':<30} {'Base Model':<25} {'Fine-Tuned Model':<25} {'Difference':<20}")
    print_separator("-", 100)
    print(f"{'Word Count':<30} {base_words:<25} {ft_words:<25} {ft_words - base_words:+<20}")
    print(f"{'Character Count':<30} {base_chars:<25} {ft_chars:<25} {ft_chars - base_chars:+<20}")
    print(f"{'Generation Time (s)':<30} {base_time:<25.2f} {ft_time:<25.2f} {ft_time - base_time:+<20.2f}")
    print_separator("=", 100)

    # Observations
    print("\n")
    print_separator("=", 100)
    print("OBSERVATIONS")
    print_separator("=", 100)

    observations = []

    # Length comparison
    if abs(ft_words - base_words) < 10:
        observations.append("✓ Response lengths are similar (difference < 10 words)")
    elif ft_words > base_words * 1.5:
        observations.append("⚠ Fine-tuned response is significantly longer (>50% more words)")
    elif ft_words < base_words * 0.5:
        observations.append("⚠ Fine-tuned response is significantly shorter (<50% of base words)")

    # Check for formatting (markdown indicators)
    ft_has_bold = "**" in ft_response
    ft_has_lists = "\n-" in ft_response or "\n*" in ft_response or "\n1." in ft_response
    base_has_bold = "**" in base_response
    base_has_lists = "\n-" in base_response or "\n*" in base_response or "\n1." in base_response

    if ft_has_bold and not base_has_bold:
        observations.append("✓ Fine-tuned response uses bold formatting (expected from Jason Fung style)")

    if ft_has_lists and not base_has_lists:
        observations.append("✓ Fine-tuned response uses lists (expected from Jason Fung style)")

    if not ft_has_bold and not ft_has_lists and (base_has_bold or base_has_lists):
        observations.append("⚠ Base model has formatting but fine-tuned doesn't (unexpected)")

    # Speed comparison
    if ft_time < base_time * 0.9:
        observations.append(f"✓ Fine-tuned model is faster ({((1 - ft_time/base_time) * 100):.1f}% faster)")
    elif ft_time > base_time * 1.1:
        observations.append(f"⚠ Fine-tuned model is slower ({((ft_time/base_time - 1) * 100):.1f}% slower)")

    # Correctness validation observations
    if verify and base_validation and ft_validation:
        base_correctness = base_validation.get("correctness")
        ft_correctness = ft_validation.get("correctness")

        # More nuanced correctness reporting
        if base_correctness == "correct" and ft_correctness == "correct":
            observations.append(f"{GREEN}✓ Both models answered correctly{RESET}")
        elif base_correctness == "partial" and ft_correctness == "partial":
            observations.append(f"{YELLOW}⚠ Both models answered partially (some correct info, some inaccuracies){RESET}")
        elif base_correctness == "correct" and ft_correctness == "partial":
            observations.append(f"{YELLOW}⚠ Base model fully correct, fine-tuned model partially correct (minor degradation){RESET}")
        elif base_correctness == "partial" and ft_correctness == "correct":
            observations.append(f"{GREEN}✓ Fine-tuned model fully correct, base model partial (improvement!){RESET}")
        elif base_correctness in ["correct", "partial"] and ft_correctness == "incorrect":
            observations.append(f"{RED}✗ CATASTROPHIC FORGETTING: Base model {base_correctness}, fine-tuned model incorrect{RESET}")
        elif base_correctness == "incorrect" and ft_correctness in ["correct", "partial"]:
            observations.append(f"{GREEN}✓ Fine-tuned model {ft_correctness}, base model incorrect (improvement!){RESET}")
        elif base_correctness == "incorrect" and ft_correctness == "incorrect":
            observations.append(f"{RED}✗ Both models answered incorrectly (likely question beyond model knowledge){RESET}")
        else:
            # Catch-all for any other combinations
            observations.append(f"{YELLOW}⚠ Base: {base_correctness}, Fine-tuned: {ft_correctness}{RESET}")

    for obs in observations:
        print(f"  {obs}")

    if not observations:
        print("  No significant differences observed")

    print_separator("=", 100)


def main():
    parser = argparse.ArgumentParser(
        description="Compare base model vs fine-tuned model responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python scripts/compare_models.py "What is insulin resistance?"

  # Test with longer responses
  python scripts/compare_models.py "Explain the detailed mechanism of autophagy" --max-tokens 1000

  # Test with short responses (check if model can be concise)
  python scripts/compare_models.py "What is 2+2?" --max-tokens 50

  # Test general knowledge (catastrophic forgetting check)
  python scripts/compare_models.py "Who wrote Romeo and Juliet?"

  # Verify correctness (requires OPENAI_API_KEY in .env)
  python scripts/compare_models.py "Who wrote Romeo and Juliet?" --verify
  python scripts/compare_models.py "What is 2+2?" --verify
  python scripts/compare_models.py "What is the capital of France?" --verify

  # Adjust temperature for creativity
  python scripts/compare_models.py "What is intermittent fasting?" --temp 0.3

  # Custom models
  python scripts/compare_models.py "Test prompt" --base mlx-community/Llama-3.2-3B-Instruct --ft models/jason_fung_mlx
        """
    )

    parser.add_argument("prompt", type=str, help="The prompt to test on both models")
    parser.add_argument("--max-tokens", type=int, default=512,
                       help="Maximum tokens to generate (default: 512)")
    parser.add_argument("--temp", type=float, default=0.1,
                       help="Temperature for sampling (default: 0.1, very low for factual accuracy)")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Top-p (nucleus) sampling (default: 0.9)")
    parser.add_argument("--top-k", type=int, default=None,
                       help="Top-k sampling - only sample from top K tokens (optional)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility (optional)")
    parser.add_argument("--base", type=str, default=BASE_MODEL,
                       help=f"Base model path (default: {BASE_MODEL})")
    parser.add_argument("--ft", type=str, default=FINETUNED_MODEL,
                       help=f"Fine-tuned model path (default: {FINETUNED_MODEL})")
    parser.add_argument("--verify", action="store_true",
                       help="Validate answers for correctness using OpenAI (requires OPENAI_API_KEY)")

    args = parser.parse_args()

    compare_models(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temp=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        base_model=args.base,
        finetuned_model=args.ft,
        verify=args.verify
    )


if __name__ == "__main__":
    main()
