#!/usr/bin/env python3
"""
Step 09 – Convert to GGUF
─────────────────────────
Uses llama.cpp tooling to transform the Hugging Face export (`models/jason_fung_mlx_hf`)
into GGUF binaries for deployment (e.g., `models/jason-fung-granite-Q4_K_M.gguf`).
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# ─────────────────────────────
# Configuration
# ─────────────────────────────
DEFAULT_HF_MODEL = "models/jason_fung_mlx_hf"
DEFAULT_OUTPUT = "models/jason-fung-granite"
DEFAULT_QUANTIZATION = "Q4_K_M"  # Matches your existing GGUF file


def check_llama_cpp():
    """Check if llama.cpp conversion tools are available."""
    # Check for llama.cpp in common locations
    possible_paths = [
        Path("llama.cpp"),
        Path.home() / "llama.cpp",
        Path("/usr/local/llama.cpp"),
    ]

    convert_script = None
    for path in possible_paths:
        # Try both naming conventions
        script_path = path / "convert_hf_to_gguf.py"
        if not script_path.exists():
            script_path = path / "convert-hf-to-gguf.py"
        if script_path.exists():
            convert_script = script_path
            print(f"✓ Found llama.cpp at: {path}")
            return str(convert_script), str(path)

    # Check if installed via pip
    try:
        import llama_cpp
        print("✓ llama-cpp-python found (may support conversion)")
        return "llama_cpp", None
    except ImportError:
        pass

    print("⚠️  llama.cpp not found in standard locations")
    return None, None


def prepare_model_for_gguf(hf_model_path: str):
    """
    Prepare HuggingFace model for GGUF conversion.

    GGUF format expects embedding weights in [hidden_size, vocab_size] format,
    but HuggingFace uses [vocab_size, hidden_size]. llama.cpp should handle this,
    but we'll verify the model is in the correct format.
    """
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        import json
        import numpy as np

        hf_path_obj = Path(hf_model_path)

        # Check if embedding weight needs transposing
        index_file = hf_path_obj / "model.safetensors.index.json"
        if not index_file.exists():
            return  # No index file, can't check

        with open(index_file) as f:
            index_data = json.load(f)

        weight_map = index_data.get("weight_map", {})
        embed_key = "model.embed_tokens.weight"

        if embed_key not in weight_map:
            return  # No embedding weight found

        # Check the current shape
        embed_file = hf_path_obj / weight_map[embed_key]
        with safe_open(embed_file, framework="pt") as f:
            if embed_key in f.keys():
                weight = f.get_tensor(embed_key)
                if len(weight.shape) == 2 and weight.shape[0] > weight.shape[1]:
                    print(f"  ✓ Embedding weight is in HuggingFace format: {weight.shape}")
                    print(f"    llama.cpp will transpose to GGUF format during conversion")

    except Exception as e:
        print(f"⚠️  Warning: Could not verify embedding weight format: {e}")
        print("   Conversion will proceed, llama.cpp should handle format conversion")


def convert_to_gguf(
    hf_model_path: str,
    output_path: str,
    quantization: str = "Q4_K_M"
):
    """
    Convert HuggingFace model to GGUF format.

    Args:
        hf_model_path: Path to HuggingFace model directory
        output_path: Output path for GGUF file (without .gguf extension)
        quantization: Quantization type (Q4_K_M, Q8_0, F16, etc.)
    """
    print(f"\n{'='*80}")
    print("Converting HuggingFace Model to GGUF Format")
    print(f"{'='*80}")
    print(f"HuggingFace model: {hf_model_path}")
    print(f"Output: {output_path}.gguf")
    print(f"Quantization: {quantization}")

    hf_path_obj = Path(hf_model_path)
    if not hf_path_obj.exists():
        print(f"\n❌ Error: HuggingFace model not found: {hf_model_path}")
        print(f"   Please run conversion first: python3 newscripts/08_convert_to_hf.py")
        sys.exit(1)

    # Prepare model for GGUF conversion
    print(f"\nPreparing model for GGUF conversion...")
    prepare_model_for_gguf(hf_model_path=str(hf_path_obj))

    # Check for llama.cpp
    convert_script, llama_cpp_path = check_llama_cpp()

    if not convert_script:
        print("\n" + "=" * 80)
        print("llama.cpp Not Found")
        print("=" * 80)
        print("\nTo convert to GGUF, you need llama.cpp:")
        print("\nOption 1: Install llama.cpp")
        print("  git clone https://github.com/ggerganov/llama.cpp.git")
        print("  cd llama.cpp")
        print("  make")
        print("\nOption 2: Use LM Studio")
        print("  1. Open LM Studio")
        print("  2. Go to Settings > Models")
        print("  3. Click 'Convert Model'")
        print(f"  4. Select: {hf_model_path}")
        print(f"  5. Choose quantization: {quantization}")
        print("\nOption 3: Use online converter")
        print("  Some online services can convert HuggingFace to GGUF")
        print("=" * 80)
        sys.exit(1)

    # Build output path
    output_file = f"{output_path}-{quantization}.gguf"
    output_path_obj = Path(output_file)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Convert using llama.cpp
    if convert_script == "llama_cpp":
        print("\n⚠️  llama-cpp-python may not support direct conversion")
        print("   Recommend using llama.cpp convert script instead")
        sys.exit(1)

    # Use llama.cpp convert script
    print(f"\nUsing llama.cpp conversion script: {convert_script}")

    # First convert to F16 (unquantized)
    f16_output = output_file.replace(f"-{quantization}.gguf", "-F16.gguf")

    cmd_convert = [
        sys.executable,
        str(convert_script),
        str(hf_path_obj),
        "--outfile",
        f16_output,
        "--outtype",
        "f16",
    ]

    print(f"\nStep 1: Converting to GGUF (F16)...")
    print(f"Executing: {' '.join(cmd_convert)}\n")

    try:
        result = subprocess.run(cmd_convert, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print("✓ Converted to GGUF (F16)")

        # Now quantize if needed
        if quantization.upper() != "F16":
            print(f"\nStep 2: Quantizing to {quantization}...")

            # Check for quantize binary in build/bin or bin directory
            llama_cpp_base = Path(convert_script).parent
            quantize_script = llama_cpp_base / "build" / "bin" / "llama-quantize"
            if not quantize_script.exists():
                quantize_script = llama_cpp_base / "bin" / "llama-quantize"
            if not quantize_script.exists():
                # Try old location
                quantize_script = llama_cpp_base / "quantize"
            if not quantize_script.exists():
                # Try python quantize script
                quantize_script = llama_cpp_base / "quantize.py"

            if quantize_script.exists():
                # Check if it's a binary or Python script
                if quantize_script.suffix == ".py":
                    cmd_quantize = [
                        sys.executable,
                        str(quantize_script),
                        f16_output,
                        output_file,
                        quantization.upper(),  # Use uppercase for quantize
                    ]
                else:
                    # Binary quantize tool
                    cmd_quantize = [
                        str(quantize_script),
                        f16_output,
                        output_file,
                        quantization.upper(),
                    ]

                print(f"Executing: {' '.join(cmd_quantize)}\n")
                result = subprocess.run(cmd_quantize, check=True, capture_output=True, text=True)
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr)
                print(f"✓ Quantized to {quantization}")

            else:
                print(f"⚠️  Quantize script not found. Keeping F16 version.")
                print(f"   You can quantize manually or use the F16 file.")
                output_file = f16_output
        else:
            output_file = f16_output

        print(f"\n{'='*80}")
        print("Conversion Complete!")
        print(f"{'='*80}")
        print(f"\n✓ GGUF model saved to: {output_file}")
        print(f"\nFile size: {Path(output_file).stat().st_size / (1024**3):.2f} GB")
        print(f"\nYou can now use this model with:")
        print(f"  - LM Studio")
        print(f"  - llama.cpp")
        print(f"  - Ollama (after conversion)")
        print("=" * 80)

        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: Conversion failed")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        print(f"\nYou can try running manually:")
        print(f"  {' '.join(cmd_convert)}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n❌ Error: llama.cpp convert script not found")
        print(f"   Please install llama.cpp or use alternative conversion method")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace model to GGUF format"
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default=DEFAULT_HF_MODEL,
        help=f"HuggingFace model directory (default: {DEFAULT_HF_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output file path without extension (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=DEFAULT_QUANTIZATION,
        choices=["Q4_K_M", "Q4_0", "Q8_0", "F16", "Q5_K_M", "Q6_K"],
        help=f"Quantization type (default: {DEFAULT_QUANTIZATION})",
    )
    parser.add_argument(
        "--llama-cpp-path",
        type=str,
        help="Path to llama.cpp directory (auto-detected if not specified)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("HuggingFace to GGUF Converter")
    print("=" * 80)

    # Resolve paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    hf_model = project_root / args.hf_model
    output = project_root / args.output

    # Convert
    convert_to_gguf(
        hf_model_path=str(hf_model),
        output_path=str(output),
        quantization=args.quantization,
    )


if __name__ == "__main__":
    main()


