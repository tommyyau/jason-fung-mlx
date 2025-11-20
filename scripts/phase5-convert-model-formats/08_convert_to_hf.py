#!/usr/bin/env python3
"""
Step 08 – Export to Hugging Face
────────────────────────────────
Transforms the fused MLX model (`models/jason_fung_mlx_fused`) into a Hugging Face-
compatible directory (`models/jason_fung_mlx_hf`), including config, tokenizer, and
properly transposed safetensors for downstream conversions.

NOTE: If you're on Apple Silicon Mac and using LM Studio, you can use the MLX model
directly without conversion! See docs/LM_STUDIO_MLX_GUIDE.md for exact steps.
This conversion is only needed for:
- Cross-platform compatibility (Windows/Linux)
- Converting to GGUF format
- Uploading to HuggingFace Hub
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

# ─────────────────────────────
# Configuration
# ─────────────────────────────
DEFAULT_MLX_MODEL = "models/jason_fung_mlx_fused"
DEFAULT_HF_OUTPUT = "models/jason_fung_mlx_hf"


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import mlx_lm
        print("✓ mlx-lm found")
    except ImportError:
        print("❌ Error: mlx-lm not found. Install with: pip install mlx-lm")
        sys.exit(1)

    try:
        import transformers
        print("✓ transformers found")
    except ImportError:
        print("⚠️  Warning: transformers not found. May be needed for some conversions.")
        print("   Install with: pip install transformers")

    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        print("✓ torch and safetensors found")
    except ImportError as e:
        print(f"❌ Error: Missing required library: {e}")
        print("   Install with: pip install torch safetensors")
        sys.exit(1)


def convert_mlx_to_huggingface(mlx_path: str, hf_path: str, base_model_name: str = None):
    """
    Convert MLX model to HuggingFace format.

    This properly converts MLX weights to HuggingFace format, handling:
    - Weight format differences (conv1d transposition)
    - Config file alignment
    - Tokenizer files

    Args:
        mlx_path: Path to MLX model directory
        hf_path: Output path for HuggingFace model
        base_model_name: Optional base model name for downloading structure
    """
    print(f"\n{'='*80}")
    print("Converting MLX Model to HuggingFace Format")
    print(f"{'='*80}")
    print(f"MLX model: {mlx_path}")
    print(f"HuggingFace output: {hf_path}")

    mlx_path_obj = Path(mlx_path)
    if not mlx_path_obj.exists():
        print(f"\n❌ Error: MLX model not found: {mlx_path}")
        print(f"   Please run fusion first: python3 newscripts/07_fuse_lora.py")
        sys.exit(1)

    # Create output directory
    hf_path_obj = Path(hf_path)
    hf_path_obj.mkdir(parents=True, exist_ok=True)

    # Import conversion utilities
    import json
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    # First, download base model structure if needed
    if base_model_name:
        try:
            from huggingface_hub import snapshot_download
            print(f"\nDownloading base model structure: {base_model_name}")
            snapshot_download(
                repo_id=base_model_name,
                local_dir=str(hf_path_obj),
                local_dir_use_symlinks=False,
                ignore_patterns=["*.safetensors", "*.bin"],  # We'll replace weights
            )
            print(f"✓ Base model structure downloaded")
        except Exception as e:
            print(f"⚠️  Could not download base model structure: {e}")
            print("   Will use MLX model structure instead")

    # Now do the actual weight conversion
    print(f"\nConverting MLX weights to HuggingFace format...")

    # Load MLX model index
    mlx_index = mlx_path_obj / "model.safetensors.index.json"
    if not mlx_index.exists():
        print(f"\n❌ Error: Index file not found: {mlx_index}")
        sys.exit(1)

    with open(mlx_index) as f:
        mlx_index_data = json.load(f)

    weight_map = mlx_index_data.get("weight_map", {})
    print(f"Found {len(weight_map)} weight tensors to convert")

    # Group weights by file
    files_to_process = {}
    for tensor_name, file_name in weight_map.items():
        if file_name not in files_to_process:
            files_to_process[file_name] = []
        files_to_process[file_name].append(tensor_name)

    # Process each file and convert weights
    all_tensors = {}
    for file_name, tensor_names in files_to_process.items():
        mlx_file = mlx_path_obj / file_name
        if not mlx_file.exists():
            print(f"⚠️  Warning: File not found: {mlx_file}")
            continue

        print(f"  Processing {file_name}...")
        with safe_open(mlx_file, framework="pt") as f:
            for tensor_name in tensor_names:
                if tensor_name not in f.keys():
                    continue

                tensor = f.get_tensor(tensor_name)
                original_shape = tensor.shape

                # Convert to numpy for manipulation
                if isinstance(tensor, torch.Tensor):
                    if tensor.dtype == torch.bfloat16:
                        tensor = tensor.float()
                    tensor_np = tensor.cpu().numpy()
                else:
                    tensor_np = np.array(tensor)

                # Transpose conv1d weights: MLX [out, in, kernel] -> HF [out, kernel, in]
                if "conv1d.weight" in tensor_name and len(tensor_np.shape) == 3:
                    print(f"    Transposing {tensor_name}: {original_shape} -> ", end="")
                    tensor_np = np.transpose(tensor_np, (0, 2, 1))
                    print(f"{tensor_np.shape}")

                # Convert back to torch tensor
                if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.bfloat16:
                    tensor_converted = torch.from_numpy(tensor_np).to(torch.bfloat16)
                else:
                    tensor_converted = torch.from_numpy(tensor_np)

                all_tensors[tensor_name] = tensor_converted

    # Save converted weights
    print(f"\nSaving converted weights...")
    total_size = sum(t.numel() * t.element_size() for t in all_tensors.values())
    chunk_size = 4 * 1024 * 1024 * 1024  # 4GB chunks
    current_chunk = 1
    current_size = 0
    current_tensors = {}
    hf_weight_map = {}
    num_chunks = max(len(files_to_process), 1)

    for tensor_name, tensor in all_tensors.items():
        tensor_size = tensor.numel() * tensor.element_size()

        if current_size + tensor_size > chunk_size and current_tensors:
            chunk_file = f"model-{current_chunk:05d}-of-{num_chunks:05d}.safetensors"
            chunk_path = hf_path_obj / chunk_file
            save_file(current_tensors, str(chunk_path))

            for name in current_tensors.keys():
                hf_weight_map[name] = chunk_file

            current_chunk += 1
            current_size = 0
            current_tensors = {}

        current_tensors[tensor_name] = tensor
        current_size += tensor_size

    # Save final chunk
    if current_tensors:
        chunk_file = f"model-{current_chunk:05d}-of-{num_chunks:05d}.safetensors"
        chunk_path = hf_path_obj / chunk_file
        save_file(current_tensors, str(chunk_path))

        for name in current_tensors.keys():
            hf_weight_map[name] = chunk_file

    # Save index file
    hf_index = {
        "metadata": {
            "total_size": total_size,
        },
        "weight_map": hf_weight_map,
    }

    hf_index_path = hf_path_obj / "model.safetensors.index.json"
    with open(hf_index_path, "w") as f:
        json.dump(hf_index, f, indent=2)

    # Copy config and tokenizer files from MLX model
    mlx_config = mlx_path_obj / "config.json"
    if mlx_config.exists():
        shutil.copy2(mlx_config, hf_path_obj / "config.json")
        print(f"  ✓ Copied config.json")

    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ]
    for tokenizer_file in tokenizer_files:
        mlx_tokenizer = mlx_path_obj / tokenizer_file
        if mlx_tokenizer.exists():
            shutil.copy2(mlx_tokenizer, hf_path_obj / tokenizer_file)

    print(f"\n{'='*80}")
    print("Conversion Complete!")
    print(f"{'='*80}")
    print(f"\n✓ MLX model converted to HuggingFace format: {hf_path}")
    print(f"\nThe model is now in proper HuggingFace format with correct weight shapes.")
    print(f"\nNext step: Convert to GGUF if needed (python3 newscripts/09_convert_to_gguf.py)")
    print("=" * 80)

    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Convert MLX model to HuggingFace format"
    )
    parser.add_argument(
        "--mlx-model",
        type=str,
        default=DEFAULT_MLX_MODEL,
        help=f"Path to MLX model directory (default: {DEFAULT_MLX_MODEL})",
    )
    parser.add_argument(
        "--hf-output",
        "--output-dir",
        type=str,
        default=DEFAULT_HF_OUTPUT,
        help=f"Output directory for HuggingFace model (default: {DEFAULT_HF_OUTPUT})",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="ibm-granite/granite-4.0-h-micro",
        help="Base HuggingFace model ID for structure (default: ibm-granite/granite-4.0-h-micro)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("MLX to HuggingFace Converter")
    print("=" * 80)

    # Check dependencies
    check_dependencies()

    # Resolve paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    mlx_path = project_root / args.mlx_model
    hf_path = project_root / args.hf_output

    # Convert
    convert_mlx_to_huggingface(
        mlx_path=str(mlx_path),
        hf_path=str(hf_path),
        base_model_name=args.base_model,
    )


if __name__ == "__main__":
    main()


