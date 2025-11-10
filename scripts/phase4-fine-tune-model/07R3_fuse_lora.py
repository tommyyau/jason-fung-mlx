#!/usr/bin/env python3
"""
Step 07 – Fuse LoRA Adapters
────────────────────────────
Combines the LoRA adapters produced during training with the original base model,
producing a fully fused MLX model under `models/jason_fung_mlx_fused`.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# ─────────────────────────────
# Configuration
# ─────────────────────────────
DEFAULT_ADAPTER_PATH = "models/granite-4.0-h-tiny-run1"
DEFAULT_BASE_MODEL = "lmstudio-community/granite-4.0-h-tiny-MLX-4bit"  # Match the model used for training
DEFAULT_OUTPUT_DIR = "models/granite-4.0-h-tiny-run1-fused"


def check_mlx_installed():
    """Check if mlx-lm is installed."""
    try:
        import mlx_lm
        print("✓ mlx-lm found")
        return True
    except ImportError:
        print("❌ Error: mlx-lm not found. Install with: pip install mlx-lm")
        sys.exit(1)


def find_adapter_path(adapter_path: str) -> str:
    """Find the actual adapter path (may be in a checkpoint subdirectory)."""
    adapter_path = Path(adapter_path)

    if not adapter_path.exists():
        return str(adapter_path)

    # Check for checkpoint directories
    checkpoint_dirs = [
        d for d in adapter_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")
    ]

    if checkpoint_dirs:
        # Use the latest checkpoint
        latest = max(
            checkpoint_dirs,
            key=lambda x: int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0,
        )
        print(f"  Found checkpoint: {latest.name}")
        return str(latest)

    return str(adapter_path)


def fuse_lora_adapters(
    base_model: str,
    adapter_path: str,
    output_dir: str,
    dequantize: bool = True,
    export_gguf: bool = False,
    gguf_path: str = None,
):
    """
    Fuse LoRA adapters with base model using mlx_lm.fuse.

    Args:
        base_model: Path to base model or HuggingFace model ID
        adapter_path: Path to LoRA adapter directory
        output_dir: Output directory for fused model
        dequantize: Whether to dequantize the model (default: True)
    """
    print(f"\n{'='*80}")
    print("Fusing LoRA Adapters with Base Model")
    print(f"{'='*80}")
    print(f"Base model: {base_model}")
    print(f"Adapter path: {adapter_path}")
    print(f"Output directory: {output_dir}")
    print(f"Dequantize: {dequantize}")

    # Check if adapter exists
    adapter_path_obj = Path(adapter_path)
    if not adapter_path_obj.exists():
        print(f"\n❌ Error: Adapter path not found: {adapter_path}")
        print(f"   Please run training first: python3 newscripts/06_train_mlx.py")
        sys.exit(1)

    # Find actual adapter path (may be in checkpoint)
    actual_adapter_path = find_adapter_path(adapter_path)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build fuse command
    cmd = [
        "python",
        "-m",
        "mlx_lm",
        "fuse",
        "--model",
        base_model,
        "--adapter-path",
        actual_adapter_path,
        "--save-path",
        str(output_path),
    ]

    if dequantize:
        cmd.append("--de-quantize")

    # Add GGUF export if requested
    if export_gguf:
        cmd.append("--export-gguf")
        if gguf_path:
            cmd.extend(["--gguf-path", gguf_path])

    print(f"\nExecuting: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)

        print(f"\n{'='*80}")
        print("Fusion Complete!")
        print(f"{'='*80}")
        print(f"\n✓ Fused model saved to: {output_dir}")
        print(f"\nNext step:")
        print(f"  Convert to HuggingFace: python3 newscripts/08_convert_to_hf.py")
        print(f"{'='*80}")

        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: Fusion failed with exit code {e.returncode}")
        print(f"\nYou can try running manually:")
        print(f"  {' '.join(cmd)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during fusion: {e}")
        sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Fuse LoRA adapters with base model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base model path or HuggingFace ID (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=DEFAULT_ADAPTER_PATH,
        help=f"Path to LoRA adapter directory (default: {DEFAULT_ADAPTER_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for fused model (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-dequantize",
        dest="dequantize",
        action="store_false",
        help="Don't dequantize the model (keep quantized if base model is quantized)",
    )
    parser.add_argument(
        "--export-gguf",
        action="store_true",
        help="Export GGUF format directly from fused model",
    )
    parser.add_argument(
        "--gguf-path",
        type=str,
        help="Path for GGUF export (default: models/jason-fung-granite-F16.gguf)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("MLX LoRA Adapter Fusion")
    print("=" * 80)

    # Check MLX installation
    check_mlx_installed()

    # Resolve paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    adapter_path = project_root / args.adapter_path
    output_dir = project_root / args.output_dir

    # Fuse adapters
    gguf_path = args.gguf_path if hasattr(args, "gguf_path") and args.gguf_path else None
    fuse_lora_adapters(
        base_model=args.base_model,
        adapter_path=str(adapter_path),
        output_dir=str(output_dir),
        dequantize=args.dequantize,
        export_gguf=args.export_gguf if hasattr(args, "export_gguf") else False,
        gguf_path=gguf_path,
    )


if __name__ == "__main__":
    main()


