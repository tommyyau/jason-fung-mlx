#!/usr/bin/env python3
"""
Compare Training Runs
─────────────────────
Extracts and compares training metrics across multiple training runs.
Supports parsing from log files, manual input, or adapter configs.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

def load_adapter_config(run_path: Path) -> Optional[Dict]:
    """Load adapter_config.json from a run directory."""
    config_file = run_path / "adapter_config.json"
    if not config_file.exists():
        return None
    
    with open(config_file, 'r') as f:
        return json.load(f)

def parse_training_output(text: str) -> Dict:
    """
    Parse MLX training output to extract metrics.
    
    Expected format examples:
    - "Iter 1800: Val loss 1.629, Val took 28.928s"
    - "Iter 1800: Train loss 1.443, Learning Rate 1.000e-05, It/sec 0.417, Tokens/sec 180.235, Trained Tokens 751673, Peak mem 8.082 GB"
    """
    metrics = {
        'iterations': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'tokens_per_sec': [],
        'iterations_per_sec': [],
        'trained_tokens': [],
        'peak_memory_gb': [],
        'val_time_sec': []
    }
    
    # Pattern for validation loss lines
    val_pattern = r"Iter (\d+):\s*Val loss ([\d.]+)(?:,\s*Val took ([\d.]+)s)?"
    for match in re.finditer(val_pattern, text):
        iter_num = int(match.group(1))
        val_loss = float(match.group(2))
        val_time = float(match.group(3)) if match.group(3) else None
        
        metrics['iterations'].append(iter_num)
        metrics['val_loss'].append(val_loss)
        if val_time:
            metrics['val_time_sec'].append(val_time)
    
    # Pattern for training loss lines with full metrics
    train_pattern = r"Iter (\d+):\s*Train loss ([\d.]+)(?:,\s*Learning Rate ([\d.e-]+))?(?:,\s*It/sec ([\d.]+))?(?:,\s*Tokens/sec ([\d.]+))?(?:,\s*Trained Tokens ([\d]+))?(?:,\s*Peak mem ([\d.]+) GB)?"
    for match in re.finditer(train_pattern, text):
        iter_num = int(match.group(1))
        train_loss = float(match.group(2))
        lr = float(match.group(3)) if match.group(3) else None
        it_per_sec = float(match.group(4)) if match.group(4) else None
        tokens_per_sec = float(match.group(5)) if match.group(5) else None
        trained_tokens = int(match.group(6)) if match.group(6) else None
        peak_mem = float(match.group(7)) if match.group(7) else None
        
        if iter_num not in metrics['iterations']:
            metrics['iterations'].append(iter_num)
            metrics['train_loss'].append(train_loss)
        else:
            # Update existing entry
            idx = metrics['iterations'].index(iter_num)
            metrics['train_loss'][idx] = train_loss
        
        if lr:
            if len(metrics['learning_rate']) <= len(metrics['iterations']) - 1:
                metrics['learning_rate'].append(lr)
        if it_per_sec:
            if len(metrics['iterations_per_sec']) <= len(metrics['iterations']) - 1:
                metrics['iterations_per_sec'].append(it_per_sec)
        if tokens_per_sec:
            if len(metrics['tokens_per_sec']) <= len(metrics['iterations']) - 1:
                metrics['tokens_per_sec'].append(tokens_per_sec)
        if trained_tokens:
            if len(metrics['trained_tokens']) <= len(metrics['iterations']) - 1:
                metrics['trained_tokens'].append(trained_tokens)
        if peak_mem:
            if len(metrics['peak_memory_gb']) <= len(metrics['iterations']) - 1:
                metrics['peak_memory_gb'].append(peak_mem)
    
    return metrics

def get_final_metrics(parsed: Dict) -> Dict:
    """Extract final iteration metrics."""
    if not parsed['iterations']:
        return {}
    
    final_idx = -1
    return {
        'final_iteration': parsed['iterations'][final_idx],
        'final_train_loss': parsed['train_loss'][final_idx] if parsed['train_loss'] else None,
        'final_val_loss': parsed['val_loss'][final_idx] if parsed['val_loss'] else None,
        'final_learning_rate': parsed['learning_rate'][final_idx] if parsed['learning_rate'] else None,
        'final_tokens_per_sec': parsed['tokens_per_sec'][final_idx] if parsed['tokens_per_sec'] else None,
        'final_iterations_per_sec': parsed['iterations_per_sec'][final_idx] if parsed['iterations_per_sec'] else None,
        'total_trained_tokens': parsed['trained_tokens'][final_idx] if parsed['trained_tokens'] else None,
        'peak_memory_gb': parsed['peak_memory_gb'][final_idx] if parsed['peak_memory_gb'] else None,
        'final_val_time_sec': parsed['val_time_sec'][final_idx] if parsed['val_time_sec'] else None,
    }

def collect_run_data(run_name: str, run_path: Path, log_file: Optional[Path] = None) -> Dict:
    """Collect all data for a training run."""
    data = {
        'run_name': run_name,
        'run_path': str(run_path),
        'timestamp': datetime.now().isoformat(),
    }
    
    # Load adapter config for hyperparameters
    adapter_config = load_adapter_config(run_path)
    if adapter_config:
        data['hyperparameters'] = {
            'model': adapter_config.get('model'),
            'learning_rate': adapter_config.get('learning_rate'),
            'batch_size': adapter_config.get('batch_size'),
            'iters': adapter_config.get('iters'),
            'max_seq_length': adapter_config.get('max_seq_length'),
            'num_layers': adapter_config.get('num_layers'),
            'lora_rank': adapter_config.get('lora_rank'),
            'grad_accumulation_steps': adapter_config.get('grad_accumulation_steps'),
            'grad_checkpoint': adapter_config.get('grad_checkpoint'),
            'steps_per_eval': adapter_config.get('steps_per_eval'),
            'steps_per_report': adapter_config.get('steps_per_report'),
            'save_every': adapter_config.get('save_every'),
        }
    
    # Parse training output if log file provided
    if log_file and log_file.exists():
        with open(log_file, 'r') as f:
            log_text = f.read()
        parsed = parse_training_output(log_text)
        data['training_metrics'] = parsed
        data['final_metrics'] = get_final_metrics(parsed)
    else:
        data['training_metrics'] = {}
        data['final_metrics'] = {}
    
    return data

def manual_input_metrics(run_name: str) -> Dict:
    """Prompt user for manual input of training metrics."""
    print(f"\n📊 Enter metrics for {run_name}")
    print("(Press Enter to skip any field)")
    
    metrics = {}
    
    try:
        final_iter = input("Final iteration: ").strip()
        if final_iter:
            metrics['final_iteration'] = int(final_iter)
        
        train_loss = input("Final train loss: ").strip()
        if train_loss:
            metrics['final_train_loss'] = float(train_loss)
        
        val_loss = input("Final validation loss: ").strip()
        if val_loss:
            metrics['final_val_loss'] = float(val_loss)
        
        lr = input("Learning rate: ").strip()
        if lr:
            metrics['final_learning_rate'] = float(lr)
        
        tokens_sec = input("Tokens/sec: ").strip()
        if tokens_sec:
            metrics['final_tokens_per_sec'] = float(tokens_sec)
        
        it_sec = input("Iterations/sec: ").strip()
        if it_sec:
            metrics['final_iterations_per_sec'] = float(it_sec)
        
        total_tokens = input("Total trained tokens: ").strip()
        if total_tokens:
            metrics['total_trained_tokens'] = int(total_tokens)
        
        peak_mem = input("Peak memory (GB): ").strip()
        if peak_mem:
            metrics['peak_memory_gb'] = float(peak_mem)
        
        val_time = input("Validation time (seconds): ").strip()
        if val_time:
            metrics['final_val_time_sec'] = float(val_time)
    
    except (ValueError, KeyboardInterrupt):
        print("\n⚠️  Invalid input or cancelled. Skipping manual entry.")
        return {}
    
    return metrics

def plot_loss_curves(runs_data: List[Dict], output_file: Optional[Path] = None, project_root: Optional[Path] = None) -> bool:
    """
    Plot training and validation loss curves for all runs.
    Returns True if plotting succeeded, False otherwise.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("⚠️  matplotlib not available. Install with: pip install matplotlib")
        return False
    
    # Check if we have training metrics to plot
    has_metrics = False
    for run in runs_data:
        metrics = run.get('training_metrics', {})
        if metrics.get('iterations') and (metrics.get('train_loss') or metrics.get('val_loss')):
            has_metrics = True
            break
    
    if not has_metrics:
        print("⚠️  No training metrics available for plotting. Need training logs with loss data.")
        return False
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Training Loss
    ax1 = axes[0]
    for run in runs_data:
        metrics = run.get('training_metrics', {})
        iterations = metrics.get('iterations', [])
        train_loss = metrics.get('train_loss', [])
        
        if iterations and train_loss:
            # Align lengths
            min_len = min(len(iterations), len(train_loss))
            ax1.plot(
                iterations[:min_len],
                train_loss[:min_len],
                label=f"{run['run_name']} (Train)",
                linewidth=2,
                marker='o',
                markersize=3
            )
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale often better for loss
    
    # Plot 2: Validation Loss
    ax2 = axes[1]
    for run in runs_data:
        metrics = run.get('training_metrics', {})
        iterations = metrics.get('iterations', [])
        val_loss = metrics.get('val_loss', [])
        
        if iterations and val_loss:
            # Align lengths
            min_len = min(len(iterations), len(val_loss))
            ax2.plot(
                iterations[:min_len],
                val_loss[:min_len],
                label=f"{run['run_name']} (Val)",
                linewidth=2,
                marker='s',
                markersize=3
            )
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale often better for loss
    
    plt.tight_layout()
    
    # Save plot
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"📊 Loss curves saved to: {output_file}")
    else:
        # Default filename
        if project_root:
            default_file = project_root / "training_loss_comparison.png"
        else:
            default_file = Path("training_loss_comparison.png")
        plt.savefig(default_file, dpi=150, bbox_inches='tight')
        print(f"📊 Loss curves saved to: {default_file}")
    
    plt.close()
    return True

def format_comparison_table(runs_data: List[Dict]) -> str:
    """Format a comparison table of all runs."""
    lines = []
    lines.append("=" * 120)
    lines.append("TRAINING RUNS COMPARISON")
    lines.append("=" * 120)
    lines.append("")
    
    # Hyperparameters section
    lines.append("HYPERPARAMETERS")
    lines.append("-" * 120)
    header = ["Run", "LR", "Batch", "Iters", "Seq Len", "LoRA Layers", "LoRA Rank", "Grad Acc", "Grad Check"]
    lines.append(" | ".join(f"{h:>12}" for h in header))
    lines.append("-" * 120)
    
    for run in runs_data:
        hp = run.get('hyperparameters', {})
        row = [
            run['run_name'],
            f"{hp.get('learning_rate', 'N/A'):.2e}" if hp.get('learning_rate') else 'N/A',
            str(hp.get('batch_size', 'N/A')),
            str(hp.get('iters', 'N/A')),
            str(hp.get('max_seq_length', 'N/A')),
            str(hp.get('num_layers', 'N/A')),
            str(hp.get('lora_rank', 'N/A')),
            str(hp.get('grad_accumulation_steps', 'N/A')),
            "✓" if hp.get('grad_checkpoint') else "✗",
        ]
        lines.append(" | ".join(f"{r:>12}" for r in row))
    
    lines.append("")
    lines.append("")
    
    # Training metrics section
    lines.append("TRAINING METRICS")
    lines.append("-" * 120)
    header = ["Run", "Final Iter", "Train Loss", "Val Loss", "Tokens/sec", "It/sec", "Peak Mem (GB)"]
    lines.append(" | ".join(f"{h:>15}" for h in header))
    lines.append("-" * 120)
    
    for run in runs_data:
        fm = run.get('final_metrics', {})
        row = [
            run['run_name'],
            str(fm.get('final_iteration', 'N/A')),
            f"{fm.get('final_train_loss', 0):.4f}" if fm.get('final_train_loss') else 'N/A',
            f"{fm.get('final_val_loss', 0):.4f}" if fm.get('final_val_loss') else 'N/A',
            f"{fm.get('final_tokens_per_sec', 0):.1f}" if fm.get('final_tokens_per_sec') else 'N/A',
            f"{fm.get('final_iterations_per_sec', 0):.3f}" if fm.get('final_iterations_per_sec') else 'N/A',
            f"{fm.get('peak_memory_gb', 0):.2f}" if fm.get('peak_memory_gb') else 'N/A',
        ]
        lines.append(" | ".join(f"{r:>15}" for r in row))
    
    lines.append("")
    lines.append("=" * 120)
    
    return "\n".join(lines)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compare training metrics across multiple MLX training runs"
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["SmolLM3-3B", "SmolLM3-3B_run2", "SmolLM3-3B_run3", "SmolLM3-3B_run4"],
        help="List of run names to compare (default: all SmolLM3 runs)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Directory containing log files (named like {run_name}.log)",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Prompt for manual input of metrics (interactive)",
    )
    parser.add_argument(
        "--training-output",
        type=str,
        nargs="+",
        help="File(s) containing training output text to parse. Can specify multiple files (will try to match to runs by name) or use '-' for stdin",
    )
    parser.add_argument(
        "--run-metrics",
        type=str,
        help="JSON file with metrics for specific runs (format: {\"run_name\": {\"final_train_loss\": 1.443, ...}})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training_runs_comparison.json",
        help="Output JSON file for comparison data (default: training_runs_comparison.json)",
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Output markdown report file (optional)",
    )
    parser.add_argument(
        "--plot",
        type=str,
        help="Output file for loss curve plot (e.g., loss_comparison.png). Requires matplotlib.",
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "models"
    
    runs_data = []
    
    print("=" * 80)
    print("Training Runs Comparison Tool")
    print("=" * 80)
    print()
    
    for run_name in args.runs:
        run_path = models_dir / run_name
        
        if not run_path.exists():
            print(f"⚠️  Run directory not found: {run_path}")
            print(f"   Skipping {run_name}")
            continue
        
        print(f"📁 Processing {run_name}...")
        
        # Check for log file
        log_file = None
        if args.log_dir:
            log_file = Path(args.log_dir) / f"{run_name}.log"
            if not log_file.exists():
                log_file = None
        
        # Collect data
        run_data = collect_run_data(run_name, run_path, log_file)
        
        # Try to parse training output if provided
        if args.training_output:
            # Try to find a matching log file for this run
            training_file = None
            for log_path in args.training_output:
                if log_path == '-':
                    # Read from stdin (only use once)
                    import sys
                    training_text = sys.stdin.read()
                    parsed = parse_training_output(training_text)
                    if parsed['iterations']:
                        run_data['training_metrics'] = parsed
                        run_data['final_metrics'] = get_final_metrics(parsed)
                    break
                else:
                    log_file = Path(log_path)
                    # Try exact match first, then try if filename contains run name
                    if log_file.exists():
                        if run_name in log_file.stem or log_file.stem in run_name:
                            training_file = log_file
                            break
                        # If only one file provided, use it for all runs
                        elif len(args.training_output) == 1:
                            training_file = log_file
                            break
            
            if training_file and training_file.exists():
                with open(training_file, 'r') as f:
                    training_text = f.read()
                parsed = parse_training_output(training_text)
                if parsed['iterations']:
                    run_data['training_metrics'] = parsed
                    run_data['final_metrics'] = get_final_metrics(parsed)
        
        # Load metrics from JSON file if provided
        if args.run_metrics:
            metrics_file = Path(args.run_metrics)
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    all_metrics = json.load(f)
                if run_name in all_metrics:
                    run_data['final_metrics'] = {**run_data.get('final_metrics', {}), **all_metrics[run_name]}
        
        # Manual input if requested and no metrics found (only if interactive)
        if args.manual and not run_data.get('final_metrics') and sys.stdin.isatty():
            try:
                manual_metrics = manual_input_metrics(run_name)
                if manual_metrics:
                    run_data['final_metrics'] = {**run_data.get('final_metrics', {}), **manual_metrics}
            except (EOFError, KeyboardInterrupt):
                print(f"   ⚠️  Skipping manual input for {run_name} (non-interactive)")
        
        runs_data.append(run_data)
        print(f"   ✓ Collected data for {run_name}")
    
    if not runs_data:
        print("\n❌ No run data collected. Exiting.")
        sys.exit(1)
    
    # Generate comparison
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    comparison_table = format_comparison_table(runs_data)
    print(comparison_table)
    
    # Save JSON output
    output_file = project_root / args.output
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'runs': runs_data,
            'comparison_table': comparison_table
        }, f, indent=2)
    print(f"\n💾 Comparison data saved to: {output_file}")
    
    # Save markdown report if requested
    if args.report:
        report_file = project_root / args.report
        with open(report_file, 'w') as f:
            f.write(f"# Training Runs Comparison\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write("```\n")
            f.write(comparison_table)
            f.write("\n```\n\n")
            f.write("## Detailed Data\n\n")
            f.write("```json\n")
            json.dump(runs_data, f, indent=2)
            f.write("\n```\n")
        print(f"📄 Markdown report saved to: {report_file}")
    
    # Generate loss curve plot if requested
    if args.plot:
        plot_file = project_root / args.plot
        plot_loss_curves(runs_data, plot_file, project_root)
    elif any(run.get('training_metrics', {}).get('iterations') for run in runs_data):
        # Auto-generate plot if training metrics are available
        plot_loss_curves(runs_data, project_root=project_root)

if __name__ == "__main__":
    main()

