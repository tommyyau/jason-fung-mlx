#!/usr/bin/env python3
"""
Step 10 – Train DPO (Memory Optimized)
──────────────────────────────────────
Fine-tunes the model using Direct Preference Optimization (DPO).
Uses "Offline DPO" to save memory:
1. Precompute reference logps (run with --stage precompute)
2. Train policy model (run with --stage train)
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import List, Dict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate
from mlx_lm.tuner.lora import LoRALinear

def save_adapters(model, output_path):
    """Save LoRA adapters in MLX format."""
    from mlx.utils import tree_flatten
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get trainable parameters (LoRA weights)
    trainable_params = model.trainable_parameters()
    flat_params = dict(tree_flatten(trainable_params))
    
    print(f"Saving {len(flat_params)} adapter parameters to {output_path}")
    mx.save_safetensors(str(output_path / "adapters.safetensors"), flat_params)

# ─────────────────────────────
# DPO Loss Implementation
# ─────────────────────────────
def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    Compute DPO loss.
    """
    policy_log_ratios = policy_chosen_logps - policy_rejected_logps
    ref_log_ratios = ref_chosen_logps - ref_rejected_logps
    
    logits = policy_log_ratios - ref_log_ratios
    
    losses = -nn.log_sigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    
    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()

def get_batch_logps(model, input_ids, attention_mask, labels):
    """
    Compute log probabilities for the given labels.
    """
    logits = model(input_ids)
    
    # Shift logits and labels for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    
    # Compute log probs
    log_probs = nn.losses.cross_entropy(shift_logits, shift_labels, reduction='none')
    
    # Apply mask (ignore padding)
    shift_mask = attention_mask[:, 1:]
    
    # Invert cross entropy to get log prob (CE = -log_prob)
    log_probs = -log_probs * shift_mask
    
    # Sum over sequence length
    sum_log_probs = log_probs.sum(axis=1)
    
    return sum_log_probs

# ─────────────────────────────
# Data Loading
# ─────────────────────────────
def load_dpo_data(path: str):
    """Load DPO dataset."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                data.append(entry)
    return data

def prepare_batch(batch_data, tokenizer, max_length=1024):
    """Prepare a batch of DPO data."""
    chosen_input_ids = []
    rejected_input_ids = []
    chosen_masks = []
    rejected_masks = []
    chosen_labels = []
    rejected_labels = []
    
    for item in batch_data:
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        
        chosen_text = prompt + chosen
        rejected_text = prompt + rejected
        
        chosen_tokens = tokenizer.encode(chosen_text)
        rejected_tokens = tokenizer.encode(rejected_text)
        
        # Truncate
        chosen_tokens = chosen_tokens[:max_length]
        rejected_tokens = rejected_tokens[:max_length]
        
        chosen_input_ids.append(mx.array(chosen_tokens)[None, :])
        rejected_input_ids.append(mx.array(rejected_tokens)[None, :])
        
        chosen_masks.append(mx.ones((1, len(chosen_tokens))))
        rejected_masks.append(mx.ones((1, len(rejected_tokens))))
        
        chosen_labels.append(mx.array(chosen_tokens)[None, :])
        rejected_labels.append(mx.array(rejected_tokens)[None, :])

    return chosen_input_ids, rejected_input_ids, chosen_masks, rejected_masks, chosen_labels, rejected_labels

# ─────────────────────────────
# Main
# ─────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    # Config file (preferred method)
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    
    # Individual arguments (override config if provided)
    parser.add_argument("--model", type=str, help="Model path")
    parser.add_argument("--data", type=str, help="Training data path")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--beta", type=float, help="DPO beta parameter")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--steps", type=int, help="Number of training steps")
    parser.add_argument("--stage", type=str, choices=["precompute", "train"], required=True, help="Training stage")
    
    # New configurable parameters
    parser.add_argument("--lora-rank", type=int, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, help="LoRA dropout")
    parser.add_argument("--lora-scale", type=float, help="LoRA scale")
    parser.add_argument("--num-layers", type=int, help="Number of layers for LoRA")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--grad-accumulation-steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-length", type=int, help="Maximum sequence length")
    parser.add_argument("--steps-per-report", type=int, help="Steps between progress reports")
    parser.add_argument("--save-every", type=int, help="Steps between checkpoints")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--optimizer", type=str, help="Optimizer (adamw)")
    
    args = parser.parse_args()
    
    # Load config from YAML if provided
    config = {}
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Helper function to get value (CLI args override config)
    def get_arg(name, default=None):
        cli_value = getattr(args, name.replace('-', '_'), None)
        if cli_value is not None:
            return cli_value
        return config.get(name.replace('_', '-'), config.get(name, default))
    
    # Get all parameters with proper type conversion
    model_path = get_arg('model')
    data_path = get_arg('data')
    output_dir = get_arg('output_dir', 'models/dpo_adapter')
    learning_rate = float(get_arg('learning_rate', 1e-6))
    beta = float(get_arg('beta', 0.1))
    epochs = int(get_arg('epochs', 1))
    steps = int(get_arg('steps', 100))
    
    # LoRA parameters
    lora_rank = int(get_arg('lora_rank', 8))
    lora_alpha = int(get_arg('lora_alpha', 16))
    lora_dropout = float(get_arg('lora_dropout', 0.0))
    lora_scale = float(get_arg('lora_scale', 10.0))
    num_layers = int(get_arg('num_layers', 16))
    
    # Training parameters
    batch_size = int(get_arg('batch_size', 1))
    grad_accumulation_steps = int(get_arg('grad_accumulation_steps', 8))
    max_seq_length = int(get_arg('max_seq_length', 1024))
    steps_per_report = int(get_arg('steps_per_report', 10))
    save_every = int(get_arg('save_every', 50))
    seed = int(get_arg('seed', 42))
    optimizer_name = get_arg('optimizer', 'adamw')
    
    # Validate required parameters
    if not model_path:
        raise ValueError("--model or config['model'] is required")
    if not data_path:
        raise ValueError("--data or config['data'] is required")
    
    # Set random seed
    mx.random.seed(seed)
    
    # Print configuration
    print(f"{'='*70}")
    print(f"DPO Training Configuration")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print(f"Stage: {args.stage}")
    print(f"\nDPO Parameters:")
    print(f"  Beta: {beta}")
    print(f"\nLoRA Parameters:")
    print(f"  Rank: {lora_rank}, Alpha: {lora_alpha}, Dropout: {lora_dropout}")
    print(f"  Scale: {lora_scale}, Layers: {num_layers}")
    print(f"\nTraining Parameters:")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Steps: {steps}, Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}, Grad Accumulation: {grad_accumulation_steps}")
    print(f"  Max Seq Length: {max_seq_length}")
    print(f"  Seed: {seed}")
    print(f"{'='*70}\n")
    
    # Load Data
    data = load_dpo_data(data_path)
    print(f"Loaded {len(data)} examples\n")
    
    if args.stage == "precompute":
        print("=== STAGE 1: Precomputing Reference Logps ===")
        print(f"Loading reference model: {model_path}")
        model, tokenizer = load(model_path)
        
        updated_data = []
        
        print("Computing logps...")
        for i, item in enumerate(data):
            c_ids, r_ids, c_mask, r_mask, c_lbls, r_lbls = prepare_batch([item], tokenizer, max_seq_length)
            
            # Compute logps
            c_logp = get_batch_logps(model, c_ids[0], c_mask[0], c_lbls[0]).item()
            r_logp = get_batch_logps(model, r_ids[0], r_mask[0], r_lbls[0]).item()
            
            item['ref_chosen_logps'] = c_logp
            item['ref_rejected_logps'] = r_logp
            updated_data.append(item)
            
            if i % 10 == 0:
                print(f"Processed {i}/{len(data)}")
                
        # Save updated data
        output_path = Path(data_path).with_suffix('.with_logps.jsonl')
        print(f"Saving to {output_path}")
        with open(output_path, 'w') as f:
            for item in updated_data:
                json.dump(item, f)
                f.write('\n')
                
    elif args.stage == "train":
        print("=== STAGE 2: Training Policy Model ===")
        
        # Check for precomputed data
        data_file = Path(data_path)
        if not str(data_file).endswith('.with_logps.jsonl'):
            precomputed_path = data_file.with_suffix('.with_logps.jsonl')
            if precomputed_path.exists():
                print(f"Found precomputed data: {precomputed_path}")
                data = load_dpo_data(str(precomputed_path))
            else:
                print("❌ Error: Precomputed logps not found. Run with --stage precompute first.")
                return
        
        print(f"Loading policy model: {model_path}")
        model, tokenizer = load(model_path)
        
        # Freeze base
        model.freeze()
        
        # Convert to LoRA
        from mlx_lm.tuner.utils import linear_to_lora_layers
        linear_to_lora_layers(model, num_layers=num_layers, config={"rank": lora_rank, "alpha": lora_alpha, "dropout": lora_dropout, "scale": lora_scale})
        
        optimizer = optim.AdamW(learning_rate=learning_rate)
        
        # Loss function
        def loss_fn(model, chosen_ids, rejected_ids, chosen_mask, rejected_mask, chosen_lbls, rejected_lbls, ref_c_logp, ref_r_logp):
            policy_chosen_logps = get_batch_logps(model, chosen_ids, chosen_mask, chosen_lbls)
            policy_rejected_logps = get_batch_logps(model, rejected_ids, rejected_mask, rejected_lbls)
            
            loss, chosen_reward, rejected_reward = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                mx.array([ref_c_logp]), mx.array([ref_r_logp]),
                beta=beta
            )
            return loss, chosen_reward, rejected_reward

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        
        from mlx.utils import tree_map

        print("Starting training...")
        current_step = 0
        steps_since_update = 0
        accumulated_grads = None
        
        for epoch in range(epochs):
            for item in data:
                if current_step >= steps:
                    break
                
                c_ids, r_ids, c_mask, r_mask, c_lbls, r_lbls = prepare_batch([item], tokenizer, max_seq_length)
                
                ref_c = item['ref_chosen_logps']
                ref_r = item['ref_rejected_logps']
                
                (loss, c_reward, r_reward), grads = loss_and_grad_fn(
                    model, c_ids[0], r_ids[0], c_mask[0], r_mask[0], c_lbls[0], r_lbls[0], ref_c, ref_r
                )
                
                # Accumulate gradients
                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    accumulated_grads = tree_map(lambda x, y: x + y, accumulated_grads, grads)
                
                steps_since_update += 1
                
                # Perform update if accumulation is complete
                if steps_since_update >= grad_accumulation_steps:
                    # Scale gradients by 1 / accumulation_steps
                    scale = 1.0 / grad_accumulation_steps
                    accumulated_grads = tree_map(lambda x: x * scale, accumulated_grads)
                    
                    optimizer.update(model, accumulated_grads)
                    mx.eval(model.parameters(), optimizer.state)
                    
                    # Reset accumulation
                    accumulated_grads = None
                    steps_since_update = 0
                    current_step += 1
                    
                    if current_step % steps_per_report == 0:
                        print(f"Step {current_step}: Loss={loss.item():.4f}, Reward Diff={(c_reward - r_reward).item():.4f}")
                    
                    # Save checkpoint if needed
                    if save_every > 0 and current_step % save_every == 0:
                        checkpoint_dir = Path(output_dir) / f"checkpoint-{current_step}"
                        print(f"\nSaving checkpoint to {checkpoint_dir}")
                        save_adapters(model, checkpoint_dir)
                        with open(checkpoint_dir / "adapter_config.json", "w") as f:
                            json.dump({"lora_parameters": {"rank": lora_rank, "alpha": lora_alpha, "dropout": lora_dropout, "scale": lora_scale}, "num_layers": num_layers}, f)
                        print("Checkpoint saved\n")
                    
        print(f"Training complete. Saving adapters to {output_dir}")
        save_adapters(model, output_dir)
        
        with open(Path(output_dir) / "adapter_config.json", "w") as f:
            json.dump({"lora_parameters": {"rank": lora_rank, "alpha": lora_alpha, "dropout": lora_dropout, "scale": lora_scale}, "num_layers": num_layers}, f)

if __name__ == "__main__":
    main()
