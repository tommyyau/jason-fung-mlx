#!/usr/bin/env python3
"""
Debug script to check if DPO training is actually working
"""
import mlx.core as mx
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

print("="*70)
print("DEBUGGING DPO TRAINING")
print("="*70)

# Load model
model_path = "ibm-granite/granite-4.0-h-micro"
print(f"\n1. Loading model: {model_path}")
model, tokenizer = load(model_path)

print("\n2. Checking trainable parameters BEFORE freezing:")
from mlx.utils import tree_flatten
trainable_before = model.trainable_parameters()
flat_before = tree_flatten(trainable_before)
print(f"   Number of parameter tensors: {len(flat_before)}")

# Freeze model
print("\n3. Freezing model...")
model.freeze()

print("\n4. Checking trainable parameters AFTER freezing:")
trainable_after_freeze = model.trainable_parameters()
flat_after_freeze = tree_flatten(trainable_after_freeze)
print(f"   Number of parameter tensors: {len(flat_after_freeze)}")

# Add LoRA layers
print("\n5. Adding LoRA layers (rank=8, alpha=16, 16 layers)...")
linear_to_lora_layers(
    model,
    num_layers=16,
    config={"rank": 8, "alpha": 16, "dropout": 0.0, "scale": 10.0}
)

print("\n6. Checking trainable parameters AFTER adding LoRA:")
trainable_after_lora = model.trainable_parameters()
flat_after_lora = tree_flatten(trainable_after_lora)
num_tensors = len(flat_after_lora)
print(f"   Number of parameter tensors: {num_tensors}")

if num_tensors == 0:
    print("\n❌ CRITICAL BUG: No trainable parameters after adding LoRA!")
    print("   This means gradients won't update anything!")
else:
    # Calculate total parameters
    total_params = sum(mx.prod(mx.array(v.shape)).item() for k, v in flat_after_lora)

    print(f"✅ Model has {total_params:,.0f} trainable LoRA parameters")
    print(f"\nSample trainable parameter keys:")
    for i, (k, v) in enumerate(list(flat_after_lora)[:5]):
        print(f"   - {k}: shape {v.shape}")

# Test gradient computation
print("\n7. Testing gradient computation...")
from mlx_lm.tuner.lora import LoRALinear
import mlx.nn as nn

def simple_loss_fn(model, x):
    """Simple loss function for testing"""
    # Just compute a dummy forward pass
    output = model(x)
    return output.sum()

# Create loss and grad function
loss_and_grad_fn = nn.value_and_grad(model, simple_loss_fn)

# Try computing gradients
test_input = mx.array([[1, 2, 3, 4, 5]])
try:
    loss, grads = loss_and_grad_fn(model, test_input)
    print(f"   Loss computed: {loss.item():.4f}")
    print(f"   Gradients computed: {len(grads)} tensors")

    # Check if any gradients are non-zero
    grad_norms = {k: mx.linalg.norm(v).item() for k, v in grads.items()}
    max_grad = max(grad_norms.values()) if grad_norms else 0
    print(f"   Max gradient norm: {max_grad:.6f}")

    if max_grad == 0:
        print("\n❌ CRITICAL BUG: All gradients are zero!")
    else:
        print(f"\n✅ Gradients are non-zero")

except Exception as e:
    print(f"\n❌ GRADIENT COMPUTATION FAILED: {e}")

print("\n" + "="*70)
