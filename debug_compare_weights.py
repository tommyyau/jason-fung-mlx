
import mlx.core as mx
from mlx_lm import load
from mlx.utils import tree_flatten

def main():
    print("Loading base model...")
    base_model, _ = load("ibm-granite/granite-4.0-h-micro")
    
    print("Loading fused model...")
    fused_model, _ = load("models/granite-4.0-h-micro-dpo-fused")
    
    print("\nComparing weights...")
    
    base_params = dict(tree_flatten(base_model.parameters()))
    fused_params = dict(tree_flatten(fused_model.parameters()))
    
    differences = []
    total_params = 0
    different_params = 0
    
    for name, base_param in base_params.items():
        if name in fused_params:
            fused_param = fused_params[name]
            
            # Check if shapes match
            if base_param.shape != fused_param.shape:
                print(f"Shape mismatch for {name}: {base_param.shape} vs {fused_param.shape}")
                continue
                
            # Check for equality
            if not mx.array_equal(base_param, fused_param):
                diff = mx.abs(base_param - fused_param).max().item()
                differences.append((name, diff))
                different_params += 1
                # print(f"Difference found in {name}. Max diff: {diff}")
            
            total_params += 1
        else:
            print(f"Parameter {name} not found in fused model")

    if len(differences) == 0:
        print("\n❌ MODELS ARE IDENTICAL! The fused model has the exact same weights as the base model.")
    else:
        print(f"\n✅ Models are different. {different_params}/{total_params} parameters have changed.")
        print("Top 5 differences:")
        differences.sort(key=lambda x: x[1], reverse=True)
        for name, diff in differences[:5]:
            print(f"  {name}: {diff}")

if __name__ == "__main__":
    main()
