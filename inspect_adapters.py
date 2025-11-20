import mlx.core as mx
from pathlib import Path

p = "models/granite-4.0-h-micro-dpo/adapters.safetensors"
if Path(p).exists():
    d = mx.load(p)
    print(f"Keys: {list(d.keys())[:5]}")
else:
    print("File not found")
