# Model Compatibility Quick Reference

## TL;DR

**✅ Use These (16GB RAM):**
- Llama 3.2 3B Instruct (production model)
- Qwen 2.5 1.5B/3B (alternative)

**❌ Don't Use These:**
- Granite 3.0/4.0 - GGUF conversion fails
- Gemma 2 9B - Out of memory (needs 24GB+ RAM)

---

## What You Found

### 1. Granite Models - GGUF Conversion Failure ⚠️

**Status:** Training works, deployment fails

**Problem:**
- MLX training: ✅ Success
- LoRA fusion: ✅ Success
- GGUF conversion: ❌ **FAILS**

**Root Cause:**
Granite uses a non-standard architecture (Multi-head Latent Attention / MoLA) that stores weights differently:
- Standard Transformer: `[num_heads * head_dim, model_dim]`
- Granite: `[num_heads, head_dim, model_dim]` (extra dimension)

The llama.cpp GGUF converter expects standard Transformer format and cannot handle Granite's architecture.

**Error:**
```
ValueError: Cannot reshape weight 'model.layers.0.self_attn.q_proj.weight'
Expected shape: [3072, 3072]
Got shape: [3072, 1, 3072]
Shape mismatch: extra dimension cannot be squeezed
```

**Conclusion:** Granite is incompatible with the GGUF deployment pipeline.

---

### 2. Gemma 2 9B - Memory Constraints ❌

**Status:** Out of memory, cannot train

**Problem:**
- Model loading: ❌ System freeze
- Training: ❌ Kernel panic / force quit
- Memory required: 18-24GB
- Available on 16GB MacBook Pro: ~12GB

**Why It Fails:**

Even with LoRA (only 0.2% trainable parameters):
```
Base model (FP16): 9B params × 2 bytes = 18GB
LoRA adapters: ~200MB
Activations: ~4-6GB
────────────────────────────
Total: 22-24GB required
Available: 12GB (after macOS overhead)
Result: OUT OF MEMORY ❌
```

**Why "Mobile Optimized" is Misleading:**

| Marketing Claim | Reality |
|----------------|---------|
| "Runs on phones" | ✅ True for **inference** (4-bit quantized, 2.3GB) |
| "Trainable on edge" | ❌ False for **training** (FP16, 18GB+) |

**Training requires full precision** - you cannot quantize during training. The 9B parameter count makes it impossible to train on 16GB RAM.

**Conclusion:** Gemma 2 9B requires 24GB+ RAM for training.

---

### 3. Qwen 2.5 - Fully Compatible ✅

**Status:** Works perfectly

**Results:**
- MLX training: ✅ Success
- GGUF conversion: ✅ Success
- Memory usage: 1.5B (~4GB), 3B (~8GB)

**Why It Works:**
- Standard architecture fully supported by MLX
- Excellent conversion pipeline support
- Multiple sizes for different RAM configurations

**Recommendation:** Use Qwen 2.5 1.5B or 3B as an alternative to Llama 3.2 3B.

---

## Memory Requirements by Model

| Model | Parameters | Training RAM | GGUF Conversion | Status |
|-------|-----------|-------------|-----------------|--------|
| Llama 3.2 3B | 3B | ~7.6GB | ✅ Works | ✅ Production |
| Qwen 2.5 1.5B | 1.5B | ~4GB | ✅ Works | ✅ Alternative |
| Qwen 2.5 3B | 3B | ~8GB | ✅ Works | ✅ Alternative |
| Granite 4.0 | 3B | ~8GB | ❌ Fails | ⚠️ Training only |
| Gemma 2 9B | 9B | ~22GB | N/A | ❌ OOM |
| Mistral 7B | 7B | ~14GB | ✅ Works | ⚠️ Tight fit |

---

## What This Means for Your Pipeline

**Current Setup (16GB RAM):**
- ✅ Llama 3.2 3B: Fully compatible, tested, production-ready
- ✅ Qwen 2.5 1.5B/3B: Fully compatible alternatives
- ❌ Granite: Cannot deploy (GGUF conversion fails)
- ❌ Gemma 2 9B: Cannot train (memory constraints)

**If You Had 24GB+ RAM:**
- ✅ Gemma 2 9B: Would work for training
- ✅ Mistral 7B: More comfortable
- ❌ Granite: Still fails (architecture issue, not memory)

**If You Had 32GB+ RAM:**
- ✅ Any model up to 13B parameters
- ❌ Granite: **Still fails** (conversion issue is fundamental)

---

## Key Takeaways

1. **MLX Training ≠ Production Deployment**
   - A model that trains doesn't guarantee conversion works
   - Always test the full pipeline: train → fuse → HF → GGUF

2. **Architecture Matters More Than Size**
   - Granite (3B): Trains but can't convert (non-standard architecture)
   - Llama (3B): Full pipeline works (standard architecture)
   - Architecture compatibility is critical

3. **"Mobile Optimized" is Marketing**
   - Gemma 2 9B "mobile optimized" = inference only
   - Training still requires 18GB+ RAM (full precision)
   - Don't be fooled by deployment claims

4. **Memory Math is Real**
   - LoRA doesn't reduce base model memory
   - You need the full model in RAM even with LoRA
   - Parameter count × 2 bytes (FP16) = minimum RAM needed

5. **Use Proven Models**
   - Llama 3.2 3B: Fully tested, works end-to-end
   - Qwen 2.5: Excellent alternative, full compatibility
   - Don't experiment with unproven architectures in production

---

## Recommendations

**For this project (16GB RAM):**
1. ✅ **Keep using Llama 3.2 3B** - fully tested, production-ready
2. ✅ **Consider Qwen 2.5 3B** - alternative with similar performance
3. ❌ **Avoid Granite** - cannot deploy to GGUF
4. ❌ **Avoid Gemma 2 9B** - insufficient memory

**Testing new models:**
1. Check parameter count (max 3B for 16GB RAM)
2. Verify architecture (standard Transformer preferred)
3. Test full pipeline (train → convert → deploy)
4. Don't assume "mobile optimized" = "trainable on edge"

---

## See Also

- [Full Compatibility Analysis](./MLX_MODEL_COMPATIBILITY.md) - Detailed technical analysis
- [Fine-Tuning Saga](./FINE_TUNING_SAGA.md) - Model selection journey
- [Training Guide](./TRAINING_GUIDE.md) - Production configuration
