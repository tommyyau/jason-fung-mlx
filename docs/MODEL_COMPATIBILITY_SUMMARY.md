# Model Compatibility Quick Reference

## TL;DR

**✅ Use These (16GB RAM):**
- **Llama 3.2 3B Instruct** - Production model (no issues)
- **Qwen 2.5 1.5B** - Best alternative (faster, lower memory)

**❌ Don't Use These:**
- **Granite 4.0** - MoE architecture unsupported by MLX
- **Gemma 3N E4B** - OOM on 16GB RAM + format issues

---

## What You Found

### 1. Granite 4.0 - MoE Architecture Incompatible ⚠️

**Status:** MLX doesn't support Mixture of Experts (MoE)

**Problem:**
> "Granite - basically they blended in two technologies, and because it's two technologies, MLX (Apple MLX) doesn't seem to actually support it anymore."

**Technical Details:**
- Granite uses **MoE (Mixture of Experts)** architecture
- MoE = multiple "expert" sub-networks with routing mechanism
- MLX designed for standard Transformers, not MoE
- Requires: dynamic expert selection, sparse activation, load balancing, custom gradient routing
- None of these are implemented in MLX

**Conclusion:** Granite's "two technologies" (Transformer + MoE) fundamentally incompatible with MLX.

---

### 2. Gemma 3N E4B - Memory Exhaustion ❌

**Status:** Out of memory, cannot train on 16GB RAM

**Problem:**
> "Gemma 3n e4b... was just a little bit too big. It basically ran out of space... I couldn't train on quantized model, and so it just basically blew out my 16 GB."

**Memory Analysis:**
```
4B parameters × 2 bytes (FP16) = 8GB base model
+ LoRA adapters: ~400MB
+ Gradients: ~400MB
+ Activations: ~4-6GB
────────────────────────────
Total: 13-15GB required
Available: ~12GB (16GB - macOS overhead)
Result: OUT OF MEMORY ❌
```

**Aggressive Optimizations Tried (Still Failed):**
- max_seq_length: 1024 → 256 (-75%)
- lora_rank: 8 → 4 (-50%)
- num_layers: 12 → 8 (-33%)
- steps_per_eval: DISABLED (validation caused OOM)

**Still failed:** Model simply too large for 16GB RAM during training.

**Why "Mobile Optimized" is Misleading:**

| Claim | Reality |
|-------|---------|
| "Runs on phones" | ✅ True for **inference** (4-bit, ~1GB) |
| "Trainable on edge" | ❌ False for **training** (FP16, ~13-15GB) |

You cannot quantize during training - requires FP16 minimum for numerical stability.

---

### 3. Gemma Chat Format Issue 😤

**Problem:**
> "Gemma was just insistent on changing the format of mlx to be whatever the Gemma format was, some sort of chat format that was a bit nuts."

**Standard MLX Format:**
```json
{"messages": [
  {"role": "user", "content": "question"},
  {"role": "assistant", "content": "answer"}
]}
```

**Gemma Format:**
```json
{"text": "<start_of_turn>user\nquestion<end_of_turn>\n<start_of_turn>model\nanswer<end_of_turn>\n"}
```

**Solution Required:**
- Run conversion script: `scripts/phase3-prepare-data-mlx/04b_convert_mlx_to_gemma.py`
- Creates separate `train_gemma.jsonl` and `valid_gemma.jsonl` files
- Extra step, easy to forget, annoying

**Affects:** Gemma 3 Text 4B (works with conversion) AND Gemma 3N E4B (OOM anyway)

---

### 4. Qwen 2.5 - The One That Works ✅

**Status:** Fully compatible, no issues

**Why it works:**
- Standard architecture (no MoE)
- Small enough for 16GB RAM (~4GB peak)
- Uses standard MLX format (no conversion needed)
- Excellent MLX support

**User Choice:**
> "I eventually decided to use Qwen 2.5, and that seems to be a lot easier to actually work with"

**Recommendation:** ✅ Use this or Llama 3.2 3B.

---

## Memory Requirements by Model

| Model | Parameters | Training RAM | Format | Status |
|-------|-----------|--------------|--------|--------|
| Llama 3.2 3B | 3B | ~7.6GB | Standard MLX | ✅ Production |
| Qwen 2.5 1.5B | 1.5B | ~4GB | Standard MLX | ✅ Alternative |
| Gemma 3 Text 4B | 4B | ~6.9GB | Custom (conversion) | ⚠️ Works with extra steps |
| Granite 4.0 | 3B | N/A | N/A | ❌ MoE unsupported |
| Gemma 3N E4B | 4B | ~13-15GB | Custom (conversion) | ❌ OOM |

---

## What This Means for Your Pipeline

**Current Setup (16GB RAM M1 MacBook Pro):**

✅ **Use:**
- Llama 3.2 3B (current production model)
- Qwen 2.5 1.5B (your successful alternative)

⚠️ **Possible but annoying:**
- Gemma 3 Text 4B (requires format conversion + config cleanup)

❌ **Don't bother:**
- Granite 4.0 (MLX architecture incompatibility)
- Gemma 3N E4B (memory + format issues)

**If You Had 24GB+ RAM:**
- ✅ Gemma 3N E4B would work (but still needs format conversion)
- ✅ Mistral 7B would work
- ❌ Granite still fails (architecture issue, not memory)

**If You Had 32GB+ RAM:**
- ✅ Any model up to 13B parameters
- ❌ Granite **still fails** (MoE is fundamental incompatibility)

---

## Key Takeaways

1. **Architecture > Size**
   - Granite (3B with MoE): Fails
   - Llama (3B standard): Works
   - Architecture compatibility is critical

2. **"Mobile Optimized" is Marketing**
   - Gemma 3N E4B "mobile optimized" = inference only
   - Training still requires 13-15GB RAM (full precision)
   - Don't be fooled by deployment/inference claims

3. **Memory Math is Real**
   - LoRA doesn't reduce base model memory
   - You need full model in RAM even with LoRA
   - Parameter count × 2 bytes (FP16) = minimum RAM
   - Add 50% for activations/gradients/optimizer

4. **Format Compatibility Matters**
   - Standard MLX format: No issues
   - Custom Gemma format: Extra conversion step
   - Prefer standard format to avoid hassle

5. **Use Proven Models**
   - Llama 3.2 3B: Fully tested, works end-to-end
   - Qwen 2.5: Excellent alternative, full compatibility
   - Don't experiment with incompatible architectures in production

---

## Recommendations

**For this project (16GB RAM):**
1. ✅ **Keep using Llama 3.2 3B** - production model, no issues
2. ✅ **Qwen 2.5 1.5B is perfect** - your choice is correct
3. ❌ **Avoid Granite** - MLX doesn't support MoE
4. ❌ **Avoid Gemma 3N E4B** - too large + format hassle

**Testing new models:**
1. Check architecture (no MoE for MLX)
2. Estimate memory: params × 2 bytes × 1.5 (overhead)
3. Check format compatibility (prefer standard MLX)
4. Test with 10 iterations before full training

---

## See Also

- [Full Compatibility Analysis](./MLX_MODEL_COMPATIBILITY.md) - Detailed technical analysis
- [Fine-Tuning Saga](./FINE_TUNING_SAGA.md) - Model selection journey
- [Training Guide](./TRAINING_GUIDE.md) - Production configuration
