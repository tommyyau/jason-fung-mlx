python -m mlx_lm lora --config config/mlx_gemma_training.yaml

==>
Loading configuration file config/mlx_gemma_training.yaml
Loading pretrained model
special_tokens_map.json: 100%|██████████████████████████████████████████| 662/662 [00:00<00:00, 10.4MB/s]
config.json: 100%|██████████████████████████████████████████████████████| 928/928 [00:00<00:00, 14.5MB/s]
model.safetensors.index.json: 79.9kB [00:00, 31.4MB/s]                       | 0.00/33.4M [00:00<?, ?B/s]
added_tokens.json: 100%|███████████████████████████████████████████████| 35.0/35.0 [00:00<00:00, 583kB/s]
tokenizer_config.json: 1.16MB [00:00, 44.3MB/s]                            | 1/8 [00:00<00:03,  2.27it/s]
tokenizer.json: 100%|███████████████████████████████████████████████| 33.4M/33.4M [00:01<00:00, 24.8MB/s]
tokenizer.model: 100%|██████████████████████████████████████████████| 4.69M/4.69M [00:01<00:00, 2.62MB/s]
model.safetensors: 100%|████████████████████████████████████████████| 4.84G/4.84G [01:58<00:00, 40.7MB/s]
Fetching 8 files: 100%|████████████████████████████████████████████████████| 8/8 [01:59<00:00, 14.92s/it]
Loading datasets 100%|██████████████████████████████████████████████| 4.69M/4.69M [00:01<00:00, 2.62MB/s]
Training
Trainable parameters: 0.116% (5.259M/4551.516M)
Starting training..., iters: 1367
Calculating loss...: 100%|███████████████████████████████████████████████| 25/25 [00:25<00:00,  1.02s/it]
Iter 1: Val loss 6.400, Val took 25.458s
Iter 50: Train loss 4.510, Learning Rate 1.000e-05, It/sec 0.491, Tokens/sec 93.137, Trained Tokens 9488, Peak mem 6.354 GB
Iter 100: Train loss 2.861, Learning Rate 1.000e-05, It/sec 0.497, Tokens/sec 102.573, Trained Tokens 19808, Peak mem 6.676 GB

Iter 150: Train loss 2.631, Learning Rate 1.000e-05, It/sec 0.540, Tokens/sec 106.255, Trained Tokens 29640, Peak mem 6.925 GB
Calculating loss...: 100%|███████████████████████████████████████████████| 25/25 [00:25<00:00,  1.02s/it]
Iter 200: Val loss 2.423, Val took 25.415s
Iter 200: Train loss 2.417, Learning Rate 1.000e-05, It/sec 0.527, Tokens/sec 104.640, Trained Tokens 39568, Peak mem 6.925 GB
Iter 250: Train loss 2.334, Learning Rate 1.000e-05, It/sec 0.527, Tokens/sec 108.559, Trained Tokens 49860, Peak mem 6.925 GB
Iter 300: Train loss 2.302, Learning Rate 1.000e-05, It/sec 0.515, Tokens/sec 103.435, Trained Tokens 59897, Peak mem 6.925 GB
Iter 350: Train loss 2.262, Learning Rate 1.000e-05, It/sec 0.524, Tokens/sec 102.263, Trained Tokens 69662, Peak mem 6.925 GB
Calculating loss...: 100%|███████████████████████████████████████████████| 25/25 [00:22<00:00,  1.11it/s]
Iter 400: Val loss 2.100, Val took 22.493s
Iter 400: Train loss 2.151, Learning Rate 1.000e-05, It/sec 0.493, Tokens/sec 104.097, Trained Tokens 80215, Peak mem 6.925 GB
Iter 450: Train loss 2.069, Learning Rate 1.000e-05, It/sec 0.542, Tokens/sec 105.975, Trained Tokens 89996, Peak mem 6.925 GB
Iter 500: Train loss 2.062, Learning Rate 1.000e-05, It/sec 0.558, Tokens/sec 100.474, Trained Tokens 99005, Peak mem 6.925 GB
Iter 500: Saved adapter weights to models/gemma-3-text-4b-it-q8-mlx/adapters.safetensors and models/gemma-3-text-4b-it-q8-mlx/0000500_adapters.safetensors.
Iter 550: Train loss 2.081, Learning Rate 1.000e-05, It/sec 0.528, Tokens/sec 99.331, Trained Tokens 108417, Peak mem 6.925 GB
Calculating loss...: 100%|███████████████████████████████████████████████| 25/25 [00:25<00:00,  1.03s/it]
Iter 600: Val loss 2.054, Val took 25.675s
Iter 600: Train loss 2.044, Learning Rate 1.000e-05, It/sec 0.550, Tokens/sec 100.671, Trained Tokens 117561, Peak mem 6.925 GB
Iter 650: Train loss 2.044, Learning Rate 1.000e-05, It/sec 0.528, Tokens/sec 105.138, Trained Tokens 127514, Peak mem 6.925 GB
Iter 700: Train loss 2.081, Learning Rate 1.000e-05, It/sec 0.592, Tokens/sec 104.065, Trained Tokens 136310, Peak mem 6.925 GB
Iter 750: Train loss 2.084, Learning Rate 1.000e-05, It/sec 0.543, Tokens/sec 104.676, Trained Tokens 145945, Peak mem 6.925 GB
Calculating loss...: 100%|███████████████████████████████████████████████| 25/25 [00:23<00:00,  1.08it/s]
Iter 800: Val loss 2.042, Val took 23.131s
Iter 800: Train loss 2.037, Learning Rate 1.000e-05, It/sec 0.559, Tokens/sec 106.730, Trained Tokens 155497, Peak mem 6.925 GB
Iter 850: Train loss 2.006, Learning Rate 1.000e-05, It/sec 0.554, Tokens/sec 105.860, Trained Tokens 165050, Peak mem 6.925 GB
Iter 900: Train loss 1.988, Learning Rate 1.000e-05, It/sec 0.571, Tokens/sec 102.819, Trained Tokens 174052, Peak mem 6.925 GB
Iter 950: Train loss 1.950, Learning Rate 1.000e-05, It/sec 0.528, Tokens/sec 106.645, Trained Tokens 184150, Peak mem 6.925 GB
Calculating loss...: 100%|███████████████████████████████████████████████| 25/25 [00:23<00:00,  1.07it/s]
Iter 1000: Val loss 2.035, Val took 23.401s
Iter 1000: Train loss 1.977, Learning Rate 1.000e-05, It/sec 0.516, Tokens/sec 108.454, Trained Tokens 194659, Peak mem 6.925 GB
Iter 1000: Saved adapter weights to models/gemma-3-text-4b-it-q8-mlx/adapters.safetensors and models/gemma-3-text-4b-it-q8-mlx/0001000_adapters.safetensors.
Iter 1050: Train loss 1.957, Learning Rate 1.000e-05, It/sec 0.530, Tokens/sec 103.828, Trained Tokens 204447, Peak mem 6.925 GB
Iter 1100: Train loss 2.017, Learning Rate 1.000e-05, It/sec 0.512, Tokens/sec 104.905, Trained Tokens 214696, Peak mem 6.925 GB
Iter 1150: Train loss 2.005, Learning Rate 1.000e-05, It/sec 0.503, Tokens/sec 102.962, Trained Tokens 224929, Peak mem 6.925 GB
Calculating loss...: 100%|███████████████████████████████████████████████| 25/25 [00:23<00:00,  1.05it/s]
Iter 1200: Val loss 1.894, Val took 23.915s
Iter 1200: Train loss 2.020, Learning Rate 1.000e-05, It/sec 0.472, Tokens/sec 99.709, Trained Tokens 235483, Peak mem 6.925 GB
Iter 1250: Train loss 1.941, Learning Rate 1.000e-05, It/sec 0.419, Tokens/sec 88.462, Trained Tokens 246027, Peak mem 6.925 GB
Iter 1300: Train loss 1.921, Learning Rate 1.000e-05, It/sec 0.531, Tokens/sec 97.488, Trained Tokens 255210, Peak mem 6.925 GB
Iter 1350: Train loss 1.921, Learning Rate 1.000e-05, It/sec 0.493, Tokens/sec 99.358, Trained Tokens 265289, Peak mem 6.925 GB
Calculating loss...: 100%|███████████████████████████████████████████████| 25/25 [00:26<00:00,  1.06s/it]
Iter 1367: Val loss 2.013, Val took 26.597s
Iter 1367: Train loss 1.964, Learning Rate 1.000e-05, It/sec 1.530, Tokens/sec 104.308, Trained Tokens 268698, Peak mem 6.925 GB
Saved final weights to models/gemma-3-text-4b-it-q8-mlx/adapters.safetensors.


python -m mlx_lm fuse \                                      
  --model alexgusevski/gemma-3-text-4b-it-q8-mlx \
  --adapter-path models/gemma-3-text-4b-it-q8-mlx \
  --save-path models/gemma-3-text-4b-it-q8-mlx_fused \
  --de-quantize

==>
Loading pretrained model
Fetching 8 files: 100%|█████████████████████████████████████████████████| 8/8 [00:00<00:00, 50006.61it/s]
De-quantizing model
README.md: 1.20kB [00:00, 1.20MB/s]


# CORRECT: Single line (recommended)
python -m mlx_lm generate --model models/gemma-3-text-4b-it-q8-mlx_fused --prompt "What is insulin resistance?"

# OR Multi-line (backslash MUST be last character, no trailing spaces!)
python -m mlx_lm generate \
  --model models/gemma-3-text-4b-it-q8-mlx_fused \
  --prompt "What is insulin resistance?"

## TROUBLESHOOTING: Line Continuation Issues

**Problem:** If you get `error: unrecognized arguments:` or `zsh: command not found: --model`, it's because:
- Trailing spaces after the backslash (`\`) break line continuation in zsh
- The backslash must be the **absolute last character** with no spaces after it

**Solutions:**

1. **Use single-line commands (easiest):**
```bash
python -m mlx_lm generate --model models/gemma-3-text-4b-it-q8-mlx_fused --prompt "What is insulin resistance?"
```

2. **Multi-line with correct backslash (no trailing spaces):**
```bash
python -m mlx_lm generate \
  --model models/gemma-3-text-4b-it-q8-mlx_fused \
  --prompt "What is insulin resistance?"
```

3. **Check for trailing spaces:** If copying from notes, make sure there are NO spaces after the `\`

---

## ROOT CAUSE ANALYSIS (Original quantization_config Error)

**The Problem:**
1. Base model (`alexgusevski/gemma-3-text-4b-it-q8-mlx`) has `quantization_config` with `group_size` and `bits` but NO `quant_method`
2. MLX's `fuse --de-quantize` dequantizes the weights but does NOT clean up `config.json`
3. MLX's `load()` function expects `quant_method` if `quantization_config` exists
4. Result: KeyError when trying to load the fused model

**Why This Happens:**
- MLX's fuse command copies the base model's `config.json` to the output directory
- When `--de-quantize` is used, it dequantizes weights but doesn't modify the config file
- This is a known limitation/bug in MLX - the config file doesn't reflect the dequantized state

**The Fix:**
1. **Immediate fix**: Manually remove `quantization_config` (and `quantization` if present) from `config.json`
2. **Permanent fix**: Updated `07_fuse_lora.py` to automatically clean up config.json after fusion when dequantizing

**Solution Applied:**
- Removed `quantization_config` from `models/gemma-3-text-4b-it-q8-mlx_fused/config.json`
    "quantization_config": {
        "group_size": 64,
        "bits": 8
    },

- Updated `scripts/phase4-fine-tune-model/07_fuse_lora.py` to automatically remove quantization fields post-fusion


Quick test (basic)
python -m mlx_lm generate \
  --model models/gemma-3-text-4b-it-q8-mlx_fused \
  --prompt "What is insulin resistance?"
python -m mlx_lm generate \  --model models/gemma-3-text-4b-it-q8-mlx_fused \  --prompt "What is insulin resistance?"

Common testing options
Adjust response length
# Short response (50 tokens)
python -m mlx_lm generate \
  --model models/gemma-3-text-4b-it-q8-mlx_fused \
  --prompt "What is fasting?" \
  --max-tokens 50

# Longer response (512 tokens - default)
python -m mlx_lm generate \
  --model models/gemma-3-text-4b-it-q8-mlx_fused \
  --prompt "Explain autophagy in detail" \
  --max-tokens 512

# Short response (50 tokens)
python -m mlx_lm generate \  --model models/gemma-3-text-4b-it-q8-mlx_fused \  --prompt "What is fasting?" \  --max-tokens 50

# Longer response (512 tokens - default)
python -m mlx_lm generate \  --model models/gemma-3-text-4b-it-q8-mlx_fused \  --prompt "Explain autophagy in detail" \  --max-tokens 512

Control randomness (temperature)
# More focused/deterministic (good for factual questions)
python -m mlx_lm generate \
  --model models/gemma-3-text-4b-it-q8-mlx_fused \
  --prompt "What is insulin resistance?" \
  --temp 0.3

# More creative (default is 0.7)
python -m mlx_lm generate \
  --model models/gemma-3-text-4b-it-q8-mlx_fused \
  --prompt "Explain the benefits of intermittent fasting" \
  --temp 0.9

# More focused/deterministic (good for factual questions)python -m mlx_lm generate \  --model models/gemma-3-text-4b-it-q8-mlx_fused \  --prompt "What is insulin resistance?" \  --temp 0.3# More creative (default is 0.7)python -m mlx_lm generate \  --model models/gemma-3-text-4b-it-q8-mlx_fused \  --prompt "Explain the benefits of intermittent fasting" \  --temp 0.9

Test before fusion (with LoRA adapters)
If you haven't fused yet, you can test with adapters:

python -m mlx_lm generate \
  --model alexgusevski/gemma-3-text-4b-it-q8-mlx \
  --adapter-path models/gemma-3-text-4b-it-q8-mlx \
  --prompt "What is insulin resistance?"

****

python -m mlx_lm generate \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --adapter-path models/Qwen2.5-1.5B-Instruct-4bit \
  --prompt "What is insulin resistance?"

****



python -m mlx_lm generate \  --model alexgusevski/gemma-3-text-4b-it-q8-mlx \  --adapter-path models/gemma-3-text-4b-it-q8-mlx \  --prompt "What is insulin resistance?"

Compare base vs fine-tuned
If you have the comparison script:

python scripts/compare_models.py "What is insulin resistance?" \
  --base-model alexgusevski/gemma-3-text-4b-it-q8-mlx \
  --fine-tuned-model models/gemma-3-text-4b-it-q8-mlx_fused

python scripts/compare_models.py "What is insulin resistance?" \  --base-model alexgusevski/gemma-3-text-4b-it-q8-mlx \  --fine-tuned-model models/gemma-3-text-4b-it-q8-mlx_fused


Quick reference
Basic command structure:
python -m mlx_lm generate \
  --model <model_path> \
  --prompt "<your question>" \
  [--max-tokens 512] \
  [--temp 0.7] \
  [--top-p 0.9]
python -m mlx_lm generate \  --model <model_path> \  --prompt "<your question>" \  [--max-tokens 512] \  [--temp 0.7] \  [--top-p 0.9]

For your Gemma model:
Fused model: models/gemma-3-text-4b-it-q8-mlx_fused
Base model: alexgusevski/gemma-3-text-4b-it-q8-mlx
Adapters: models/gemma-3-text-4b-it-q8-mlx
Start with a simple prompt to verify it's working.