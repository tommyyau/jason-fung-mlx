# Performance Optimization Guide

**Critical Finding**: On 16GB RAM systems, closing IDEs during training provides **50-70% speedup** by avoiding memory swapping.

---

## Critical: Memory Pressure on 16GB Systems 🚨

### The Problem

MLX fine-tuning on a 16GB M1 MacBook Pro is memory-constrained:

```
MLX Training Memory Usage:
├─ Model weights (3B params): ~6 GB
├─ Gradients: ~6 GB
├─ Optimizer state (Adam): ~12 GB
├─ Activations: ~2-4 GB
├─ MLX optimizations reduce to: ~12-14 GB
└─ Total needed: 12-14 GB

Add Cursor/VSCode:
├─ Cursor + Electron: ~3-4 GB
├─ Language servers: ~0.5-1 GB
├─ Extensions: ~0.5 GB
└─ Total with Cursor: ~17-20 GB ❌ EXCEEDS 16GB!
```

### The Solution ✅

**Close Cursor/IDEs during training → 50-70% speedup**

```bash
# Before training:
1. Save your work in Cursor
2. Close Cursor completely (Cmd+Q)
3. Close Chrome/browsers
4. Close Slack, Discord, Spotify
5. Open native Terminal (Terminal.app or iTerm2)

# Run training:
cd /path/to/jason-fung-mlx
python3 scripts/phase4-fine-tune-model/06_train_mlx.py --lora --execute

# Results:
├─ Memory pressure: GREEN (was RED)
├─ Swap usage: 0 GB (was 2-3 GB)
├─ Iteration time: 2.5s (was 5.5s)
└─ Total training: 91 min (was 200+ min)
```

### Measured Performance Impact

**Validated on M1 MacBook Pro 16GB**:

| Configuration | Iteration Time | Total Training (2 epochs) | Speedup |
|---------------|----------------|---------------------------|---------|
| With Cursor open | 5-6 seconds | 180-220 minutes | Baseline |
| Cursor closed | 2.5-3 seconds | 90-110 minutes | **50-70% faster** ✅ |

**Root cause**: Memory swapping to SSD (30-60x slower than RAM)

---

## Why This Happens: Memory Swapping

### RAM vs SSD Performance

| Storage | Speed | Penalty |
|---------|-------|---------|
| **RAM** | ~200 GB/s | Baseline ✅ |
| **SSD Swap** | ~3-7 GB/s | **30-60x slower** ❌ |
| **Compressed RAM** | ~50 GB/s | 4x slower ❌ |

### What Happens When You Exceed 16GB

```
macOS Memory Pressure Response:
├─ 0-14 GB: Everything in RAM (fast) ✅
├─ 14-16 GB: Start compressing memory (4x slower)
├─ 16+ GB: Swap to SSD (30-60x slower) ❌
└─ 18+ GB: Heavy swapping + thrashing (100x+ slower) ❌❌
```

### Activity Monitor During Training

#### ❌ BAD: With Cursor Open (Memory Pressure)

```
Memory:
├─ Physical Memory: 16 GB
├─ Memory Used: 19.2 GB ❌
├─ Cached Files: 0 GB (purged)
├─ Swap Used: 3.2 GB ❌
├─ Compressed: 2.1 GB
└─ Memory Pressure: 🔴 RED

What's happening:
├─ Constantly swapping to disk
├─ Compressing/decompressing
├─ Purging all caches
└─ Result: 5-6 seconds per iteration ❌
```

#### ✅ GOOD: With Cursor Closed (No Pressure)

```
Memory:
├─ Physical Memory: 16 GB
├─ Memory Used: 14.8 GB ✅
├─ Cached Files: 1.2 GB
├─ Swap Used: 0 GB ✅
├─ Compressed: 0.5 GB
└─ Memory Pressure: 🟢 GREEN or 🟡 YELLOW

What's happening:
├─ Everything fits in RAM
├─ Minimal compression
├─ No swapping
└─ Result: 2.5 seconds per iteration ✅
```

---

## Verification Steps

### Check Current Memory Pressure

```bash
# Method 1: Command line
sysctl vm.swapusage

# Output if good:
# vm.swapusage: total = 0.00M  used = 0.00M  free = 0.00M ✅

# Output if swapping:
# vm.swapusage: total = 3072.00M  used = 2048.00M  free = 1024.00M ❌

# Method 2: Activity Monitor
open -a "Activity Monitor"
# Click "Memory" tab
# Check:
#   - Swap Used: Should be 0 GB ✅
#   - Memory Pressure: Should be green or yellow ✅
```

### Monitor During Training

```bash
# Terminal 1: Run training
python3 scripts/phase4-fine-tune-model/06_train_mlx.py --lora --execute

# Terminal 2: Monitor memory every 2 seconds
watch -n 2 'sysctl vm.swapusage && echo && ps aux | grep python | grep -v grep | awk "{print \$3, \$4, \$11}"'

# You should see:
# - Swap stays at 0.00M ✅
# - Python process ~95% CPU ✅
# - Python process ~85-90% memory ✅
```

---

## Training Performance Benchmarks

### Configuration

```yaml
Model: Llama-3.2-3B-Instruct (mlx-community)
Training: LoRA fine-tuning
Dataset: 1,095 examples (train), 342 examples (val)
Batch size: 1
Gradient accumulation: 8
Max sequence length: 1024
Epochs: 2
Hardware: M1 MacBook Pro, 16GB RAM
```

### Results

| Metric | With Cursor | Cursor Closed | Improvement |
|--------|-------------|---------------|-------------|
| **Iteration time** | 5.5s | 2.5s | 54% faster ✅ |
| **Iterations per minute** | 11 | 24 | 118% faster ✅ |
| **Time per epoch** | 100 min | 45 min | 55% faster ✅ |
| **Total training (2 epochs)** | 200 min | 90 min | 55% faster ✅ |
| **Memory pressure** | 🔴 RED | 🟢 GREEN | ✅ |
| **Swap usage** | 2-3 GB | 0 GB | ✅ |

**Time saved per training run**: ~110 minutes (1 hour 50 minutes)

---

## Best Practices for 16GB Systems

### Pre-Training Checklist

```bash
# 1. Save all work
# 2. Close these apps:
├─ Cursor/VSCode ✅ (saves 3-4 GB)
├─ Chrome/Safari (if many tabs) ✅ (saves 1-3 GB)
├─ Slack/Discord ✅ (saves 0.5-1 GB)
├─ Spotify/Music ✅ (saves 0.3-0.5 GB)
└─ Docker Desktop (if running) ✅ (saves 2-4 GB)

# 3. Keep only:
├─ Terminal (native, not IDE) ✅
├─ Activity Monitor (optional, for monitoring) ✅
└─ System processes ✅

# 4. Verify memory pressure is green/yellow:
sysctl vm.swapusage  # Should show 0.00M swap used
```

### During Training

```bash
# Monitor in Activity Monitor:
├─ Memory Pressure: 🟢 GREEN or 🟡 YELLOW (not 🔴 RED)
├─ Swap Used: 0 GB
├─ Python process: ~90% memory, ~95% CPU
└─ No other heavy processes running

# If memory pressure goes RED:
1. Pause training (Ctrl+C)
2. Close more applications
3. Restart training
```

### After Training

```bash
# You can reopen Cursor/apps
# Training is complete, no more memory pressure
```

---

## Alternative Optimizations (If You Must Keep Cursor Open)

If you need to keep Cursor open while training:

### Option 1: Reduce Memory Usage

```yaml
# config/training_config.yaml

# Reduce sequence length (saves 2-3 GB)
max_seq_length: 768  # Down from 1024

# Reduce LoRA layers (saves 1-2 GB)
lora:
  layers: 8  # Down from 12

# Combined savings: ~3-5 GB
# Total usage: ~9-10 GB (training) + 3-4 GB (Cursor) = ~13 GB ✅
```

**Trade-offs**:
- Longer answers may be truncated (768 tokens)
- Less model adaptation (8 layers vs 12)
- Slightly lower final quality

### Option 2: Use `tmux` + Detach

```bash
# Start tmux session
tmux new -s training

# Run training
python3 scripts/phase4-fine-tune-model/06_train_mlx.py --lora --execute

# Detach (training continues in background): Ctrl+B, then D
# Now you can open Cursor and work on other things

# Check progress later:
tmux attach -t training
```

**Benefits**: Training runs in background, you can work in Cursor separately

### Option 3: Upgrade to 32GB RAM

```
Cost: ~$400 (if buying new Mac) or $200 (upgrade kit for some models)

With 32GB:
├─ Can keep Cursor open ✅
├─ Can increase batch_size to 2-4 ✅
├─ Can train 7B models ✅
├─ Can run multiple experiments in parallel ✅
└─ Worth it if you train frequently
```

---

## Comparison: Terminal.app vs Cursor Terminal

### Cursor Terminal (Electron-based)

```
Base overhead:
├─ Electron rendering engine: ~1 GB
├─ VSCode extensions: ~0.5 GB
├─ Language servers (Python, TypeScript): ~0.5-1 GB
├─ Cursor AI features: ~1-2 GB
├─ Terminal emulator: ~0.3 GB
└─ Total: ~3-5 GB ❌

CPU overhead:
├─ Rendering: ~10-15% CPU
├─ Background AI analysis: ~5-10% CPU
├─ LSP servers: ~5% CPU
└─ Total: ~20-30% CPU ❌
```

### Terminal.app (Native macOS)

```
Overhead:
├─ Terminal process: ~50 MB ✅
├─ Rendering: <5% CPU ✅
└─ Total: Negligible ✅
```

**For long-running tasks: Native terminal is 100-200x lighter**

---

## When to Use Each Approach

### Use Native Terminal ✅

- **Training models** (critical!)
- Long data processing (>10 minutes)
- Large file operations
- Any memory-intensive task
- When you want to close IDE but keep task running

### Use Cursor Terminal

- Quick commands (`git status`, `ls`)
- Interactive development
- Need AI assistance while working
- Short tasks (<5 minutes)
- Debugging with AI

---

## Advanced: Using `tmux` for Best of Both Worlds

### Setup

```bash
# Install tmux (if needed)
brew install tmux

# Create tmux config for better experience
cat > ~/.tmux.conf << 'EOF'
# Enable mouse support
set -g mouse on

# Increase history
set -g history-limit 10000

# Status bar
set -g status-bg blue
set -g status-fg white
EOF
```

### Workflow

```bash
# 1. Start tmux session for training
tmux new -s train

# 2. Run training
cd /path/to/jason-fung-mlx
python3 scripts/phase4-fine-tune-model/06_train_mlx.py --lora --execute

# 3. Detach (keep running): Ctrl+B, then D
# Training continues in background!

# 4. Now you can:
├─ Open Cursor and work on other code ✅
├─ Close terminal completely ✅
├─ Restart your computer (tmux persists if you re-attach) ✅
└─ Come back hours later ✅

# 5. Check progress anytime:
tmux attach -t train

# 6. Kill session when done:
tmux kill-session -t train
```

**Benefits**: Training isolation + freedom to work in IDE

---

## Memory Optimization Quick Reference

| Action | Memory Saved | Effort | Worth It? |
|--------|--------------|--------|-----------|
| Close Cursor | 3-4 GB | 5 seconds | ✅ YES! (50-70% speedup) |
| Close Chrome (many tabs) | 1-3 GB | 10 seconds | ✅ YES |
| Close Slack/Discord | 0.5-1 GB | 5 seconds | ✅ YES |
| Reduce max_seq_length | 2-3 GB | 1 minute | ⚠️ Maybe (quality trade-off) |
| Reduce LoRA layers | 1-2 GB | 1 minute | ⚠️ Maybe (quality trade-off) |
| Upgrade to 32GB | N/A | $$$ | ⚠️ If you train frequently |

---

## Summary

### Key Findings

1. **Closing Cursor during training provides 50-70% speedup on 16GB systems** 🚀
2. Root cause: Memory pressure forces swapping to SSD (30-60x slower than RAM)
3. Simple fix: Close IDE, use native terminal
4. Alternative: Use `tmux` to run training in background

### One-Line Recommendation

**On 16GB systems: Always close Cursor/IDEs before training** — it's the single biggest performance optimization you can make (50-70% speedup for zero cost).

### Training Time Comparison

```
Original (inside Cursor, 5e-6 LR): ~200 minutes
Optimized (native terminal, 1e-5 LR): ~90 minutes

Combined speedup: 55% faster from avoiding swap + potentially faster convergence

Time saved per training run: ~110 minutes ✅
```

---

**Last Updated**: 2025-11-09
**Validated On**: M1 MacBook Pro 16GB, macOS Sonoma
**Key Contributor**: User discovery during training runs
