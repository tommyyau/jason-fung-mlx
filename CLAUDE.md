# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a complete pipeline for fine-tuning language models on domain-specific content using MLX (Apple's ML framework optimized for Apple Silicon). The project fine-tunes models on Dr. Jason Fung's medical education content to produce a model that mimics his teaching style and communication patterns.

**Key Architecture Decision**: This is a two-pass question-answer extraction pipeline that converts raw video transcripts into high-quality, well-formatted Q&A pairs suitable for instruction tuning. Fine-tuning teaches style and patterns, not facts - for factual knowledge, use RAG (Retrieval Augmented Generation).

## Project Structure

The pipeline is organized into 5 sequential phases:

- `scripts/phase1-extract-transcripts/` - Extract video transcripts from YouTube
- `scripts/phase2-refine-raw-data/` - Two-pass Q&A extraction (questions → answers)
- `scripts/phase3-prepare-data-mlx/` - Convert to MLX training format and split datasets
- `scripts/phase4-fine-tune-model/` - MLX LoRA training and adapter fusion
- `scripts/phase5-convert-model-formats/` - Convert to HuggingFace and GGUF formats

- `data/` - All datasets (transcripts, questions, answers, training splits)
- `models/` - Fine-tuned models in various formats (MLX, fused, HuggingFace, GGUF)
- `docs/` - Detailed documentation of the journey and lessons learned

## Development Commands

### Environment Setup

```bash
# Install all dependencies
pip install -r requirements.txt

# Verify MLX installation
python3 -c "import mlx.core as mx; import mlx_lm; print('✓ MLX installed successfully')"
```

### Data Pipeline Execution

Run scripts in order. Each phase depends on the previous phase's output:

```bash
# Phase 1: Extract transcripts from YouTube videos
python3 scripts/phase1-extract-transcripts/01_get_channel_videos.py
python3 scripts/phase1-extract-transcripts/02_fetch_videos.py

# Phase 2: Two-pass Q&A extraction
python3 scripts/phase2-refine-raw-data/01_extract_questions.py
python3 scripts/phase2-refine-raw-data/02_verify-json-format-from-extracted-questions.py
python3 scripts/phase2-refine-raw-data/03_generate_answers.py
python3 scripts/phase2-refine-raw-data/04_make_answers_readable_in_markdown.py

# Phase 3: Prepare MLX training data
python3 scripts/phase3-prepare-data-mlx/04_convert_answers_to_mlx.py
python3 scripts/phase3-prepare-data-mlx/05_split_train_val.py

# Phase 4: Training
python3 scripts/phase4-fine-tune-model/06_train_mlx.py --model mlx-community/Llama-3.2-3B-Instruct --lora
python3 scripts/phase4-fine-tune-model/07_fuse_lora.py

# Phase 5: Model conversion
python3 scripts/phase5-convert-model-formats/08_convert_to_hf.py
python3 scripts/phase5-convert-model-formats/09_convert_to_gguf.py
```

### Testing the Fine-Tuned Model

```bash
# Test with MLX
python -m mlx_lm.generate \
  --model models/jason_fung_mlx \
  --prompt "What is insulin resistance?"

# Test fused model
python -m mlx_lm.generate \
  --model models/jason_fung_mlx_fused \
  --prompt "What is insulin resistance?"
```

## Training Configuration

The training setup is optimized for 16GB RAM Apple Silicon Macs:

**Model**: `mlx-community/Llama-3.2-3B-Instruct` (3B parameters, Llama 3.2 architecture)
**Training Method**: LoRA (Low-Rank Adaptation) - only 0.216% of parameters are trainable
**Memory Optimizations**:
- Batch size: 1 (with gradient accumulation of 8 = effective batch size of 8)
- Max sequence length: 1024 tokens
- Gradient checkpointing: enabled
- LoRA layers: 12 (reduced from 16 to preserve base model capabilities)

**Key Parameters** (see `docs/TRAINING_GUIDE.md` for details):
- Learning rate: 1e-5 (conservative to prevent catastrophic forgetting)
- Epochs: 2 (reduced from 3 to prevent overfitting)
- LoRA rank: 8, alpha: 8 (balanced for style learning without forgetting)
- Gradient accumulation: 8 steps (for stable gradients)

**Training Monitoring**:
- Validation every 50 steps (`--steps-per-eval 50`)
- Progress reports every 50 steps (`--steps-per-report 50`)
- Checkpoints saved every 500 steps (`--save-every 500`)

## Critical Architecture Insights

### Two-Pass Question-Answer Pipeline

The pipeline uses a two-pass approach that is critical to quality:

1. **Pass 1: Question Extraction** (`01_extract_questions.py`)
   - Extracts numbered questions from full transcripts
   - Maintains full context (no chunking - chunking breaks continuity)
   - Questions are numbered for reliable batch synchronization

2. **Pass 2: Answer Generation** (`03_generate_answers.py`)
   - Generates fully formatted answers for each question
   - Uses full transcript as context for each answer
   - Applies explicit formatting instructions (markdown, bold, lists, paragraphs)
   - Critical: Format is learned from training data, NOT from inference prompts

**Why Two-Pass?**
- Single-pass extraction produced messy, unformatted outputs
- Format quality in training data directly impacts model output format
- Separating concerns allows better quality control at each stage

### Fine-Tuning vs RAG

**What Fine-Tuning Does**: Teaches communication style, response patterns, formatting habits
**What Fine-Tuning Doesn't Do**: Inject new factual knowledge (use RAG for this)

**Evidence from This Project**:
- Attempted medical fact validation in Phase 5 - wrong approach
- Fine-tuning learns "how to respond" not "what to respond"
- For factual accuracy, use RAG with validated sources

### Data Format Requirements

Training data must be in MLX chat format:

```json
{"messages": [
  {"role": "user", "content": "question text"},
  {"role": "assistant", "content": "formatted answer with **bold**, lists, etc."}
]}
```

The conversion happens in `scripts/phase3-prepare-data-mlx/04_convert_answers_to_mlx.py`.

### Catastrophic Forgetting Prevention

The model is tuned with several safeguards against catastrophic forgetting:

1. Conservative learning rate (1e-5 vs typical 5e-5)
2. Fewer training epochs (2 vs 3)
3. Reduced LoRA layers (12 vs 16) - preserves more base model
4. Lower LoRA alpha (8 vs 16) - reduces adaptation strength
5. Gradient accumulation (8 steps) - smoother updates

See `docs/FINE_TUNING_SAGA.md` for the full story of discovering and addressing this issue.

## Model Outputs

After training completes, the following artifacts are produced:

- `models/jason_fung_mlx/` - LoRA adapters (6.947M trainable parameters)
  - `adapters.safetensors` - Final LoRA weights
  - `adapter_config.json` - LoRA configuration
  - Checkpoints: `0000500_adapters.safetensors`, `0001000_adapters.safetensors`, etc.

- `models/jason_fung_mlx_fused/` - Fused model (LoRA merged with base)
  - Full 3B parameter model with adapters merged
  - Ready for MLX inference

- `models/jason_fung_mlx_hf/` - HuggingFace format
  - Compatible with transformers library
  - Can be uploaded to HuggingFace Hub

- `models/*.gguf` - GGUF format files
  - `jason-fung-llama-F16.gguf` - Full precision (6.4GB)
  - `jason-fung-llama-Q4_K_M.gguf` - Quantized (2GB)
  - Compatible with llama.cpp, LM Studio, and other inference engines
  - Note: Naming may show "granite" from earlier testing, but contains Llama 3.2 3B weights

## Environment Variables

Required in `.env` file:

```
OPENAI_API_KEY=your_key_here  # For question/answer generation (Phase 2)
```

The scripts search for `.env` in multiple locations:
- Project root
- `data/.env`
- Current directory

## Common Pitfalls

1. **Don't chunk transcripts for fine-tuning** - Use full context. Chunking works for RAG, not fine-tuning.

2. **Don't skip the two-pass pipeline** - Single-pass extraction produces poor formatting.

3. **Don't increase learning rate without careful testing** - Catastrophic forgetting is real.

4. **Don't expect factual knowledge injection** - Fine-tuning teaches style, use RAG for facts.

5. **Don't skip monitoring during training** - Use `--steps-per-report` and `--steps-per-eval` to catch issues early.

6. **Don't train without validation data** - The 80/20 train/val split is intentional.

7. **Don't use incompatible models** - Granite fails GGUF conversion, Gemma 2 9B exceeds 16GB RAM. See `docs/MLX_MODEL_COMPATIBILITY.md`.

## Key Documentation

- `docs/TRAINING_GUIDE.md` - Complete MLX training guide with all parameters
- `docs/MLX_MODEL_COMPATIBILITY.md` - **Model compatibility guide** (why Granite and Gemma don't work)
- `docs/DATA_REFINEMENT_JOURNEY.md` - Evolution of the data pipeline (10 lessons learned)
- `docs/FINE_TUNING_SAGA.md` - Training challenges and solutions
- `docs/MISSING_LESSONS.md` - What guides don't tell you about fine-tuning

These documents contain hard-won lessons from the development process and should be consulted when making changes to the pipeline or training configuration.
