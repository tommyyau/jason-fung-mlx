# Jason Fung MLX Fine-Tuning Pipeline

A complete pipeline for fine-tuning language models on domain-specific content using MLX (Apple's ML framework optimized for Apple Silicon). This project fine-tunes models on Dr. Jason Fung's medical education content to produce a model that mimics his teaching style and communication patterns.

## Overview

This pipeline demonstrates a sophisticated **two-pass question-answer extraction approach** that converts raw video transcripts into high-quality, well-formatted Q&A pairs suitable for instruction tuning.

**Key Achievement**: 99.9% properly formatted training data (1,366/1,367 examples) through careful two-pass processing.

## Features

- ✅ Two-pass Q&A extraction (questions → answers) for superior data quality
- ✅ MLX LoRA training optimized for 16GB Apple Silicon Macs
- ✅ Memory-efficient configuration (batch size 1 + gradient accumulation)
- ✅ Catastrophic forgetting prevention (conservative learning rate, reduced epochs)
- ✅ Multi-format model export (MLX, HuggingFace, GGUF)
- ✅ Comprehensive documentation of lessons learned

## Project Structure

```
jason-fung-mlx/
├── scripts/
│   ├── phase1-extract-transcripts/    # YouTube transcript extraction
│   ├── phase2-refine-raw-data/        # Two-pass Q&A generation
│   ├── phase3-prepare-data-mlx/       # MLX format conversion & splits
│   ├── phase4-fine-tune-model/        # Training & LoRA fusion
│   └── phase5-convert-model-formats/  # HuggingFace & GGUF export
├── data/                              # Datasets (gitignored)
├── models/                            # Fine-tuned models (gitignored)
├── docs/                              # Journey documentation
│   ├── TRAINING_GUIDE.md
│   ├── DATA_REFINEMENT_JOURNEY.md
│   ├── FINE_TUNING_SAGA.md
│   └── MISSING_LESSONS.md
├── CLAUDE.md                          # Claude Code guidance
├── README_DATA_REFINEMENT.md          # Expert audit & critique
└── AUDIT_*.md                         # Comprehensive audit reports
```

## Quick Start

### Prerequisites

- Apple Silicon Mac (M1/M2/M3 or newer)
- 16GB RAM minimum
- Python 3.10+
- OpenAI API key (for Q&A generation)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/jason-fung-mlx.git
cd jason-fung-mlx

# Install dependencies
pip install -r requirements.txt

# Verify MLX installation
python3 -c "import mlx.core as mx; import mlx_lm; print('✓ MLX installed successfully')"

# Create .env file
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Training Pipeline

Run scripts in order (each phase depends on the previous):

```bash
# Phase 1: Extract transcripts
python3 scripts/phase1-extract-transcripts/01_get_channel_videos.py
python3 scripts/phase1-extract-transcripts/02_fetch_videos.py

# Phase 2: Generate Q&A pairs (two-pass approach)
python3 scripts/phase2-refine-raw-data/01_extract_questions.py
python3 scripts/phase2-refine-raw-data/03_generate_answers.py

# Phase 3: Prepare training data
python3 scripts/phase3-prepare-data-mlx/04_convert_answers_to_mlx.py
python3 scripts/phase3-prepare-data-mlx/05_split_train_val.py

# Phase 4: Train model
python3 scripts/phase4-fine-tune-model/06_train_mlx.py --model mlx-community/Llama-3.2-3B-Instruct --lora

# Phase 5: Export to other formats
python3 scripts/phase4-fine-tune-model/07_fuse_lora.py
python3 scripts/phase5-convert-model-formats/08_convert_to_hf.py
python3 scripts/phase5-convert-model-formats/09_convert_to_gguf.py
```

### Testing the Model

```bash
# Test with MLX
python -m mlx_lm.generate \
  --model models/jason_fung_mlx \
  --prompt "What is insulin resistance?"
```

## Training Configuration

Optimized for 16GB RAM Apple Silicon:

- **Model**: Llama-3.2-3B-Instruct (3B parameters)
- **Method**: LoRA (0.216% trainable parameters)
- **Batch Size**: 1 (with gradient accumulation of 8)
- **Learning Rate**: 5e-6 (conservative to prevent forgetting)
- **Epochs**: 2
- **Max Sequence Length**: 1024 tokens

See `docs/TRAINING_GUIDE.md` for detailed parameters.

## Key Insights

### Two-Pass Pipeline (Critical Architecture Decision)

**Pass 1**: Extract numbered questions from full transcripts
**Pass 2**: Generate formatted answers with explicit markdown instructions

**Why this matters**: Single-pass extraction produces unformatted, inconsistent data. Two-pass allows independent quality control and achieves 99.9% formatting compliance.

### Fine-Tuning vs RAG

**Fine-tuning teaches**: Communication style, response patterns, formatting habits
**Fine-tuning does NOT teach**: New factual knowledge (use RAG for this)

See `docs/DATA_REFINEMENT_JOURNEY.md` for the full story of how we learned this.

## Documentation

- **`CLAUDE.md`** - Guidance for Claude Code when working in this repo
- **`README_DATA_REFINEMENT.md`** - Expert critique and recommendations (start here for improvements)
- **`docs/TRAINING_GUIDE.md`** - Complete MLX training guide
- **`docs/DATA_REFINEMENT_JOURNEY.md`** - 10 lessons learned from data pipeline evolution
- **`docs/FINE_TUNING_SAGA.md`** - Training challenges and solutions
- **`AUDIT_REPORT.md`** - Comprehensive technical audit (1,216 lines)

## Current Status & Known Issues

**Status**: B+ (Strong foundation with critical gaps)

**What Works**:
- ✅ Two-pass pipeline produces high-quality data
- ✅ Training completes successfully on 16GB RAM
- ✅ Multiple export formats supported

**What Needs Work** (see `README_DATA_REFINEMENT.md`):
- ⚠️ No training metrics capture (can't see if training worked)
- ⚠️ No evaluation script (can't test model quality systematically)
- ⚠️ Learning rate not validated (may be too conservative)
- ⚠️ ~5% duplicate data in dataset

**Roadmap**: See `README_DATA_REFINEMENT.md` for 3-week improvement plan (8-10 hours for critical fixes).

## Results

From actual training run:
- **Final Training Loss**: ~1.438 (started at 3.396)
- **Final Validation Loss**: ~1.804
- **Training Time**: Several hours on M1 MacBook Pro
- **Peak Memory**: ~7.6 GB
- **Trainable Parameters**: 6.947M / 3,212.750M (0.216%)

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Built with [MLX](https://github.com/ml-explore/mlx) by Apple
- Uses [MLX-LM](https://github.com/ml-explore/mlx-examples/tree/main/llms) for training
- Fine-tuned on educational content from Dr. Jason Fung

## Contributing

This is a personal learning project, but feedback and suggestions are welcome via issues.

## Citation

If you use this pipeline or methodology in your work, please cite:

```bibtex
@software{jason_fung_mlx_2025,
  title = {Jason Fung MLX Fine-Tuning Pipeline},
  author = {Tommy Yau},
  year = {2025},
  url = {https://github.com/yourusername/jason-fung-mlx}
}
```
