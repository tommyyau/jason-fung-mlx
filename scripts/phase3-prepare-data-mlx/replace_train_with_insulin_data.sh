#!/bin/bash
# Replace train.jsonl with insulin-focused data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/mlx_training_data"

echo "Replacing train.jsonl with insulin-focused data..."
echo ""

# Check if insulin-focused data exists
if [ ! -f "$DATA_DIR/train_insulin_focused.jsonl" ]; then
    echo "❌ Error: train_insulin_focused.jsonl not found!"
    echo "   Please run: python3 scripts/phase3-prepare-data-mlx/04d_generate_insulin_focused_data.py"
    exit 1
fi

# Backup existing train.jsonl
if [ -f "$DATA_DIR/train.jsonl" ]; then
    BACKUP_FILE="$DATA_DIR/train.jsonl.backup.$(date +%Y%m%d_%H%M%S)"
    echo "📦 Backing up existing train.jsonl to: $BACKUP_FILE"
    cp "$DATA_DIR/train.jsonl" "$BACKUP_FILE"
fi

# Replace train.jsonl
echo "🔄 Replacing train.jsonl with insulin-focused data..."
cp "$DATA_DIR/train_insulin_focused.jsonl" "$DATA_DIR/train.jsonl"

# Count lines
LINE_COUNT=$(wc -l < "$DATA_DIR/train.jsonl")
echo ""
echo "✅ Done! train.jsonl now contains $LINE_COUNT insulin-focused examples"
echo ""
echo "Next steps:"
echo "  1. Review a few examples: head -n 3 $DATA_DIR/train.jsonl"
echo "  2. Train your model: ./train_and_test_run9.sh"








