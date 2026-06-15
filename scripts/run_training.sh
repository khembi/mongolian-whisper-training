#!/usr/bin/env bash
# Full training pipeline: train -> merge -> evaluate
set -euo pipefail

cd "$(dirname "$0")/.."

BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
PREPARED_DATASET="${PREPARED_DATASET:-}"

TRAIN_ARGS=(
  --output-dir "$OUTPUT_DIR"
  --batch-size "$BATCH_SIZE"
  --grad-accum 2
  --epochs "$EPOCHS"
  --bf16
)
if [ -n "$PREPARED_DATASET" ]; then
  TRAIN_ARGS+=(--prepared-dataset "$PREPARED_DATASET")
fi

echo "==> Step 1/3: Training LoRA..."
python train.py "${TRAIN_ARGS[@]}"

echo "==> Step 2/3: Merging LoRA weights..."
python merge_lora.py \
  --adapter-dir "$OUTPUT_DIR/best" \
  --output-dir ./merged-model

echo "==> Step 3/3: Evaluating merged model..."
python run_eval.py --model-dir ./merged-model

echo ""
echo "Training pipeline complete."
echo "  LoRA checkpoint: $OUTPUT_DIR/best"
echo "  Merged model:    ./merged-model"
echo ""
echo "Next: bash scripts/convert_to_ggml.sh ./merged-model ./ggml-large-v3-mn.bin"
