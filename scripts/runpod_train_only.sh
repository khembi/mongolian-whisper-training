#!/usr/bin/env bash
# RunPod GPU-only training — use after uploading prepared-dataset/ from your PC.
#
# On pod:
#   cd /workspace/mongolian-whisper-training
#   bash scripts/setup_cuda_blackwell.sh   # or setup_cuda_ampere.sh
#   bash scripts/runpod_train_only.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

PREPARED_DATASET="${PREPARED_DATASET:-./prepared-dataset}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
CONVERT_GGML="${CONVERT_GGML:-1}"

if [ ! -d "$PREPARED_DATASET/train" ] && [ ! -f "$PREPARED_DATASET/dataset_dict.json" ]; then
  echo "ERROR: Prepared dataset not found at $PREPARED_DATASET"
  echo "Upload from your PC first:"
  echo "  scp -r prepared-dataset root@POD_IP:/workspace/mongolian-whisper-training/"
  exit 1
fi

echo "==> Step 1/3: Training LoRA (prepared data, no preprocessing)..."
python train.py \
  --prepared-dataset "$PREPARED_DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum 2 \
  --epochs "$EPOCHS" \
  --bf16

echo "==> Step 2/3: Merging LoRA weights..."
python merge_lora.py \
  --adapter-dir "$OUTPUT_DIR/best" \
  --output-dir ./merged-model

echo "==> Step 3/3: Evaluating merged model..."
if [ -d "$PREPARED_DATASET/test" ] || [ -f "$PREPARED_DATASET/dataset_dict.json" ]; then
  python run_eval.py --model-dir ./merged-model
else
  echo "(skipped WER eval — test split not in prepared data)"
fi

if [ "$CONVERT_GGML" = "1" ]; then
  echo "==> Converting to GGML for Remotion..."
  apt-get update -qq && apt-get install -y -qq git >/dev/null 2>&1 || true
  bash scripts/convert_to_ggml.sh ./merged-model ./ggml-large-v3-mn.bin
fi

echo ""
echo "Training pipeline complete."
echo "  LoRA checkpoint: $OUTPUT_DIR/best"
echo "  Merged model:    ./merged-model"
[ "$CONVERT_GGML" = "1" ] && echo "  Remotion model:  ./ggml-large-v3-mn.bin"
