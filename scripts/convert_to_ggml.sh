#!/usr/bin/env bash
# Convert merged Hugging Face Whisper model to whisper.cpp ggml format for Remotion.
#
# Usage:
#   bash scripts/convert_to_ggml.sh ./merged-model ./ggml-large-v3-mn.bin
#
set -euo pipefail

MODEL_DIR="${1:-./merged-model}"
OUTPUT_BIN="${2:-./ggml-large-v3-mn.bin}"
WORK_DIR="${3:-./whisper-convert}"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

if [ ! -d "whisper.cpp" ]; then
  echo "==> Cloning whisper.cpp..."
  git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git
fi

if [ ! -d "whisper" ]; then
  echo "==> Cloning OpenAI whisper (needed for mel filters)..."
  git clone --depth 1 https://github.com/openai/whisper.git
fi

MODEL_DIR_ABS="$(cd "$(dirname "$MODEL_DIR")" && pwd)/$(basename "$MODEL_DIR")"
OUTPUT_BIN_ABS="$(cd "$(dirname "$OUTPUT_BIN")" 2>/dev/null && pwd)/$(basename "$OUTPUT_BIN")" || OUTPUT_BIN_ABS="$(pwd)/$(basename "$OUTPUT_BIN")"

echo "==> Converting HF model to ggml..."
python3 whisper.cpp/models/convert-h5-to-ggml.py \
  "$MODEL_DIR_ABS" \
  ./whisper \
  ./whisper.cpp/models

# Find the generated bin (convert script names it based on model type)
GENERATED=$(find ./whisper.cpp/models -name "ggml-*.bin" -newer ./whisper.cpp/models/convert-h5-to-ggml.py 2>/dev/null | head -1)
if [ -z "$GENERATED" ]; then
  GENERATED=$(ls -t ./whisper.cpp/models/ggml-*.bin 2>/dev/null | head -1)
fi

if [ -z "$GENERATED" ]; then
  echo "ERROR: Could not find generated ggml bin file."
  exit 1
fi

cp "$GENERATED" "$OUTPUT_BIN_ABS"
echo "==> Saved: $OUTPUT_BIN_ABS"
echo "==> Copy this file to your Remotion project's whisper.cpp/models/ folder."
