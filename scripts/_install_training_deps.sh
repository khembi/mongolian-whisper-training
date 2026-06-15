#!/usr/bin/env bash
# Shared: install pinned torch, full training deps, then re-pin torch.
# Usage: _install_training_deps.sh <torch-version> <torchaudio-version> <cu-index>
# Example: _install_training_deps.sh 2.7.1+cu128 2.7.1+cu128 cu128
set -euo pipefail

TORCH_VER="${1:?torch version required}"
TORCHAUDIO_VER="${2:?torchaudio version required}"
CUDA_INDEX="${3:?cuda index required, e.g. cu128}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INDEX_URL="https://download.pytorch.org/whl/${CUDA_INDEX}"

pip install --upgrade pip

echo "==> Installing PyTorch ${TORCH_VER} (${CUDA_INDEX})..."
pip install "torch==${TORCH_VER}" "torchaudio==${TORCHAUDIO_VER}" \
  --index-url "$INDEX_URL" --force-reinstall

echo "==> Installing training dependencies (with transitive deps)..."
pip install -r "${ROOT}/requirements-training.txt"

echo "==> Removing template torchvision (not used; conflicts with pinned torch)..."
pip uninstall -y torchvision 2>/dev/null || true

echo "==> Re-pinning PyTorch (prevent pip from swapping CUDA build)..."
pip install "torch==${TORCH_VER}" "torchaudio==${TORCHAUDIO_VER}" \
  --index-url "$INDEX_URL" --force-reinstall --no-deps

echo "==> Verifying imports..."
python -c "
import lazy_loader, librosa, peft, transformers, datasets, evaluate, torch
print('OK:', torch.__version__, 'librosa', librosa.__version__)
"
