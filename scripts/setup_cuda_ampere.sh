#!/usr/bin/env bash
# Linux + CUDA 12.4 for Ampere / Ada / Hopper (A100, L40S, RTX 4090, etc.)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Installing PyTorch 2.6.0+cu124 (Ampere–Hopper)..."
pip install --upgrade pip
pip install "torch==2.6.0+cu124" "torchaudio==2.6.0+cu124" \
  --index-url https://download.pytorch.org/whl/cu124 --force-reinstall

echo "==> Installing other deps (without upgrading torch)..."
pip install transformers "datasets[audio]>=2.19.0,<3.0.0" accelerate peft evaluate jiwer \
  soundfile librosa tensorboard safetensors huggingface_hub --no-deps
pip install numpy packaging psutil pyyaml regex tokenizers tqdm filelock requests httpx hf-xet

echo "==> Verifying CUDA..."
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print('OK:', torch.__version__, torch.cuda.get_device_name(0), 'CC', torch.cuda.get_device_capability(0))
"

echo "==> Setup complete."
