#!/usr/bin/env bash
# Linux + CUDA 12.4 for Ampere / Ada / Hopper (A100, L40S, RTX 4090, etc.)
set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/_install_training_deps.sh 2.6.0+cu124 2.6.0+cu124 cu124

echo "==> Verifying CUDA..."
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print('OK:', torch.__version__, torch.cuda.get_device_name(0), 'CC', torch.cuda.get_device_capability(0))
"

echo "==> Setup complete."
