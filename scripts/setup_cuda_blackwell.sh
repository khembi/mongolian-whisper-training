#!/usr/bin/env bash
# Linux + CUDA 12.8 for Blackwell (RTX PRO 6000, RTX 5090, sm_120)
set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/_install_training_deps.sh 2.7.1+cu128 2.7.1+cu128 cu128

echo "==> Verifying CUDA + Blackwell arch..."
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
cap = torch.cuda.get_device_capability(0)
assert cap[0] >= 12, f'Expected Blackwell (CC 12.x), got {cap}'
print('OK:', torch.__version__, torch.cuda.get_device_name(0), 'CC', cap)
"

echo "==> Setup complete."
