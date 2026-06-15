# Windows + CUDA 12.8 for Blackwell (RTX PRO 6000, RTX 5090, sm_120)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "Installing PyTorch 2.7.1+cu128 (Blackwell / sm_120)..."
pip install "torch==2.7.1+cu128" "torchaudio==2.7.1+cu128" --index-url https://download.pytorch.org/whl/cu128 --force-reinstall

Write-Host "Installing other deps (without upgrading torch)..."
pip install transformers "datasets[audio]>=2.19.0,<3.0.0" accelerate peft evaluate jiwer soundfile librosa tensorboard safetensors huggingface_hub --no-deps
pip install numpy packaging psutil pyyaml regex tokenizers tqdm filelock requests httpx hf-xet

Write-Host "Verifying CUDA + Blackwell arch..."
python -c @"
import torch
assert torch.cuda.is_available(), 'CUDA not available'
cap = torch.cuda.get_device_capability(0)
assert cap[0] >= 12, f'Expected Blackwell (CC 12.x), got {cap}'
print('OK:', torch.__version__, torch.cuda.get_device_name(0), 'CC', cap)
"@
