# Windows + CUDA 12.8 for Blackwell (RTX PRO 6000, RTX 5090, sm_120)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "Installing PyTorch 2.7.1+cu128 (Blackwell / sm_120)..."
pip install "torch==2.7.1+cu128" "torchaudio==2.7.1+cu128" --index-url https://download.pytorch.org/whl/cu128 --force-reinstall

Write-Host "Installing training dependencies (with transitive deps)..."
pip install -r requirements-training.txt

Write-Host "Removing template torchvision (not used; conflicts with pinned torch)..."
pip uninstall -y torchvision 2>$null

Write-Host "Re-pinning PyTorch..."
pip install "torch==2.7.1+cu128" "torchaudio==2.7.1+cu128" --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps

Write-Host "Verifying CUDA + Blackwell arch..."
python -c @"
import torch
assert torch.cuda.is_available(), 'CUDA not available'
cap = torch.cuda.get_device_capability(0)
assert cap[0] >= 12, f'Expected Blackwell (CC 12.x), got {cap}'
print('OK:', torch.__version__, torch.cuda.get_device_name(0), 'CC', cap)
"@
