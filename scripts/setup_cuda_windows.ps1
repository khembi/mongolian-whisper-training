# Windows + CUDA setup (run from project folder)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "Installing CUDA PyTorch..."
pip install "torch==2.6.0+cu124" "torchaudio==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124 --force-reinstall

Write-Host "Installing other deps (without upgrading torch)..."
pip install transformers "datasets[audio]>=2.19.0,<3.0.0" accelerate peft evaluate jiwer soundfile librosa tensorboard safetensors huggingface_hub --no-deps
pip install numpy packaging psutil pyyaml regex tokenizers tqdm filelock requests httpx hf-xet

Write-Host "Verifying CUDA..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('OK:', torch.__version__, torch.cuda.get_device_name(0))"
