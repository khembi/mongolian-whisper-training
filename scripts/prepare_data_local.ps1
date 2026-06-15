# Prepare dataset locally on Windows (CPU, no GPU required).
# Run overnight; then upload prepared-dataset/ to RunPod for GPU training only.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "==> Installing CPU dependencies (if needed)..."
pip install -q -r requirements.txt

Write-Host "==> Preparing train + validation splits (skip test for faster prep)..."
Write-Host "    This takes ~4-5 hours on CPU. Safe to run overnight."
Write-Host ""

python prepare_data.py `
  --output-dir ./prepared-dataset `
  --splits train,validation

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nPrepare FAILED." -ForegroundColor Red
    exit 1
}

Write-Host "`n==> Done. Upload prepared-dataset/ to RunPod:"
Write-Host "  scp -r prepared-dataset root@POD_IP:/workspace/mongolian-whisper-training/"
Write-Host "  bash scripts/runpod_train_only.sh"
