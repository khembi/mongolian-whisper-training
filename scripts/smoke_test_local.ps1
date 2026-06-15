# Local smoke test on Windows (before RunPod)
Set-Location $PSScriptRoot\..

Write-Host "Installing dependencies..."
pip install -r requirements.txt -q

Write-Host "`nRunning smoke test (dataset + inference)..."
python smoke_test.py --skip-training

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nSmoke test PASSED. You can proceed to RunPod."
} else {
    Write-Host "`nSmoke test FAILED. Fix errors above before RunPod."
    exit 1
}
