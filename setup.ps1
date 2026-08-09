# One-time setup: create venv + install deps
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ""
Write-Host "Setup OK. Next:"
Write-Host "  1) .\pull_model.ps1   (download the OpenVINO vision model for the NPU)"
Write-Host "  2) .\run.ps1          (start the analyzer on http://localhost:31027)"