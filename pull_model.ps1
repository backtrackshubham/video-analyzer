# Download a pre-exported OpenVINO vision-language checkpoint for the NPU.
# Defaults to an INT4 variant that runs on both Meteor Lake and Lunar Lake NPUs.
param(
    [string]$Repo = "llmware/qwen2-vl-2b-instruct-ov",
    [string]$OutDir = "$PSScriptRoot\models"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Venv missing. Run .\setup.ps1 first."
}

& .\.venv\Scripts\python.exe -m pip show huggingface_hub *> $null
if ($LASTEXITCODE -ne 0) {
    & .\.venv\Scripts\python.exe -m pip install huggingface_hub
}

$target = Join-Path $OutDir ([IO.Path]::GetFileName($Repo))
$env:HF_REPO = $Repo
$env:HF_TARGET = $target
& .\.venv\Scripts\python.exe -c "import os; from huggingface_hub import snapshot_download; print(snapshot_download(os.environ['HF_REPO'], local_dir=os.environ['HF_TARGET']))"

Write-Host ""
Write-Host "Model -> $target"