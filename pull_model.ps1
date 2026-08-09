# Download a pre-exported OpenVINO vision-language checkpoint for the NPU.
# Default (NF4) requires a Core Ultra Series 2 (Lunar Lake) NPU or newer.
# For older Meteor Lake NPUs use an INT4 export (see export_model.ps1).
# Falls back to https://hf-mirror.com if huggingface.co is unreachable.
param(
    [string]$Repo = "0ldev/Qwen2.5-VL-3B-Instruct-ov-nf4-npu",
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
& .\.venv\Scripts\python.exe download_model.py
if ($LASTEXITCODE -ne 0) {
    throw "Model download failed on all endpoints. Check the network / proxy, then re-run."
}

# Validate the checkpoint is an OpenVINO GenAI VLM (so VLMPipeline can load it)
$markers = @("openvino_config.json", "openvino_model.xml", "openvino_language_model.xml")
$found = $markers | Where-Object { Test-Path (Join-Path $target $_) }
if (-not $found) {
    throw "Downloaded folder does not look like an OpenVINO GenAI VLM: $target"
}

Write-Host ""
Write-Host "Model -> $target"