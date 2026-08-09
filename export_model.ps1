# Export a Hugging Face VLM to OpenVINO IR (INT4) for the NPU, using a
# separate export venv so the runtime venv stays lean.
param(
    [string]$ModelId = "Qwen/Qwen2.5-VL-3B-Instruct",
    [string]$OutDir = "$PSScriptRoot\models\qwen2.5-vl-3b-instruct-ov"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv-export")) {
    python -m venv .venv-export
}

& .\.venv-export\Scripts\python.exe -m pip install --upgrade pip
& .\.venv-export\Scripts\python.exe -m pip install "optimum-intel[openvino,export-transformers]" openvino nncf onnx torch torchvision

& .\.venv-export\Scripts\python.exe -m optimum.cli export openvino `
    --model $ModelId `
    --trust-remote-code `
    --weight-format int4 `
    --sym `
    --ratio 1.0 `
    --group-size -1 `
    $OutDir

Write-Host ""
Write-Host "Exported -> $OutDir"
Write-Host "Point the analyzer at it with the env var VLM_MODEL."