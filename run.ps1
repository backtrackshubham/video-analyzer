# Start the analyzer (FastAPI on http://localhost:31027)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Venv missing. Run .\setup.ps1 first."
}

& .\.venv\Scripts\python.exe main.py