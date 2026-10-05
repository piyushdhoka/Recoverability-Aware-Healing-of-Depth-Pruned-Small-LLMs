# Run the whole pipeline for one model on this laptop (resumable: re-run after any crash).
#   Qwen laptop (RTX 4070):  .\scripts\run_all.ps1 -Model qwen
#   Llama laptop (RTX 4060): .\scripts\run_all.ps1 -Model llama
#   SmolLM2 (any laptop):    .\scripts\run_all.ps1 -Model smollm
param([Parameter(Mandatory = $true)][string]$Model, [int]$From = 0)
# "Continue", not "Stop": in Windows PowerShell 5.1 any stderr text from python (warnings, progress bars)
# would otherwise abort the run. Failures are detected from python's exit code instead.
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
Set-Location (Split-Path $PSScriptRoot -Parent)
$stages = @("00_check_env", "01_build_eval", "02_build_pools", "03_teacher", "04_diagnose",
            "05_pilots", "06_fit_allocate", "07_main")
for ($i = $From; $i -lt $stages.Count; $i++) {
    Write-Host "=== stage $($stages[$i]) ($Model) $(Get-Date -Format 'HH:mm:ss') ==="
    python "scripts/$($stages[$i]).py" --model $Model
    if ($LASTEXITCODE -ne 0) { Write-Host "STAGE FAILED: $($stages[$i]) (exit $LASTEXITCODE). Re-run the same command to resume."; exit 1 }
}
Write-Host "Done. Commit results/$Model and push; then run: python scripts/08_analyze.py"
