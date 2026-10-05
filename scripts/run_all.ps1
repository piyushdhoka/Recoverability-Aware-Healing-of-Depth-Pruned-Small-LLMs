# Run the whole pipeline for one model on this laptop (resumable: re-run after any crash).
#   Qwen laptop (RTX 4070):  .\scripts\run_all.ps1 -Model qwen
#   Llama laptop (RTX 4060): .\scripts\run_all.ps1 -Model llama
param([Parameter(Mandatory = $true)][string]$Model, [int]$From = 0)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$stages = @("00_check_env", "01_build_eval", "02_build_pools", "03_teacher", "04_diagnose",
            "05_pilots", "06_fit_allocate", "07_main")
for ($i = $From; $i -lt $stages.Count; $i++) {
    Write-Host "=== stage $($stages[$i]) ($Model) ==="
    python "scripts/$($stages[$i]).py" --model $Model
    if ($LASTEXITCODE -ne 0) { throw "stage $($stages[$i]) failed" }
}
Write-Host "Done. Commit results/$Model and push; then run: python scripts/08_analyze.py"
