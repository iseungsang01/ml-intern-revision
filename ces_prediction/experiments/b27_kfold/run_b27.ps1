# B.27 launcher: start or resume the batch (B.27 + B.27c: 10 repeats x 5 folds x 2 populations = 100 runs; 20 done in B.27), then write data/.b27_summary.json.
# Safe to rerun at any time: finished folds (bootstrap_summary.json present) are skipped,
# a fold killed mid-training is retrained from scratch (weights are written only at the end).
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File ces_prediction\experiments\b27_kfold\run_b27.ps1
$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $repo
$log = Join-Path $repo "data\.b27_batch.log"

$running = Get-CimInstance Win32_Process -Filter "Name like 'py%.exe'" |
    Where-Object { $_.CommandLine -match "run_b27.py" }
if ($running) {
    Write-Host "B.27 is already running (PID $($running.ProcessId -join ', ')). Not starting a second batch."
    exit 1
}

$done = (Get-ChildItem -Path "data" -Directory -Filter ".b27_seqv2_*" -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName "bootstrap_summary.json") }).Count
Write-Host "B.27: $done / 100 folds finished. Resuming; log -> $log"
py ces_prediction\experiments\b27_kfold\run_b27.py --resume *>> $log
if ($LASTEXITCODE -ne 0) {
    Write-Host "B.27 exited with $LASTEXITCODE -- see $log (rerun this script to resume)."
    exit $LASTEXITCODE
}
Write-Host "B.27 complete: data\.b27_summary.json"
