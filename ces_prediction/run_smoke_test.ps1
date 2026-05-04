$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $scriptDir

$env:CES_EPOCHS = "1"
$env:CES_MAX_TRAIN_SAMPLES = "2000"
$env:CES_MAX_VAL_SAMPLES = "500"
$env:CES_BATCH_SIZE = "128"
$env:CES_TEMPORAL_SUBSETS = "1"
$env:CES_SPLIT_DIR = Join-Path $rootDir "data\.smoke_splits"
$env:CES_OUTPUT_DIR = Join-Path $rootDir "data\.smoke_outputs"

Write-Host "Running KSTAR CES smoke test..."
Write-Host "  CES_EPOCHS=$env:CES_EPOCHS"
Write-Host "  CES_MAX_TRAIN_SAMPLES=$env:CES_MAX_TRAIN_SAMPLES"
Write-Host "  CES_MAX_VAL_SAMPLES=$env:CES_MAX_VAL_SAMPLES"
Write-Host "  CES_BATCH_SIZE=$env:CES_BATCH_SIZE"
Write-Host "  CES_SPLIT_DIR=$env:CES_SPLIT_DIR"
Write-Host "  CES_OUTPUT_DIR=$env:CES_OUTPUT_DIR"

Push-Location $rootDir
try {
    python -m pytest -q
    python .\ces_prediction\train.py
}
finally {
    Pop-Location
}
