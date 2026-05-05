param(
    [string]$PythonExe = "python",
    [string]$DataRoot = "",
    [string]$SourceCheckpoint = "",
    [string]$OutputDir = "",
    [string[]]$Domains = @("sunlamp", "lightbox", "shirt"),
    [double]$BenefitDelta = 0.05,
    [int]$Folds = 5
)

$RepoRoot = Split-Path -Parent $PSScriptRoot

if (-not $DataRoot) { $DataRoot = Join-Path $RepoRoot "speedplusv2" }
if (-not $SourceCheckpoint) { $SourceCheckpoint = Join-Path $RepoRoot "output\dinov3_heatmap_source_v2\best_source_dino_heatmap.pth" }
if (-not $OutputDir) { $OutputDir = Join-Path $RepoRoot "output\analysis_table1_risk_benefit" }

$env:PYTHONPATH = $RepoRoot
Set-Location $RepoRoot

& $PythonExe -m risk_controlled_otta.experiments.analyze_table1_risk_prediction `
    --data_root $DataRoot `
    --domains $Domains `
    --source_checkpoint $SourceCheckpoint `
    --output_dir $OutputDir `
    --label_mode beneficial_update `
    --benefit_delta $BenefitDelta `
    --folds $Folds `
    --num_workers 0

exit $LASTEXITCODE


