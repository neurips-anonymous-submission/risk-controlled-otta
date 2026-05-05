param(
    [string]$PythonExe = "python",
    [string]$ProjectRoot = "",
    [string]$OutputDir = "",
    [string[]]$Domains = @("sunlamp", "lightbox", "shirt")
)

$RepoRoot = Split-Path -Parent $PSScriptRoot

if (-not $ProjectRoot) { $ProjectRoot = $RepoRoot }
if (-not $OutputDir) { $OutputDir = "output\analysis_table3_group" }

$env:PYTHONPATH = $RepoRoot
Set-Location $RepoRoot

& $PythonExe -m risk_controlled_otta.experiments.analyze_table3_group_analysis `
    --project_root $ProjectRoot `
    --domains $Domains `
    --output_dir $OutputDir

exit $LASTEXITCODE


