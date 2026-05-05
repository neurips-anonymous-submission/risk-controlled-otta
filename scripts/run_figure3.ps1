param(
    [string]$PythonExe = "python"
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = $RepoRoot
Set-Location $RepoRoot

& $PythonExe ".\risk_controlled_otta\visualize\plot_figure3_main_v2.py"

exit $LASTEXITCODE


