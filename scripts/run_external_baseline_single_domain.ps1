param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("cotta", "eata")]
    [string]$Method,

    [Parameter(Mandatory = $true)]
    [ValidateSet("sunlamp", "lightbox", "shirt")]
    [string]$Domain,

    [string]$PythonExe = "python",
    [string]$DataRoot = "",
    [string]$SourceCheckpoint = "",
    [string]$OutputDir = ""
)

$RepoRoot = Split-Path -Parent $PSScriptRoot

if (-not $DataRoot) { $DataRoot = Join-Path $RepoRoot "speedplusv2" }
if (-not $SourceCheckpoint) { $SourceCheckpoint = Join-Path $RepoRoot "output\dinov3_heatmap_source_v2\best_source_dino_heatmap.pth" }
if (-not $OutputDir) { $OutputDir = Join-Path $RepoRoot ("output\table2_redo_external\{0}_{1}" -f $Method, $Domain) }

$env:PYTHONPATH = $RepoRoot
Set-Location $RepoRoot

switch ($Method) {
    "cotta" {
        & $PythonExe ".\table2_redo_external_baselines\run_cotta_single_domain.py" `
            --data_root $DataRoot `
            --domain $Domain `
            --source_checkpoint $SourceCheckpoint `
            --output_dir $OutputDir
    }
    "eata" {
        & $PythonExe ".\table2_redo_external_baselines\run_eata_single_domain.py" `
            --data_root $DataRoot `
            --domain $Domain `
            --source_checkpoint $SourceCheckpoint `
            --output_dir $OutputDir
    }
}

exit $LASTEXITCODE

