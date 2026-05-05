param(
    [string]$PythonExe = "python",
    [string]$DataRoot = "",
    [string]$OutputDir = "",
    [string]$ModelName = "vit_base_patch16_dinov3.lvd1689m",
    [string]$PretrainedPath = "",
    [int]$BatchSize = 32
)

$RepoRoot = Split-Path -Parent $PSScriptRoot

if (-not $DataRoot) { $DataRoot = Join-Path $RepoRoot "speedplusv2" }
if (-not $OutputDir) { $OutputDir = Join-Path $RepoRoot "output\dinov3_heatmap_source_v2" }
if (-not $PretrainedPath) { $PretrainedPath = Join-Path $RepoRoot "weights\dinov3\vit_base_patch16_dinov3\model.safetensors" }

$env:PYTHONPATH = $RepoRoot
Set-Location $RepoRoot

& $PythonExe -m risk_controlled_otta.train.train_dino_heatmap `
    --data_root $DataRoot `
    --output_dir $OutputDir `
    --model_name $ModelName `
    --pretrained_path $PretrainedPath `
    --batch_size $BatchSize

exit $LASTEXITCODE


