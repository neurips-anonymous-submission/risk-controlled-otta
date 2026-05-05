param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("source_only", "original_otta", "ours_threshold", "ours_mlp_geo", "ours_dual", "risk_controlled_threshold", "risk_controlled_mlp", "risk_controlled_dual", "risk_controlled_learnable_geo", "risk_controlled_dual_branch", "strict_ltta", "strict_petta", "strict_hybrid")]
    [string]$Method,

    [Parameter(Mandatory = $true)]
    [ValidateSet("validation", "sunlamp", "lightbox", "shirt")]
    [string]$Domain,

    [string]$PythonExe = "python",
    [string]$DataRoot = "",
    [string]$SourceCheckpoint = "",
    [string]$OutputDir = ""
)

$RepoRoot = Split-Path -Parent $PSScriptRoot

$ResolvedMethod = switch ($Method) {
    "risk_controlled_threshold" { "ours_threshold" }
    "risk_controlled_mlp" { "ours_mlp_geo" }
    "risk_controlled_dual" { "ours_dual" }
    "risk_controlled_learnable_geo" { "ours_mlp_geo" }
    "risk_controlled_dual_branch" { "ours_dual" }
    default { $Method }
}

if (-not $DataRoot) { $DataRoot = Join-Path $RepoRoot "speedplusv2" }
if (-not $SourceCheckpoint) { $SourceCheckpoint = Join-Path $RepoRoot "output\dinov3_heatmap_source_v2\best_source_dino_heatmap.pth" }
if (-not $OutputDir) { $OutputDir = Join-Path $RepoRoot ("output\step_history_{0}_{1}" -f $ResolvedMethod, $Domain) }

$env:PYTHONPATH = $RepoRoot
Set-Location $RepoRoot

switch ($ResolvedMethod) {
    "source_only" {
        & $PythonExe -m risk_controlled_otta.experiments.generate_source_only_step_history `
            --data_root $DataRoot `
            --domain $Domain `
            --source_checkpoint $SourceCheckpoint `
            --output_dir $OutputDir `
            --num_workers 0
    }
    "original_otta" {
        & $PythonExe -m risk_controlled_otta.experiments.generate_original_otta_step_history `
            --data_root $DataRoot `
            --domain $Domain `
            --source_checkpoint $SourceCheckpoint `
            --output_dir $OutputDir `
            --num_workers 0
    }
    "ours_threshold" {
        & $PythonExe -m risk_controlled_otta.experiments.generate_ours_step_history `
            --data_root $DataRoot `
            --domain $Domain `
            --source_checkpoint $SourceCheckpoint `
            --output_dir $OutputDir `
            --trigger_mode threshold `
            --gate_usage hard `
            --num_workers 0
    }
    "ours_mlp_geo" {
        & $PythonExe -m risk_controlled_otta.experiments.generate_ours_step_history `
            --data_root $DataRoot `
            --domain $Domain `
            --source_checkpoint $SourceCheckpoint `
            --output_dir $OutputDir `
            --trigger_mode mlp_geo `
            --gate_usage hard `
            --num_workers 0
    }
    "ours_dual" {
        & $PythonExe -m risk_controlled_otta.experiments.generate_ours_step_history `
            --data_root $DataRoot `
            --domain $Domain `
            --source_checkpoint $SourceCheckpoint `
            --output_dir $OutputDir `
            --trigger_mode dual_branch `
            --gate_usage hard `
            --num_workers 0
    }
    "strict_ltta" {
        & $PythonExe -m risk_controlled_otta.experiments.generate_strict_step_history `
            --data_root $DataRoot `
            --source_checkpoint $SourceCheckpoint `
            --method strict_ltta `
            --domain $Domain `
            --output_dir $OutputDir `
            --update_scope stem `
            --lr 1e-6 `
            --weight_decay 0.0 `
            --adapt_steps 1 `
            --temperature 1.0 `
            --lambda_entropy 1.0 `
            --lambda_confidence 0.02 `
            --lambda_dwt 0.05 `
            --grad_clip_norm 0.5 `
            --pseudo_bbox_min_confidence 0.10 `
            --pseudo_bbox_expand_ratio 1.50 `
            --pseudo_bbox_min_size 96.0 `
            --quality_reprojection_cap 50.0 `
            --nms_kernel 3 `
            --subpixel_radius 2 `
            --min_confidence 0.05 `
            --top_k 8 `
            --min_points 6 `
            --ransac_reproj_error 6.0 `
            --ransac_iterations 100 `
            --ransac_confidence 0.999 `
            --num_workers 0
    }
    "strict_petta" {
        & $PythonExe -m risk_controlled_otta.experiments.generate_strict_step_history `
            --data_root $DataRoot `
            --source_checkpoint $SourceCheckpoint `
            --method strict_petta `
            --domain $Domain `
            --output_dir $OutputDir `
            --normal_update_scope stem `
            --guarded_update_scope decoder `
            --lr 5e-6 `
            --guarded_lr 1e-6 `
            --weight_decay 0.0 `
            --guarded_weight_decay 0.0 `
            --adapt_steps 1 `
            --guarded_adapt_steps 1 `
            --temperature 1.0 `
            --lambda_entropy 1.0 `
            --lambda_confidence 0.05 `
            --lambda_dwt 0.1 `
            --lambda_mim 0.25 `
            --lambda_anchor 0.01 `
            --mim_mask_ratio 0.35 `
            --mim_patch_size 32 `
            --teacher_momentum 0.999 `
            --grad_clip_norm 1.0 `
            --guarded_grad_clip_norm 0.5 `
            --pseudo_bbox_min_confidence 0.05 `
            --pseudo_bbox_expand_ratio 1.50 `
            --pseudo_bbox_min_size 96.0 `
            --petta_window_size 32 `
            --petta_warmup_steps 8 `
            --petta_threshold 0.75 `
            --petta_freeze_threshold 1.25 `
            --petta_quality_weight 1.0 `
            --petta_confidence_weight 0.5 `
            --petta_entropy_weight 0.25 `
            --petta_reprojection_weight 0.5 `
            --petta_inlier_weight 0.5 `
            --petta_fallback_weight 0.5 `
            --petta_min_inlier_ratio 0.5 `
            --soft_reset_momentum 0.25 `
            --reset_to_last_stable `
            --use_ema_teacher `
            --quality_reprojection_cap 50.0 `
            --nms_kernel 3 `
            --subpixel_radius 2 `
            --min_confidence 0.05 `
            --top_k 8 `
            --min_points 6 `
            --ransac_reproj_error 6.0 `
            --ransac_iterations 100 `
            --ransac_confidence 0.999 `
            --num_workers 0
    }
    "strict_hybrid" {
        & $PythonExe -m risk_controlled_otta.experiments.generate_strict_step_history `
            --data_root $DataRoot `
            --source_checkpoint $SourceCheckpoint `
            --method strict_hybrid_tta_lite `
            --domain $Domain `
            --output_dir $OutputDir `
            --efficient_update_scope stem `
            --full_update_scope decoder `
            --lr 2e-6 `
            --full_lr 5e-7 `
            --weight_decay 0.0 `
            --full_weight_decay 0.0 `
            --adapt_steps 1 `
            --temperature 1.0 `
            --lambda_entropy 1.0 `
            --lambda_confidence 0.05 `
            --lambda_dwt 0.1 `
            --lambda_mim 0.25 `
            --mim_mask_ratio 0.35 `
            --mim_patch_size 32 `
            --teacher_momentum 0.999 `
            --grad_clip_norm 1.0 `
            --full_grad_clip_norm 0.25 `
            --pseudo_bbox_min_confidence 0.05 `
            --pseudo_bbox_expand_ratio 1.50 `
            --pseudo_bbox_min_size 96.0 `
            --ddsd_window_size 32 `
            --ddsd_warmup_steps 8 `
            --ddsd_threshold 1.10 `
            --ddsd_cooldown_steps 8 `
            --ddsd_quality_weight 1.0 `
            --ddsd_confidence_weight 0.5 `
            --ddsd_entropy_weight 0.25 `
            --ddsd_reprojection_weight 0.5 `
            --ddsd_inlier_weight 0.5 `
            --ddsd_min_inlier_ratio 0.5 `
            --use_ema_teacher `
            --quality_reprojection_cap 50.0 `
            --nms_kernel 3 `
            --subpixel_radius 2 `
            --min_confidence 0.05 `
            --top_k 8 `
            --min_points 6 `
            --ransac_reproj_error 6.0 `
            --ransac_iterations 100 `
            --ransac_confidence 0.999 `
            --num_workers 0
    }
}

exit $LASTEXITCODE

