param(
    [string]$PythonExe = "python",
    [string]$OutputPath = ""
)

$RepoRoot = Split-Path -Parent $PSScriptRoot

if (-not $OutputPath) { $OutputPath = Join-Path $RepoRoot "visualization_results_dinov3_heatmap\motivation_curve_sunlamp_keypoint_localization_reliability_clean.png" }

$env:PYTHONPATH = $RepoRoot
Set-Location $RepoRoot

& $PythonExe ".\risk_controlled_otta\visualize\plot_online_motivation_curve.py" `
    --source_only_history ".\output\step_history_source_only_sunlamp\step_history.json" `
    --cotta_history ".\output\table2_redo_external\cotta_sunlamp\step_history.json" `
    --eata_history ".\output\table2_redo_external\eata_sunlamp\step_history.json" `
    --ours_history ".\output\step_history_ours_dual_sunlamp\step_history.json" `
    --metric_key quality_like `
    --metric_label "Geometry-Aware Keypoint Quality" `
    --rolling_window 200 `
    --line_smooth_window 81 `
    --highlight_method ours `
    --band_alpha_scale 0.16 `
    --band_lower_q 25 `
    --band_upper_q 75 `
    --start_step 25 `
    --marker_every 240 `
    --marker_size 6.8 `
    --line_width_main 4.5 `
    --line_width_other 3.0 `
    --auto_tight_ylim `
    --tight_ylim_pad 0.08 `
    --legend_labelspacing 0.16 `
    --legend_handlelength 2.2 `
    --legend_borderpad 0.10 `
    --annotate_end_values `
    --end_value_decimals 3 `
    --end_value_font_size 15 `
    --end_value_line_width 2.2 `
    --end_value_dot_size 56 `
    --end_value_right_pad_ratio 0.045 `
    --end_value_min_sep_ratio 0.050 `
    --cotta_label "Recent selective method" `
    --eata_label "Lightweight method" `
    --source_only_label "Source-only/No TTA" `
    --ours_label "Geometric risk-aware OTTA (ours)" `
    --output_path $OutputPath

exit $LASTEXITCODE


