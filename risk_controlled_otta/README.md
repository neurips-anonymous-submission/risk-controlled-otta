# Risk-Controlled-OTTA

This package contains the DINO-based heatmap pose pipeline together with the
online adaptation code used in the Risk-Controlled-OTTA paper.

## Method family

Risk-Controlled-OTTA is instantiated in three variants:

- `Risk-Controlled-OTTA-Threshold`
- `Risk-Controlled-OTTA-Learnable-Geo`
- `Risk-Controlled-OTTA-Dual-Branch`

These correspond to:

- rule-based geometric trigger
- learnable geometry-risk trigger
- geometry-feature risk trigger

## Main files

- `models/dino_pose_model.py`
- `losses/heatmap_loss.py`
- `data/dino_heatmap_dataset.py`
- `train/train_dino_heatmap.py`
- `eval/evaluate_dino_heatmap.py`
- `eval/sweep_pose_params.py`
- `adapt/triggered_single_model_tta_dino_heatmap.py`
- `adapt/learnable_trigger_single_model_tta_dino_heatmap.py`

## Core idea

Risk-Controlled-OTTA treats adaptation as a risk-controlled intervention problem.
Instead of relying on prediction confidence alone, it estimates task-level
geometric risk from cues such as inlier ratio, reprojection consistency, and
PnP failure signals, and only adapts when that estimated risk is high.

## Defaults

- backbone: `vit_base_patch16_dinov3.lvd1689m`
- input size: `384`
- heatmap size: `96`
- sigma: `3.0`
- encoder lr: `5e-5`
- decoder lr: `5e-4`
- warmup steps: `1000`

## Example

```bash
python -m risk_controlled_otta.train.train_dino_heatmap \
  --data_root speedplusv2 \
  --output_dir output/dinov3_heatmap_source \
  --model_name vit_base_patch16_dinov3.lvd1689m \
  --pretrained_path weights/dinov3/vit_base_patch16_dinov3/model.safetensors \
  --batch_size 32
```

