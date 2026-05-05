Table 2 red-box redo: single-domain CoTTA / EATA

This folder contains standalone runners to recompute the Table 2 stability
metrics for the external baselines on single target domains:

- CoTTA
- EATA

Metrics saved in each `summary.json`:

- `p95_e_star_p`
- `collapse_rate`
- `adapt_ratio`

Typical usage from `D:\code`:

```powershell
python .\table2_redo_external_baselines\run_cotta_single_domain.py --data_root D:\code\speedplusv2 --domain sunlamp --source_checkpoint D:\code\output\dinov3_heatmap_source_v2\best_source_dino_heatmap.pth --output_dir D:\code\output\table2_redo_external\cotta_sunlamp
python .\table2_redo_external_baselines\run_eata_single_domain.py  --data_root D:\code\speedplusv2 --domain sunlamp --source_checkpoint D:\code\output\dinov3_heatmap_source_v2\best_source_dino_heatmap.pth --output_dir D:\code\output\table2_redo_external\eata_sunlamp
```

Expected domain names:

- `sunlamp`
- `lightbox`
- `shirt`

The SHIRT domain is expected to be available as `speedplusv2/shirt/test.json`
and `speedplusv2/shirt/images/`.

