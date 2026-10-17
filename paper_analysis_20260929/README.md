# Training-data scarcity paper analysis

CPU-only post-hoc analysis; no training performed.

- Main report: [results/analysis_report.md](results/analysis_report.md)
- Numeric outputs: [results/analysis_data](results/analysis_data)
- 15 figures, each vector PDF and 320-dpi PNG: [results/paper_figures](results/paper_figures)
- Three ACM double-column LaTeX tables: [results/paper_tables](results/paper_tables)
- Independent validation: [results/independent_validation.json](results/independent_validation.json)

The 48 original batch16 scarcity runs are the main analysis; 12 Full runs are separate references. All 108 completed original/follow-up runs are included in the protocol-separated long table. No corrected-pilot results replace points on the original scarcity curves.

## Reproduce

Use a **new**, nonexistent output directory under `/ceph_data/yeonwoo`; the script refuses overwrite.

```bash
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B \
  /ceph_data/yeonwoo/MHW_subset/paper_analysis_20260929/analyze_data_scarcity_paper.py \
  --results-root /ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency \
  --output-dir /ceph_data/yeonwoo/MHW_subset/paper_analysis_20260929/reproduction \
  --bootstrap 20000
```

Validate numeric pairing, bootstrap implementation, captions, figures and source preservation:

```bash
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B \
  /ceph_data/yeonwoo/MHW_subset/paper_analysis_20260929/validate_analysis_outputs.py \
  /ceph_data/yeonwoo/MHW_subset/paper_analysis_20260929/results
```

`requirements-analysis.txt` records the existing environment's analysis versions. No package installation was performed. Pillow, used by the validation script to inspect PNG metadata, is a matplotlib dependency. Code definitions are read from the preserved latest home backup; large source arrays remain in their original Ceph paths.

## Interpretation

Main improvement = 100 × (mean baseline RMSE − mean PINN RMSE) / mean baseline RMSE. Seedwise improvement is separately computed before averaging. Main 95% CIs use 20000 paired 90-day circular block draws; 1/30/180-day variants are provided. They condition on fitted seed pairs and do not estimate new-training-seed uncertainty.

MHW scores and counts come from existing saved evaluations. Missing threshold/event masks prevent MHW date-bootstrap and fresh MHW/non-MHW separation. Missing raw geographic coordinates prevent lat/lon maps; supplementary maps explicitly use saved grid indices. All unavailable analyses are listed in the report and unavailable_analyses.csv.

2026-10-01: requested SST figures use 2×1 panels (7-day above 14-day): `fig_relative_improvement.pdf` and `fig_rmse_vs_fraction.pdf`, with matching PNGs.


## Figure 4 geospatial (2026-10-01)

Actual NetCDF coordinates and native-grid mapping are now verified. See [Figure 4 PDF](results/figure4_geospatial/fig4_spatial_delta_rmse_10pct_geospatial.pdf), [provenance report](results/figure4_geospatial/provenance_report.md), and [reproducible CPU script](generate_figure4_geospatial.py). Original index-grid figures and metrics are preserved.
