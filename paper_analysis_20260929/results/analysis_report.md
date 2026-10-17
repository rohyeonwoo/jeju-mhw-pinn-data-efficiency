# Training-data scarcity: paper analysis

CPU-only post-hoc analysis; no training performed

## 1. 실제 실험 구조 / scope

- Main: allocation_batch16_5plus5, 48 completed runs = 4 block fractions × 2 leads × 2 models × 3 seeds. Full batch16: 12 separate references. Follow-ups: 18 lambda + 30 corrected-physics runs retained only in protocol-separated long inventory; never pooled into main figures.
- Nested block fractions verified against every saved actual training-index list. 100% block subset = 3995 windows; Full = 7644/7639. Fractions are percentages of 85 eligible 90-day blocks (rounded), not percentages of original training-window count.
- Original P0 includes lambda_physics=0.1 and auxiliary lambda_rres=10. Original land-adjacent spatial differencing and possible head-induced random-order differences remain. This is not the later matched clean physics-only protocol.
- Train/validation/test metadata: sample-based chronological 70/15/15. Full training normalization (including unselected input/target information) shared across fractions. Subset seed equals training seed.

## 2. 사용 파일 / provenance

- Results root: `/ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency`. Latest code definitions: `/ceph_data/yeonwoo/server_backups/node1_20260928T054310Z/home/mhw_subset`.
- `analysis_data/file_inventory.csv`: discovered result files, sizes, timestamps, array keys, capabilities. `experiment_inventory.csv`: selected/excluded batch jobs. `source_provenance.csv`: before/after SHA-256 for every used source.
- No checkpoint opened; no NetCDF read; no model import or inference. Target calendar recovered exactly from saved dataset-origin metadata plus target_index, checked against recorded test endpoints.

## 3. 계산 가능 지표

- SST RMSE/MAE/bias recomputed from 60 prediction arrays and checked against saved evaluation within 2e-6 K. ACC retained from saved evaluator. Seed mean and sample SD (ddof=1).
- MHW CSI/POD/FAR/Strong+ CSI and TP/FP/FN reused from existing evaluation and checked against saved contingency counts with original zero-denominator=0 convention. No new MHW masks generated.
- SST bootstrap, per-cell errors, error quantiles, seasons, paired error ECDF, training/validation history. Saved observed-MHW grid-day RMSE retained; it is not domain-wide MHW-day RMSE.

## 4. 계산하지 못한 항목

| item | status | reason |
|---|---|---|
| MHW threshold/climatology array and fresh event masks | 계산 불가 | No saved arrays in discovered result NPZ/NPY; evaluator originally recomputed from training NetCDF, which this saved-results-only analysis does not read. |
| MHW CSI/POD/FAR paired date bootstrap | 계산 불가 | Aggregate counts exist but not per-date event masks/counts; cannot resample aggregate scores as dates. |
| Non-MHW-conditioned RMSE and fresh MHW-conditioned recomputation | 계산 불가 | Exact MHW masks absent. Existing saved MHW-conditioned RMSE is retained and labeled. |
| Latitude/longitude geographic maps | 계산 불가 | Saved prediction/graph NPZ has mask/topology but no raw lat/lon. Grid-index maps supplied instead; normalized node features not treated as coordinates. |
| ACC independent recomputation | 계산 불가 | Saved ACC reused; climatology arrays absent. |
| New-training-seed population CI | 계산 불가 | Only 3 fitted seed pairs; bootstrap describes conditional test-period variation. |
| Reliable isolated wall-clock efficiency comparison | 계산 불가 | Concurrent shared-GPU original jobs; logged elapsed time is descriptive only. |

## 5. SST 결과

| data_regime | training_fraction | lead_day | baseline_rmse_mean | baseline_rmse_std | pinn_rmse_mean | pinn_rmse_std | delta_rmse | relative_improvement_pct | mean_seed_relative_improvement_pct | n_pinn_better | n_seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| block_subset | 10.000000 | 7 | 0.956765 | 0.035757 | 0.912458 | 0.026829 | -0.044307 | 4.630928 | 4.611815 | 3 | 3 |
| block_subset | 10.000000 | 14 | 1.161726 | 0.019248 | 1.128563 | 0.028133 | -0.033163 | 2.854626 | 2.839837 | 2 | 3 |
| block_subset | 25.000000 | 7 | 0.847586 | 0.015982 | 0.844894 | 0.019332 | -0.002692 | 0.317636 | 0.299962 | 1 | 3 |
| block_subset | 25.000000 | 14 | 1.074555 | 0.006415 | 1.046804 | 0.032418 | -0.027751 | 2.582572 | 2.571134 | 2 | 3 |
| block_subset | 50.000000 | 7 | 0.827970 | 0.006475 | 0.815282 | 0.010942 | -0.012688 | 1.532482 | 1.533045 | 3 | 3 |
| block_subset | 50.000000 | 14 | 1.027531 | 0.011395 | 1.012537 | 0.011139 | -0.014994 | 1.459218 | 1.459117 | 3 | 3 |
| block_subset | 100.000000 | 7 | 0.808885 | 0.014499 | 0.798599 | 0.004259 | -0.010286 | 1.271566 | 1.253321 | 3 | 3 |
| block_subset | 100.000000 | 14 | 0.987276 | 0.009220 | 0.983304 | 0.027059 | -0.003973 | 0.402383 | 0.413010 | 2 | 3 |
| Full | NaN | 7 | 0.777313 | 0.010947 | 0.784929 | 0.009872 | 0.007616 | -0.979760 | -0.985766 | 0 | 3 |
| Full | NaN | 14 | 0.947605 | 0.011963 | 0.956207 | 0.008497 | 0.008602 | -0.907789 | -0.913653 | 1 | 3 |

Relative improvement in summary/figure B = ratio of seed-mean RMSEs. Seedwise figure uses individual ratios and their arithmetic mean; both estimands are separately named in CSV.

## 6. MHW 결과와 기존 정의

- Existing evaluation builds a common threshold using the first round(10957×0.70)=7670 daily records: 1993-01-01 to 2013-12-31. This daily reference differs from lead-specific training target endpoints (2014-01-10/12). It is unchanged across fractions; no subset-specific threshold.
- 90th percentile, ±5 day-of-year pooling, minimum 5 consecutive exceedance days. No extra smoothing and no gap merging in existing implementation.
- Leap handling is the original pandas Gregorian dayofyear with a 366-day circular distance. There is no explicit Feb-29 interpolation or post-February alignment correction in non-leap years. Report this implementation, not an idealized standard.
- CSI/POD/FAR pool ocean grid-time cells within each run; Strong+ uses normalized intensity ≥2 on duration-filtered events. Thresholds themselves were not saved, so their numeric arrays cannot be re-audited here.
| data_regime | training_fraction | lead_day | model | n_seeds | mhw_csi_mean | mhw_csi_std | mhw_csi_n | mhw_pod_mean | mhw_pod_std | mhw_pod_n | mhw_far_mean | mhw_far_std | mhw_far_n | strong_csi_mean | strong_csi_std | strong_csi_n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| block_subset | 10.000000 | 7 | baseline | 3 | 0.248399 | 0.042469 | 3 | 0.302755 | 0.083309 | 3 | 0.383157 | 0.080262 | 3 | 0.060432 | 0.031920 | 3 |
| block_subset | 10.000000 | 7 | PINN | 3 | 0.255361 | 0.022994 | 3 | 0.305137 | 0.044837 | 3 | 0.375906 | 0.056653 | 3 | 0.069447 | 0.031113 | 3 |
| block_subset | 10.000000 | 14 | baseline | 3 | 0.180575 | 0.011272 | 3 | 0.249150 | 0.018312 | 3 | 0.603220 | 0.018516 | 3 | 0.057294 | 0.014195 | 3 |
| block_subset | 10.000000 | 14 | PINN | 3 | 0.209751 | 0.027471 | 3 | 0.306992 | 0.049545 | 3 | 0.600077 | 0.023742 | 3 | 0.073706 | 0.016067 | 3 |
| block_subset | 25.000000 | 7 | baseline | 3 | 0.315305 | 0.056422 | 3 | 0.387949 | 0.104013 | 3 | 0.347645 | 0.057126 | 3 | 0.073805 | 0.039566 | 3 |
| block_subset | 25.000000 | 7 | PINN | 3 | 0.321590 | 0.015040 | 3 | 0.401288 | 0.041144 | 3 | 0.373801 | 0.051269 | 3 | 0.083027 | 0.023881 | 3 |
| block_subset | 25.000000 | 14 | baseline | 3 | 0.181031 | 0.018978 | 3 | 0.228038 | 0.036324 | 3 | 0.521805 | 0.037709 | 3 | 0.033350 | 0.003185 | 3 |
| block_subset | 25.000000 | 14 | PINN | 3 | 0.230766 | 0.047744 | 3 | 0.331772 | 0.100626 | 3 | 0.558157 | 0.023763 | 3 | 0.061201 | 0.023118 | 3 |
| block_subset | 50.000000 | 7 | baseline | 3 | 0.319497 | 0.043016 | 3 | 0.380886 | 0.068713 | 3 | 0.324373 | 0.035795 | 3 | 0.072746 | 0.031066 | 3 |
| block_subset | 50.000000 | 7 | PINN | 3 | 0.349424 | 0.048351 | 3 | 0.435720 | 0.089668 | 3 | 0.349659 | 0.031374 | 3 | 0.099805 | 0.048653 | 3 |
| block_subset | 50.000000 | 14 | baseline | 3 | 0.214552 | 0.016416 | 3 | 0.276957 | 0.011275 | 3 | 0.510110 | 0.064864 | 3 | 0.052140 | 0.003601 | 3 |
| block_subset | 50.000000 | 14 | PINN | 3 | 0.212325 | 0.033614 | 3 | 0.288908 | 0.061194 | 3 | 0.547710 | 0.032046 | 3 | 0.046565 | 0.025909 | 3 |
| block_subset | 100.000000 | 7 | baseline | 3 | 0.333330 | 0.038895 | 3 | 0.407786 | 0.064055 | 3 | 0.347767 | 0.017764 | 3 | 0.080251 | 0.020161 | 3 |
| block_subset | 100.000000 | 7 | PINN | 3 | 0.362899 | 0.018035 | 3 | 0.454352 | 0.037617 | 3 | 0.354225 | 0.018938 | 3 | 0.089641 | 0.020739 | 3 |
| block_subset | 100.000000 | 14 | baseline | 3 | 0.195447 | 0.029374 | 3 | 0.246618 | 0.054498 | 3 | 0.501107 | 0.029953 | 3 | 0.039651 | 0.018557 | 3 |
| block_subset | 100.000000 | 14 | PINN | 3 | 0.221030 | 0.008707 | 3 | 0.280355 | 0.012083 | 3 | 0.488867 | 0.014789 | 3 | 0.052478 | 0.003353 | 3 |
| Full | NaN | 7 | baseline | 3 | 0.361271 | 0.043201 | 3 | 0.433264 | 0.083551 | 3 | 0.299532 | 0.048085 | 3 | 0.104362 | 0.036764 | 3 |
| Full | NaN | 7 | PINN | 3 | 0.342465 | 0.022957 | 3 | 0.396807 | 0.040427 | 3 | 0.279958 | 0.035621 | 3 | 0.084782 | 0.013149 | 3 |
| Full | NaN | 14 | baseline | 3 | 0.241451 | 0.013414 | 3 | 0.297393 | 0.020353 | 3 | 0.437284 | 0.003809 | 3 | 0.054516 | 0.025868 | 3 |
| Full | NaN | 14 | PINN | 3 | 0.194287 | 0.026034 | 3 | 0.234549 | 0.043747 | 3 | 0.451213 | 0.064087 | 3 | 0.027582 | 0.000397 | 3 |

## 7. Paired bootstrap

- 20,000 replicates per condition/method. Existing circular moving-block algorithm adapted into chunked CPU computation. Primary 90 days; sensitivity 30/180 days; iid target-date cluster variant (block=1) also included.
- Same date samples for all cells, seed pairs and fractions within each lead. RMSE first calculated per fitted seed, then averaged; no seed or spatial-cell resampling. 95% percentile CIs are conditional on existing fitted models, not confidence over new training seeds. No p-values.
- IID-date intervals ignore serial dependence; use 90-day primary with block-length sensitivities. The 90-day choice follows the existing analysis and is not claimed optimal.
| training_fraction | lead_day | delta_rmse | ci_low | ci_high | relative_ci_low | relative_ci_high |
|---|---|---|---|---|---|---|
| 10.000000 | 7 | -0.044307 | -0.061451 | -0.027295 | 2.915694 | 6.208753 |
| 25.000000 | 7 | -0.002692 | -0.015484 | 0.010036 | -1.220015 | 1.779816 |
| 50.000000 | 7 | -0.012688 | -0.022265 | -0.003565 | 0.431829 | 2.679219 |
| 100.000000 | 7 | -0.010286 | -0.017393 | -0.002812 | 0.329256 | 2.272854 |
| 10.000000 | 14 | -0.033163 | -0.057020 | -0.011276 | 1.000019 | 4.758980 |
| 25.000000 | 14 | -0.027751 | -0.059007 | 0.004664 | -0.428379 | 5.477213 |
| 50.000000 | 14 | -0.014994 | -0.042148 | 0.012807 | -1.240160 | 4.009038 |
| 100.000000 | 14 | -0.003973 | -0.018972 | 0.010829 | -1.047543 | 2.065471 |

## 8. 가장 중요한 5개 quantitative findings

1. 10% / 7-day: baseline 0.9568 K, PINN 0.9125 K; relative RMSE improvement 4.63%; improved seeds 3/3.
2. 10% / 14-day: baseline 1.1617 K, PINN 1.1286 K; relative RMSE improvement 2.85%; improved seeds 2/3.
3. 25% / 7-day: relative RMSE improvement 0.32%; improved seeds 1/3.
4. Separate Full reference / 7-day: baseline 0.7773 K, PINN 0.7849 K; relative improvement -0.98%.
5. Separate Full reference / 14-day: baseline 0.9476 K, PINN 0.9562 K; relative improvement -0.91%.

## 9. Figure / Table 목록

- [PDF: fig_rmse_vs_fraction_lead7](paper_figures/fig_rmse_vs_fraction_lead7.pdf) · [PNG](paper_figures/fig_rmse_vs_fraction_lead7.png)
- [PDF: fig_rmse_vs_fraction_lead14](paper_figures/fig_rmse_vs_fraction_lead14.pdf) · [PNG](paper_figures/fig_rmse_vs_fraction_lead14.png)
- [PDF: fig_relative_improvement](paper_figures/fig_relative_improvement.pdf) · [PNG](paper_figures/fig_relative_improvement.png)
- [PDF: fig_seedwise_improvement_lead7](paper_figures/fig_seedwise_improvement_lead7.pdf) · [PNG](paper_figures/fig_seedwise_improvement_lead7.png)
- [PDF: fig_seedwise_improvement_lead14](paper_figures/fig_seedwise_improvement_lead14.pdf) · [PNG](paper_figures/fig_seedwise_improvement_lead14.png)
- [PDF: fig_csi_vs_fraction](paper_figures/fig_csi_vs_fraction.pdf) · [PNG](paper_figures/fig_csi_vs_fraction.png)
- [PDF: fig_pod_vs_fraction](paper_figures/fig_pod_vs_fraction.pdf) · [PNG](paper_figures/fig_pod_vs_fraction.png)
- [PDF: fig_far_vs_fraction](paper_figures/fig_far_vs_fraction.pdf) · [PNG](paper_figures/fig_far_vs_fraction.png)
- [PDF: fig_bootstrap_delta_rmse](paper_figures/fig_bootstrap_delta_rmse.pdf) · [PNG](paper_figures/fig_bootstrap_delta_rmse.png)
- [PDF: fig_spatial_delta_rmse_10pct_lead7](paper_figures/fig_spatial_delta_rmse_10pct_lead7.pdf) · [PNG](paper_figures/fig_spatial_delta_rmse_10pct_lead7.png)
- [PDF: fig_spatial_delta_rmse_10pct_lead14](paper_figures/fig_spatial_delta_rmse_10pct_lead14.pdf) · [PNG](paper_figures/fig_spatial_delta_rmse_10pct_lead14.png)
- [PDF: fig_spatial_delta_rmse_100pct_lead7](paper_figures/fig_spatial_delta_rmse_100pct_lead7.pdf) · [PNG](paper_figures/fig_spatial_delta_rmse_100pct_lead7.png)
- [PDF: fig_spatial_delta_rmse_100pct_lead14](paper_figures/fig_spatial_delta_rmse_100pct_lead14.pdf) · [PNG](paper_figures/fig_spatial_delta_rmse_100pct_lead14.png)
- [PDF: fig_paired_error_ecdf](paper_figures/fig_paired_error_ecdf.pdf) · [PNG](paper_figures/fig_paired_error_ecdf.png)
- `paper_figures/figure_metadata.json`: captions, signs, denominators and source CSVs. Every figure also has its plotted-data CSV.
- LaTeX: `paper_tables/table_setup.tex`, `table_sst.tex`, `table_mhw.tex` (requires booktabs; SST/MHW use table*).

## 10. Scarcity trend와 해석 주의

| lead_day | improvement_10pct | improvement_25pct | improvement_50pct | improvement_100pct | maximum_improvement_fraction | benefit_monotonically_increases_as_fraction_decreases |
|---|---|---|---|---|---|---|
| 7 | 4.630928 | 0.317636 | 1.532482 | 1.271566 | 10 | False |
| 14 | 2.854626 | 2.582572 | 1.459218 | 0.402383 | 10 | True |
- The largest relative improvement was observed under the most data-scarce regime for both leads. Across the four observed seed-mean point estimates, lead7 is non-monotonic and lead14 is monotonic as data decreases. This descriptive lead14 ordering is not a statistically established population trend; do not generalize monotonicity across leads.
- Effects refer to the combined original P0/auxiliary training setup, not isolated physical constraints. Later pilot results are different protocols, not substitute points for this curve.
- Only three seeds; subset-selection and training randomness are confounded. Fractions are nested, all share test data and Full statistics. No causal mechanism or general significance claim.
- Spatial improvement is the fraction of 661 cells with negative difference of seed-mean per-cell RMSE; not area weighted. Maps use saved row/column coordinates only.
- Error quantiles/ECDF are descriptive grid-time distributions. Seasonal counts are target dates, not independent grid samples. Training times are affected by concurrent GPU allocation and are not an efficiency claim.

## 11. 검증 / reproducibility

- Verified 551 used source hashes before/after; all unchanged. 60 run configurations, actual subsets, paired target/index/mask arrays, saved MHW contingency formulas, SST recalculations, numeric CSV round trips and shared figure/table/report inputs checked.
- See `verification.json`, `analysis_data/pairing_audit.csv`, `analysis_data/source_provenance.csv`.
- Every analysis output is in this new Ceph directory. Script refuses to overwrite an existing output directory. CUDA hidden; torch not imported; no subprocess training/inference calls.

```bash
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/paper_analysis_20260929/analyze_data_scarcity_paper.py \
  --results-root /ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency \
  --output-dir /ceph_data/yeonwoo/MHW_subset/paper_analysis_20260929/another_new_analysis --bootstrap 20000
```

독립 최종 검증: `independent_validation.json`의 모든 검사 통과. 원본 결과 파일 1,869개의 크기·기록 정밀도 내 수정 시각과 사용 원본 551개의 SHA-256 보존 확인. Figure 14종의 PDF/PNG 및 300 dpi 이상 해상도, pairing·부호·보고서/CSV/LaTeX 수치 일치 검증 완료.

2026-10-01 layout update: [Relative reduction](paper_figures/fig_relative_improvement.pdf) and [Absolute SST RMSE](paper_figures/fig_rmse_vs_fraction.pdf) now use 2×1 panels (7-day above 14-day). Saved numerical results are unchanged; figure count is now 15. See rmse_layout_verification.json.

2026-10-01 Figure 4 update: the new request explicitly permits reading the original NetCDF coordinates. The earlier saved-result-only coordinate limitation is resolved for Figure 4: [geospatial provenance report](figure4_geospatial/provenance_report.md). This does not change the earlier MHW threshold/bootstrap limitations or any metric values.
