# Figure 4 geospatial provenance and verification

CPU-only; saved main-experiment predictions; no training or inference.

## Source files

- Main experiment: `/ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency/allocation_batch16_5plus5/ratio_10` only. 12 runs = 2 leads × 2 models × 3 seeds. Batch16, 376 training windows, original p0_original/daily_rollout, lambda_physics=0.1, lambda_rres=10. Full and corrected/follow-up runs are excluded.
- NetCDF: `/ceph_data/yeonwoo/MHW_subset/data/merged_sst_cmems_era5_wind10_qnet_jeju_1993_2022_FINAL_staticmask_filled_glorys3d_budget2d.nc`; full SHA-256 `fff22a44aeeb35fdd26f29b0d0ade7cf9549c18f5ae29aef2a415a5826708ec2` matches every run config.
- Previous verified cell values: `/ceph_data/yeonwoo/MHW_subset/paper_analysis_20260929/results/analysis_data/spatial_cell_rmse.csv`; original index-grid figures and numerical CSV are preserved.
- Every prediction/config/graph/metric source path is in `run_provenance.csv` and `run_provenance.json`; before/after SHA-256 for all used files is in `input_provenance.csv`.
| lead_day | seed | model | source_path |
|---|---|---|---|
| 7 | 7 | baseline | /ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency/allocation_batch16_5plus5/ratio_10/jeju_hilam/lead7_seed7 |
| 7 | 42 | baseline | /ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency/allocation_batch16_5plus5/ratio_10/jeju_hilam/lead7_seed42 |
| 7 | 123 | baseline | /ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency/allocation_batch16_5plus5/ratio_10/jeju_hilam/lead7_seed123 |
| 7 | 7 | P0_PINN | /ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency/allocation_batch16_5plus5/ratio_10/pi_jeju_hilam/lead7_seed7 |
| 7 | 42 | P0_PINN | /ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency/allocation_batch16_5plus5/ratio_10/pi_jeju_hilam/lead7_seed42 |
| 7 | 123 | P0_PINN | /ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency/allocation_batch16_5plus5/ratio_10/pi_jeju_hilam/lead7_seed123 |
| 14 | 7 | baseline | /ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency/allocation_batch16_5plus5/ratio_10/jeju_hilam/lead14_seed7 |
| 14 | 42 | baseline | /ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency/allocation_batch16_5plus5/ratio_10/jeju_hilam/lead14_seed42 |
| 14 | 123 | baseline | /ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency/allocation_batch16_5plus5/ratio_10/jeju_hilam/lead14_seed123 |
| 14 | 7 | P0_PINN | /ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency/allocation_batch16_5plus5/ratio_10/pi_jeju_hilam/lead14_seed7 |
| 14 | 42 | P0_PINN | /ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency/allocation_batch16_5plus5/ratio_10/pi_jeju_hilam/lead14_seed42 |
| 14 | 123 | P0_PINN | /ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency/allocation_batch16_5plus5/ratio_10/pi_jeju_hilam/lead14_seed123 |

## Coordinate provenance

- `latitude`: shape (29,), 30–37°N; `longitude`: shape (33,), 123–131°E. Both actual coordinate arrays are strictly increasing. Measured spacing: latitude 0.25°, longitude 0.25°; neither is hard-coded.
- Important metadata inconsistency: latitude `stored_direction="decreasing"` is stale and contradicts the actual ascending numeric values. It is not used to infer orientation. Every stored target grid exactly matches the native NetCDF grid at the recorded test index.
- Longitude has no units attribute in this processed file. The longitude dimension/name and source meshgrid usage establish the E-coordinate mapping; missing attrs are recorded, not silently invented.

## Grid mapping

- row 0 = south (30°N); row 28 = north (37°N). Column 0 = west (123°E); column 32 = east (131°E). `array[i,j] ↔ (latitude[i], longitude[j])` verified.
- Source SST dimensions are `(time, latitude, longitude)`. Base target_array uses raw values in that dimension order; target_idx selects time only, followed by nan_to_num on excluded missing values.
- Graph mapping is C-order: flat index = i*33+j. `flatnonzero(mask.reshape(-1))` defines ocean nodes; `unravel_index` creates node_to_grid. All 12 saved graphs match the original helper output exactly.
- Dataset permute(2,0,1) changes ocean-node/time/channel axes after spatial flattening; it is not a latitude/longitude transpose. collect_predictions scatters nodes back using the same ocean_flat_indices then reshapes to [N,1,29,33]. No spatial transpose or flip.
- Original `load_time_grid_array` can move time first but preserves spatial-dimension order; target_array uses ds.sst_mean directly. Pure NumPy mapping helpers were extracted from hash-matched source; torch modules and model inference were not imported or run.
- Source locations and exact snippets: `source_mapping_evidence.md`. All-cell marker roundtrip verifies ordering even where geographic corners are masked.
| lead_day | corner | row | column | latitude | longitude | in_ocean_mask | delta_rmse |
|---|---|---|---|---|---|---|---|
| 7 | SW | 0 | 0 | 30 | 123 | False | NaN / masked |
| 7 | SE | 0 | 32 | 30 | 131 | False | NaN / masked |
| 7 | NW | 28 | 0 | 37 | 123 | False | NaN / masked |
| 7 | NE | 28 | 32 | 37 | 131 | False | NaN / masked |
| 14 | SW | 0 | 0 | 30 | 123 | False | NaN / masked |
| 14 | SE | 0 | 32 | 30 | 131 | False | NaN / masked |
| 14 | NW | 28 | 0 | 37 | 123 | False | NaN / masked |
| 14 | NE | 28 | 32 | 37 | 131 | False | NaN / masked |
- Masked corner NaNs are not evidence of coordinate failure; nearest valid ocean anchors are provided in `ocean_anchor_checks.csv`.

## Cell center / edge handling

- Coordinate arrays locate native SST samples and are treated as pixel centers. No latitude/longitude bounds variable is declared. Derived edges are visualization bounds, not a claim about archived physical grid-cell vertices.
- pcolormesh(shading="flat") uses adjacent-center midpoint edges and half-spacing extrapolation at each outer edge. The code computes spacing from arrays; no assumed 0.25° value or min/max-as-edge extent.
- Longitude edges: 122.875 to 131.125°E; latitude edges: 29.875 to 37.125°N. The actual QuadMesh vertices and center positions are asserted against these coordinate arrays.

## Spatial metric

For every ocean cell: first sqrt(mean over held-out dates of squared prediction error) for each model and each seed; then arithmetic mean of seed RMSEs; finally P0 PINN minus baseline. Float64 error reductions, original float32 predictions/targets. Not ensemble-mean predictions, pooled-seed RMSE, global scalar RMSE, MSE or MAE difference.
- Mask semantics: exactly 661 valid ocean cells, 296 excluded cells. Every valid cell has all test dates finite in saved predictions, saved targets and native SST targets. Blank cells are mask exclusions, not missing predictions/targets or insufficient valid dates.
- 7-day: 2018-07-07–2022-12-31, 1639 dates. 14-day: 2018-07-08–2022-12-31, 1638 dates. Exact daily indices verified from the original split helper and native NetCDF time; baseline/P0 and all seeds match within each lead.

## Numeric equivalence

| lead_day | shape | finite_cells | min | max | regenerated_legacy_vs_geospatial_max_abs_difference | regenerated_legacy_vs_geospatial_mean_abs_difference | historical_csv_vs_recomputed_max_abs_difference | historical_csv_vs_recomputed_mean_abs_difference | independent_native_grid_delta_max_abs_difference |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 29 x 33 | 661 | -0.231367355691 | 0.0251157466929 | 0 | 0 | 4.75883221718e-13 | 3.06850414984e-14 | 6.66133814775e-16 |
| 14 | 29 x 33 | 661 | -0.167879387327 | 0.0449451747605 | 0 | 0 | 4.80379624967e-13 | 3.24396046645e-14 | 1.33226762955e-15 |
- The earlier script did not archive an unrounded NPY grid. It saved a 12-significant-digit CSV. The old index-grid arithmetic was regenerated from the same predictions and passed unchanged to the geospatial renderer; its difference is exactly zero. Historical CSV differences are separately reported and must be ≤1e-12 to account only for text rounding. An independent full-native-grid time reduction is also checked (not the same-array identity check).
- The exact arrays actually inside the Matplotlib QuadMesh are checked for zero value difference and identical NaN mask. New full-precision .npy/.npz arrays avoid future CSV rounding ambiguity. Plot generation is aborted before rendering on any source, mapping or numerical mismatch.

## Figure output

- [PDF](fig4_spatial_delta_rmse_10pct_geospatial.pdf) and [600-dpi PNG](fig4_spatial_delta_rmse_10pct_geospatial.png). Fixed canvas 3.35 × 6.3 inches; single-column 2×1; (a) 7-day above (b) 14-day.
- Colormap RdBu_r; common symmetric limits [-0.231367355691, 0.231367355691] K, zero at white. Existing four-map common limit reconstructed from verified spatial CSV: 0.231367355691 K; target two-panel maximum magnitude: 0.231367355690687 K.
- Common geographic edge extent in both panels; longitude increases left→right and latitude bottom→top. Aspect uses 1/cos(mean latitude) as a fixed visual correction, with no coordinate transformation or regridding. No geographic basemap/coastline inference was added.
- Colorbar: Seed-mean ΔRMSE (K). No grid-coordinate/unavailable-coordinate titles. No implication of statistical significance from color.

## Reproducibility

```bash
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/paper_analysis_20260929/generate_figure4_geospatial.py --project-root /ceph_data/yeonwoo/MHW_subset --analysis-root /ceph_data/yeonwoo/MHW_subset/paper_analysis_20260929 --output-dir /ceph_data/yeonwoo/MHW_subset/paper_analysis_20260929/results/figure4_geospatial_reproduced
```

- Use a new output directory; existing results are never overwritten. The script reads original source/predictions plus the earlier spatial CSV and real NetCDF. All computation is CPU-only. Checkpoints are not loaded.
- `figure4_arrays.npz`: actual coordinates, edges, mask, graph indexing, plotted ΔRMSE arrays. `cell_rmse_*`: 12 seed/model-specific cell grids. `cellwise_rmse_by_seed.csv`: full-precision cell-level calculation records. `figure_metadata.json`, `verification.json`, `numeric_equivalence.csv` record the audit.

## Final verdict

| Check | Yes/No |
|---|---|
| run provenance correct | Yes |
| cellwise metric correct | Yes |
| lat/lon mapping verified | Yes |
| no transpose/flip | Yes |
| center/edge handling correct | Yes |
| ocean mask semantics correct | Yes |
| test dates correct | Yes |
| existing spatial values numerically preserved | Yes |
| final figure reproducible | Yes |
