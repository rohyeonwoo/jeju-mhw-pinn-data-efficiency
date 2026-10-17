Figure 1. Study grid and matched Oracle120 design. The dashed box is the new exploratory SW-Jeju region, not a prior dynamical definition. The 100% subset contains 3,995 retained windows and is smaller than the original full training set. All five models use oracle forcing only where specified in their training losses; Test is exploratory.


Figure 2. Test SST RMSE across training subsets and forecast leads. Points show means of three matched seeds; bars show sample SD (not confidence intervals). The 100% subset comprises 3,995 windows. All values are from fixed Validation-best checkpoints on the previously inspected Test period.


Figure 3. Paired horizontal-advection constraint comparison. Open markers identify seeds; black diamonds are means. Blue intervals reuse the existing 2,000-replicate circular 90-day temporal block bootstrap, conditional on the same three seeds. Positive Adv−NoAdv RMSE favors NoAdv. Temporal intervals and between-seed dispersion quantify different uncertainties; no multiplicity adjustment or population-of-seeds inference is implied.

Only the matched NoAdv–Adv comparison isolates inclusion of the horizontal-advection term in this implemented training constraint. P0 versus either NoAdv or Adv also changes active heads and physical-term definitions: the we head is active only in P0; the Rres head is inactive in all three. Shortwave penetration, learned entrainment versus an MLD-deepening proxy, and arithmetic versus thickness-weighted mixed-layer velocity definitions also differ. Sharing the same network architecture does not imply identical active objectives. This ablation concerns predictive utility of the implemented constraint, not the independent causal role of advection in the ocean.


Figure 4. Pooled grid-day MHW CSI, POD, and FAR across subsets and leads, with three-seed sample SD. FAR is the false-alarm ratio FP/(TP+FP), not the false-positive rate. A single fixed threshold and common observed labels are used across all ratios and models. Five-day events join predictions issued at different times at a fixed lead, rather than a single-issuance five-day alarm.


Figure 5. Spatial and seasonal SST differences at all ratios. Maps average seedwise pointwise Adv−NoAdv RMSE (K); the shared color scale is clipped at the 98th percentile of absolute mapped values to reveal structure. Seasonal points average seedwise regional RMSE differences, with sample SD. Positive values favor NoAdv. Maps are descriptive and contain spatially dependent cells. Seasonal plot values are in statistics/region_season_paired.csv (region=All) and regional_seasonal_seed_metrics.csv.


Figure S1. Saved auxiliary diagnostic batches: median and interquartile range at common scheduled epochs 1, 5, 10, 20 and 80, each with all 24 runs. Irregular best-candidate diagnostics are retained in CSV but excluded from the plotted aggregate trend; epoch-specific counts are provided. Gradient ratio is 100×0.0001||∇shared U||/||∇shared D||. Epoch 1 is after one training epoch. Declining magnitudes do not establish a causal mechanism or exact zero in the Rres head.


Figure S2. Seedwise RMSE improvement percentages for six prespecified paired comparisons: 100(reference−candidate)/reference. Each dot represents an identical ratio/lead/seed pair; positive favors the candidate. Percentage averages in tables are averages of these seedwise values, not percentages computed from group means.


Figure S3. Physical diagnostic from observed Test-day SST and saved mixed-layer velocities using the documented ocean-neighbor stencil, signed metre spacing and −86400(uTx+vTy). These observed-field tendencies are not the model-interpolated training-path advection terms. Panel (a) displays −2 to 2 K/day without renormalizing the omitted tails; tail counts are saved separately. Panel (b) compares seasonal RMS and mean absolute magnitude. Panel (c) is a descriptive spatial association, not independent samples or a causal test. Companion CSVs: advection_histogram.csv, observed_advection_rms_map.csv, advection_spatial_association.csv.


Figure S4. Paired regional SST and MHW differences at all ratios and leads, with three-seed sample SD. Positive signs favor NoAdv for both displayed definitions. The SW-Jeju box is newly specified and exploratory (see spatial_case_protocol.md); coast/offshore follow the saved scoring definition. Regional and seasonal TP/FP/FN values are provided in regional_seasonal_seed_metrics.csv and region_season_paired.csv.


Figure S5. Reproducible calendar-selected case: earliest June–August day of the final complete Test year, fixed 100% subset and seed 42. Selection does not depend on skill or event intensity, so this is an illustrative day rather than a claim of a representative extreme event. MHW labels are taken from the already evaluated complete fixed-lead series. All five-model predictions for this date are retained in the companion case_all_models.csv.


Figure S6. Existing common-equation posthoc residual RMS and SST RMSE for all 120 frozen predictions. A excludes horizontal advection and B includes it. Daily within-run residual/error correlations are also provided. Descriptive associations do not show independent physical closure or causal improvement.
