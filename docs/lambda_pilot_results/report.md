# Lambda pilot: complete results

Lead7; batch16; 10% block subset (376 samples) and original Full (7644 samples). Three seeds. All arms reported.
Aux0 retains lambda_rres=10 and physics heads. Baseline has neither; legacy baseline/P0 DataLoader order need not be identical despite the same seed.
Lambda is selected by mean validation RMSE, never by test score. Existing test results informed this follow-up design: interpretation is exploratory.
Time-block intervals are conditional on the fitted models, not confidence intervals over new training seeds. Three paired seeds cannot yield exact two-sided p<0.05. Multiple comparisons are descriptive, without a family-wide significance claim.

| Regime | Arm | Mean validation RMSE | Test RMSE mean ± SD | CSI | POD | FAR |
|---|---|---|---|---|---|---|
| ratio_10 | baseline | 0.94681 | 0.95677 ± 0.03576 | 0.2484 | 0.3028 | 0.3832 |
| ratio_10 | aux0 | 0.92504 | 0.88348 ± 0.02035 | 0.3548 | 0.4918 | 0.4383 |
| ratio_10 | p0_0.01 | 0.92074 | 0.89827 ± 0.02245 | 0.3142 | 0.4073 | 0.4144 |
| ratio_10 | p0_0.1 | 0.91717 | 0.91246 ± 0.02683 | 0.2554 | 0.3051 | 0.3759 |
| ratio_10 | p0_0.3 | 0.91655 | 0.88425 ± 0.04352 | 0.3207 | 0.4212 | 0.4190 |
| full | baseline | 0.81234 | 0.77731 ± 0.01095 | 0.3613 | 0.4333 | 0.2995 |
| full | aux0 | 0.81062 | 0.78368 ± 0.00782 | 0.3561 | 0.4219 | 0.3014 |
| full | p0_0.01 | 0.81033 | 0.78188 ± 0.01172 | 0.3463 | 0.4020 | 0.2836 |
| full | p0_0.1 | 0.80888 | 0.78493 ± 0.00987 | 0.3425 | 0.3968 | 0.2800 |
| full | p0_0.3 | 0.81526 | 0.79052 ± 0.02401 | 0.3528 | 0.4199 | 0.2968 |

## Validation-selected setting

- ratio_10: p0_0.3 (mean validation RMSE=0.91655).
- full: p0_0.1 (mean validation RMSE=0.80888).

See paired_comparisons.csv for every seed difference, exact sign-flip p, and 30/90/180-day conditional time-block intervals. No automatic expansion is triggered by favorable test scores.
