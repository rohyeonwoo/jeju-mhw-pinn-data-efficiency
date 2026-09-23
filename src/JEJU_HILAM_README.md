# JejuHiLAM Baseline

This baseline is a Jeju-adapted, SeaCast-inspired HiLAM comparison model for
the existing MHW/PINN experiments.  It is intended as a modern GNN comparison
group, not as a full reproduction of SeaCast.

## Scope

- Model label: `SeaCast-inspired HiLAM / JejuHiLAM`
- Internal model id: `jeju_hilam`
- Forecast task: direct lead-specific SST prediction
- Leads: 7 and 14 days
- Seeds: 42, 123, 2026
- History: 30 days
- Physics loss: none
- Future forcing: none
- Residual feature: past `f_theta_maps.npy` sequence is included by default and can be disabled with `--no-residual-feature`

## Input Mapping

The wrapper uses the same base dataset and split logic as the PINN experiments.
The base sample has `x=[30,22,29,33]`, `f_seq=[30,1,29,33]`, and
`x_aug=[30,23,29,33]`.

Temporal dynamic node features:

| Channel | Name |
|---:|---|
| 0 | `sst_mean` |
| 1 | `sst_anom` |
| 2 | `u_ml` |
| 3 | `v_ml` |
| 4 | `mlotst` |
| 5 | `ssr` |
| 6 | `str` |
| 7 | `slhf` |
| 8 | `sshf` |
| 9 | `mslp` |
| 10 | `tcc` |
| 11 | `Td_glorys` |
| 12 | `wind10` |
| 13 | `thetao_z0m_anom` |
| 14 | `thetao_z16m_anom` |
| 15 | `thetao_z34m_anom` |
| 16 | `thetao_z78m_anom` |
| 17 | `thetao_z110m_anom` |
| 18 | `thetao_z156m_anom` |
| 22 | `f_theta_residual_map` |

`ocean_mask_static`, `lat2d`, and `lon2d` are not repeated as temporal node
features.  The ocean mask defines the node set, while lat/lon/row/col are graph
static node features.

## Graph

- Native grid: 29 x 33 = 957 cells
- Ocean nodes: 661
- Land or masked cells: 296
- SW analysis box: lat[33.5,35.0] lon[125.5,127.5]
- SW analysis nodes: 36
- Level 0: ocean-only native grid
- Level 1: deterministic 2x2 native-grid coarsening
- Level 2: deterministic 2x2 coarsening of level 1
- Same-level edges: directed 8-neighbor adjacency
- Cross-level edges: parent-child edges in both directions

Graph artifacts are saved per run:

- `jeju_hilam_graph.npz`
- `jeju_hilam_graph_summary.json`
- optional `jeju_hilam_graph.png`

## Architecture

- Shared 1-layer GRU temporal encoder over `[30, input_dim]` for each ocean node
- Hidden dimension: 256
- Processor layers: 3
- Encoder: grid-to-level-0 mesh interaction
- Processor: level 2 to level 1 to level 0, then level 0 to level 1 to level 2
- Decoder: level 2 to level 0 readout, then mesh-to-grid interaction
- Output: normalized target SST at ocean nodes

The implementation is pure PyTorch and does not require PyG or Lightning.

## Commands

Smoke test, lead 14 seed 42:

```powershell
cd D:\MHW
.\MHW\run_jeju_hilam.ps1 -SmokeTest -Leads 14 -Seeds 42
```

Optional tiny overfit check:

```powershell
cd D:\MHW
.\MHW\run_jeju_hilam.ps1 -SmokeTest -TinyOverfit -Leads 14 -Seeds 42 -BatchSize 4
```

Full 3-seed comparison for lead 7 and 14:

```powershell
cd D:\MHW
.\MHW\run_jeju_hilam.ps1
```

Outputs are written under:

```text
D:\MHW\outputs\jeju_hilam\lead{lead}_seed{seed}
```

Each run saves:

- `jeju_hilam_config.json`
- `jeju_hilam_epoch_history.csv`
- `jeju_hilam_best.pt`
- `jeju_hilam_lead_{lead}_history_30_test_fields.npz`
- `jeju_hilam_lead_{lead}_history_30_metrics.json`

## Comparison Notes

For reporting, compare JejuHiLAM against the existing P0/P1/P2/P3 results under
the same dataset, split, lead, history and seed settings.  The most important
rows are lead 14 SW-summer RMSE, ACC and Bias, followed by lead 7 as a secondary
check.

Suggested language:

> We implemented a SeaCast-inspired hierarchical GNN baseline adapted to the
> Jeju ocean-only grid.  The model follows the HiLAM-style encode-process-decode
> hierarchy but performs direct 7-day and 14-day SST prediction using the same
> historical inputs and train/validation/test split as the PINN ablation models.

## PI-JejuHiLAM Pilot

The PI pilot keeps the JejuHiLAM forecasting backbone, input, split, optimizer,
and lead-specific setup fixed, then enables the validated P1 physics loss:

```text
L = L_data + alpha L_physics
P1 physics = original + vertical advection
```

Horizontal diffusion is intentionally excluded.  The P1 physics path reuses the
existing daily-rollout implementation with train-fixed RHS normalization,
vertical advection enabled, horizontal diffusion disabled, and the existing
strict Rres handling.

Pilot run:

```powershell
cd D:\MHW
.\MHW\run_pi_jeju_hilam_hold.cmd -LogFile D:\MHW\outputs\pi_jeju_hilam\pilot.log
```

Summarize the four pilot checks after the run:

```powershell
cd D:\MHW\MHW
conda run -n mhw_win python summarize_pi_jeju_hilam_pilot.py
```

The summary reports full-domain RMSE/ACC, SW-summer RMSE/ACC, common final
equation residual, and training stability.

## Upstream Reference

SeaCast is based on Neural-LAM and supports hierarchical graph models such as
Hi-LAM.  This project uses the public SeaCast/Neural-LAM architecture as a
design reference, but the JejuHiLAM code here is a local pure-PyTorch baseline
adapted to the Jeju grid and experiment protocol.
