#!/usr/bin/env python3
"""Aggregate final main-experiment SST/MHW metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from mhw.datasets import make_time_splits
from mhw.mhw_skill import compute_climatology_threshold, day_of_year
from summarize_hybrid_fields_mhw_skill import add_baseline_skill_scores, row_from_arrays, squeeze_txy


MODEL_SPECS = [
    {
        "model": "persistence",
        "family": "reference",
        "architecture": "Persistence",
        "physics": "none",
        "deterministic": True,
    },
    {
        "model": "convlstm",
        "family": "ConvLSTM",
        "architecture": "ConvLSTM",
        "physics": "none",
        "metric_pattern": "hybrid_learned_lead_{lead:02d}_history_{history:02d}_metrics.json",
        "field_key": "test_fields_npz",
    },
    {
        "model": "convlstm_pinn",
        "family": "ConvLSTM-PINN",
        "architecture": "ConvLSTM",
        "physics": "P0",
        "metric_pattern": "hybrid_mt_pinn_lead_{lead:02d}_history_{history:02d}_metrics.json",
        "field_key": "test_fields_npz",
    },
    {
        "model": "jeju_hilam",
        "family": "JejuHiLAM",
        "architecture": "GRU + hierarchical GNN",
        "physics": "none",
        "metric_pattern": "jeju_hilam_lead_{lead}_history_{history}_metrics.json",
        "field_key": "fields_path",
    },
    {
        "model": "pi_jeju_hilam",
        "family": "PI-JejuHiLAM",
        "architecture": "GRU + hierarchical GNN",
        "physics": "P0",
        "metric_pattern": "pi_jeju_hilam_lead_{lead}_history_{history}_metrics.json",
        "field_key": "fields_path",
    },
]

HEADLINE_METRICS = [
    "test_rmse",
    "test_mae",
    "test_bias",
    "test_acc",
    "rmse_skill_vs_persistence",
    "rmse_skill_vs_climatology",
    "mhw_grid_csi",
    "mhw_grid_pod",
    "mhw_grid_far",
    "tail_moderate_plus_csi",
    "tail_moderate_plus_pod",
    "tail_strong_plus_csi",
    "tail_strong_plus_pod",
    "fixed_mhw_obs_event_rmse",
    "fixed_mhw_obs_event_peak_bias",
    "mhw_event_onset_bias_days",
    "mhw_event_duration_bias_days",
    "test_physics_loss",
    "test_rres_l2",
    "test_budget_rres_absmean_kday",
    "test_budget_rres_l2_kday",
    "test_common_res_absmean_kday",
    "test_common_res_rmse_kday",
]

LOWER_IS_BETTER = {
    "test_rmse",
    "test_mae",
    "mhw_grid_far",
    "fixed_mhw_obs_event_rmse",
    "test_physics_loss",
    "test_rres_l2",
    "test_budget_rres_absmean_kday",
    "test_budget_rres_l2_kday",
    "test_common_res_absmean_kday",
    "test_common_res_rmse_kday",
}
HIGHER_IS_BETTER = {
    "test_acc",
    "rmse_skill_vs_persistence",
    "rmse_skill_vs_climatology",
    "mhw_grid_csi",
    "mhw_grid_pod",
    "tail_moderate_plus_csi",
    "tail_moderate_plus_pod",
    "tail_strong_plus_csi",
    "tail_strong_plus_pod",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize outputs/main_experiment_final")
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("D:/MHW/outputs/main_experiment_final"))
    p.add_argument("--history", type=int, default=30)
    p.add_argument("--leads", type=str, default="1,3,7,14")
    p.add_argument("--seeds", type=str, default="42,7,123")
    p.add_argument("--final-physics", choices=["P0", "P1", "P2", "P3"], default="P0")
    p.add_argument("--clim-frac", type=float, default=0.70)
    p.add_argument("--quantile", type=float, default=0.90)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--min-duration", type=int, default=5)
    p.add_argument("--area-frac", type=float, default=0.15)
    p.add_argument("--onset-tolerance", type=int, default=3)
    return p.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(item) for item in text.replace(",", " ").split() if item.strip()]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def metric_path(root: Path, spec: dict[str, Any], lead: int, seed: int, history: int) -> Path:
    model_root = root / str(spec["model"])
    if str(spec["model"]) in {"convlstm", "convlstm_pinn"}:
        run_dir = model_root / f"lead{lead}" / f"seed{seed}"
    else:
        run_dir = model_root / f"lead{lead}_seed{seed}"
    return run_dir / str(spec["metric_pattern"]).format(lead=lead, history=history)


def load_prediction_row(
    *,
    spec: dict[str, Any],
    metrics_path: Path,
    metrics: dict[str, Any],
    lead: int,
    seed: int,
    history: int,
    target_doy: np.ndarray,
    clim: np.ndarray,
    thresh: np.ndarray,
    default_mask: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    field_key = str(spec["field_key"])
    fields_text = metrics.get(field_key) or metrics.get("fields_path") or metrics.get("test_fields_npz")
    base = {
        "model": spec["model"],
        "family": spec["family"],
        "architecture": spec["architecture"],
        "physics": spec["physics"],
        "lead_days": lead,
        "history_days": history,
        "seed": seed,
        "status": "ok",
        "metrics_path": str(metrics_path),
    }
    if not fields_text or not Path(fields_text).exists():
        return {**base, "status": "missing_fields", "fields_path": fields_text}

    data = np.load(fields_text)
    pred = squeeze_txy(data["pred_raw"]).astype(np.float32)
    obs = squeeze_txy(data["target_raw"]).astype(np.float32)
    if "mask" in data:
        mask_arr = squeeze_txy(data["mask"]).astype(np.float32)
        mask2d = mask_arr[0] if mask_arr.ndim == 3 else mask_arr
    else:
        mask2d = default_mask
    if "target_index" in data:
        local_target_doy = day_of_year_time_subset(args.data_path, data["target_index"])
    else:
        local_target_doy = target_doy
    n = min(pred.shape[0], obs.shape[0], local_target_doy.shape[0])

    eval_args = SimpleNamespace(
        min_duration=args.min_duration,
        area_frac=args.area_frac,
        onset_tolerance=args.onset_tolerance,
    )
    row = row_from_arrays(
        label="main_experiment_final",
        lead=lead,
        history=history,
        group=str(spec["model"]),
        pred=pred[:n],
        obs=obs[:n],
        mask2d=mask2d,
        target_doy=local_target_doy[:n],
        clim=clim,
        thresh=thresh,
        args=eval_args,
    )
    row.update(base)
    row["fields_path"] = str(fields_text)
    for key in [
        "best_epoch",
        "best_selection_value",
        "best_val_selection_score",
        "best_val_total_loss",
        "best_val_total_loss_at_selection",
        "lambda_physics",
        "lambda_physics_effective",
        "lambda_rres",
        "physics_preset",
        "physics_components",
        "physics_mode",
        "physics_loss",
        "physics_time_mode",
        "physics_normalize",
        "physics_rhs_scale",
        "zero_rres_physics",
        "input_dim",
        "include_residual_feature",
        "test_physics_loss",
        "test_rres_l2",
        "test_we_tie_loss",
        "test_budget_rres_absmean_kday",
        "test_budget_rres_l2_kday",
        "test_budget_vadv_absmean_kday",
        "test_budget_hdiff_absmean_kday",
        "test_common_res_absmean_kday",
        "test_common_res_rmse_kday",
        "sw_summer_rmse",
        "sw_summer_acc",
        "sw_summer_bias",
        "model_path",
        "checkpoint_path",
        "reused",
        "reused_from_ablation",
    ]:
        if key in metrics:
            row[key] = metrics[key]
    row["checkpoint_path"] = row.get("checkpoint_path") or row.get("model_path")
    row.setdefault("reused", False)
    row.setdefault("reused_from_ablation", None)
    return row


_TIME_CACHE: dict[Path, np.ndarray] = {}


def day_of_year_time_subset(data_path: Path, target_index: np.ndarray) -> np.ndarray:
    path = Path(data_path)
    if path not in _TIME_CACHE:
        with xr.open_dataset(path) as ds:
            _TIME_CACHE[path] = day_of_year(ds["time"].values)
    return _TIME_CACHE[path][target_index.astype(np.int64)]


def add_skill_scores(rows: list[dict[str, Any]]) -> None:
    slim_rows = []
    for row in rows:
        if row.get("status") != "ok":
            slim_rows.append(row)
            continue
        slim_rows.append({**row, "group": row.get("model")})
    add_baseline_skill_scores(slim_rows)
    for original, updated in zip(rows, slim_rows, strict=False):
        for key in [
            "rmse_skill_vs_persistence",
            "anom_rmse_skill_vs_persistence",
            "acc_delta_vs_persistence",
            "csi_delta_vs_persistence",
            "rmse_skill_vs_climatology",
            "anom_rmse_skill_vs_climatology",
            "csi_delta_vs_climatology",
        ]:
            if key in updated:
                original[key] = updated[key]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"].eq("ok")].copy()
    rows: list[dict[str, Any]] = []
    for (model, lead), group in ok.groupby(["model", "lead_days"], dropna=False):
        first = group.iloc[0]
        out: dict[str, Any] = {
            "model": model,
            "family": first.get("family"),
            "architecture": first.get("architecture"),
            "physics": first.get("physics"),
            "lead_days": lead,
            "n_runs": int(len(group)),
        }
        for metric in HEADLINE_METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["lead_days", "model"])


def make_deltas(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = df[df["status"].eq("ok")].copy()
    rows: list[dict[str, Any]] = []
    pairs = [
        ("convlstm_pinn", "convlstm", "ConvLSTM-PINN(P0) - ConvLSTM"),
        ("pi_jeju_hilam", "jeju_hilam", "PI-JejuHiLAM(P0) - JejuHiLAM"),
        ("convlstm", "persistence", "ConvLSTM - Persistence"),
        ("convlstm_pinn", "persistence", "ConvLSTM-PINN(P0) - Persistence"),
        ("jeju_hilam", "persistence", "JejuHiLAM - Persistence"),
        ("pi_jeju_hilam", "persistence", "PI-JejuHiLAM(P0) - Persistence"),
    ]
    metrics = [
        "test_rmse",
        "test_acc",
        "mhw_grid_csi",
        "mhw_grid_pod",
        "mhw_grid_far",
        "tail_strong_plus_csi",
        "fixed_mhw_obs_event_rmse",
        "test_physics_loss",
        "test_common_res_rmse_kday",
    ]
    for lead in sorted(ok["lead_days"].dropna().unique()):
        lead_df = ok[ok["lead_days"].eq(lead)]
        for model, ref, label in pairs:
            model_df = lead_df[lead_df["model"].eq(model)]
            ref_df = lead_df[lead_df["model"].eq(ref)]
            if model_df.empty or ref_df.empty:
                continue
            if ref == "persistence":
                ref_row = ref_df.iloc[0]
                for _, model_row in model_df.iterrows():
                    seed = model_row.get("seed")
                    add_delta_rows(rows, label, lead, seed, model_row, ref_row, metrics)
            else:
                for seed in sorted(set(model_df["seed"].dropna()).intersection(set(ref_df["seed"].dropna()))):
                    model_row = model_df[model_df["seed"].eq(seed)].iloc[0]
                    ref_row = ref_df[ref_df["seed"].eq(seed)].iloc[0]
                    add_delta_rows(rows, label, lead, seed, model_row, ref_row, metrics)

    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw, raw
    summary_rows: list[dict[str, Any]] = []
    for (comparison, lead, metric), group in raw.groupby(["comparison", "lead_days", "metric"], dropna=False):
        values = pd.to_numeric(group["delta"], errors="coerce").dropna()
        summary_rows.append(
            {
                "comparison": comparison,
                "lead_days": lead,
                "metric": metric,
                "n_pairs": int(len(values)),
                "delta_mean": float(values.mean()) if not values.empty else np.nan,
                "delta_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "improved_pairs": int(pd.to_numeric(group["improved"], errors="coerce").fillna(0).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["lead_days", "comparison", "metric"])
    return raw.sort_values(["lead_days", "comparison", "seed", "metric"]), summary


def add_delta_rows(
    rows: list[dict[str, Any]],
    comparison: str,
    lead: int,
    seed: Any,
    model_row: pd.Series,
    ref_row: pd.Series,
    metrics: list[str],
) -> None:
    for metric in metrics:
        if metric not in model_row.index or metric not in ref_row.index:
            continue
        model_value = finite_float(model_row.get(metric))
        ref_value = finite_float(ref_row.get(metric))
        if model_value is None or ref_value is None:
            continue
        delta = model_value - ref_value
        if metric in LOWER_IS_BETTER:
            improved = delta < 0.0
            direction = "lower"
        elif metric in HIGHER_IS_BETTER:
            improved = delta > 0.0
            direction = "higher"
        else:
            improved = None
            direction = "neutral"
        rows.append(
            {
                "comparison": comparison,
                "lead_days": int(lead),
                "seed": seed,
                "metric": metric,
                "reference_value": ref_value,
                "model_value": model_value,
                "delta": delta,
                "direction": direction,
                "improved": improved,
            }
        )


def write_per_run_csvs(root: Path, rows: list[dict[str, Any]]) -> None:
    """One CSV per completed run, named per the paper's _final.csv convention
    (e.g. convlstm_lead7_seed42_final.csv, persistence_lead7_final.csv)."""
    per_run_dir = root / "per_run_csvs_final"
    per_run_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row.get("status") != "ok":
            continue
        model = str(row.get("model"))
        lead = int(row.get("lead_days"))
        if model == "persistence":
            path = per_run_dir / f"persistence_lead{lead}_final.csv"
        else:
            seed = row.get("seed")
            path = per_run_dir / f"{model}_lead{lead}_seed{seed}_final.csv"
        pd.DataFrame([row]).to_csv(path, index=False)


def write_markdown(summary_df: pd.DataFrame, delta_summary: pd.DataFrame, out_path: Path) -> None:
    def table(df: pd.DataFrame, cols: list[str]) -> str:
        if df.empty:
            return "_No rows._"
        view = df[cols].copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        rows = [list(map(str, view.columns))]
        rows.extend([[str(value) for value in row] for row in view.to_numpy()])
        widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
        header = "| " + " | ".join(rows[0][i].ljust(widths[i]) for i in range(len(widths))) + " |"
        sep = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
        body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(widths))) + " |" for row in rows[1:]]
        return "\n".join([header, sep, *body])

    lines = [
        "# Main Experiment Final Summary",
        "",
        "Final physics for PINN models: P0.",
        "",
        "## SST Headline",
        "",
    ]
    if not summary_df.empty:
        cols = [
            "model",
            "lead_days",
            "n_runs",
            "test_rmse_mean",
            "test_rmse_std",
            "test_acc_mean",
            "test_acc_std",
            "mhw_grid_csi_mean",
            "mhw_grid_csi_std",
            "tail_strong_plus_csi_mean",
            "tail_strong_plus_csi_std",
        ]
        have = [col for col in cols if col in summary_df.columns]
        lines.append(table(summary_df, have))
        lines.append("")
    if not delta_summary.empty:
        lines.extend(["## Paired Deltas", ""])
        focus = delta_summary[delta_summary["metric"].isin(["test_rmse", "test_acc", "mhw_grid_csi", "mhw_grid_far"])]
        lines.append(table(focus, list(focus.columns)))
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    leads = parse_int_list(args.leads)
    seeds = parse_int_list(args.seeds)

    ds = xr.open_dataset(args.data_path)
    sst = ds["sst_mean"].values.astype(np.float32)
    doy = day_of_year(ds["time"].values)
    ocean_mask = ds["ocean_mask_static"].values.astype(np.float32)
    mask2d = (np.isfinite(ocean_mask) & (ocean_mask > 0.5)).astype(np.float32)
    time_size = int(ds.sizes["time"])

    ref_mask = np.zeros(time_size, dtype=bool)
    ref_mask[: int(round(time_size * args.clim_frac))] = True
    clim, thresh = compute_climatology_threshold(
        sst=sst,
        doy=doy,
        reference_time_mask=ref_mask,
        window=args.window,
        quantile=args.quantile,
    )

    eval_args = SimpleNamespace(
        min_duration=args.min_duration,
        area_frac=args.area_frac,
        onset_tolerance=args.onset_tolerance,
    )
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for lead in leads:
        splits = make_time_splits(time_size=time_size, history=args.history, lead=lead)
        input_idx = splits.test + args.history - 1
        target_idx = input_idx + lead
        target_doy = doy[target_idx]
        persistence_row = row_from_arrays(
            label="main_experiment_final",
            lead=lead,
            history=args.history,
            group="persistence",
            pred=sst[input_idx],
            obs=sst[target_idx],
            mask2d=mask2d,
            target_doy=target_doy,
            clim=clim,
            thresh=thresh,
            args=eval_args,
        )
        persistence_row.update(
            {
                "model": "persistence",
                "family": "reference",
                "architecture": "Persistence",
                "physics": "none",
                "lead_days": lead,
                "history_days": args.history,
                "seed": "deterministic",
                "status": "ok",
                "checkpoint_path": None,
                "reused": False,
                "reused_from_ablation": None,
            }
        )
        rows.append(persistence_row)

        for spec in MODEL_SPECS:
            if spec["model"] == "persistence":
                continue
            for seed in seeds:
                path = metric_path(args.root, spec, lead, seed, args.history)
                if not path.exists():
                    missing_row = {
                        "model": spec["model"],
                        "family": spec["family"],
                        "architecture": spec["architecture"],
                        "physics": spec["physics"],
                        "lead_days": lead,
                        "history_days": args.history,
                        "seed": seed,
                        "status": "missing_metrics",
                        "metrics_path": str(path),
                    }
                    rows.append(missing_row)
                    missing.append(missing_row)
                    continue
                metrics = read_json(path)
                rows.append(
                    load_prediction_row(
                        spec=spec,
                        metrics_path=path,
                        metrics=metrics,
                        lead=lead,
                        seed=seed,
                        history=args.history,
                        target_doy=target_doy,
                        clim=clim,
                        thresh=thresh,
                        default_mask=mask2d,
                        args=args,
                    )
                )

    add_skill_scores(rows)
    raw_df = pd.DataFrame(rows)
    summary_df = summarize(raw_df)
    delta_raw, delta_summary = make_deltas(raw_df)

    args.root.mkdir(parents=True, exist_ok=True)
    raw_path = args.root / "main_experiment_raw_results_final.csv"
    summary_path = args.root / "main_experiment_summary_final.csv"
    delta_raw_path = args.root / "main_experiment_seed_paired_deltas_final.csv"
    delta_summary_path = args.root / "main_experiment_deltas_final.csv"
    md_path = args.root / "main_experiment_summary_final.md"
    manifest_path = args.root / "experiment_manifest_final.json"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    delta_raw.to_csv(delta_raw_path, index=False)
    delta_summary.to_csv(delta_summary_path, index=False)
    write_markdown(summary_df, delta_summary, md_path)
    write_per_run_csvs(args.root, rows)

    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = read_json(manifest_path)
    manifest.update(
        {
            "summary_status": "complete" if not missing else "incomplete",
            "missing_run_count": len(missing),
            "missing_runs": missing,
            "summary_files": {
                "raw_results": str(raw_path),
                "summary": str(summary_path),
                "seed_paired_deltas": str(delta_raw_path),
                "delta_summary": str(delta_summary_path),
                "markdown": str(md_path),
                "per_run_csv_dir": str(args.root / "per_run_csvs_final"),
            },
            "mhw_definition": (
                f"train-only DOY {args.quantile:.0%} threshold, +/-{args.window} day centered window, "
                f"min_duration={args.min_duration}, no threshold smoothing, no gap merge"
            ),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"saved raw: {raw_path}")
    print(f"saved summary: {summary_path}")
    print(f"saved deltas: {delta_summary_path}")
    print(f"saved paired deltas: {delta_raw_path}")
    print(f"saved markdown: {md_path}")
    print(f"saved per-run CSVs: {args.root / 'per_run_csvs_final'}")
    if missing:
        print(f"missing runs: {len(missing)}")
    else:
        print("all expected runs found")


if __name__ == "__main__":
    main()
