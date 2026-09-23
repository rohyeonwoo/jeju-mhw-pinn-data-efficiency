#!/usr/bin/env python3
"""Score saved hybrid/PINN test fields with fixed-threshold MHW metrics.

Training scripts such as ``run_pinn_bo_alpha_gpu01_leadpack.sh`` save
``*_test_fields.npz`` files but do not always attach the Hobday-style MHW skill
columns to their compact summaries. This script rebuilds the train-only
climatology/threshold and scores those saved fields without retraining.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import xarray as xr

from mhw.datasets import make_time_splits
from mhw.eval_metrics import bias, mae, rmse
from mhw.mhw_skill import (
    contingency_skill,
    compute_climatology_threshold,
    compute_mhw_skill_metrics,
    day_of_year,
    mhw_binary_field,
)


TAIL_LEVELS = {
    "moderate": 1.0,
    "strong": 2.0,
    "severe": 3.0,
    "extreme": 4.0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MHW skill from saved hybrid/PINN field npz files")
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--root-label", type=str, default=None)
    p.add_argument("--leads", type=str, default="1,3,7,14")
    p.add_argument("--history", type=int, default=30)
    p.add_argument("--groups", type=str, default="pinn_fmapoff,pinn_fmapon")
    p.add_argument("--clim-frac", type=float, default=0.70)
    p.add_argument("--quantile", type=float, default=0.90)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--min-duration", type=int, default=5)
    p.add_argument("--area-frac", type=float, default=0.15)
    p.add_argument("--onset-tolerance", type=int, default=3)
    p.add_argument("--include-persistence", action="store_true", help="Deprecated alias; baseline rows are on by default")
    p.add_argument("--skip-baseline-rows", action="store_true")
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-md", type=Path, default=None)
    return p.parse_args()


def squeeze_txy(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[1] == 1:
        return arr[:, 0]
    if arr.ndim == 3:
        return arr
    return np.squeeze(arr)


def finite_float(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0.0:
        return None
    return float(num / den)


def skill_score(score: float | None, reference: float | None) -> float | None:
    if score is None or reference is None or reference == 0.0:
        return None
    return float(1.0 - score / reference)


def regression_metrics(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    mask_eval = np.broadcast_to(mask[None, ...], obs.shape).astype(np.float32) if mask.ndim == 2 else mask
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "test_rmse": rmse(pred, obs, mask_eval),
        "test_mae": mae(pred, obs, mask_eval),
        "test_bias": bias(pred, obs, mask_eval),
    }


def anomaly_metrics(
    pred: np.ndarray,
    obs: np.ndarray,
    clim_target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    mask_eval = np.broadcast_to(mask[None, ...], obs.shape).astype(np.float32) if mask.ndim == 2 else mask
    pred_anom = np.nan_to_num(pred - clim_target, nan=0.0, posinf=0.0, neginf=0.0)
    obs_anom = np.nan_to_num(obs - clim_target, nan=0.0, posinf=0.0, neginf=0.0)
    # NOTE: test_anom_rmse == test_rmse by construction (the same clim_target is
    # subtracted from both pred and obs, so the difference is unchanged). It is
    # kept only for backward-compatible columns; the non-redundant anomaly skill
    # metric is the anomaly correlation coefficient (ACC) below.
    w = mask_eval.astype(np.float64)
    pa = pred_anom.astype(np.float64)
    oa = obs_anom.astype(np.float64)
    den = math.sqrt(float(np.sum(w * pa * pa)) * float(np.sum(w * oa * oa)))
    acc = float(np.sum(w * pa * oa) / den) if den > 0.0 else None
    return {
        "test_acc": acc,
        "test_anom_rmse": rmse(pred_anom, obs_anom, mask_eval),
        "test_anom_mae": mae(pred_anom, obs_anom, mask_eval),
        "test_anom_bias": bias(pred_anom, obs_anom, mask_eval),
    }


def normalized_intensity(sst: np.ndarray, doy: np.ndarray, clim: np.ndarray, thresh: np.ndarray) -> np.ndarray:
    clim_t = clim[doy]
    thresh_t = thresh[doy]
    denom = np.maximum(thresh_t - clim_t, 1e-6)
    return (sst - clim_t) / denom


def tail_category_skill(
    pred: np.ndarray,
    obs: np.ndarray,
    target_doy: np.ndarray,
    clim: np.ndarray,
    thresh: np.ndarray,
    mask2d: np.ndarray,
    min_duration: int,
) -> dict[str, float]:
    pred_mhw = mhw_binary_field(pred, target_doy, thresh, min_duration=min_duration)
    obs_mhw = mhw_binary_field(obs, target_doy, thresh, min_duration=min_duration)
    pred_i = normalized_intensity(pred, target_doy, clim, thresh)
    obs_i = normalized_intensity(obs, target_doy, clim, thresh)
    valid = mask2d.astype(bool)[None, ...]

    out: dict[str, float] = {}
    next_level = {"moderate": "strong", "strong": "severe", "severe": "extreme"}
    for name, lower in TAIL_LEVELS.items():
        pred_plus = pred_mhw & (pred_i >= lower)
        obs_plus = obs_mhw & (obs_i >= lower)
        plus = contingency_skill(pred_plus, obs_plus, mask=valid)
        for key, value in asdict(plus).items():
            out[f"tail_{name}_plus_{key}"] = value
        out[f"tail_{name}_plus_obs_positive_count"] = float(plus.hits + plus.misses)
        out[f"tail_{name}_plus_pred_positive_count"] = float(plus.hits + plus.false_alarms)

        upper = TAIL_LEVELS.get(next_level.get(name, ""), np.inf)
        pred_exact = pred_mhw & (pred_i >= lower) & (pred_i < upper)
        obs_exact = obs_mhw & (obs_i >= lower) & (obs_i < upper)
        exact = contingency_skill(pred_exact, obs_exact, mask=valid)
        for key, value in asdict(exact).items():
            out[f"tail_{name}_exact_{key}"] = value
        out[f"tail_{name}_exact_obs_positive_count"] = float(exact.hits + exact.misses)
        out[f"tail_{name}_exact_pred_positive_count"] = float(exact.hits + exact.false_alarms)
    return out


def metrics_path(root: Path, lead: int, group: str) -> Path | None:
    paths = sorted((root / f"lead{lead}" / group).glob("*_metrics.json"))
    return paths[-1] if paths else None


def iter_metric_paths(root: Path, lead: int, groups: list[str]) -> list[Path]:
    allowed = set(groups)
    paths = []
    for path in sorted((root / f"lead{lead}").glob("**/*_metrics.json")):
        if path.parent.name in allowed:
            paths.append(path)
    return paths


def row_from_arrays(
    *,
    label: str,
    lead: int,
    history: int,
    group: str,
    pred: np.ndarray,
    obs: np.ndarray,
    mask2d: np.ndarray,
    target_doy: np.ndarray,
    clim: np.ndarray,
    thresh: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    row = {
        "root_label": label,
        "lead_days": lead,
        "history_days": history,
        "group": group,
        "status": "ok",
    }
    row.update(regression_metrics(pred, obs, mask2d))
    row.update(anomaly_metrics(pred, obs, clim[target_doy], mask2d))
    row.update(
        compute_mhw_skill_metrics(
            pred_sst=pred,
            obs_sst=obs,
            target_doy=target_doy,
            thresh_by_doy=thresh,
            mask=mask2d,
            min_duration=args.min_duration,
            area_frac=args.area_frac,
            onset_tolerance_days=args.onset_tolerance,
        )
    )
    row.update(
        tail_category_skill(
            pred=pred,
            obs=obs,
            target_doy=target_doy,
            clim=clim,
            thresh=thresh,
            mask2d=mask2d,
            min_duration=args.min_duration,
        )
    )
    return row


def add_baseline_skill_scores(rows: list[dict]) -> None:
    by_split: dict[tuple[int, int], dict[str, dict]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (int(row["lead_days"]), int(row["history_days"]))
        by_split.setdefault(key, {})[str(row.get("group"))] = row

    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (int(row["lead_days"]), int(row["history_days"]))
        baselines = by_split.get(key, {})
        persistence = baselines.get("persistence")
        climatology = baselines.get("climatology")
        if persistence:
            row["rmse_skill_vs_persistence"] = skill_score(
                finite_float(row.get("test_rmse")), finite_float(persistence.get("test_rmse"))
            )
            row["anom_rmse_skill_vs_persistence"] = skill_score(
                finite_float(row.get("test_anom_rmse")), finite_float(persistence.get("test_anom_rmse"))
            )
            p_acc = finite_float(persistence.get("test_acc"))
            r_acc = finite_float(row.get("test_acc"))
            row["acc_delta_vs_persistence"] = (
                (r_acc - p_acc) if (r_acc is not None and p_acc is not None) else None
            )
            row["csi_delta_vs_persistence"] = (
                finite_float(row.get("mhw_grid_csi")) or 0.0
            ) - (finite_float(persistence.get("mhw_grid_csi")) or 0.0)
        if climatology:
            row["rmse_skill_vs_climatology"] = skill_score(
                finite_float(row.get("test_rmse")), finite_float(climatology.get("test_rmse"))
            )
            row["anom_rmse_skill_vs_climatology"] = skill_score(
                finite_float(row.get("test_anom_rmse")), finite_float(climatology.get("test_anom_rmse"))
            )
            row["csi_delta_vs_climatology"] = (
                finite_float(row.get("mhw_grid_csi")) or 0.0
            ) - (finite_float(climatology.get("mhw_grid_csi")) or 0.0)


def main() -> None:
    args = parse_args()
    leads = [int(x) for x in args.leads.replace(",", " ").split()]
    groups = [x for x in args.groups.replace(",", " ").split() if x]
    label = args.root_label or args.output_root.name
    write_baseline_rows = (not args.skip_baseline_rows) or args.include_persistence

    ds = xr.open_dataset(args.data_path)
    sst = ds["sst_mean"].values.astype(np.float32)
    doy = day_of_year(ds["time"].values)
    mask2d = np.isfinite(ds["ocean_mask_static"].values).astype(np.float32)
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

    rows: list[dict] = []
    for lead in leads:
        splits = make_time_splits(time_size=time_size, history=args.history, lead=lead)
        target_idx = splits.test + args.history - 1 + lead
        target_doy = doy[target_idx]

        if write_baseline_rows:
            input_idx = splits.test + args.history - 1
            rows.append(
                row_from_arrays(
                    label="reference",
                    lead=lead,
                    history=args.history,
                    group="persistence",
                    pred=sst[input_idx],
                    obs=sst[target_idx],
                    mask2d=mask2d,
                    target_doy=target_doy,
                    clim=clim,
                    thresh=thresh,
                    args=args,
                )
            )
            rows.append(
                row_from_arrays(
                    label="reference",
                    lead=lead,
                    history=args.history,
                    group="climatology",
                    pred=clim[target_doy],
                    obs=sst[target_idx],
                    mask2d=mask2d,
                    target_doy=target_doy,
                    clim=clim,
                    thresh=thresh,
                    args=args,
                )
            )

        found_groups: set[str] = set()
        for mpath in iter_metric_paths(args.output_root, lead, groups):
            group = mpath.parent.name
            found_groups.add(group)
            run_id = str(mpath.parent.relative_to(args.output_root))
            base = {
                "root_label": label,
                "lead_days": lead,
                "history_days": args.history,
                "group": group,
                "run_id": run_id,
            }
            metrics = json.loads(mpath.read_text(encoding="utf-8"))
            fields = metrics.get("test_fields_npz")
            if not fields or not Path(fields).exists():
                rows.append({**base, "status": "missing_fields", "metrics_path": str(mpath)})
                continue

            data = np.load(fields)
            pred = squeeze_txy(data["pred_raw"]).astype(np.float32)
            obs = squeeze_txy(data["target_raw"]).astype(np.float32)
            mask = squeeze_txy(data["mask"]).astype(np.float32)
            if mask.ndim == 3:
                mask2d_eval = mask[0]
            else:
                mask2d_eval = mask
            n = min(pred.shape[0], obs.shape[0], target_doy.shape[0])
            row = row_from_arrays(
                label=label,
                lead=lead,
                history=args.history,
                group=group,
                pred=pred[:n],
                obs=obs[:n],
                mask2d=mask2d_eval,
                target_doy=target_doy[:n],
                clim=clim,
                thresh=thresh,
                args=args,
            )
            row.update(
                {
                    "run_id": run_id,
                    "metrics_path": str(mpath),
                    "fields_path": str(fields),
                    "alpha": finite_float(metrics.get("lambda_physics")),
                    "train_test_rmse": finite_float(metrics.get("test_rmse")),
                    "test_data_loss": finite_float(metrics.get("test_data_loss")),
                    "test_physics_loss": finite_float(metrics.get("test_physics_loss")),
                    "lambda_physics": finite_float(metrics.get("lambda_physics")),
                    "lambda_physics_effective": finite_float(metrics.get("lambda_physics_effective")),
                    "lambda_physics_best": finite_float(metrics.get("lambda_physics_best")),
                    "lambda_physics_final": finite_float(metrics.get("lambda_physics_final")),
                    "lambda_rres": finite_float(metrics.get("lambda_rres")),
                    "lambda_we_tie": finite_float(metrics.get("lambda_we_tie")),
                    "physics_alpha_times_loss": safe_div(
                        (
                            finite_float(metrics.get("lambda_physics_effective"))
                            if finite_float(metrics.get("lambda_physics_effective")) is not None
                            else finite_float(metrics.get("lambda_physics"))
                        )
                        * finite_float(metrics.get("test_physics_loss"))
                        if finite_float(metrics.get("test_physics_loss")) is not None
                        and (
                            finite_float(metrics.get("lambda_physics_effective")) is not None
                            or finite_float(metrics.get("lambda_physics")) is not None
                        )
                        else None,
                        1.0,
                    ),
                    "physics_contrib_ratio": safe_div(
                        (
                            (
                                finite_float(metrics.get("lambda_physics_effective"))
                                if finite_float(metrics.get("lambda_physics_effective")) is not None
                                else finite_float(metrics.get("lambda_physics"))
                            )
                            * finite_float(metrics.get("test_physics_loss"))
                            if finite_float(metrics.get("test_physics_loss")) is not None
                            and (
                                finite_float(metrics.get("lambda_physics_effective")) is not None
                                or finite_float(metrics.get("lambda_physics")) is not None
                            )
                            else None
                        ),
                        finite_float(metrics.get("test_data_loss")),
                    ),
                    "test_rres_frac_of_tendency": finite_float(metrics.get("test_rres_frac_of_tendency")),
                    "test_rres_l2_frac_of_tendency": finite_float(metrics.get("test_rres_l2_frac_of_tendency")),
                    "test_rres_frac_of_physics": finite_float(metrics.get("test_rres_frac_of_physics")),
                    "test_rres_l2_frac_of_physics": finite_float(metrics.get("test_rres_l2_frac_of_physics")),
                    "test_rres_crest_factor": finite_float(metrics.get("test_rres_crest_factor")),
                    "test_rres_abs_mean": finite_float(metrics.get("test_rres_abs_mean")),
                    "test_budget_hdiff_absmean_kday": finite_float(metrics.get("test_budget_hdiff_absmean_kday")),
                    "test_budget_vadv_absmean_kday": finite_float(metrics.get("test_budget_vadv_absmean_kday")),
                    "test_share_hdiff": finite_float(metrics.get("test_share_hdiff")),
                    "test_share_vadv": finite_float(metrics.get("test_share_vadv")),
                    "test_share_rres": finite_float(metrics.get("test_share_rres")),
                    "physics_preset": metrics.get("physics_preset"),
                    "physics_components": metrics.get("physics_components"),
                    "physics_normalize": metrics.get("physics_normalize"),
                    "adaptive_weighting": metrics.get("adaptive_weighting"),
                    "adaptive_sampling": metrics.get("adaptive_sampling"),
                    "adaptive_temperature": finite_float(metrics.get("adaptive_temperature")),
                    "adaptive_samples": metrics.get("adaptive_samples"),
                    "adaptive_weight_min": finite_float(metrics.get("adaptive_weight_min")),
                    "adaptive_weight_max": finite_float(metrics.get("adaptive_weight_max")),
                    "physics_tendency_scale": finite_float(metrics.get("physics_tendency_scale")),
                    "physics_rhs_scale": finite_float(metrics.get("physics_rhs_scale")),
                    "physics_loss": metrics.get("physics_loss"),
                    "physics_time_mode": metrics.get("physics_time_mode"),
                    "forecast_mode": metrics.get("forecast_mode"),
                    "forcing_mode": metrics.get("forcing_mode"),
                    "selection_metric": metrics.get("selection_metric"),
                    "best_val_selection_score": finite_float(metrics.get("best_val_selection_score")),
                    "best_val_total_loss": finite_float(metrics.get("best_val_total_loss")),
                    "best_val_total_loss_at_selection": finite_float(metrics.get("best_val_total_loss_at_selection")),
                    "epoch_log_csv": metrics.get("epoch_log_csv"),
                    "zero_feature_map": metrics.get("zero_feature_map"),
                    "zero_rres_physics": metrics.get("zero_rres_physics"),
                    "rres_penalty_normalize": metrics.get("rres_penalty_normalize"),
                    "best_epoch": metrics.get("best_epoch"),
                }
            )
            rows.append(row)

        simple_missing = [group for group in groups if group not in found_groups and metrics_path(args.output_root, lead, group) is None]
        for group in simple_missing:
            rows.append(
                {
                    "root_label": label,
                    "lead_days": lead,
                    "history_days": args.history,
                    "group": group,
                    "status": "missing_metrics",
                }
            )

    add_baseline_skill_scores(rows)

    summary = {
        "config": {
            "data_path": str(args.data_path),
            "output_root": str(args.output_root),
            "root_label": label,
            "history_days": args.history,
            "leads": leads,
            "groups": groups,
            "baseline_rows": write_baseline_rows,
            "threshold": (
                f"train-first-{args.clim_frac:.0%} calendar-day DOY "
                f"{args.quantile:.0%} percentile, +/-{args.window}d, "
                f"min_duration={args.min_duration}"
            ),
            "mhw_counting": "pooled_time_space",
            "regression_note": (
                "test_rmse/mae/bias are raw SST. test_anom_* subtract the same train "
                "day-of-year climatology from BOTH pred and obs, so test_anom_rmse == "
                "test_rmse (and _mae/_bias likewise) by construction -- they are kept for "
                "backward compatibility but excluded from the headline. The non-redundant "
                "anomaly-skill metric is test_acc (anomaly correlation coefficient): "
                "sum(w*pa*oa)/sqrt(sum(w*pa^2)*sum(w*oa^2)) over ocean cells, pa=pred-clim, "
                "oa=obs-clim. ACC in [-1,1], higher is better; a climatology forecast has "
                "pa==0 so its ACC is undefined, hence acc is compared to persistence only."
            ),
            "skill_score_note": (
                "RMSE skill score = 1 - model_rmse / baseline_rmse. "
                "acc_delta_vs_persistence = model_acc - persistence_acc."
            ),
            "tail_note": (
                "tail_*_plus columns are Hobday category-or-higher thresholds; "
                "tail_*_exact columns are non-overlapping bins. Check obs_positive_count "
                "before making severe/extreme claims."
            ),
            "physics_note": (
                "test_physics_loss is stored in the normalization used by that run. "
                "physics_normalize/physics_*_scale columns record whether it is tendency, rhs, or raw. "
                "physics_contrib_ratio = lambda_physics_effective * test_physics_loss / test_data_loss."
            ),
        },
        "rows": rows,
    }

    out_json = args.output_json or (args.output_root / "hybrid_fields_mhw_skill_summary.json")
    out_md = args.output_md or (args.output_root / "hybrid_fields_mhw_skill_table.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cols = [
        "root_label",
        "lead_days",
        "run_id",
        "group",
        "alpha",
        "test_rmse",
        "test_mae",
        "test_bias",
        "test_acc",
        "rmse_skill_vs_persistence",
        "rmse_skill_vs_climatology",
        "acc_delta_vs_persistence",
        "mhw_grid_csi",
        "mhw_grid_pod",
        "mhw_grid_far",
        "mhw_grid_hits",
        "mhw_grid_misses",
        "mhw_grid_false_alarms",
        "mhw_grid_obs_positive_count",
        "mhw_grid_pred_positive_count",
        "tail_moderate_plus_obs_positive_count",
        "tail_moderate_plus_csi",
        "tail_moderate_plus_pod",
        "tail_strong_plus_obs_positive_count",
        "tail_strong_plus_csi",
        "tail_strong_plus_pod",
        "tail_severe_plus_obs_positive_count",
        "tail_severe_plus_csi",
        "tail_severe_plus_pod",
        "fixed_mhw_obs_event_rmse",
        "fixed_mhw_obs_event_mean_bias",
        "fixed_mhw_obs_event_peak_bias",
        "mhw_event_onset_bias_days",
        "mhw_event_duration_bias_days",
        "test_physics_loss",
        "physics_preset",
        "physics_components",
        "physics_normalize",
        "adaptive_weighting",
        "adaptive_sampling",
        "adaptive_temperature",
        "adaptive_samples",
        "adaptive_weight_min",
        "adaptive_weight_max",
        "lambda_physics",
        "physics_contrib_ratio",
        "lambda_rres",
        "test_rres_frac_of_tendency",
        "test_rres_l2_frac_of_tendency",
        "test_rres_frac_of_physics",
        "test_rres_l2_frac_of_physics",
        "test_rres_abs_mean",
        "test_budget_hdiff_absmean_kday",
        "test_budget_vadv_absmean_kday",
        "test_share_hdiff",
        "test_share_vadv",
        "test_share_rres",
        "best_epoch",
        "best_val_selection_score",
    ]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in sorted(rows, key=lambda r: (r.get("lead_days", 0), r.get("root_label", ""), r.get("group", ""))):
        cells = []
        for col in cols:
            value = row.get(col, "")
            cells.append(f"{value:.4g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"saved: {out_json}")
    print(f"saved: {out_md}")
    print("\n".join(lines[: min(len(lines), 24)]))


if __name__ == "__main__":
    main()
