#!/usr/bin/env python3
"""MHW event skill from saved JejuHiLAM/PI-JejuHiLAM test fields.

This is a no-retraining evaluator. It reuses the same Hobday-style threshold
and MHW categorical metrics used for the saved ConvLSTM/PINN field evaluator,
but adapts to the JejuHiLAM output layout:

    root/lead14_seed42/*_metrics.json
    root/lead14_seed42/*_test_fields.npz

Headline metrics:
    CSI, POD, FAR, Strong+ CSI, onset bias, duration bias.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import xarray as xr

from mhw.eval_metrics import bias, mae, rmse
from mhw.mhw_skill import compute_climatology_threshold, compute_mhw_skill_metrics, day_of_year
from summarize_hybrid_fields_mhw_skill import anomaly_metrics, tail_category_skill


DEFAULT_DATA_PATH = Path(
    "D:/MHW/data/merged_sst_cmems_era5_wind10_qnet_jeju_1993_2022_FINAL_staticmask_filled_glorys3d_budget2d.nc"
)
HEADLINE_METRICS = [
    "mhw_grid_csi",
    "mhw_grid_pod",
    "mhw_grid_far",
    "tail_strong_plus_csi",
    "mhw_event_onset_bias_days",
    "mhw_event_duration_bias_days",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MHW skill from saved JejuHiLAM test_fields.npz")
    p.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    p.add_argument(
        "--roots",
        type=Path,
        nargs="+",
        default=[Path("D:/MHW/outputs/jeju_hilam")],
        help="One or more output roots containing lead*_seed* run directories.",
    )
    p.add_argument("--leads", type=str, default="7,14")
    p.add_argument("--seeds", type=str, default="", help="Optional seed filter, e.g. '42,123,2026'")
    p.add_argument("--clim-frac", type=float, default=0.70)
    p.add_argument("--quantile", type=float, default=0.90)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--min-duration", type=int, default=5)
    p.add_argument("--area-frac", type=float, default=0.15)
    p.add_argument("--onset-tolerance", type=int, default=3)
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("D:/MHW/outputs/jeju_hilam/jeju_hilam_mhw_skill"),
    )
    p.add_argument("--skip-reference", action="store_true", help="Do not add persistence/climatology rows")
    return p.parse_args()


def parse_int_set(text: str) -> set[int] | None:
    values = [x for x in text.replace(",", " ").split() if x.strip()]
    if not values:
        return None
    return {int(x) for x in values}


def squeeze_txy(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[1] == 1:
        return arr[:, 0]
    if arr.ndim == 3:
        return arr
    return np.squeeze(arr)


def finite_float(value: object) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def regression_metrics(pred: np.ndarray, obs: np.ndarray, mask2d: np.ndarray) -> dict[str, float]:
    mask_eval = np.broadcast_to(mask2d[None, ...], obs.shape).astype(np.float32)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "test_rmse": rmse(pred, obs, mask_eval),
        "test_mae": mae(pred, obs, mask_eval),
        "test_bias": bias(pred, obs, mask_eval),
    }


def row_from_arrays(
    *,
    root_label: str,
    run_id: str,
    group: str,
    model: str,
    model_label: str,
    lead: int,
    history: int,
    seed: int | None,
    alpha: float | None,
    pred: np.ndarray,
    obs: np.ndarray,
    mask2d: np.ndarray,
    target_idx: np.ndarray,
    doy: np.ndarray,
    clim: np.ndarray,
    thresh: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    target_doy = doy[target_idx]
    row = {
        "root_label": root_label,
        "run_id": run_id,
        "group": group,
        "model": model,
        "model_label": model_label,
        "lead_days": int(lead),
        "history_days": int(history),
        "seed": seed,
        "alpha": alpha,
        "n_test_days": int(pred.shape[0]),
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


def iter_metric_files(root: Path, leads: set[int] | None, seeds: set[int] | None) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.glob("**/*_metrics.json")):
        parent = path.parent.name
        match = re.search(r"lead(\d+)_seed(\d+)", parent)
        if not match:
            continue
        lead = int(match.group(1))
        seed = int(match.group(2))
        if leads is not None and lead not in leads:
            continue
        if seeds is not None and seed not in seeds:
            continue
        files.append(path)
    return files


def load_run(metric_path: Path) -> tuple[dict, Path | None]:
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    fields = metrics.get("fields_path") or metrics.get("test_fields_npz")
    fields_path = Path(fields) if fields else None
    if fields_path is None or not fields_path.exists():
        return metrics, None
    return metrics, fields_path


def add_reference_rows(
    rows: list[dict],
    *,
    ds_sst: np.ndarray,
    doy: np.ndarray,
    clim: np.ndarray,
    thresh: np.ndarray,
    mask2d: np.ndarray,
    args: argparse.Namespace,
) -> None:
    seen: set[tuple[int, int, int, int]] = set()
    for row in list(rows):
        if row.get("status") != "ok" or row.get("seed") is None:
            continue
        key = (int(row["lead_days"]), int(row["history_days"]), int(row["seed"]), int(row["n_test_days"]))
        if key in seen:
            continue
        seen.add(key)
        lead, history, seed, _ = key
        target_idx = np.asarray(row["_target_idx"], dtype=np.int64)
        input_idx = target_idx - lead
        rows.append(
            row_from_arrays(
                root_label="reference",
                run_id=f"lead{lead}_seed{seed}/persistence",
                group="persistence",
                model="persistence",
                model_label="Persistence",
                lead=lead,
                history=history,
                seed=seed,
                alpha=None,
                pred=ds_sst[input_idx],
                obs=ds_sst[target_idx],
                mask2d=mask2d,
                target_idx=target_idx,
                doy=doy,
                clim=clim,
                thresh=thresh,
                args=args,
            )
        )
        rows.append(
            row_from_arrays(
                root_label="reference",
                run_id=f"lead{lead}_seed{seed}/climatology",
                group="climatology",
                model="climatology",
                model_label="Train climatology",
                lead=lead,
                history=history,
                seed=seed,
                alpha=None,
                pred=clim[doy[target_idx]],
                obs=ds_sst[target_idx],
                mask2d=mask2d,
                target_idx=target_idx,
                doy=doy,
                clim=clim,
                thresh=thresh,
                args=args,
            )
        )


def aggregate_rows(rows: list[dict]) -> list[dict]:
    keys = ["root_label", "group", "model", "lead_days", "history_days", "alpha"]
    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("group") not in {"persistence", "climatology"}]
    groups: dict[tuple, list[dict]] = {}
    for row in ok_rows:
        key = tuple(row.get(k) for k in keys)
        groups.setdefault(key, []).append(row)

    out: list[dict] = []
    for key, vals in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        rec = dict(zip(keys, key))
        rec["n_runs"] = len(vals)
        rec["seeds"] = ",".join(str(v.get("seed")) for v in sorted(vals, key=lambda x: int(x.get("seed") or 0)))
        for metric in ["test_rmse", "test_acc", *HEADLINE_METRICS]:
            xs = np.asarray([finite_float(v.get(metric)) for v in vals if finite_float(v.get(metric)) is not None])
            if xs.size:
                rec[f"{metric}_mean"] = float(np.mean(xs))
                rec[f"{metric}_std"] = float(np.std(xs, ddof=1)) if xs.size > 1 else 0.0
        out.append(rec)
    return out


def fmt(value: object, ndigits: int = 4) -> str:
    x = finite_float(value)
    if x is None:
        return "" if value is None else str(value)
    return f"{x:.{ndigits}f}"


def write_outputs(rows: list[dict], aggregates: list[dict], args: argparse.Namespace) -> None:
    clean_rows = []
    for row in rows:
        clean = {k: v for k, v in row.items() if not k.startswith("_")}
        clean_rows.append(clean)

    payload = {
        "config": {
            "data_path": str(args.data_path),
            "roots": [str(p) for p in args.roots],
            "leads": sorted(parse_int_set(args.leads) or []),
            "seeds": sorted(parse_int_set(args.seeds) or []) if parse_int_set(args.seeds) else None,
            "threshold": (
                f"train-first-{args.clim_frac:.0%} DOY {args.quantile:.0%} percentile, "
                f"+/-{args.window}d, min_duration={args.min_duration}"
            ),
            "mhw_counting": "CSI/POD/FAR are pooled over ocean grid-time cells.",
            "strong_plus_definition": (
                "tail_strong_plus_* uses Hobday normalized intensity >= 2.0 "
                "on MHW days after min-duration filtering."
            ),
            "event_definition": (
                f"Onset/duration are from the domain MHW index using area_frac={args.area_frac} "
                f"and onset_tolerance={args.onset_tolerance}d."
            ),
        },
        "aggregate_rows": aggregates,
        "rows": clean_rows,
    }

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    md_path = args.output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_rows = aggregates + clean_rows
    if csv_rows:
        fieldnames = sorted({k for row in csv_rows for k in row})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    lines = [
        "# JejuHiLAM MHW skill",
        "",
        "- No retraining: scores are computed from saved `*_test_fields.npz`.",
        "- CSI/POD/FAR: pooled ocean grid-time MHW detection.",
        "- Strong+ CSI: Hobday normalized intensity >= 2.0.",
        "",
        "## Seed Mean",
        "",
        "| root | model | lead | alpha | seeds | CSI | POD | FAR | Strong+ CSI | onset bias | duration bias | RMSE | ACC |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row.get('root_label')} | {row.get('model')} | {row.get('lead_days')} | "
            f"{fmt(row.get('alpha'))} | {row.get('seeds')} | "
            f"{fmt(row.get('mhw_grid_csi_mean'))}+-{fmt(row.get('mhw_grid_csi_std'))} | "
            f"{fmt(row.get('mhw_grid_pod_mean'))}+-{fmt(row.get('mhw_grid_pod_std'))} | "
            f"{fmt(row.get('mhw_grid_far_mean'))}+-{fmt(row.get('mhw_grid_far_std'))} | "
            f"{fmt(row.get('tail_strong_plus_csi_mean'))}+-{fmt(row.get('tail_strong_plus_csi_std'))} | "
            f"{fmt(row.get('mhw_event_onset_bias_days_mean'))}+-{fmt(row.get('mhw_event_onset_bias_days_std'))} | "
            f"{fmt(row.get('mhw_event_duration_bias_days_mean'))}+-{fmt(row.get('mhw_event_duration_bias_days_std'))} | "
            f"{fmt(row.get('test_rmse_mean'))}+-{fmt(row.get('test_rmse_std'))} | "
            f"{fmt(row.get('test_acc_mean'))}+-{fmt(row.get('test_acc_std'))} |"
        )

    lines += [
        "",
        "## Runs",
        "",
        "| root | run | model | lead | seed | alpha | CSI | POD | FAR | Strong+ CSI | onset bias | duration bias | obs+ | strong obs+ |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(clean_rows, key=lambda r: (r.get("lead_days", 0), str(r.get("root_label")), str(r.get("run_id")))):
        if row.get("status") != "ok" or row.get("group") in {"persistence", "climatology"}:
            continue
        lines.append(
            f"| {row.get('root_label')} | {row.get('run_id')} | {row.get('model')} | "
            f"{row.get('lead_days')} | {row.get('seed')} | {fmt(row.get('alpha'))} | "
            f"{fmt(row.get('mhw_grid_csi'))} | {fmt(row.get('mhw_grid_pod'))} | "
            f"{fmt(row.get('mhw_grid_far'))} | {fmt(row.get('tail_strong_plus_csi'))} | "
            f"{fmt(row.get('mhw_event_onset_bias_days'))} | "
            f"{fmt(row.get('mhw_event_duration_bias_days'))} | "
            f"{fmt(row.get('mhw_grid_obs_positive_count'), 0)} | "
            f"{fmt(row.get('tail_strong_plus_obs_positive_count'), 0)} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {md_path}")
    print(f"saved: {json_path}")
    print(f"saved: {csv_path}")


def main() -> None:
    args = parse_args()
    leads = parse_int_set(args.leads)
    seeds = parse_int_set(args.seeds)

    ds = xr.open_dataset(args.data_path)
    sst = ds["sst_mean"].values.astype(np.float32)
    doy = day_of_year(ds["time"].values)
    mask2d = np.isfinite(ds["ocean_mask_static"].values).astype(np.float32)
    ref_mask = np.zeros(int(ds.sizes["time"]), dtype=bool)
    ref_mask[: int(round(ref_mask.size * args.clim_frac))] = True
    clim, thresh = compute_climatology_threshold(
        sst=sst,
        doy=doy,
        reference_time_mask=ref_mask,
        window=args.window,
        quantile=args.quantile,
    )

    rows: list[dict] = []
    for root in args.roots:
        root = root.resolve()
        for metric_path in iter_metric_files(root, leads, seeds):
            metrics, fields_path = load_run(metric_path)
            parent_match = re.search(r"lead(\d+)_seed(\d+)", metric_path.parent.name)
            lead = int(metrics.get("lead") or parent_match.group(1))
            seed = int(metrics.get("seed") or parent_match.group(2))
            history = int(metrics.get("history") or 30)
            run_id = str(metric_path.parent.relative_to(root))
            root_label = root.name
            if fields_path is None:
                rows.append(
                    {
                        "root_label": root_label,
                        "run_id": run_id,
                        "lead_days": lead,
                        "history_days": history,
                        "seed": seed,
                        "status": "missing_fields",
                        "metrics_path": str(metric_path),
                    }
                )
                continue

            data = np.load(fields_path)
            pred = squeeze_txy(data["pred_raw"]).astype(np.float32)
            obs = squeeze_txy(data["target_raw"]).astype(np.float32)
            mask = squeeze_txy(data["mask"]).astype(np.float32)
            mask2d_eval = mask[0] if mask.ndim == 3 else mask
            target_idx = np.asarray(data["target_index"], dtype=np.int64)
            n = min(pred.shape[0], obs.shape[0], target_idx.shape[0])
            row = row_from_arrays(
                root_label=root_label,
                run_id=run_id,
                group=str(metrics.get("model") or root_label),
                model=str(metrics.get("model") or root_label),
                model_label=str(metrics.get("model_label") or metrics.get("model") or root_label),
                lead=lead,
                history=history,
                seed=seed,
                alpha=finite_float(metrics.get("lambda_physics")),
                pred=pred[:n],
                obs=obs[:n],
                mask2d=mask2d_eval,
                target_idx=target_idx[:n],
                doy=doy,
                clim=clim,
                thresh=thresh,
                args=args,
            )
            row.update(
                {
                    "metrics_path": str(metric_path),
                    "fields_path": str(fields_path),
                    "best_epoch": metrics.get("best_epoch"),
                    "best_val_rmse": finite_float(metrics.get("best_selection_value")),
                    "physics_mode": metrics.get("physics_mode"),
                    "physics_components": metrics.get("physics_components"),
                    "physics_normalize": metrics.get("physics_normalize"),
                    "_target_idx": target_idx[:n],
                }
            )
            rows.append(row)

    if not args.skip_reference:
        add_reference_rows(rows, ds_sst=sst, doy=doy, clim=clim, thresh=thresh, mask2d=mask2d, args=args)

    aggregates = aggregate_rows(rows)
    write_outputs(rows, aggregates, args)


if __name__ == "__main__":
    main()
