#!/usr/bin/env python3
"""Preflight checks for the final main MHW forecasting experiment."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr

from mhw.datasets import (
    DEFAULT_STATIC_VARS,
    JejuForecastDataset,
    compute_train_stats,
    make_time_splits,
)
from mhw.jeju_hilam_dataset import BASE_DYNAMIC_VARS as HILAM_DYNAMIC_VARS
from mhw.jeju_hilam_dataset import make_jeju_hilam_bundle
from train_hybrid_jeju_forecast import configure_physics_components, set_seed


FINAL_DYNAMIC_VARS = [
    "sst_mean",
    "sst_anom",
    "u_ml",
    "v_ml",
    "mlotst",
    "ssr",
    "str",
    "slhf",
    "sshf",
    "mslp",
    "tcc",
    "Td_glorys",
    "wind10",
    "thetao_z0m_anom",
    "thetao_z16m_anom",
    "thetao_z34m_anom",
    "thetao_z78m_anom",
    "thetao_z110m_anom",
    "thetao_z156m_anom",
]

PHYSICS_PRESETS = ("P0", "P1", "P2", "P3")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate final main-experiment settings before training")
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--residual-map", type=Path, default=None)
    p.add_argument("--history", type=int, default=30)
    p.add_argument("--leads", type=str, default="1,3,7,14")
    p.add_argument("--seeds", type=str, default="42,7,123")
    p.add_argument("--final-physics", choices=PHYSICS_PRESETS, default="P0")
    p.add_argument("--expected-convlstm-channels", type=int, default=22)
    p.add_argument("--expected-hilam-temporal-channels", type=int, default=19)
    p.add_argument("--lambda-physics", type=float, default=0.1)
    p.add_argument("--lambda-rres", type=float, default=10.0)
    p.add_argument("--k-diff", type=float, default=100.0)
    p.add_argument("--hm-min", type=float, default=5.0)
    p.add_argument("--forcing-mode", choices=["perfect", "persist"], default="perfect")
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def parse_list(text: str) -> list[str]:
    return [item.strip() for item in text.replace(",", " ").split() if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item) for item in parse_list(text)]


def time_str(values: np.ndarray, index: int) -> str:
    return np.datetime_as_string(values[int(index)], unit="D")


def split_row(ds: xr.Dataset, indices: np.ndarray, history: int, lead: int, name: str) -> dict[str, Any]:
    times = ds["time"].values
    if indices.size == 0:
        return {"split": name, "n_samples": 0}
    input_start = indices
    input_end = indices + history - 1
    target = input_end + lead
    return {
        "split": name,
        "n_samples": int(indices.size),
        "input_start": time_str(times, int(input_start[0])),
        "input_end": time_str(times, int(input_end[-1])),
        "target_start": time_str(times, int(target[0])),
        "target_end": time_str(times, int(target[-1])),
    }


def physics_config(preset: str) -> argparse.Namespace:
    args = argparse.Namespace(
        physics_preset=preset,
        physics_components="",
        physics_vertical=False,
        physics_hdiff=False,
    )
    return configure_physics_components(args)


def seed_probe(seed: int) -> dict[str, Any]:
    set_seed(seed)
    return {
        "seed": seed,
        "python_random": round(random.random(), 8),
        "numpy_random": round(float(np.random.rand()), 8),
        "torch_random": round(float(torch.rand(1).item()), 8),
    }


def finite_minmax(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": float("nan"), "max": float("nan")}
    return {"min": float(np.min(finite)), "max": float(np.max(finite))}


def main() -> None:
    args = parse_args()
    leads = parse_int_list(args.leads)
    seeds = parse_int_list(args.seeds)
    dynamic_vars = list(FINAL_DYNAMIC_VARS)
    static_vars = list(DEFAULT_STATIC_VARS)

    ds = xr.open_dataset(args.data_path)
    required_dataset_vars = dynamic_vars + ["sst_mean", "ocean_mask_static", "latitude", "longitude"]
    missing = [name for name in required_dataset_vars if name not in ds.variables]
    if missing:
        raise SystemExit(f"Missing final experiment variables: {missing}")

    required_physics = ["qnet", "mlotst", "u_ml", "v_ml", "Td_glorys", "vadv_kday"]
    missing_physics = [name for name in required_physics if name not in ds.variables]
    if missing_physics:
        raise SystemExit(f"Missing physics variables: {missing_physics}")

    if dynamic_vars != list(HILAM_DYNAMIC_VARS):
        raise SystemExit(
            "ConvLSTM and JejuHiLAM dynamic-variable orders differ:\n"
            f"ConvLSTM: {dynamic_vars}\nHiLAM: {list(HILAM_DYNAMIC_VARS)}"
        )

    final_cfg = physics_config(args.final_physics)
    qnet_minmax = finite_minmax(ds["qnet"].values)
    vadv_minmax = finite_minmax(ds["vadv_kday"].values)
    failures: list[str] = []

    report: dict[str, Any] = {
        "data_path": str(args.data_path),
        "dataset_dims": {name: int(size) for name, size in ds.sizes.items()},
        "experiment_set": "main_experiment_final",
        "target": "sst_mean",
        "split": "chronological 70/15/15 by window sample count",
        "history_days": args.history,
        "leads": leads,
        "seeds": seeds,
        "persistence": "deterministic, no seed",
        "final_physics": args.final_physics,
        "feature_map_enabled": False,
        "residual_map_loaded": False,
        "dynamic_vars": dynamic_vars,
        "dynamic_var_count": len(dynamic_vars),
        "static_vars": static_vars,
        "static_var_count": len(static_vars),
        "convlstm_expected_channels": args.expected_convlstm_channels,
        "hilam_expected_temporal_channels": args.expected_hilam_temporal_channels,
        "physics_settings": {
            "qnet_formula": "qnet = ssr + str + slhf + sshf",
            "qnet_positive_direction": "ocean warming",
            "u_v": "u_ml, v_ml",
            "zero_rres_physics": True,
            "lambda_physics": args.lambda_physics,
            "lambda_rres": args.lambda_rres,
            "rhs_normalization": "train-fixed",
            "physics_time_mode": "daily_rollout",
            "k_diff_m2_s": args.k_diff,
            "hm_min_m": args.hm_min,
            "vadv_kday_cap": "+/-2.0 K day^-1",
        },
        "final_physics_terms": {
            "physics_preset": final_cfg.physics_preset,
            "physics_components": final_cfg.physics_components,
            "vertical_enabled": bool(final_cfg.physics_vertical),
            "diffusion_enabled": bool(final_cfg.physics_hdiff),
        },
        "available_ablation_terms": [],
        "split_rows": [],
        "channel_checks": [],
        "seed_probes": [seed_probe(seed) for seed in seeds],
        "torch_determinism": {
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
        "qnet_minmax": qnet_minmax,
        "vadv_kday_minmax": vadv_minmax,
    }

    for preset in PHYSICS_PRESETS:
        cfg = physics_config(preset)
        report["available_ablation_terms"].append(
            {
                "preset": preset,
                "physics_components": cfg.physics_components,
                "vertical_enabled": bool(cfg.physics_vertical),
                "diffusion_enabled": bool(cfg.physics_hdiff),
            }
        )

    if args.final_physics != "P0":
        failures.append(f"Final main experiment is expected to use P0, got {args.final_physics}")
    if final_cfg.physics_components != "original":
        failures.append(f"P0 final physics should be original only, got {final_cfg.physics_components}")

    for lead in leads:
        splits = make_time_splits(time_size=int(ds.sizes["time"]), history=args.history, lead=lead)
        report["split_rows"].extend(
            [
                {"lead": lead, **split_row(ds, splits.train, args.history, lead, "train")},
                {"lead": lead, **split_row(ds, splits.val, args.history, lead, "val")},
                {"lead": lead, **split_row(ds, splits.test, args.history, lead, "test")},
            ]
        )

        stats = compute_train_stats(
            ds=ds,
            dynamic_vars=dynamic_vars,
            static_vars=static_vars,
            history=args.history,
            lead=lead,
            train_indices=splits.train,
            residual_maps=None,
        )
        sample_ds = JejuForecastDataset(
            ds=ds,
            history=args.history,
            lead=lead,
            indices=splits.train[:1],
            dynamic_vars=dynamic_vars,
            static_vars=static_vars,
            stats=stats,
            residual_maps=None,
            forcing_mode=args.forcing_mode,
        )
        sample = sample_ds[0]
        convlstm_channels = int(sample["x"].shape[1])
        f_seq_max_abs = float(sample["f_seq"].abs().max().item())

        bundle = make_jeju_hilam_bundle(
            data_path=args.data_path,
            residual_map_path=args.residual_map,
            history=args.history,
            lead_days=lead,
            include_residual_feature=False,
        )
        hilam_channels = len(bundle.train.input_feature_names)

        channel_row = {
            "lead": lead,
            "convlstm_x_channels_without_feature_map": convlstm_channels,
            "convlstm_model_input_channels": convlstm_channels,
            "convlstm_expected_channels": args.expected_convlstm_channels,
            "f_seq_placeholder_channels": int(sample["f_seq"].shape[1]),
            "f_seq_placeholder_max_abs": f_seq_max_abs,
            "hilam_temporal_channels_without_residual": hilam_channels,
            "hilam_expected_temporal_channels": args.expected_hilam_temporal_channels,
            "hilam_node_count": int(bundle.graph.num_ocean_nodes),
            "hilam_static_feature_dim": int(bundle.graph.static_feature_dim),
            "hilam_edge_feature_dim": int(bundle.graph.edge_feature_dim),
            "hilam_input_features": bundle.train.input_feature_names,
        }
        report["channel_checks"].append(channel_row)

        if convlstm_channels != args.expected_convlstm_channels:
            failures.append(
                f"lead{lead} ConvLSTM channels={convlstm_channels}, expected {args.expected_convlstm_channels}"
            )
        if abs(f_seq_max_abs) > 0.0:
            failures.append(f"lead{lead} residual f_seq placeholder is not all-zero")
        if hilam_channels != args.expected_hilam_temporal_channels:
            failures.append(
                f"lead{lead} HiLAM temporal channels={hilam_channels}, "
                f"expected {args.expected_hilam_temporal_channels}"
            )
        if any("residual" in name.lower() for name in bundle.train.input_feature_names):
            failures.append(f"lead{lead} HiLAM still includes a residual input channel")

    print("=== MAIN EXPERIMENT FINAL PREFLIGHT ===")
    print(f"data: {report['data_path']}")
    print(f"target: {report['target']}")
    print(f"split: {report['split']}")
    print(f"final physics: {args.final_physics} -> {final_cfg.physics_components}")
    print(f"feature map enabled: {report['feature_map_enabled']}")
    print(f"dynamic channels ({len(dynamic_vars)}): {', '.join(dynamic_vars)}")
    print(f"static channels ({len(static_vars)}): {', '.join(static_vars)}")
    for row in report["channel_checks"]:
        print(
            f"lead{row['lead']}: ConvLSTM={row['convlstm_model_input_channels']}ch, "
            f"HiLAM temporal={row['hilam_temporal_channels_without_residual']}ch, "
            f"nodes={row['hilam_node_count']}, f_seq_max_abs={row['f_seq_placeholder_max_abs']:.3g}"
        )
    print("target-date ranges:")
    for row in report["split_rows"]:
        print(
            f"  lead{row['lead']} {row['split']}: n={row['n_samples']} "
            f"target={row.get('target_start')}..{row.get('target_end')}"
        )
    print("seed probes:")
    for row in report["seed_probes"]:
        print(
            f"  seed={row['seed']} python={row['python_random']} "
            f"numpy={row['numpy_random']} torch={row['torch_random']}"
        )
    print(f"qnet min/max: {qnet_minmax}")
    print(f"vadv_kday min/max: {vadv_minmax}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"saved: {args.output_json}")

    if failures:
        print("PREFLIGHT FAILED:")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)
    print("PREFLIGHT OK")


if __name__ == "__main__":
    main()
