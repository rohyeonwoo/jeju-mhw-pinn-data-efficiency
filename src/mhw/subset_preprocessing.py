"""Subset-fitted normalization artifacts for the existing SeaCast/P0 datasets.

No model or optimizer is constructed here. Existing NetCDF features (including
anomalies) are retained; their upstream climatologies are not refitted here.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from .datasets import JejuForecastDataset, build_lat_lon_2d, make_time_splits
from .jeju_hilam_dataset import (
    BASE_DYNAMIC_VARS, STATIC_VARS_FOR_BASE_DATASET,
    JejuHiLAMDataBundle, JejuHiLAMDataset,
)
from .jeju_hilam_graph import build_jeju_hilam_graph

SCHEMA = "subset_normalization_v1"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def selected_time_indices(starts, history: int, lead: int, time_size: int):
    raw = np.asarray(starts)
    if raw.ndim != 1 or not raw.size or raw.dtype.kind not in "iu":
        raise ValueError("Expected nonempty integer sample-start indices")
    starts = raw.astype(np.int64)
    if history < 1 or lead < 1 or np.any(np.diff(starts) <= 0):
        raise ValueError("Sample starts must be strictly increasing; history/lead must be positive")
    targets = starts + history - 1 + lead
    if starts[0] < 0 or targets[-1] >= time_size:
        raise ValueError("Sample window is outside the dataset")
    inputs = np.unique((starts[:, None] + np.arange(history)).ravel())
    return inputs, targets


def _mean_std(values):
    if not np.isfinite(values).all():
        raise ValueError("Non-finite ocean values in selected normalization population")
    # Population std (ddof=0), matching the original normalization convention.
    mean = float(np.mean(values, dtype=np.float64))
    std = float(np.std(values, dtype=np.float64))
    return (mean, std if std > 1e-6 else 1.0)


def fit_subset_stats(ds: xr.Dataset, starts, history: int, lead: int) -> dict:
    """Fit inputs on unique history days and target scaling on target days only."""
    inputs, targets = selected_time_indices(starts, history, lead, ds.sizes["time"])
    ocean = np.isfinite(ds["ocean_mask_static"].values)
    if not ocean.any():
        raise ValueError("Empty ocean mask")
    stats = {}
    for name in BASE_DYNAMIC_VARS:
        values = ds[name].isel(time=inputs).values.astype(np.float32)
        stats[name] = _mean_std(values[:, ocean])
    sst_target = ds["sst_mean"].isel(time=targets).values.astype(np.float32)
    stats["target"] = _mean_std(sst_target[:, ocean])
    current = np.asarray(starts) + history - 1
    sst_current = ds["sst_mean"].isel(time=current).values.astype(np.float32)
    tendency = ((sst_target - sst_current) / float(lead))[:, ocean]
    if not np.isfinite(tendency).all():
        raise ValueError("Non-finite selected observed tendency")
    mse = float(np.mean(tendency ** 2))
    stats["observed_tendency_mse_kday2"] = (max(mse, 1e-12), 1.0)
    stats["observed_tendency_rms_kday"] = (max(float(np.sqrt(mse)), 1e-6), 1.0)
    # Fixed geographical information is not estimated from temporal observations.
    lat, lon = build_lat_lon_2d(ds)
    stats.update(ocean_mask_static=(0.0, 1.0), lat2d=_mean_std(lat), lon2d=_mean_std(lon))
    return stats


def p0_scale_args():
    return SimpleNamespace(
        physics_component_list=["original"], physics_components="original",
        physics_preset="P0", physics_hdiff=False, physics_vertical=False,
        entrainment_dt=1.0, hm_min=5.0, physics_normalize_eps=1e-8,
    )


class PhysicsScaleDataset(Dataset):
    """Raw fields only; matches JejuForecastDataset's physics tensors exactly."""
    def __init__(self, base: JejuForecastDataset, starts, lead: int):
        self.base = base
        self.starts = np.asarray(starts, dtype=np.int64)
        self.lead = lead

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, item):
        base = self.base
        current = int(self.starts[item]) + base.history - 1
        clean = lambda x: np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        result = {
            "Tm": clean(base.target_array[current])[None],
            "y_raw": clean(base.target_array[current + self.lead])[None],
            "mask": base.ocean_mask[None], "dx": base.dx[None], "dy": base.dy[None],
        }
        for key, name in (("qnet_seq", "qnet"), ("hm_seq", "mlotst"),
                          ("u_seq", "u_ml"), ("v_seq", "v_ml")):
            result[key] = clean(base.physics_arrays[name][current:current + self.lead])[:, None]
        return {k: torch.from_numpy(v).float() for k, v in result.items()}


def compute_subset_rhs_scale(base, starts, lead: int, batch_size: int = 64) -> float:
    # Import locally to keep dataset loading independent of the trainer module.
    from train_hybrid_jeju_forecast import compute_fixed_rhs_scale
    loader = DataLoader(PhysicsScaleDataset(base, starts, lead), batch_size=batch_size,
                        shuffle=False, num_workers=0, drop_last=False)
    return compute_fixed_rhs_scale(loader, p0_scale_args(), lead, torch.device("cpu"))


def make_raw_base(ds, history: int = 30, lead: int = 7):
    return JejuForecastDataset(
        ds=ds, history=history, lead=lead, indices=np.array([0], dtype=np.int64),
        dynamic_vars=BASE_DYNAMIC_VARS, static_vars=STATIC_VARS_FOR_BASE_DATASET,
        stats={}, residual_maps=None,
    )


def bundle_from_manifest(ds, manifest: dict, base=None) -> JejuHiLAMDataBundle:
    """Apply already-fitted stats to train/val/test without fitting on held-out data."""
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported subset preprocessing manifest")
    history, lead = int(manifest["history"]), int(manifest["lead"])
    starts = np.asarray(manifest["train_start_indices"], dtype=np.int64)
    inputs, targets = selected_time_indices(starts, history, lead, ds.sizes["time"])
    splits = make_time_splits(ds.sizes["time"], history, lead)
    if not np.isin(starts, splits.train).all():
        raise ValueError("Subset contains held-out sample starts")
    for key, expected in (("input_time_indices", inputs), ("target_time_indices", targets),
                          ("val_start_indices", splits.val), ("test_start_indices", splits.test)):
        if not np.array_equal(manifest[key], expected):
            raise ValueError(f"Manifest index mismatch: {key}")
    if manifest["input_feature_names"] != BASE_DYNAMIC_VARS:
        raise ValueError("Input feature schema changed")
    stats = {key: tuple(value) for key, value in manifest["statistics"].items()}
    for key in [*BASE_DYNAMIC_VARS, *STATIC_VARS_FOR_BASE_DATASET, "target"]:
        if key not in stats or len(stats[key]) != 2 or not np.isfinite(stats[key]).all() or stats[key][1] <= 0:
            raise ValueError(f"Invalid statistics: {key}")
    scale = manifest["physics_rhs_scale"]
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid fixed RHS scale")
    graph = build_jeju_hilam_graph(ds, levels=3)
    base = make_raw_base(ds, history, lead) if base is None else base
    def wrapped(indices):
        # Share immutable raw arrays, but keep indices and normalization state local.
        view = copy.copy(base)
        view.history, view.lead = history, lead
        view.indices, view.stats = indices.copy(), stats
        return JejuHiLAMDataset(view, graph, include_residual_feature=False)
    return JejuHiLAMDataBundle(ds, graph, wrapped(starts), wrapped(splits.val),
                              wrapped(splits.test), stats, starts, splits.val, splits.test)


def load_subset_bundle(manifest_path: str | Path, data_path: str | Path | None = None):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") in ("subset_normalization_v2", "subset_normalization_v3"):
        from .subset_climatology import load_v2_bundle
        return load_v2_bundle(manifest_path, data_path)
    path = Path(data_path or manifest["data_path"])
    if sha256_file(path) != manifest["data_sha256"]:
        raise ValueError("Source NetCDF hash differs from the fitted dataset")
    ds = xr.open_dataset(path)
    try:
        return bundle_from_manifest(ds, manifest), manifest
    except Exception:
        ds.close()
        raise


def apply_manifest_physics(args, physics_args, manifest):
    """Prevent a cached Full scale or incompatible budget from overriding this fit."""
    if (args.history, args.lead) != (manifest["history"], manifest["lead"]):
        raise ValueError("CLI history/lead does not match subset manifest")
    args.include_residual_feature = False
    if physics_args is None:
        return
    schema = manifest.get("schema")
    if schema in ("subset_normalization_v1", "subset_normalization_v2"):
        raise ValueError("Legacy RHS scale uses land-contaminated gradients; use a v3 manifest")
    if schema == "subset_normalization_v3":
        from .ocean_stencil import STENCIL
        if manifest.get("physics_stencil") != STENCIL:
            raise ValueError("Manifest stencil differs from the running physics code")
    if args.physics_mode != "p0_original" or args.hm_min != 5.0 or args.entrainment_dt != 1.0:
        raise ValueError("This preprocessing artifact was fitted for P0, hm_min=5, deltaT=1")
    scale = manifest["physics_rhs_scale"]
    if args.physics_rhs_scale is not None and args.physics_rhs_scale != scale:
        raise ValueError("CLI RHS scale conflicts with subset-fitted scale")
    args.physics_rhs_scale = physics_args.physics_rhs_scale = scale
