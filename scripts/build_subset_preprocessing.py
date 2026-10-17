#!/usr/bin/env python3
"""Build 24 subset-fitted dataset manifests on CPU. Never starts training."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

from mhw.datasets import make_time_splits
from mhw.subset_preprocessing import (
    SCHEMA, BASE_DYNAMIC_VARS, PhysicsScaleDataset, bundle_from_manifest,
    compute_subset_rhs_scale, fit_subset_stats, make_raw_base, p0_scale_args,
    selected_time_indices, sha256_file,
)
from train_hybrid_jeju_forecast import compute_fixed_rhs_scale


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def validate_bundle(bundle, manifest, base):
    """Check actual dataset transforms, inverse target transform and raw physics."""
    mean, std = bundle.stats["target"]
    ocean = bundle.graph.ocean_flat_indices
    for dataset in (bundle.train, bundle.val, bundle.test):
        assert dataset.base.stats is bundle.stats
        assert (dataset.target_mean, dataset.target_std) == (mean, std)
        for i in (0, len(dataset) - 1):
            sample = dataset[i]
            a = int(sample["start_index"])
            for channel, name in enumerate(BASE_DYNAMIC_VARS):
                mu, sig = bundle.stats[name]
                raw = base.dynamic_arrays[name][a:a + manifest["history"]]
                expected = ((raw - mu) / sig).reshape(manifest["history"], -1)[:, ocean].T
                np.testing.assert_allclose(sample["x_nodes"][:, :, channel], expected, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(sample["y_nodes"] * std + mean, sample["y_raw_nodes"], rtol=1e-6, atol=3e-6)
    raw_view = copy.copy(base)
    raw_view.indices = bundle.train_indices[:3]
    raw_view.lead = manifest["lead"]
    raw_view.stats = bundle.stats
    quick = PhysicsScaleDataset(base, raw_view.indices, manifest["lead"])
    for i in range(len(raw_view)):
        reference = raw_view[i]
        for key, value in quick[i].items():
            torch.testing.assert_close(value, reference[key], rtol=0, atol=0)
    reference_scale = compute_fixed_rhs_scale(
        DataLoader(raw_view, batch_size=3), p0_scale_args(), manifest["lead"], torch.device("cpu"))
    fast_scale = compute_subset_rhs_scale(base, raw_view.indices, manifest["lead"], batch_size=3)
    np.testing.assert_allclose(fast_scale, reference_scale, rtol=0, atol=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", type=Path, default=ROOT / "data/merged_sst_cmems_era5_wind10_qnet_jeju_1993_2022_FINAL_staticmask_filled_glorys3d_budget2d.nc")
    p.add_argument("--output-dir", type=Path, default=ROOT / "data/subsets/preprocessing_subset_v1")
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory; existing artifacts are never overwritten")
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True)
    data_sha = sha256_file(args.data_path)
    source_hashes = {str(f.relative_to(ROOT)): sha256_file(f) for f in (
        ROOT / "src/mhw/subset_preprocessing.py", ROOT / "src/mhw/datasets.py",
        ROOT / "src/mhw/jeju_hilam_dataset.py", ROOT / "src/train_hybrid_jeju_forecast.py",
        ROOT / "src/train_jeju_hilam.py", Path(__file__),
    )}
    rows = []
    with xr.open_dataset(args.data_path) as ds:
        dates = ds.time.values.astype("datetime64[D]")
        assert np.all(np.diff(dates) == np.timedelta64(1, "D"))
        needed = list(dict.fromkeys([*BASE_DYNAMIC_VARS, "qnet", "ocean_mask_static",
                                    *[n for n in ("u_mld", "v_mld", "vadv_kday") if n in ds]]))
        ds = ds[needed].load()
        base = make_raw_base(ds)
        for seed in (42, 7, 123):
            subset_path = ROOT / f"data/subsets/train_subsets_v2_win44_seed{seed}.json"
            subsets = json.loads(subset_path.read_text())
            assert subsets["_metadata"]["source"]["data_sha256"] == data_sha
            previous = set()
            for ratio in (10, 25, 50, 100):
                starts = np.asarray(subsets[str(ratio)]["start_indices"], dtype=np.int64)
                assert len(starts) == {10: 376, 25: 987, 50: 1974, 100: 3995}[ratio]
                assert previous < set(starts)
                previous = set(starts)
                for lead in (7, 14):
                    print(f"Fitting seed={seed} ratio={ratio} lead={lead}", flush=True)
                    inputs, targets = selected_time_indices(starts, 30, lead, len(dates))
                    splits = make_time_splits(len(dates), 30, lead)
                    assert np.isin(starts, splits.train).all()
                    heldout_targets = np.concatenate([splits.val, splits.test]) + 29 + lead
                    assert not np.intersect1d(inputs, heldout_targets).size
                    assert not np.intersect1d(targets, heldout_targets).size
                    stats = fit_subset_stats(ds, starts, 30, lead)
                    scale = compute_subset_rhs_scale(base, starts, lead)
                    manifest = {
                        "schema": SCHEMA, "history": 30, "lead": lead, "subset_seed": seed,
                        "subset_ratio": ratio, "data_path": str(args.data_path.resolve()),
                        "data_sha256": data_sha, "subset_path": str(subset_path.resolve()),
                        "subset_sha256": sha256_file(subset_path), "source_sha256": source_hashes,
                        "input_feature_names": BASE_DYNAMIC_VARS, "include_residual_feature": False,
                        "train_start_indices": starts.tolist(), "val_start_indices": splits.val.tolist(),
                        "test_start_indices": splits.test.tolist(), "input_time_indices": inputs.tolist(),
                        "target_time_indices": targets.tolist(), "statistics": stats,
                        "statistics_basis": "selected unique history days for inputs; selected target days for target; ocean cells; ddof=0",
                        "input_date_range": [str(dates[inputs[0]]), str(dates[inputs[-1]])],
                        "target_date_range": [str(dates[targets[0]]), str(dates[targets[-1]])],
                        "physics_rhs_scale": scale, "physics_rhs_scale_unit": "(K/day)^2",
                        "physics_scale_basis": "selected training samples only; observed linear path; we=0; pooled mean(RHS^2)",
                        "physics_scale_definition": {"preset": "P0", "hm_min": 5.0, "entrainment_dt": 1.0, "eps": 1e-8},
                        "upstream_features": "Existing NetCDF anomaly features retained; upstream climatologies not refitted or certified subset-only",
                    }
                    bundle = bundle_from_manifest(ds, manifest, base=base)
                    validate_bundle(bundle, manifest, base)
                    name = f"ratio_{ratio}_lead{lead}_seed{seed}.json"
                    manifest["validation"] = "passed: indices, held-out targets excluded, transforms, inverse transform, raw physics parity, RHS reference parity"
                    write_json(args.output_dir / name, manifest)
                    row = {"manifest": name, "sha256": sha256_file(args.output_dir / name),
                           "ratio": ratio, "seed": seed, "lead": lead,
                           "train_samples": len(starts), "input_unique_days": len(inputs),
                           "target_days": len(targets), "val_samples": len(splits.val),
                           "test_samples": len(splits.test), "target_mean": stats["target"][0],
                           "target_std": stats["target"][1], "physics_rhs_scale": scale}
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    write_json(args.output_dir / "index.json", {
        "schema": SCHEMA, "status": "validated", "training_executed": False,
        "artifact_count": len(rows), "datasets": rows,
    })


if __name__ == "__main__":
    main()
