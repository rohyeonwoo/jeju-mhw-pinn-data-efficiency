"""Inspect Full statistics and subset mappings without training or editing src.

Run from the mhw_subset environment with python -B. Reports go to
docs/server_preflight; original data, source, and Full results are read only.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

# Set the server-required name before importing GPU modules.
import setproctitle
setproctitle.setproctitle("yeonwoo")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import Subset
import train_jeju_hilam as training
from mhw.datasets import make_time_splits
from mhw.jeju_hilam_dataset import make_jeju_hilam_bundle


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = ROOT / "docs" / "server_preflight"
    output.mkdir(exist_ok=True)
    snapshot = json.loads((output / "original_snapshot_sha256.json").read_text())
    protected = {p: h for p, h in snapshot.items()
                 if p.startswith(("src/", "data/", "outputs/main_experiment_final/"))
                 or p == "scripts/make_train_subsets.py"}
    for name, expected in protected.items():
        assert sha256(ROOT / name) == expected, name

    environment = {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "packages": {p: importlib.metadata.version(p) for p in
                     ("torch", "numpy", "pandas", "xarray", "h5py", "h5netcdf", "scikit-learn", "matplotlib")},
        "gpus": [{"index": i, "name": torch.cuda.get_device_name(i),
                  "memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                 for i in range(torch.cuda.device_count())],
    }
    device = torch.device(args.device)
    assert device.type == "cuda" and torch.cuda.is_available()
    # Exercise actual CUDA kernels and autograd, without a forecasting model.
    tensor = torch.arange(16, device=device, dtype=torch.float32).requires_grad_()
    tensor.square().sum().backward()
    torch.testing.assert_close(tensor.grad, 2 * tensor.detach())
    environment["cuda_autograd_check"] = "passed"
    save(output / "environment.json", environment)

    data = next((ROOT / "data").glob("*.nc"))
    subsets = {seed: json.loads((ROOT / "data/subsets" /
                                f"train_subsets_v2_win44_seed{seed}.json").read_text())
               for seed in (42, 7, 123)}
    report = {"workspace": str(ROOT), "data_sha256": sha256(data),
              "training_performed": False, "leads": {}, "protected_file_count": len(protected)}
    for lead in (7, 14):
        print(f"lead={lead}: building original Full bundle", flush=True)
        bundle = make_jeju_hilam_bundle(data, None, 30, lead, include_residual_feature=False)
        split = make_time_splits(int(bundle.ds.sizes["time"]), 30, lead)
        before = {name: getattr(bundle, name + "_indices").copy()
                  for name in ("train", "val", "test")}
        for name in before:
            np.testing.assert_array_equal(before[name], getattr(split, name))
            np.testing.assert_array_equal(getattr(bundle, name).base.indices, before[name])
        stats_before = json.dumps(bundle.stats, sort_keys=True)
        full_configs = []
        for model in ("jeju_hilam", "pi_jeju_hilam"):
            for seed in (42, 7, 123):
                path = ROOT / "outputs/main_experiment_final" / model / f"lead{lead}_seed{seed}" / "jeju_hilam_config.json"
                config = json.loads(path.read_text())
                expected = {"history": 30, "lead": lead, "seed": seed, "batch_size": 8,
                            "epochs": 80, "patience": 12, "hidden_dim": 256,
                            "temporal_hidden_dim": 256, "processor_layers": 3,
                            "mlp_hidden_layers": 1, "lr": 2e-4, "initial_lr": 2e-5,
                            "weight_decay": 0.1, "include_residual_feature": False,
                            "prediction_mode": "direct", "input_dim": 19,
                            "physics_mode": "none" if model == "jeju_hilam" else "p0_original"}
                if model == "pi_jeju_hilam":
                    expected.update(lambda_physics=0.1, lambda_rres=10.0,
                                    zero_rres_physics=True, physics_normalize="rhs")
                for key, value in expected.items():
                    assert config[key] == value, (path, key, config[key], value)
                for name in before:
                    assert config[name + "_samples"] == len(before[name])
                np.testing.assert_allclose(bundle.stats["target"],
                                           [config["target_mean"], config["target_std"]],
                                           rtol=1e-6, atol=1e-6)
                full_configs.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                                     "saved_rhs_scale": config.get("physics_rhs_scale")})

        subset_checks = []
        for seed, payload in subsets.items():
            previous = set()
            for ratio, expected_count in ((10, 376), (25, 987), (50, 1974), (100, 3995)):
                starts = np.asarray(payload[str(ratio)]["start_indices"], dtype=np.int64)
                positions = np.searchsorted(before["train"], starts)
                assert len(starts) == expected_count
                assert np.all(positions < len(before["train"]))
                np.testing.assert_array_equal(before["train"][positions], starts)
                assert previous <= set(starts)
                previous = set(starts)
                selected = Subset(bundle.train, positions.tolist())
                assert selected.dataset.base.stats is bundle.stats
                # Check real loader samples, including both ends and an interior point.
                for index in (0, len(selected) // 2, len(selected) - 1):
                    sample = selected[index]
                    assert int(sample["current_index"]) == int(starts[index]) + 29
                    assert int(sample["target_index"]) == int(starts[index]) + 29 + lead
                subset_checks.append({"seed": seed, "ratio": ratio, "samples": len(selected),
                                      "start_indices_sha256": hashlib.sha256(starts.astype("<i8").tobytes()).hexdigest()})
        for name in before:
            np.testing.assert_array_equal(getattr(bundle, name + "_indices"), before[name])
        assert json.dumps(bundle.stats, sort_keys=True) == stats_before

        # Parse original defaults without modifying the original CLI or source.
        saved_argv = sys.argv
        try:
            sys.argv = ["train_jeju_hilam.py", "--lead", str(lead),
                        "--physics-mode", "p0_original", "--no-residual-feature"]
            train_args = training.parse_args()
        finally:
            sys.argv = saved_argv
        physics = training.build_physics_args(train_args)
        print(f"lead={lead}: computing RHS scale over {len(bundle.train)} Full samples", flush=True)
        rhs = training.compute_fixed_rhs_scale(training.make_loader(bundle.train, train_args, False),
                                               physics, lead, device)
        saved_rhs = [c["saved_rhs_scale"] for c in full_configs if c["saved_rhs_scale"] is not None]
        np.testing.assert_allclose(rhs, saved_rhs, rtol=1e-6, atol=1e-6)
        S = (len(bundle.train) + 7) // 8
        ref_end = round(int(bundle.ds.sizes["time"]) * 0.70) - 1
        dates = bundle.ds.time.values.astype("datetime64[D]")
        lead_report = {
            "counts": {name: len(indices) for name, indices in before.items()},
            "split_sha256": {name: hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()
                             for name, indices in before.items()},
            "full_stats": bundle.stats, "full_rhs_scale": rhs, "saved_rhs_scales": saved_rhs,
            "statistics_end_date": str(dates[int(before["train"][-1]) + 29 + lead]),
            "evaluation_reference_end_date": str(dates[ref_end]),
            "S": S, "max_steps": 80 * S, "warmup_steps": 5 * S,
            "early_stopping_patience_validations": 12,
            "subset_checks": subset_checks, "full_configs": full_configs,
        }
        report["leads"][str(lead)] = lead_report
        save(output / f"full_statistics_lead{lead}.json", lead_report)
        bundle.ds.close()
        del bundle, selected, sample
        gc.collect()
        torch.cuda.empty_cache()
        print(f"lead={lead}: PASS, RHS={rhs:.10g}, S={S}", flush=True)

    for a, b in zip(report["leads"]["7"]["subset_checks"], report["leads"]["14"]["subset_checks"]):
        assert a == b
    for name, expected in protected.items():
        assert sha256(ROOT / name) == expected, name
    report["protected_files_unchanged"] = True
    report["status"] = "passed"
    save(output / "audit_report.json", report)
    print("PASS: Full configurations/statistics, train-only subset mappings, original val/test, source preservation", flush=True)


if __name__ == "__main__":
    main()
