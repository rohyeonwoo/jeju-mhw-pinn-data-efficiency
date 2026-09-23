#!/usr/bin/env python3
"""Train a Jeju-adapted HiLAM baseline for lead-specific SST prediction."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader, Subset

from train_hybrid_jeju_forecast import (
    SECONDS_PER_DAY,
    _linear_rhs_per_day,
    compute_fixed_rhs_scale,
    daily_rollout_physics_loss,
    finalize_budget_metrics,
    identifiability_terms,
    physics_budget_diagnostics,
    physics_lambda_factor,
)
from mhw.eval_metrics import compute_forecast_metrics
from mhw.jeju_hilam_dataset import make_jeju_hilam_bundle
from mhw.jeju_hilam_graph import (
    DEFAULT_SW_BOUNDS,
    maybe_plot_graph,
    nodes_to_grid,
    save_graph_artifacts,
)
from mhw.mhw_skill import compute_climatology_threshold, day_of_year
from mhw.models.jeju_hilam import JejuHiLAM


DEFAULT_DATA_PATH = Path("D:/MHW/data/merged_sst_cmems_era5_wind10_qnet_jeju_1993_2022_FINAL_staticmask_filled_glorys3d_budget2d.nc")
DEFAULT_RESIDUAL_PATH = Path("D:/MHW/outputs/newphys_inverse_feature_map/f_theta_maps.npy")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Jeju-adapted HiLAM baseline")
    p.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    p.add_argument("--residual-map", type=Path, default=DEFAULT_RESIDUAL_PATH)
    p.add_argument("--output-root", type=Path, default=Path("D:/MHW/outputs/jeju_hilam"))
    p.add_argument("--history", type=int, default=30)
    p.add_argument("--lead", type=int, required=True, choices=[1, 3, 7, 14])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--temporal-hidden-dim", type=int, default=256)
    p.add_argument("--processor-layers", type=int, default=3)
    p.add_argument("--mlp-hidden-layers", type=int, default=1)
    p.add_argument("--lr", type=float, default=2.0e-4)
    p.add_argument("--initial-lr", type=float, default=2.0e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--selection-metric", choices=["val_rmse", "val_loss"], default="val_rmse")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--prediction-mode", choices=["direct", "residual"], default="direct")
    p.add_argument("--physics-mode", choices=["none", "p0_original", "p1_vertical"], default="none")
    p.add_argument("--lambda-physics", type=float, default=0.1)
    p.add_argument("--physics-warmup-epochs", type=int, default=5)
    p.add_argument("--physics-rhs-scale", type=float, default=None)
    p.add_argument("--physics-normalize-eps", type=float, default=1.0e-8)
    p.add_argument("--physics-clip", type=float, default=10.0)
    p.add_argument("--lambda-rres", type=float, default=10.0)
    p.add_argument("--rres-penalty-normalize", choices=["head", "tendency", "rhs"], default="rhs")
    p.add_argument("--lambda-we-tie", type=float, default=0.0)
    p.add_argument("--entrainment-dt", type=float, default=1.0)
    p.add_argument("--hm-min", type=float, default=5.0)
    p.add_argument("--we-scale", type=float, default=1.0e-5)
    p.add_argument("--rres-scale", type=float, default=1.0e-4)
    p.add_argument("--we-softplus-shift", type=float, default=1.0)
    p.add_argument("--no-zero-rres-physics", dest="zero_rres_physics", action="store_false", default=True)
    p.add_argument("--include-residual-feature", dest="include_residual_feature", action="store_true", default=True)
    p.add_argument("--no-residual-feature", dest="include_residual_feature", action="store_false")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--tiny-overfit", action="store_true")
    p.add_argument("--tiny-samples", type=int, default=32)
    p.add_argument("--tiny-overfit-steps", type=int, default=20)
    p.add_argument("--plot-graph", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def build_physics_args(args: argparse.Namespace) -> argparse.Namespace | None:
    if args.physics_mode == "none":
        return None
    if args.physics_mode == "p0_original":
        components = ["original"]
        preset = "P0"
    elif args.physics_mode == "p1_vertical":
        components = ["original", "vertical"]
        preset = "P1"
    else:
        raise ValueError(f"Unsupported physics mode: {args.physics_mode}")

    return argparse.Namespace(
        physics_component_list=components,
        physics_components=",".join(components),
        physics_preset=preset,
        physics_hdiff=False,
        physics_vertical="vertical" in components,
        k_diff=100.0,
        physics_loss="mse",
        physics_huber_beta=0.25,
        physics_clip=float(args.physics_clip),
        physics_normalize="rhs",
        physics_normalize_eps=float(args.physics_normalize_eps),
        physics_tendency_scale=None,
        physics_rhs_scale=args.physics_rhs_scale,
        adaptive_sampling=False,
        adaptive_weighting=False,
        adaptive_temperature=1.0,
        adaptive_samples=256,
        adaptive_weight_min=0.1,
        adaptive_weight_max=10.0,
        zero_rres_physics=bool(args.zero_rres_physics),
        entrainment_dt=float(args.entrainment_dt),
        hm_min=float(args.hm_min),
        physics_warmup_epochs=int(args.physics_warmup_epochs),
        rres_penalty_normalize=str(args.rres_penalty_normalize),
        lambda_rres=float(args.lambda_rres),
        lambda_we_tie=float(args.lambda_we_tie),
    )


def nodes_to_grid_torch(node_values: torch.Tensor, graph, fill_value: float = 0.0) -> torch.Tensor:
    if node_values.ndim != 2:
        raise ValueError(f"Expected [B,N] node values, got {tuple(node_values.shape)}")
    flat = node_values.new_full(
        (node_values.shape[0], graph.height * graph.width),
        float(fill_value),
    )
    ocean_idx = torch.as_tensor(graph.ocean_flat_indices, device=node_values.device, dtype=torch.long)
    flat[:, ocean_idx] = node_values
    return flat.view(node_values.shape[0], 1, graph.height, graph.width)


def physics_raw_residual_stats(
    pred_raw: torch.Tensor,
    tm: torch.Tensor,
    mask: torch.Tensor,
    batch: dict[str, torch.Tensor],
    physics_args: argparse.Namespace,
    lead_days: int,
    we_pred: torch.Tensor,
    rres_pred: torch.Tensor,
) -> dict[str, float]:
    rhs_per_day, rhs_mask = _linear_rhs_per_day(
        pred_raw, tm, mask, batch, physics_args, lead_days, we_pred
    )
    steps = max(int(lead_days), 1)
    dtdt_terms = []
    prev = tm
    for step in range(1, steps + 1):
        frac = float(step) / float(steps)
        next_state = tm + frac * (pred_raw - tm)
        dtdt_terms.append((next_state - prev) * mask)
        prev = next_state
    dtdt_per_day = torch.cat(dtdt_terms, dim=0)
    rres_term = torch.zeros_like(rres_pred) if getattr(physics_args, "zero_rres_physics", False) else rres_pred
    residual = dtdt_per_day - rhs_per_day - rres_term.repeat(steps, 1, 1, 1) * SECONDS_PER_DAY
    valid = rhs_mask
    count = float(valid.sum().detach().cpu().item())
    if count <= 0:
        return {"common_res_abs_sum": 0.0, "common_res_sq_sum": 0.0, "common_res_count": 0.0}
    return {
        "common_res_abs_sum": float((residual.abs() * valid).sum().detach().cpu().item()),
        "common_res_sq_sum": float(((residual**2) * valid).sum().detach().cpu().item()),
        "common_res_count": count,
    }


def make_loader(dataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(args.seed + (17 if shuffle else 0))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        generator=generator,
    )


def set_epoch_lr(optimizer: torch.optim.Optimizer, args: argparse.Namespace, epoch: int) -> float:
    if args.epochs <= 1:
        lr = args.lr
    elif epoch <= args.warmup_epochs:
        frac = epoch / max(args.warmup_epochs, 1)
        lr = args.initial_lr + frac * (args.lr - args.initial_lr)
    else:
        denom = max(args.epochs - args.warmup_epochs, 1)
        progress = min(max((epoch - args.warmup_epochs) / denom, 0.0), 1.0)
        lr = 0.5 * args.lr * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def batch_raw_stats(
    pred_norm: torch.Tensor,
    y_norm: torch.Tensor,
    y_raw: torch.Tensor,
    target_mean: float,
    target_std: float,
) -> dict[str, float]:
    pred_raw = pred_norm * target_std + target_mean
    err = pred_raw - y_raw
    count = float(err.numel())
    return {
        "norm_sse": float(F.mse_loss(pred_norm, y_norm, reduction="sum").detach().cpu().item()),
        "raw_sse": float((err**2).sum().detach().cpu().item()),
        "raw_abs": float(err.abs().sum().detach().cpu().item()),
        "raw_bias": float(err.sum().detach().cpu().item()),
        "count": count,
    }


def run_epoch(
    model: JejuHiLAM,
    loader: DataLoader,
    graph,
    device: torch.device,
    target_mean: float,
    target_std: float,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
    precision: str = "fp32",
    physics_args: argparse.Namespace | None = None,
    lambda_physics: float = 0.0,
    epoch: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"norm_sse": 0.0, "raw_sse": 0.0, "raw_abs": 0.0, "raw_bias": 0.0, "count": 0.0}
    loss_totals = {"total": 0.0, "data": 0.0, "physics": 0.0, "rres": 0.0, "we_tie": 0.0}
    budget_sums: dict[str, float] = {}
    common_sums = {"common_res_abs_sum": 0.0, "common_res_sq_sum": 0.0, "common_res_count": 0.0}
    batch_count = 0
    lambda_eff = (
        float(lambda_physics) * physics_lambda_factor(epoch, int(getattr(physics_args, "physics_warmup_epochs", 0)))
        if physics_args is not None
        else 0.0
    )
    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if precision == "bf16" and device.type == "cuda"
        else nullcontext()
    )

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x_nodes = batch["x_nodes"].to(device, non_blocking=True)
        y_nodes = batch["y_nodes"].to(device, non_blocking=True)
        y_raw_nodes = batch["y_raw_nodes"].to(device, non_blocking=True)
        base_nodes = batch["tm_norm_nodes"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)
        with autocast:
            output = model(x_nodes, base_nodes=base_nodes, return_aux=physics_args is not None)
            if physics_args is None:
                pred_norm = output
                data_loss = F.mse_loss(pred_norm, y_nodes)
                phys_loss = pred_norm.new_zeros(())
                rres_l2 = pred_norm.new_zeros(())
                we_tie = pred_norm.new_zeros(())
                total_loss = data_loss
            else:
                pred_norm = output["pred"]
                data_loss = F.mse_loss(pred_norm, y_nodes)
                pred_raw_grid = nodes_to_grid_torch(pred_norm * target_std + target_mean, graph)
                we_grid = nodes_to_grid_torch(output["we"], graph)
                rres_grid = nodes_to_grid_torch(output["rres"], graph)
                tm = batch["Tm"].to(device, non_blocking=True)
                y_raw_grid = batch["y_raw"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                phys_loss = daily_rollout_physics_loss(
                    pred_raw=pred_raw_grid,
                    Tm=tm,
                    target_raw=y_raw_grid,
                    mask=mask,
                    batch=batch,
                    args=physics_args,
                    lead_days=int(batch["target_index"][0].item() - batch["current_index"][0].item()),
                    we_pred=we_grid,
                    rres_pred=rres_grid,
                )
                rres_l2, we_tie, _ = identifiability_terms(
                    we_pred=we_grid,
                    rres_pred=rres_grid,
                    pred_raw=pred_raw_grid,
                    batch=batch,
                    mask=mask,
                    lead_days=int(batch["target_index"][0].item() - batch["current_index"][0].item()),
                    args=physics_args,
                    we_scale=float(getattr(model, "we_scale", 1.0e-5)),
                    rres_scale=float(getattr(model, "rres_scale", 1.0e-4)),
                )
                total_loss = (
                    data_loss
                    + lambda_eff * phys_loss
                    + float(getattr(physics_args, "lambda_rres", 0.0)) * rres_l2
                    + float(getattr(physics_args, "lambda_we_tie", 0.0)) * we_tie
                )
        if training:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        stats = batch_raw_stats(pred_norm.detach(), y_nodes, y_raw_nodes, target_mean, target_std)
        for key, value in stats.items():
            totals[key] += value
        loss_totals["total"] += float(total_loss.detach().cpu().item())
        loss_totals["data"] += float(data_loss.detach().cpu().item())
        loss_totals["physics"] += float(phys_loss.detach().cpu().item())
        loss_totals["rres"] += float(rres_l2.detach().cpu().item())
        loss_totals["we_tie"] += float(we_tie.detach().cpu().item())
        batch_count += 1

        if physics_args is not None:
            with torch.no_grad():
                pred_raw_grid = nodes_to_grid_torch(pred_norm.detach() * target_std + target_mean, graph)
                we_grid = nodes_to_grid_torch(output["we"].detach(), graph)
                rres_grid = nodes_to_grid_torch(output["rres"].detach(), graph)
                tm = batch["Tm"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                lead_days = int(batch["target_index"][0].item() - batch["current_index"][0].item())
                for key, value in physics_budget_diagnostics(
                    pred_raw=pred_raw_grid,
                    Tm=tm,
                    mask=mask,
                    batch=batch,
                    args=physics_args,
                    lead_days=lead_days,
                    we_pred=we_grid,
                    rres_pred=rres_grid,
                ).items():
                    budget_sums[key] = budget_sums.get(key, 0.0) + value
                for key, value in physics_raw_residual_stats(
                    pred_raw=pred_raw_grid,
                    tm=tm,
                    mask=mask,
                    batch=batch,
                    physics_args=physics_args,
                    lead_days=lead_days,
                    we_pred=we_grid,
                    rres_pred=rres_grid,
                ).items():
                    common_sums[key] += value

    count = max(totals["count"], 1.0)
    denom = max(batch_count, 1)
    result = {
        "loss": totals["norm_sse"] / count,
        "total_loss": loss_totals["total"] / denom,
        "data_loss": loss_totals["data"] / denom,
        "physics_loss": loss_totals["physics"] / denom,
        "rres_l2": loss_totals["rres"] / denom,
        "we_tie_loss": loss_totals["we_tie"] / denom,
        "lambda_physics_effective": lambda_eff,
        "rmse": math.sqrt(totals["raw_sse"] / count),
        "mae": totals["raw_abs"] / count,
        "bias": totals["raw_bias"] / count,
    }
    if physics_args is not None:
        pooled_budget = finalize_budget_metrics(budget_sums)
        result.update(pooled_budget)
        common_count = max(common_sums["common_res_count"], 1.0)
        result["common_res_absmean_kday"] = common_sums["common_res_abs_sum"] / common_count
        result["common_res_rmse_kday"] = math.sqrt(common_sums["common_res_sq_sum"] / common_count)
    return result


def collect_predictions(
    model: JejuHiLAM,
    loader: DataLoader,
    graph,
    device: torch.device,
    target_mean: float,
    target_std: float,
    precision: str,
    physics_args: argparse.Namespace | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    pred_grids: list[np.ndarray] = []
    target_grids: list[np.ndarray] = []
    mask_grids: list[np.ndarray] = []
    we_grids: list[np.ndarray] = []
    rres_grids: list[np.ndarray] = []
    target_indices: list[np.ndarray] = []
    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if precision == "bf16" and device.type == "cuda"
        else nullcontext()
    )

    with torch.no_grad():
        for batch in loader:
            x_nodes = batch["x_nodes"].to(device, non_blocking=True)
            base_nodes = batch["tm_norm_nodes"].to(device, non_blocking=True)
            with autocast:
                output = model(x_nodes, base_nodes=base_nodes, return_aux=physics_args is not None)
            pred_norm = output["pred"] if physics_args is not None else output
            pred_raw_nodes = (pred_norm.float().cpu().numpy() * target_std) + target_mean
            pred_grids.append(nodes_to_grid(pred_raw_nodes, graph, fill_value=0.0))
            if physics_args is not None:
                we_grids.append(nodes_to_grid(output["we"].float().cpu().numpy(), graph, fill_value=0.0))
                rres_grids.append(nodes_to_grid(output["rres"].float().cpu().numpy(), graph, fill_value=0.0))
            target_grids.append(batch["y_raw_grid"].numpy())
            mask_grids.append(batch["mask_grid"].numpy())
            target_indices.append(batch["target_index"].numpy())

    out = {
        "pred_raw": np.concatenate(pred_grids, axis=0).astype(np.float32),
        "target_raw": np.concatenate(target_grids, axis=0).astype(np.float32),
        "mask": np.concatenate(mask_grids, axis=0).astype(np.float32),
        "target_index": np.concatenate(target_indices, axis=0).astype(np.int64),
    }
    if physics_args is not None:
        out["we_raw"] = np.concatenate(we_grids, axis=0).astype(np.float32)
        out["rres_raw"] = np.concatenate(rres_grids, axis=0).astype(np.float32)
        out["rres_kday"] = (out["rres_raw"] * SECONDS_PER_DAY).astype(np.float32)
    return out


def anomaly_acc(pred: np.ndarray, obs: np.ndarray, clim: np.ndarray, mask: np.ndarray) -> float:
    weights = mask.astype(bool) & np.isfinite(pred) & np.isfinite(obs) & np.isfinite(clim)
    if not weights.any():
        return float("nan")
    pa = pred - clim
    oa = obs - clim
    numerator = float(np.sum(pa[weights] * oa[weights]))
    pred_norm = float(np.sum(pa[weights] ** 2))
    obs_norm = float(np.sum(oa[weights] ** 2))
    if pred_norm <= 0.0 or obs_norm <= 0.0:
        return float("nan")
    return numerator / math.sqrt(pred_norm * obs_norm)


def masked_rmse_bias(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    valid = mask.astype(bool) & np.isfinite(pred) & np.isfinite(obs)
    if not valid.any():
        return float("nan"), float("nan")
    err = pred[valid] - obs[valid]
    return float(np.sqrt(np.mean(err**2))), float(np.mean(err))


def add_acc_and_sw_metrics(metrics: dict[str, float], ds: xr.Dataset, fields: dict[str, np.ndarray]) -> None:
    pred = fields["pred_raw"][:, 0]
    obs = fields["target_raw"][:, 0]
    mask = fields["mask"][:, 0] > 0.5
    target_index = fields["target_index"]
    time_values = ds["time"].values
    doy = day_of_year(time_values)
    sst = ds["sst_mean"].values.astype(np.float32)

    ref_mask = np.zeros(int(ds.sizes["time"]), dtype=bool)
    ref_mask[: int(round(int(ds.sizes["time"]) * 0.70))] = True
    clim, _ = compute_climatology_threshold(
        sst=sst,
        doy=doy,
        reference_time_mask=ref_mask,
        window=5,
        quantile=0.90,
    )
    clim_target = clim[doy[target_index]]
    metrics["test_acc"] = anomaly_acc(pred, obs, clim_target, mask)

    lat_name = "lat" if "lat" in ds else "latitude"
    lon_name = "lon" if "lon" in ds else "longitude"
    lat = ds[lat_name].values
    lon = ds[lon_name].values
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lat2d, lon2d = lat, lon
    sw_mask = (
        mask[0]
        & (lat2d >= DEFAULT_SW_BOUNDS["lat_min"])
        & (lat2d <= DEFAULT_SW_BOUNDS["lat_max"])
        & (lon2d >= DEFAULT_SW_BOUNDS["lon_min"])
        & (lon2d <= DEFAULT_SW_BOUNDS["lon_max"])
    )
    months = time_values[target_index].astype("datetime64[M]").astype(int) % 12 + 1
    summer = np.isin(months, [6, 7, 8])
    sw_summer_mask = summer[:, None, None] & sw_mask[None, :, :]
    sw_rmse, sw_bias = masked_rmse_bias(pred, obs, sw_summer_mask)
    metrics["sw_summer_rmse"] = sw_rmse
    metrics["sw_summer_bias"] = sw_bias
    metrics["sw_summer_acc"] = anomaly_acc(pred, obs, clim_target, sw_summer_mask)
    metrics["sw_summer_target_days"] = int(summer.sum())
    metrics["sw_summer_cells"] = int(sw_mask.sum())


def save_epoch_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


def run_smoke(args: argparse.Namespace, model: JejuHiLAM, loader: DataLoader, device: torch.device) -> None:
    batch = next(iter(loader))
    x_nodes = batch["x_nodes"].to(device)
    base_nodes = batch["tm_norm_nodes"].to(device)
    with torch.no_grad():
        pred = model(x_nodes, base_nodes=base_nodes)
    print("Smoke forward ok")
    print(f"x_nodes={tuple(x_nodes.shape)} pred={tuple(pred.shape)} y={tuple(batch['y_nodes'].shape)}")
    print(f"pred finite={bool(torch.isfinite(pred).all().item())}")

    if args.tiny_overfit:
        print("Tiny overfit check")
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        subset = Subset(loader.dataset, list(range(min(args.tiny_samples, len(loader.dataset)))))
        tiny_loader = DataLoader(subset, batch_size=min(args.batch_size, args.tiny_samples), shuffle=True)
        first_loss = None
        last_loss = None
        tiny_iter = iter(tiny_loader)
        for _ in range(args.tiny_overfit_steps):
            try:
                tiny_batch = next(tiny_iter)
            except StopIteration:
                tiny_iter = iter(tiny_loader)
                tiny_batch = next(tiny_iter)
            tx = tiny_batch["x_nodes"].to(device)
            ty = tiny_batch["y_nodes"].to(device)
            tb = tiny_batch["tm_norm_nodes"].to(device)
            opt.zero_grad(set_to_none=True)
            out = model(tx, base_nodes=tb)
            loss = F.mse_loss(out, ty)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu().item())
            if first_loss is None:
                first_loss = last_loss
        print(f"tiny_loss_first={first_loss} tiny_loss_last={last_loss}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    run_dir = args.output_root / f"lead{args.lead}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    physics_args = build_physics_args(args)
    model_id = "pi_jeju_hilam" if physics_args is not None else "jeju_hilam"
    model_label = (
        f"PI-JejuHiLAM / {physics_args.physics_preset} physics"
        if physics_args is not None
        else "SeaCast-inspired HiLAM / JejuHiLAM"
    )

    bundle = make_jeju_hilam_bundle(
        data_path=args.data_path,
        residual_map_path=args.residual_map,
        history=args.history,
        lead_days=args.lead,
        include_residual_feature=args.include_residual_feature,
    )
    save_graph_artifacts(bundle.graph, run_dir)
    if args.plot_graph:
        maybe_plot_graph(bundle.graph, run_dir / "jeju_hilam_graph.png")

    input_dim = len(bundle.train.input_feature_names)
    model = JejuHiLAM(
        graph=bundle.graph,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        temporal_hidden_dim=args.temporal_hidden_dim,
        processor_layers=args.processor_layers,
        mlp_hidden_layers=args.mlp_hidden_layers,
        prediction_mode=args.prediction_mode,
        physics_heads=physics_args is not None,
        we_nonneg=True,
        we_scale=args.we_scale,
        rres_scale=args.rres_scale,
        we_softplus_shift=args.we_softplus_shift,
    ).to(device)

    target_mean, target_std = map(float, bundle.stats["target"])
    train_loader = make_loader(bundle.train, args, shuffle=True)
    val_loader = make_loader(bundle.val, args, shuffle=False)
    test_loader = make_loader(bundle.test, args, shuffle=False)

    if physics_args is not None and physics_args.physics_rhs_scale is None:
        print(f"Computing train-fixed {physics_args.physics_preset} RHS normalization scale...")
        rhs_scale_loader = make_loader(bundle.train, args, shuffle=False)
        physics_args.physics_rhs_scale = compute_fixed_rhs_scale(
            rhs_scale_loader,
            physics_args,
            lead_days=args.lead,
            device=device,
        )
        args.physics_rhs_scale = physics_args.physics_rhs_scale
        print(f"{physics_args.physics_preset} train-fixed RHS scale: {physics_args.physics_rhs_scale:.8g}")

    config = {
        "model": model_id,
        "model_label": model_label,
        "lead": args.lead,
        "history": args.history,
        "seed": args.seed,
        "input_feature_names": bundle.train.input_feature_names,
        "input_dim": input_dim,
        "include_residual_feature": args.include_residual_feature,
        "prediction_mode": args.prediction_mode,
        "physics_mode": args.physics_mode,
        "lambda_physics": args.lambda_physics if physics_args is not None else 0.0,
        "physics_preset": getattr(physics_args, "physics_preset", None),
        "physics_components": getattr(physics_args, "physics_components", None),
        "physics_normalize": getattr(physics_args, "physics_normalize", None),
        "physics_rhs_scale": getattr(physics_args, "physics_rhs_scale", None),
        "zero_rres_physics": getattr(physics_args, "zero_rres_physics", None),
        "lambda_rres": getattr(physics_args, "lambda_rres", 0.0) if physics_args is not None else 0.0,
        "rres_penalty_normalize": getattr(physics_args, "rres_penalty_normalize", None),
        "lambda_we_tie": getattr(physics_args, "lambda_we_tie", 0.0) if physics_args is not None else 0.0,
        "entrainment_dt": args.entrainment_dt,
        "hidden_dim": args.hidden_dim,
        "temporal_hidden_dim": args.temporal_hidden_dim,
        "processor_layers": args.processor_layers,
        "mlp_hidden_layers": args.mlp_hidden_layers,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "initial_lr": args.initial_lr,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "graph": bundle.graph.summary(),
        "train_samples": len(bundle.train),
        "val_samples": len(bundle.val),
        "test_samples": len(bundle.test),
        "target_mean": target_mean,
        "target_std": target_std,
    }
    save_json(run_dir / "jeju_hilam_config.json", config)

    print(json.dumps({"run_dir": str(run_dir), **config}, indent=2, allow_nan=True))

    if args.smoke_test:
        run_smoke(args, model, train_loader, device)
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    best_value = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history_rows: list[dict[str, Any]] = []
    best_path = run_dir / "jeju_hilam_best.pt"

    for epoch in range(1, args.epochs + 1):
        lr = set_epoch_lr(optimizer, args, epoch)
        train_stats = run_epoch(
            model,
            train_loader,
            bundle.graph,
            device,
            target_mean,
            target_std,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
            precision=args.precision,
            physics_args=physics_args,
            lambda_physics=args.lambda_physics,
            epoch=epoch,
        )
        val_stats = run_epoch(
            model,
            val_loader,
            bundle.graph,
            device,
            target_mean,
            target_std,
            optimizer=None,
            max_batches=args.max_val_batches,
            precision=args.precision,
            physics_args=physics_args,
            lambda_physics=args.lambda_physics,
            epoch=epoch,
        )
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_stats["loss"],
            "train_total_loss": train_stats["total_loss"],
            "train_data_loss": train_stats["data_loss"],
            "train_physics_loss": train_stats["physics_loss"],
            "train_rres_l2": train_stats["rres_l2"],
            "train_rmse": train_stats["rmse"],
            "train_mae": train_stats["mae"],
            "train_bias": train_stats["bias"],
            "val_loss": val_stats["loss"],
            "val_total_loss": val_stats["total_loss"],
            "val_data_loss": val_stats["data_loss"],
            "val_physics_loss": val_stats["physics_loss"],
            "val_rres_l2": val_stats["rres_l2"],
            "val_rmse": val_stats["rmse"],
            "val_mae": val_stats["mae"],
            "val_bias": val_stats["bias"],
        }
        history_rows.append(row)
        print(json.dumps(row, allow_nan=True))

        current = row[args.selection_metric]
        if current < best_value:
            best_value = current
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "best_epoch": best_epoch,
                    "best_value": best_value,
                    "stats": bundle.stats,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}")
                break

    save_epoch_history(run_dir / "jeju_hilam_epoch_history.csv", history_rows)
    try:
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_epoch_stats = run_epoch(
        model,
        test_loader,
        bundle.graph,
        device,
        target_mean,
        target_std,
        optimizer=None,
        max_batches=None,
        precision=args.precision,
        physics_args=physics_args,
        lambda_physics=args.lambda_physics,
        epoch=args.physics_warmup_epochs,
    )

    fields = collect_predictions(
        model,
        test_loader,
        bundle.graph,
        device,
        target_mean,
        target_std,
        precision=args.precision,
        physics_args=physics_args,
    )
    fields_path = run_dir / f"{model_id}_lead_{args.lead}_history_{args.history}_test_fields.npz"
    np.savez_compressed(fields_path, **fields)

    metrics = compute_forecast_metrics(fields["pred_raw"], fields["target_raw"], fields["mask"])
    add_acc_and_sw_metrics(metrics, bundle.ds, fields)
    metrics.update({f"test_{key}": value for key, value in test_epoch_stats.items()})
    metrics.update(
        {
            "model": model_id,
            "model_label": model_label,
            "lead": args.lead,
            "history": args.history,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_selection_metric": args.selection_metric,
            "best_selection_value": best_value,
            "physics_mode": args.physics_mode,
            "lambda_physics": args.lambda_physics if physics_args is not None else 0.0,
            "physics_preset": getattr(physics_args, "physics_preset", None),
            "physics_components": getattr(physics_args, "physics_components", None),
            "physics_normalize": getattr(physics_args, "physics_normalize", None),
            "physics_rhs_scale": getattr(physics_args, "physics_rhs_scale", None),
            "zero_rres_physics": getattr(physics_args, "zero_rres_physics", None),
            "lambda_rres": getattr(physics_args, "lambda_rres", 0.0) if physics_args is not None else 0.0,
            "test_samples": len(bundle.test),
            "fields_path": str(fields_path),
            "checkpoint_path": str(best_path),
        }
    )
    save_json(run_dir / f"{model_id}_lead_{args.lead}_history_{args.history}_metrics.json", metrics)
    print(json.dumps(metrics, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
