#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from mhw.datasets import (
    DEFAULT_DYNAMIC_VARS,
    DEFAULT_STATIC_VARS,
    JejuForecastDataset,
    compute_train_stats,
    make_time_splits,
)
from mhw.eval_metrics import compute_forecast_metrics
from mhw.models import ConvLSTMForecaster, ConvLSTMMultiTaskPINN, compute_residual
from mhw.process_title import set_process_title
from mhw.train_utils import cleanup_distributed, is_main_process, setup_distributed


PRESET_CONFIGS = {
    1: 30,
    3: 30,
    7: 30,
    14: 30,
}
SECONDS_PER_DAY = 86400.0
PINN_MAX_DELTA_BY_LEAD = {
    1: 2.0,
    3: 3.0,
    7: 5.0,
    14: 7.0,
}
PHYSICS_COMPONENT_ORDER = ("original", "vertical", "diffusion")
PHYSICS_COMPONENT_ALIASES = {
    "orig": "original",
    "base": "original",
    "vadv": "vertical",
    "vertical_advection": "vertical",
    "hdiff": "diffusion",
    "horizontal_diffusion": "diffusion",
}
PHYSICS_COMPONENT_PRESETS = {
    "p0": ("original",),
    "original": ("original",),
    "p1": ("original", "vertical"),
    "vertical": ("original", "vertical"),
    "p2": ("original", "diffusion"),
    "diffusion": ("original", "diffusion"),
    "p3": ("original", "vertical", "diffusion"),
    "extended": ("original", "vertical", "diffusion"),
    "both": ("original", "vertical", "diffusion"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hybrid Jeju forecast models with residual features")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(
            "/home/yeonwoo/lab/MHW/data/merged_sst_cmems_era5_wind10_qnet_jeju_1993_2022_FINAL_staticmask_filled.nc"
        ),
    )
    parser.add_argument("--residual-path", type=Path, default=None)
    parser.add_argument(
        "--disable-residual-map",
        action="store_true",
        help="Do not load or concatenate the learned f_theta residual feature-map channel.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["hybrid_naive", "hybrid_learned", "hybrid_mt_pinn"],
        required=True,
    )
    parser.add_argument("--horizon", type=int, choices=[1, 3, 7, 14], help="Forecast lead days")
    parser.add_argument("--history", type=int, default=None, help="Input window days")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=str, default="32,32")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("/home/yeonwoo/lab/MHW/outputs"))
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--selection-metric", choices=["rmse", "data", "total"], default="rmse",
                        help="Validation metric for best checkpoint and early stopping. "
                        "Use rmse/data for prediction-oriented model selection; total reproduces the old physics-weighted behavior.")
    parser.add_argument("--lambda-physics", type=float, default=0.05)
    parser.add_argument(
        "--dynamic-vars",
        type=str,
        default="",
        help="Comma list of input channels. Empty keeps DEFAULT_DYNAMIC_VARS; "
        "--extra-dynamic-vars are appended afterward for backward compatibility.",
    )
    parser.add_argument("--adaptive-weighting", action="store_true")
    parser.add_argument("--adaptive-sampling", action="store_true")
    parser.add_argument("--adaptive-temperature", type=float, default=1.0)
    parser.add_argument("--adaptive-samples", type=int, default=256)
    parser.add_argument(
        "--adaptive-weight-min",
        type=float,
        default=0.1,
        help="Lower clamp for residual-adaptive relative weights. Uniform weighting is 1.",
    )
    parser.add_argument(
        "--adaptive-weight-max",
        type=float,
        default=10.0,
        help="Upper clamp for residual-adaptive relative weights. Prevents hard-point domination.",
    )
    parser.add_argument("--physics-loss", type=str, choices=["smooth_l1", "mse"], default="smooth_l1")
    parser.add_argument("--physics-time-mode", type=str, choices=["endpoint", "daily_rollout"], default="endpoint")
    parser.add_argument(
        "--forecast-mode",
        type=str,
        choices=["free", "budget"],
        default="free",
        help="free: SST from a learned temperature head (legacy A/B/C). budget: SST is the "
        "forward-Euler integral of the mixed-layer budget, so Rres/we drive the forecast "
        "and residual structure becomes causal to drift.",
    )
    parser.add_argument("--extra-dynamic-vars", type=str, default="",
                        help="Comma list of extra input channels present in the dataset "
                        "(e.g. subsurface thetao_z0m_anom,thetao_z16m_anom,...).")
    parser.add_argument("--physics-hdiff", action="store_true",
                        help="Enable horizontal eddy diffusion term K*laplacian(T) in the budget.")
    parser.add_argument("--k-diff", type=float, default=100.0,
                        help="Horizontal eddy diffusivity K (m^2/s), fixed constant, for --physics-hdiff.")
    parser.add_argument("--physics-vertical", action="store_true",
                        help="Enable resolved vertical advection (vadv=+w*dTdz from GLORYS) in the budget.")
    parser.add_argument(
        "--physics-preset",
        type=str,
        default="",
        help="Named physics ablation preset: P0/original, P1/vertical, P2/diffusion, or P3/extended.",
    )
    parser.add_argument(
        "--physics-components",
        type=str,
        default="",
        help="Comma or plus separated physics components. Example: original,vertical. "
        "Overrides the legacy --physics-hdiff/--physics-vertical flags.",
    )
    parser.add_argument(
        "--budget-substeps",
        type=int,
        default=16,
        help="Sub-steps per day for --forecast-mode budget. Sized for the dataset's WORST-case "
        "current (max|u_mld|~1.93 m/s, dx~22.2 km => 1-day CFL~7.5): 16 sub-steps gives worst-case "
        "CFL~0.47. Centered-difference advection (FTCS) is only conditionally stable, so keep a "
        "margin; raise this if a fast-current event destabilizes the rollout.",
    )
    parser.add_argument("--forcing-mode", choices=["perfect", "persist"], default="perfect",
                        help="For daily_rollout physics: perfect uses future reanalysis forcing; persist repeats the last observed forcing.")
    parser.add_argument("--physics-huber-beta", type=float, default=0.25)
    parser.add_argument("--physics-clip", type=float, default=10.0)
    parser.add_argument("--physics-normalize", choices=["none", "tendency", "rhs"], default="none",
                        help="Normalize physics residual before loss. tendency uses one pooled scalar "
                        "mean(dTdt^2) over valid ocean cells/times, so MSE becomes "
                        "mean(residual^2) / mean(dTdt^2). rhs uses pooled mean(RHS^2), "
                        "so MSE becomes a violation ratio against the equation terms.")
    parser.add_argument("--physics-normalize-eps", type=float, default=1e-8,
                        help="Minimum pooled tendency scale for --physics-normalize tendency.")
    parser.add_argument("--physics-tendency-scale", type=float, default=None,
                        help="Override fixed observed tendency MSE scale in (K/day)^2. "
                        "Default: compute once from the train split for this lead/history.")
    parser.add_argument("--physics-rhs-scale", type=float, default=None,
                        help="Optional fixed RHS MSE scale in (K/day)^2 for --physics-normalize rhs. "
                        "Default: use the current batch/path pooled RHS scale, detached.")
    parser.add_argument("--physics-warmup-epochs", type=int, default=5)
    parser.add_argument("--pinn-max-delta", type=float, default=None)
    # --- closure identifiability controls (default off = legacy behavior) ---
    parser.add_argument("--entrainment-dt", type=float, default=0.0,
                        help="Stratification offset dT (K): Td = Tm - dT so we is identifiable. 0 disables (Td=Tm).")
    parser.add_argument("--we-nonneg", action="store_true",
                        help="Constrain entrainment velocity we >= 0 via softplus (else signed tanh).")
    parser.add_argument("--we-softplus-shift", type=float, default=0.0,
                        help="Shift b in we=we_scale*softplus(head-b); >0 lowers the ln2 floor so we can reach ~0 (dh/dt median is 0). Try 2-4 with --we-nonneg.")
    parser.add_argument("--lambda-rres", type=float, default=0.0,
                        help="L2 penalty on the free residual closure Rres (breaks the we/Rres/f_theta tie).")
    parser.add_argument("--rres-penalty-normalize", choices=["head", "tendency", "rhs"], default="head",
                        help="Normalize the lambda_rres penalty. head keeps the legacy rres_scale normalization; "
                        "tendency uses pooled observed mean(dTdt^2); rhs uses pooled mean(RHS^2), "
                        "matching --physics-normalize rhs.")
    parser.add_argument("--lambda-we-tie", type=float, default=0.0,
                        help="Penalty tying we to entrainment-during-deepening: (we - relu(dh/dt))^2.")
    parser.add_argument("--zero-feature-map", action="store_true",
                        help="Ablation: zero the train-only f_theta channel to test whether |Rres| grows without it.")
    parser.add_argument("--zero-rres-physics", action="store_true",
                        help="Use a strict heat-budget physics loss by setting the free Rres closure to zero inside the physics equation.")
    parser.add_argument("--hm-min", type=float, default=5.0,
                        help="Mixed-layer depth floor (m) in the heating term qnet/(rho cp h).")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0,
                        help="Max global grad-norm for clipping (<=0 disables). Guards against loss spikes.")
    parser.add_argument("--adaptive-lambda-mode", choices=["fixed", "bayes"], default="fixed",
                        help="fixed: keep lambda_physics constant. bayes: update lambda each epoch "
                        "from a naive log-space posterior so lambda*physics_loss tracks data_loss.")
    parser.add_argument("--bayes-lambda-target-ratio", type=float, default=1.0,
                        help="Target contribution ratio: lambda_physics * physics_loss ~= ratio * data_loss.")
    parser.add_argument("--bayes-lambda-min", type=float, default=1e-4)
    parser.add_argument("--bayes-lambda-max", type=float, default=0.2)
    parser.add_argument("--bayes-lambda-prior-strength", type=float, default=4.0,
                        help="Pseudo-count strength of the current lambda prior in the naive Bayes update.")
    parser.add_argument("--bayes-lambda-evidence-strength", type=float, default=1.0,
                        help="Pseudo-count strength of one epoch's loss-ratio evidence.")
    parser.add_argument("--bayes-lambda-source", choices=["train", "val"], default="train",
                        help="Which split's data/physics loss ratio updates lambda for the next epoch.")
    parser.add_argument("--bayes-lambda-start-epoch", type=int, default=None,
                        help="First epoch allowed to update lambda. Default: physics_warmup_epochs.")
    parser.add_argument("--bayes-opt-lambda", action="store_true")
    parser.add_argument("--bo-objective", choices=["rmse", "data", "total"], default="rmse",
                        help="Metric minimized while selecting alpha with Bayesian optimization. "
                        "rmse = validation raw-SST RMSE; data = normalized validation data loss; "
                        "total = data + alpha*physics (+ other regularizers).")
    parser.add_argument("--bo-init-values", type=str, default="0,0.001,0.01")
    parser.add_argument("--bo-candidates", type=str, default="0,0.0001,0.0003,0.001,0.003,0.01,0.03,0.05,0.1")
    parser.add_argument("--bo-iters", type=int, default=3)
    parser.add_argument("--bo-epochs", type=int, default=3)
    return parser.parse_args()


def _normalize_physics_component(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    key = PHYSICS_COMPONENT_ALIASES.get(key, key)
    if key not in PHYSICS_COMPONENT_ORDER:
        valid = ", ".join(PHYSICS_COMPONENT_ORDER)
        presets = ", ".join(["P0", "P1", "P2", "P3"])
        raise ValueError(f"Unknown physics component '{name}'. Use components [{valid}] or preset [{presets}].")
    return key


def parse_physics_components(value: str) -> tuple[str, ...]:
    text = value.strip()
    if not text:
        return ("original",)
    preset_key = text.lower().replace("-", "_")
    if preset_key in PHYSICS_COMPONENT_PRESETS:
        return PHYSICS_COMPONENT_PRESETS[preset_key]

    tokens: list[str] = []
    for part in text.replace("+", ",").replace(";", ",").split(","):
        tokens.extend(piece for piece in part.split() if piece)

    selected: list[str] = []
    for token in tokens:
        component = _normalize_physics_component(token)
        if component not in selected:
            selected.append(component)
    if "original" not in selected:
        selected.insert(0, "original")
    return tuple(component for component in PHYSICS_COMPONENT_ORDER if component in selected)


def infer_physics_preset(components: tuple[str, ...]) -> str:
    for preset in ("p0", "p1", "p2", "p3"):
        if components == PHYSICS_COMPONENT_PRESETS[preset]:
            return preset.upper()
    return "custom"


def configure_physics_components(args: argparse.Namespace) -> argparse.Namespace:
    preset = str(getattr(args, "physics_preset", "") or "").strip()
    component_text = str(getattr(args, "physics_components", "") or "").strip()
    if preset and component_text:
        raise ValueError("Use either --physics-preset or --physics-components, not both.")

    if preset:
        key = preset.lower().replace("-", "_")
        if key not in PHYSICS_COMPONENT_PRESETS:
            valid = ", ".join(["P0", "P1", "P2", "P3", "original", "vertical", "diffusion", "extended"])
            raise ValueError(f"Unknown --physics-preset '{preset}'. Valid presets: {valid}.")
        components = PHYSICS_COMPONENT_PRESETS[key]
    elif component_text:
        components = parse_physics_components(component_text)
    else:
        selected = ["original"]
        if bool(getattr(args, "physics_vertical", False)):
            selected.append("vertical")
        if bool(getattr(args, "physics_hdiff", False)):
            selected.append("diffusion")
        components = tuple(selected)

    args.physics_component_list = list(components)
    args.physics_components = ",".join(components)
    args.physics_preset = infer_physics_preset(components)
    args.physics_vertical = "vertical" in components
    args.physics_hdiff = "diffusion" in components
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Force deterministic GPU kernels so a fixed seed reproduces run-to-run.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def maybe_reduce_mean(value: float, device: torch.device) -> float:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return float(value)
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor /= torch.distributed.get_world_size()
    return float(tensor.item())


def maybe_reduce_sum(value: float, device: torch.device) -> float:
    """All-reduce a pooled SUM across DDP ranks (no averaging).

    Budget diagnostics accumulate per-cell sums; the correct cross-process
    reduction is a sum so ratios stay ratio-of-global-pooled-means.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return float(value)
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return float(tensor.item())


def _require_scale(core: torch.nn.Module, name: str) -> float:
    """Fetch a head normalization scale, failing loudly if absent.

    The identifiability penalties are normalized by these scales; a silent
    default would let an ablation train with mismatched normalization and
    quietly corrupt cross-case comparisons.
    """
    if not hasattr(core, name):
        raise AttributeError(
            f"{type(core).__name__} has no '{name}'; identifiability penalty "
            "normalization would be undefined. Refusing to fall back silently."
        )
    return float(getattr(core, name))


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff2 = (pred - target) ** 2
    return (diff2 * mask).sum() / mask.sum().clamp_min(1.0)


def masked_physics_mse(residual: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff2 = residual ** 2
    return (diff2 * mask).sum() / mask.sum().clamp_min(1.0)


def _masked_mean(loss_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (loss_map * mask).sum() / mask.sum().clamp_min(1.0)


def _pooled_mse_scale(values: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
    scale = ((values ** 2) * mask).sum() / mask.sum().clamp_min(1.0)
    return torch.clamp(scale, min=max(float(eps), 1e-12))


def _fixed_or_batch_tendency_scale(
    args: argparse.Namespace,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    eps = getattr(args, "physics_normalize_eps", 1e-8)
    fixed_scale = getattr(args, "physics_tendency_scale", None)
    if fixed_scale is not None and float(fixed_scale) > 0:
        return torch.as_tensor(
            max(float(fixed_scale), max(float(eps), 1e-12)),
            device=values.device,
            dtype=values.dtype,
        )
    return _pooled_mse_scale(values, mask, eps=eps)


def _fixed_or_batch_rhs_scale(
    args: argparse.Namespace,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    eps = getattr(args, "physics_normalize_eps", 1e-8)
    fixed_scale = getattr(args, "physics_rhs_scale", None)
    if fixed_scale is not None and float(fixed_scale) > 0:
        return torch.as_tensor(
            max(float(fixed_scale), max(float(eps), 1e-12)),
            device=values.device,
            dtype=values.dtype,
        )
    return _pooled_mse_scale(values, mask, eps=eps)


def observed_tendency_per_day(Tm: torch.Tensor, target_raw: torch.Tensor, lead_days: int) -> torch.Tensor:
    """Observed SST tendency in K/day, using the fixed target rather than the model prediction."""
    return (target_raw - Tm) / max(float(lead_days), 1.0)


def _boltzmann_weights(values: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
    masked_values = values.abs() * mask
    batch = masked_values.shape[0]
    flat_vals = masked_values.view(batch, -1)
    flat_mask = mask.view(batch, -1)
    logits = flat_vals / max(temperature, 1e-6)
    logits = logits.masked_fill(flat_mask <= 0, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _adaptive_relative_weights(
    values: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
    min_weight: float,
    max_weight: float,
) -> torch.Tensor:
    """Return clamped residual-adaptive relative weights over valid cells."""
    probs = _boltzmann_weights(values, mask, temperature=temperature)
    flat_mask = mask.view(mask.shape[0], -1)
    valid_count = flat_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)

    # Convert probabilities to relative weights: uniform sampling becomes 1.0.
    weights = probs * valid_count
    lo = max(float(min_weight), 0.0)
    hi = max(float(max_weight), lo)
    weights = torch.clamp(weights, min=lo, max=hi) * flat_mask
    return weights


def compute_physics_loss(
    residual: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
    tendency_per_day: torch.Tensor | None = None,
    rhs_per_day: torch.Tensor | None = None,
) -> torch.Tensor:
    if args.physics_clip > 0:
        residual = torch.clamp(residual, min=-args.physics_clip, max=args.physics_clip)
    normalize_mode = getattr(args, "physics_normalize", "none")
    if normalize_mode == "tendency":
        if tendency_per_day is None:
            raise ValueError("--physics-normalize tendency requires tendency_per_day")
        tendency_scale = _fixed_or_batch_tendency_scale(
            args,
            tendency_per_day,
            mask,
        ).detach()
        residual = residual / torch.sqrt(tendency_scale)
    elif normalize_mode == "rhs":
        if rhs_per_day is None:
            raise ValueError("--physics-normalize rhs requires rhs_per_day")
        rhs_scale = _fixed_or_batch_rhs_scale(
            args,
            rhs_per_day,
            mask,
        ).detach()
        residual = residual / torch.sqrt(rhs_scale)
    if args.physics_loss == "smooth_l1":
        loss_map = F.smooth_l1_loss(
            residual,
            torch.zeros_like(residual),
            reduction="none",
            beta=args.physics_huber_beta,
        )
    else:
        loss_map = residual ** 2

    if args.adaptive_sampling:
        weights = _adaptive_relative_weights(
            residual.detach(),
            mask,
            temperature=args.adaptive_temperature,
            min_weight=args.adaptive_weight_min,
            max_weight=args.adaptive_weight_max,
        )
        sample_probs = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        flat_loss = loss_map.view(loss_map.shape[0], -1)
        losses = []
        for batch_idx in range(loss_map.shape[0]):
            probs = sample_probs[batch_idx]
            if torch.count_nonzero(probs).item() == 0:
                losses.append(torch.zeros((), device=residual.device, dtype=residual.dtype))
                continue
            sample_count = min(args.adaptive_samples, probs.numel())
            sampled_idx = torch.multinomial(probs, num_samples=sample_count, replacement=True)
            sampled_loss = flat_loss[batch_idx, sampled_idx]
            # Sampling already draws from the clamped adaptive distribution.
            # Weighting again would square the emphasis and can let hard points
            # dominate the physics term.
            losses.append(sampled_loss.mean())
        return torch.stack(losses).mean()

    if args.adaptive_weighting:
        weights = _adaptive_relative_weights(
            residual.detach(),
            mask,
            temperature=args.adaptive_temperature,
            min_weight=args.adaptive_weight_min,
            max_weight=args.adaptive_weight_max,
        )
        flat_loss = loss_map.view(loss_map.shape[0], -1)
        return (flat_loss * weights).sum(dim=-1).div(weights.sum(dim=-1).clamp_min(1e-12)).mean()

    return _masked_mean(loss_map, mask)


def physics_lambda_factor(epoch: int | None, warmup_epochs: int) -> float:
    if warmup_epochs <= 0 or epoch is None:
        return 1.0
    return min(1.0, max(float(epoch), 0.0) / float(warmup_epochs))


def bayesian_lambda_update(
    current_lambda: float,
    data_loss: float,
    physics_loss: float,
    args: argparse.Namespace,
) -> tuple[float, dict[str, float]]:
    """Naive log-space Bayesian update for the physics-loss weight.

    Evidence says the next base lambda should make lambda * physics_loss roughly
    match target_ratio * data_loss. The current lambda acts as a pseudo-count
    prior, and the epoch loss ratio acts as one noisy observation.
    """
    eps = 1e-12
    current = float(np.clip(current_lambda, args.bayes_lambda_min, args.bayes_lambda_max))
    if physics_loss <= eps or data_loss <= eps:
        return current, {
            "bayes_lambda_suggested": current,
            "bayes_lambda_next": current,
            "bayes_lambda_loss_ratio": 0.0,
        }

    target_ratio = max(float(args.bayes_lambda_target_ratio), eps)
    suggested = target_ratio * float(data_loss) / max(float(physics_loss), eps)
    suggested = float(np.clip(suggested, args.bayes_lambda_min, args.bayes_lambda_max))

    prior_strength = max(float(args.bayes_lambda_prior_strength), eps)
    evidence_strength = max(float(args.bayes_lambda_evidence_strength), eps)
    posterior_log = (
        prior_strength * math.log(current)
        + evidence_strength * math.log(suggested)
    ) / (prior_strength + evidence_strength)
    next_lambda = float(np.clip(math.exp(posterior_log), args.bayes_lambda_min, args.bayes_lambda_max))
    return next_lambda, {
        "bayes_lambda_suggested": suggested,
        "bayes_lambda_next": next_lambda,
        "bayes_lambda_loss_ratio": float(data_loss) / max(float(physics_loss), eps),
    }


def bound_pinn_prediction(
    pred_norm: torch.Tensor,
    Tm: torch.Tensor,
    target_mean: float,
    target_std: float,
    lead_days: int,
    max_delta_override: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_delta = max_delta_override
    if max_delta is None:
        max_delta = PINN_MAX_DELTA_BY_LEAD.get(int(lead_days), float(lead_days))
    max_delta = max(float(max_delta), 1e-6)
    pred_raw_unbounded = pred_norm * target_std + target_mean
    delta = pred_raw_unbounded - Tm
    pred_raw = Tm + max_delta * torch.tanh(delta / max_delta)
    pred_norm_bounded = (pred_raw - target_mean) / target_std
    return pred_norm_bounded, pred_raw


def spatial_gradients_torch(field: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dTdx_num = torch.zeros_like(field)
    dTdy_num = torch.zeros_like(field)
    if field.shape[-1] > 1:
        dTdx_num[..., 1:-1] = (field[..., 2:] - field[..., :-2]) * 0.5
        dTdx_num[..., 0] = field[..., 1] - field[..., 0]
        dTdx_num[..., -1] = field[..., -1] - field[..., -2]
    if field.shape[-2] > 1:
        dTdy_num[..., 1:-1, :] = (field[..., 2:, :] - field[..., :-2, :]) * 0.5
        dTdy_num[..., 0, :] = field[..., 1, :] - field[..., 0, :]
        dTdy_num[..., -1, :] = field[..., -1, :] - field[..., -2, :]
    dx_safe = torch.where(dx.abs() > 1e-12, dx, torch.ones_like(dx))
    dy_safe = torch.where(dy.abs() > 1e-12, dy, torch.ones_like(dy))
    return dTdx_num / dx_safe, dTdy_num / dy_safe


def laplacian_torch(field: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """Horizontal Laplacian d2T/dx2 + d2T/dy2 (K/m^2), 2nd-order centred, one-sided
    at edges. Used for the horizontal eddy-diffusion term K*laplacian(T)."""
    d2x = torch.zeros_like(field)
    d2y = torch.zeros_like(field)
    if field.shape[-1] > 2:
        d2x[..., 1:-1] = field[..., 2:] - 2.0 * field[..., 1:-1] + field[..., :-2]
    if field.shape[-2] > 2:
        d2y[..., 1:-1, :] = field[..., 2:, :] - 2.0 * field[..., 1:-1, :] + field[..., :-2, :]
    dx2 = torch.clamp(dx * dx, min=1e-6)
    dy2 = torch.clamp(dy * dy, min=1e-6)
    return d2x / dx2 + d2y / dy2


def active_physics_components(args: argparse.Namespace) -> set[str]:
    components = getattr(args, "physics_component_list", None)
    if components is None:
        components = parse_physics_components(getattr(args, "physics_components", ""))
    return set(components)


def diffusion_term_ksec(
    field: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor | None:
    if "diffusion" not in active_physics_components(args):
        return None
    return float(getattr(args, "k_diff", 100.0)) * laplacian_torch(field, dx, dy)


def vertical_advection_term_ksec(
    batch: dict[str, torch.Tensor],
    step_idx: int,
    device: torch.device,
    args: argparse.Namespace,
) -> torch.Tensor | None:
    if "vertical" not in active_physics_components(args):
        return None
    if "vadv_seq" not in batch:
        raise KeyError("Physics component 'vertical' requires vadv_seq in the batch/dataset.")
    return batch["vadv_seq"].to(device)[:, step_idx] / SECONDS_PER_DAY


def add_selected_extra_physics_terms(
    rhs: torch.Tensor,
    field: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
    batch: dict[str, torch.Tensor],
    step_idx: int,
    args: argparse.Namespace,
) -> torch.Tensor:
    diffusion = diffusion_term_ksec(field, dx, dy, args)
    if diffusion is not None:
        rhs = rhs + diffusion
    vertical = vertical_advection_term_ksec(batch, step_idx, field.device, args)
    if vertical is not None:
        rhs = rhs + vertical
    return rhs


def daily_rollout_physics_loss(
    pred_raw: torch.Tensor,
    Tm: torch.Tensor,
    target_raw: torch.Tensor,
    mask: torch.Tensor,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
    lead_days: int,
    we_pred: torch.Tensor,
    rres_pred: torch.Tensor,
) -> torch.Tensor:
    qnet_seq = batch["qnet_seq"].to(pred_raw.device)
    hm_seq = batch["hm_seq"].to(pred_raw.device)
    u_seq = batch["u_seq"].to(pred_raw.device)
    v_seq = batch["v_seq"].to(pred_raw.device)
    dx = batch["dx"].to(pred_raw.device)
    dy = batch["dy"].to(pred_raw.device)
    residuals = []
    tendency_terms = []
    rhs_terms = []
    masks = []
    prev = Tm
    steps = max(int(lead_days), 1)
    obs_tendency = observed_tendency_per_day(Tm, target_raw, lead_days)
    rres_term = torch.zeros_like(rres_pred) if getattr(args, "zero_rres_physics", False) else rres_pred
    for step in range(1, steps + 1):
        frac = float(step) / float(steps)
        next_state = Tm + frac * (pred_raw - Tm)
        dTdt = (next_state - prev) / SECONDS_PER_DAY
        tendency_terms.append(obs_tendency)
        dTdx, dTdy = spatial_gradients_torch(prev, dx, dy)
        entrainment_dt = float(getattr(args, "entrainment_dt", 0.0))
        Td_step = prev - entrainment_dt if entrainment_dt > 0.0 else prev
        residual = compute_residual(
            dTdt=dTdt,
            qnet=qnet_seq[:, step - 1],
            rho=1025.0,
            Cp=3990.0,
            hm=hm_seq[:, step - 1],
            u=u_seq[:, step - 1],
            v=v_seq[:, step - 1],
            dTdx=dTdx,
            dTdy=dTdy,
            we=we_pred,
            Tm=prev,
            Td=Td_step,
            Rres=rres_term,
            hm_min=float(getattr(args, "hm_min", 5.0)),
        )
        extra_rhs = add_selected_extra_physics_terms(
            torch.zeros_like(residual),
            field=prev,
            dx=dx,
            dy=dy,
            batch=batch,
            step_idx=step - 1,
            args=args,
        )
        residual = residual - extra_rhs
        residual_per_day = residual * SECONDS_PER_DAY
        # Normalize against the resolved RHS scale, excluding the free Rres closure
        # so the catch-all head cannot inflate its own denominator.
        rhs_terms.append(dTdt * SECONDS_PER_DAY - residual_per_day - rres_term * SECONDS_PER_DAY)
        residuals.append(residual_per_day)
        masks.append(mask)
        prev = next_state
    return compute_physics_loss(
        torch.cat(residuals, dim=0),
        torch.cat(masks, dim=0),
        args,
        tendency_per_day=torch.cat(tendency_terms, dim=0),
        rhs_per_day=torch.cat(rhs_terms, dim=0),
    )


def _linear_rhs_per_day(
    pred_raw: torch.Tensor,
    Tm: torch.Tensor,
    mask: torch.Tensor,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
    lead_days: int,
    we_pred: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolved RHS in K/day (flux + adv + entrain + hdiff + vadv), EXCLUDING the
    free Rres closure, plus its step-broadcast mask. Single source of truth so the
    online rhs-scale and the fixed train rhs-scale share an identical definition."""
    device = pred_raw.device
    qnet_seq = batch["qnet_seq"].to(device)
    hm_seq = batch["hm_seq"].to(device)
    u_seq = batch["u_seq"].to(device)
    v_seq = batch["v_seq"].to(device)
    dx = batch["dx"].to(device)
    dy = batch["dy"].to(device)
    entrainment_dt = float(getattr(args, "entrainment_dt", 0.0))
    hm_min = float(getattr(args, "hm_min", 5.0))
    steps = max(int(lead_days), 1)

    rhs_terms = []
    prev = Tm
    for step in range(1, steps + 1):
        frac = float(step) / float(steps)
        next_state = Tm + frac * (pred_raw - Tm)
        dTdx, dTdy = spatial_gradients_torch(prev, dx, dy)
        hm_safe = torch.clamp(hm_seq[:, step - 1], min=hm_min)
        Td_step = prev - entrainment_dt if entrainment_dt > 0.0 else prev
        rhs = (
            qnet_seq[:, step - 1] / (1025.0 * 3990.0 * hm_safe)
            - u_seq[:, step - 1] * dTdx
            - v_seq[:, step - 1] * dTdy
            - we_pred * (prev - Td_step) / hm_safe
        )
        rhs = add_selected_extra_physics_terms(
            rhs,
            field=prev,
            dx=dx,
            dy=dy,
            batch=batch,
            step_idx=step - 1,
            args=args,
        )
        rhs_terms.append(rhs * SECONDS_PER_DAY)
        prev = next_state

    rhs_per_day = torch.cat(rhs_terms, dim=0)
    rhs_mask = mask.repeat(steps, 1, 1, 1)
    return rhs_per_day, rhs_mask


def resolved_rhs_scale_from_linear_path(
    pred_raw: torch.Tensor,
    Tm: torch.Tensor,
    mask: torch.Tensor,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
    lead_days: int,
    we_pred: torch.Tensor,
) -> torch.Tensor:
    """Pooled mean(RHS^2) in K/day units, excluding the free Rres closure."""
    rhs_per_day, rhs_mask = _linear_rhs_per_day(pred_raw, Tm, mask, batch, args, lead_days, we_pred)
    return _fixed_or_batch_rhs_scale(args, rhs_per_day, rhs_mask).detach()


def compute_fixed_rhs_scale(
    loader: DataLoader,
    args: argparse.Namespace,
    lead_days: int,
    device: torch.device,
) -> float:
    """Train-only fixed RHS MSE scale (K/day)^2, pooled over the whole train split.

    Uses the OBSERVED target as the linear-path endpoint and we=0, so the scale is
    model-independent and identical every batch/epoch (removes batch-adaptive drift).
    Definition matches _linear_rhs_per_day exactly (same terms, Rres excluded).
    """
    sq_sum = 0.0
    cell_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            Tm = batch["Tm"].to(device)
            y_raw = batch["y_raw"].to(device)
            mask = batch["mask"].to(device)
            we_zero = torch.zeros_like(Tm)
            rhs_per_day, rhs_mask = _linear_rhs_per_day(
                y_raw, Tm, mask, batch, args, lead_days, we_zero
            )
            sq_sum += float(((rhs_per_day ** 2) * rhs_mask).sum().item())
            cell_sum += float(rhs_mask.sum().item())
    eps = float(getattr(args, "physics_normalize_eps", 1e-8))
    scale = sq_sum / cell_sum if cell_sum > 0 else 1.0
    return max(scale, max(eps, 1e-12))


def integrate_budget_forecast(
    Tm: torch.Tensor,
    we_pred: torch.Tensor,
    rres_pred: torch.Tensor,
    mask: torch.Tensor,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
    lead_days: int,
) -> torch.Tensor:
    """Physics-integrated forecast: roll the mixed-layer budget forward so the
    SST prediction flows THROUGH flux/adv/entrain/Rres instead of a free head.

        T(t+n) = Tm + sum_k dt * rhs_k,
        rhs = qnet/(rho Cp h) - u.dTdx - v.dTdy - we (T - Td)/h + Rres   (K/s)

    Forward Euler, dt = 1 day, n = lead_days. Spatial gradients are recomputed
    from the evolving field each step so advection is state-dependent; land cells
    are frozen (rhs masked to 0) so the integration stays finite off-ocean. This
    makes Rres a genuine forcing on the forecast -- its spatial structure now
    compounds over the rollout, so smoothing Rres becomes a real drift lever
    (which it is not when the forecast comes from a free temperature head).
    """
    device = Tm.device
    qnet_seq = batch["qnet_seq"].to(device)
    hm_seq = batch["hm_seq"].to(device)
    u_seq = batch["u_seq"].to(device)
    v_seq = batch["v_seq"].to(device)
    dx = batch["dx"].to(device)
    dy = batch["dy"].to(device)
    entrainment_dt = float(getattr(args, "entrainment_dt", 0.0))
    hm_min = float(getattr(args, "hm_min", 5.0))
    steps = max(int(lead_days), 1)
    # Explicit centered-difference advection with a 1-day step is CFL-unstable
    # here (worst-case current ~1.93 m/s, grid ~22 km => 1-day CFL ~7.5 >> 1), so
    # integrate each day in sub-steps sized to the WORST-case current (not the
    # mean) to keep CFL < 1 during fast-current events. Forcing is held constant
    # within a day; sub-stepping only refines the state trajectory.
    substeps = max(int(getattr(args, "budget_substeps", 8)), 1)
    dt_sub = SECONDS_PER_DAY / substeps

    prev = Tm
    rres_term = torch.zeros_like(rres_pred) if getattr(args, "zero_rres_physics", False) else rres_pred
    for step in range(1, steps + 1):
        qnet_step = qnet_seq[:, step - 1]
        hm_safe = torch.clamp(hm_seq[:, step - 1], min=hm_min)
        u_step = u_seq[:, step - 1]
        v_step = v_seq[:, step - 1]
        flux = qnet_step / (1025.0 * 3990.0 * hm_safe)
        for _ in range(substeps):
            dTdx, dTdy = spatial_gradients_torch(prev, dx, dy)
            Td_step = prev - entrainment_dt if entrainment_dt > 0.0 else prev
            rhs = (
                flux
                - u_step * dTdx
                - v_step * dTdy
                - we_pred * (prev - Td_step) / hm_safe
                + rres_term
            )
            rhs = add_selected_extra_physics_terms(
                rhs,
                field=prev,
                dx=dx,
                dy=dy,
                batch=batch,
                step_idx=step - 1,
                args=args,
            )
            prev = prev + dt_sub * rhs * mask  # freeze land: rhs=0 off-ocean
    return prev


def physics_budget_diagnostics(
    pred_raw: torch.Tensor,
    Tm: torch.Tensor,
    mask: torch.Tensor,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
    lead_days: int,
    we_pred: torch.Tensor,
    rres_pred: torch.Tensor,
) -> dict[str, float]:
    """Decompose the mixed-layer budget into its terms (K/day) for the closure
    identifiability claim.

    Reducing |Rres| is only evidence that physics took over the explanation if,
    at the same time, the data fit holds AND the resolved physics terms actually
    carry more variance (i.e. they absorb what the residual gave up). This returns
    per-cell POOLED SUMS (Sigma|x|m, Sigma x m, Sigma x^2 m) for each budget term;
    run_epoch accumulates them over the split and finalize_budget_metrics turns
    them into abs-mean / RMS / std (K/day) and residual FRACTIONS, so the A/B/C
    ablation can show flux/adv/entrainment variance rising as |Rres| falls --
    distinguishing "physics explains more" from "L2 just erased the signal".
    """
    device = pred_raw.device
    qnet_seq = batch["qnet_seq"].to(device)
    hm_seq = batch["hm_seq"].to(device)
    u_seq = batch["u_seq"].to(device)
    v_seq = batch["v_seq"].to(device)
    dx = batch["dx"].to(device)
    dy = batch["dy"].to(device)
    entrainment_dt = float(getattr(args, "entrainment_dt", 0.0))
    hm_min = float(getattr(args, "hm_min", 5.0))
    steps = max(int(lead_days), 1)

    flux_terms, adv_terms, entrain_terms, rres_terms, dtdt_terms = [], [], [], [], []
    hdiff_terms, vadv_terms = [], []
    prev = Tm
    rres_term = torch.zeros_like(rres_pred) if getattr(args, "zero_rres_physics", False) else rres_pred
    for step in range(1, steps + 1):
        frac = float(step) / float(steps)
        next_state = Tm + frac * (pred_raw - Tm)
        dTdt = (next_state - prev) / SECONDS_PER_DAY
        dTdx, dTdy = spatial_gradients_torch(prev, dx, dy)
        hm_safe = torch.clamp(hm_seq[:, step - 1], min=hm_min)
        Td_step = prev - entrainment_dt if entrainment_dt > 0.0 else prev
        flux = qnet_seq[:, step - 1] / (1025.0 * 3990.0 * hm_safe)
        adv = -u_seq[:, step - 1] * dTdx - v_seq[:, step - 1] * dTdy
        entrain = -we_pred * (prev - Td_step) / hm_safe
        flux_terms.append(flux); adv_terms.append(adv)
        entrain_terms.append(entrain); rres_terms.append(rres_term); dtdt_terms.append(dTdt)
        hdiff = diffusion_term_ksec(prev, dx, dy, args)
        vadv = vertical_advection_term_ksec(batch, step - 1, device, args)
        hdiff_terms.append(hdiff if hdiff is not None else torch.zeros_like(flux))
        vadv_terms.append(vadv if vadv is not None else torch.zeros_like(flux))
        prev = next_state

    day = SECONDS_PER_DAY
    flux = torch.cat(flux_terms) * day
    adv = torch.cat(adv_terms) * day
    entrain = torch.cat(entrain_terms) * day
    rres = torch.cat(rres_terms) * day
    dtdt = torch.cat(dtdt_terms) * day
    hdiff = torch.cat(hdiff_terms) * day
    vadv = torch.cat(vadv_terms) * day
    m = mask.repeat(steps, 1, 1, 1)  # same ocean mask for every rollout step

    def _pooled(field: torch.Tensor) -> tuple[float, float, float]:
        """Masked pooled sums (Sigma|x|m, Sigma x m, Sigma x^2 m) for this batch.

        We return SUMS, not per-batch means, so run_epoch can pool over the whole
        split and form ratio-of-pooled-means (no per-batch Jensen gap) and
        cell-count-weighted means (partial final batches no longer over-weighted).
        """
        fm = field * m
        return (
            float((field.abs() * m).sum().item()),
            float(fm.sum().item()),
            float((field * fm).sum().item()),
        )

    stats: dict[str, float] = {"budget_cells": float(m.sum().item())}
    for name, field in (
        ("flux", flux),
        ("adv", adv),
        ("entrain", entrain),
        ("rres", rres),
        ("dtdt", dtdt),
        ("hdiff", hdiff),
        ("vadv", vadv),
    ):
        s_abs, s_sgn, s_sq = _pooled(field)
        stats[f"budget_{name}_abs"] = s_abs
        stats[f"budget_{name}_sgn"] = s_sgn
        stats[f"budget_{name}_sq"] = s_sq
    return stats


def finalize_budget_metrics(sums: dict[str, float]) -> dict[str, float]:
    """Reduce epoch-pooled per-cell sums to masked abs-mean / RMS / std (K/day)
    and the dimensionless residual fractions + crest factors.

    Ratios are ratio-of-pooled-means, NOT the mean of per-batch ratios: the old
    per-batch averaging had a Jensen gap that made the reported Rres/tend
    disagree with |Rres|-bar / |tend|-bar (e.g. 4.62 vs 4.26 for A_legacy).
    Crest factor = RMS / abs-mean (>= 1) is the diagnostic that separates
    residual *magnitude* from residual *spike structure* -- the axis the
    smoothness regularizer is meant to move without collapsing magnitude.
    """
    n = max(sums.get("budget_cells", 0.0), 1.0)

    def absmean(name: str) -> float:
        return sums.get(f"budget_{name}_abs", 0.0) / n

    def rms(name: str) -> float:
        return math.sqrt(max(sums.get(f"budget_{name}_sq", 0.0) / n, 0.0))

    def std(name: str) -> float:
        mu = sums.get(f"budget_{name}_sgn", 0.0) / n
        return math.sqrt(max(sums.get(f"budget_{name}_sq", 0.0) / n - mu * mu, 0.0))

    def safe(a: float, b: float) -> float:
        return a / b if b > 1e-12 else 0.0

    rres_abs, rres_rms = absmean("rres"), rms("rres")
    dtdt_abs, dtdt_rms = absmean("dtdt"), rms("dtdt")
    hdiff_abs, vadv_abs = absmean("hdiff"), absmean("vadv")
    phys_abs = absmean("flux") + absmean("adv") + absmean("entrain") + hdiff_abs + vadv_abs
    phys_rms = math.sqrt(rms("flux") ** 2 + rms("adv") ** 2 + rms("entrain") ** 2
                         + rms("hdiff") ** 2 + rms("vadv") ** 2)
    # each resolved term's share of the total absolute budget activity (C-global)
    denom_share = (absmean("flux") + absmean("adv") + absmean("entrain")
                   + hdiff_abs + vadv_abs + rres_abs)
    return {
        "budget_flux_absmean_kday": absmean("flux"),
        "budget_adv_absmean_kday": absmean("adv"),
        "budget_entrain_absmean_kday": absmean("entrain"),
        "budget_hdiff_absmean_kday": hdiff_abs,
        "budget_vadv_absmean_kday": vadv_abs,
        "budget_rres_absmean_kday": rres_abs,
        "budget_dTdt_absmean_kday": dtdt_abs,
        "budget_rres_l2_kday": rres_rms,
        "budget_dTdt_l2_kday": dtdt_rms,
        "budget_flux_std_kday": std("flux"),
        "budget_adv_std_kday": std("adv"),
        "budget_entrain_std_kday": std("entrain"),
        "budget_hdiff_std_kday": std("hdiff"),
        "budget_vadv_std_kday": std("vadv"),
        "budget_rres_std_kday": std("rres"),
        # residual FRACTIONS (dimensionless) -- the interpretable numbers
        "rres_frac_of_tendency": safe(rres_abs, dtdt_abs),
        "rres_frac_of_physics": safe(rres_abs, phys_abs),
        "rres_l2_frac_of_tendency": safe(rres_rms, dtdt_rms),
        "rres_l2_frac_of_physics": safe(rres_rms, phys_rms),
        # budget shares of total activity (flux/adv/entrain/hdiff/vadv/residual)
        "share_flux": safe(absmean("flux"), denom_share),
        "share_adv": safe(absmean("adv"), denom_share),
        "share_entrain": safe(absmean("entrain"), denom_share),
        "share_hdiff": safe(hdiff_abs, denom_share),
        "share_vadv": safe(vadv_abs, denom_share),
        "share_rres": safe(rres_abs, denom_share),
        # crest factors (RMS / abs-mean) -- magnitude-vs-structure separation
        "rres_crest_factor": safe(rres_rms, rres_abs),
        "dtdt_crest_factor": safe(dtdt_rms, dtdt_abs),
    }


def identifiability_terms(
    we_pred: torch.Tensor,
    rres_pred: torch.Tensor,
    pred_raw: torch.Tensor | None,
    batch: dict[str, torch.Tensor],
    mask: torch.Tensor,
    lead_days: int,
    args: argparse.Namespace,
    we_scale: float = 1.0e-5,
    rres_scale: float = 1.0e-4,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Closure-identifiability penalties and per-batch budget diagnostics.

    Both penalties are NORMALIZED by the head's characteristic scale so they are
    O(1) and comparable to the data loss. Without this, we/Rres/dh-dt all live at
    ~1e-5, their squared errors are ~1e-10, and any "reasonable" lambda makes the
    penalty numerically inert (this only shows up once you actually train, not in
    a forward smoke). After normalization lambda_rres / lambda_we_tie in [1e-3, 1]
    are meaningful.

    Returns (rres_l2, we_tie, diagnostics) where diagnostics report the mean
    magnitude of each closure term so an ablation can show, e.g., that |Rres|
    grows when the train-only feature map is removed.
    """
    device = we_pred.device
    we_scale = max(float(we_scale), 1e-12)
    rres_scale = max(float(rres_scale), 1e-12)

    def _mmean(field: torch.Tensor) -> torch.Tensor:
        return (field * mask).sum() / mask.sum().clamp_min(1.0)

    # L2 on the free residual closure: prefer physics over the catch-all.
    if getattr(args, "rres_penalty_normalize", "head") == "tendency":
        Tm = batch["Tm"].to(device)
        target_raw = batch["y_raw"].to(device)
        obs_tendency = observed_tendency_per_day(Tm, target_raw, lead_days)
        tendency_scale = _fixed_or_batch_tendency_scale(
            args,
            obs_tendency,
            mask,
        ).detach()
        rres_l2 = _mmean((rres_pred * SECONDS_PER_DAY) ** 2) / tendency_scale
    elif getattr(args, "rres_penalty_normalize", "head") == "rhs":
        if pred_raw is None:
            raise ValueError("--rres-penalty-normalize rhs requires pred_raw")
        Tm = batch["Tm"].to(device)
        rhs_scale = resolved_rhs_scale_from_linear_path(
            pred_raw=pred_raw,
            Tm=Tm,
            mask=mask,
            batch=batch,
            args=args,
            lead_days=lead_days,
            we_pred=we_pred,
        )
        rres_l2 = _mmean((rres_pred * SECONDS_PER_DAY) ** 2) / rhs_scale
    else:
        rres_l2 = _mmean((rres_pred / rres_scale) ** 2)

    # Tie we to entrainment during mixed-layer deepening: we ~ relu(dh/dt), normalized.
    hm = batch["hm"].to(device)                     # h at forecast start [B,1,H,W]
    hm_seq = batch["hm_seq"].to(device)             # h over the rollout   [B,L,1,H,W]
    steps = max(int(lead_days), 1)
    dhdt = (hm_seq[:, -1] - hm) / (float(steps) * SECONDS_PER_DAY)
    dhdt_deepen = torch.relu(dhdt)
    we_tie = _mmean(((we_pred - dhdt_deepen) / we_scale) ** 2)

    # Budget-term magnitudes for ablation reporting (raw units).
    entrainment_dt = float(getattr(args, "entrainment_dt", 0.0))
    hm_safe = torch.clamp(hm, min=float(getattr(args, "hm_min", 5.0)))
    entrain_term = we_pred * entrainment_dt / hm_safe  # K/s cooling magnitude proxy
    diagnostics = {
        "rres_abs_mean": float(_mmean(rres_pred.abs()).item()),
        "we_mean": float(_mmean(we_pred).item()),
        "we_abs_mean": float(_mmean(we_pred.abs()).item()),
        "entrainment_abs_mean": float(_mmean(entrain_term.abs()).item()),
        "dhdt_deepen_mean": float(_mmean(dhdt_deepen).item()),
    }
    return rres_l2, we_tie, diagnostics


def masked_rmse_raw(
    pred_norm: torch.Tensor,
    target_raw: torch.Tensor,
    mask: torch.Tensor,
    target_mean: float,
    target_std: float,
) -> torch.Tensor:
    pred_raw = pred_norm * target_std + target_mean
    diff2 = (pred_raw - target_raw) ** 2
    return torch.sqrt((diff2 * mask).sum() / mask.sum().clamp_min(1.0))


def build_model(
    name: str,
    in_channels: int,
    hidden_channels: list[int],
    we_nonneg: bool = False,
    entrainment_dt: float = 0.0,
    hm_min: float = 5.0,
    we_softplus_shift: float = 0.0,
    residual_feature_channels: int = 1,
) -> torch.nn.Module:
    total_in_channels = in_channels + int(residual_feature_channels)
    if name in {"hybrid_naive", "hybrid_learned"}:
        return ConvLSTMForecaster(in_channels=total_in_channels, hidden_channels=hidden_channels)
    if name == "hybrid_mt_pinn":
        return ConvLSTMMultiTaskPINN(
            in_channels=total_in_channels,
            hidden_channels=hidden_channels,
            we_nonneg=we_nonneg,
            entrainment_dt=entrainment_dt,
            hm_min=hm_min,
            we_softplus_shift=we_softplus_shift,
        )
    raise ValueError(f"Unknown model: {name}")


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        cleaned[key[len("module."):] if key.startswith("module.") else key] = value
    return cleaned


def run_epoch(
    model_name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    target_mean: float,
    target_std: float,
    lambda_physics: float,
    lead_days: int,
    epoch: int | None,
    max_batches: int | None,
    args: argparse.Namespace,
    collect_outputs: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, np.ndarray]]:
    is_train = optimizer is not None
    model.train(is_train)

    total_sum = 0.0
    data_sum = 0.0
    phys_sum = 0.0
    rres_sum = 0.0
    we_tie_sum = 0.0
    rmse_sum = 0.0
    batch_count = 0
    diag_sums: dict[str, float] = {}
    budget_sums: dict[str, float] = {}
    pred_raw_list: list[np.ndarray] = []
    target_raw_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    rres_raw_list: list[np.ndarray] = []
    we_raw_list: list[np.ndarray] = []
    lambda_eff = 0.0
    if model_name == "hybrid_mt_pinn":
        lambda_eff = lambda_physics * physics_lambda_factor(epoch, args.physics_warmup_epochs)

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = batch["x"].to(device)
        use_residual_feature_map = not bool(getattr(args, "disable_residual_map", False))
        if use_residual_feature_map:
            f_seq = batch["f_seq"].to(device)
            if getattr(args, "zero_feature_map", False):
                f_seq = torch.zeros_like(f_seq)
            x_aug = torch.cat([x, f_seq], dim=2)
        else:
            x_aug = x
        y = batch["y"].to(device)
        y_raw = batch["y_raw"].to(device)
        mask = batch["mask"].to(device)

        qnet = batch["qnet"].to(device)
        hm = batch["hm"].to(device)
        u = batch["u"].to(device)
        v = batch["v"].to(device)
        Tm = batch["Tm"].to(device)
        Td = batch["Td"].to(device)
        dTdx = batch["dTdx"].to(device)
        dTdy = batch["dTdy"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            if model_name in {"hybrid_naive", "hybrid_learned"}:
                output = model(x_aug)
                pred = output.prediction
                data_loss = masked_mse(pred, y, mask)
                phys_loss = torch.zeros((), device=device)
                total_loss = data_loss
            else:
                output = model(
                    x=x_aug,
                    qnet=qnet,
                    hm=hm,
                    u=u,
                    v=v,
                    dTdx=dTdx,
                    dTdy=dTdy,
                    Tm=Tm,
                    Td=Td,
                    dTdt=None,
                )
                if getattr(args, "forecast_mode", "free") == "budget":
                    # Physics-integrated forecast: Rres/we drive the prediction, so
                    # the budget is satisfied by construction and the free
                    # temperature head is unused. Data loss backprops through the
                    # rollout into we/Rres.
                    pred_raw = integrate_budget_forecast(
                        Tm=Tm,
                        we_pred=output.we_pred,
                        rres_pred=output.residual_pred,
                        mask=mask,
                        batch=batch,
                        args=args,
                        lead_days=lead_days,
                    )
                    pred = (pred_raw - target_mean) / target_std
                    data_loss = masked_mse(pred, y, mask)
                    phys_loss = torch.zeros((), device=device)
                else:
                    pred = output.T_pred
                    pred, pred_raw = bound_pinn_prediction(
                        pred_norm=pred,
                        Tm=Tm,
                        target_mean=target_mean,
                        target_std=target_std,
                        lead_days=lead_days,
                        max_delta_override=args.pinn_max_delta,
                    )
                    data_loss = masked_mse(pred, y, mask)
                    if args.physics_time_mode == "daily_rollout":
                        phys_loss = daily_rollout_physics_loss(
                            pred_raw=pred_raw,
                            Tm=Tm,
                            target_raw=y_raw,
                            mask=mask,
                            batch=batch,
                            args=args,
                            lead_days=lead_days,
                            we_pred=output.we_pred,
                            rres_pred=output.residual_pred,
                        )
                    else:
                        dTdt = (pred_raw - Tm) / max(float(lead_days) * SECONDS_PER_DAY, 1.0)
                        output = model(
                            x=x_aug,
                            qnet=qnet,
                            hm=hm,
                            u=u,
                            v=v,
                            dTdx=dTdx,
                            dTdy=dTdy,
                            Tm=Tm,
                            Td=Td,
                            dTdt=dTdt,
                        )
                        residual_per_day = output.physics_residual * SECONDS_PER_DAY
                        if getattr(args, "zero_rres_physics", False):
                            residual_per_day = residual_per_day + output.residual_pred * SECONDS_PER_DAY
                        extra_rhs = add_selected_extra_physics_terms(
                            torch.zeros_like(output.physics_residual),
                            field=Tm,
                            dx=batch["dx"].to(device),
                            dy=batch["dy"].to(device),
                            batch=batch,
                            step_idx=-1,
                            args=args,
                        )
                        residual_per_day = residual_per_day - extra_rhs * SECONDS_PER_DAY
                        phys_loss = compute_physics_loss(
                            residual_per_day,
                            mask,
                            args,
                            tendency_per_day=observed_tendency_per_day(Tm, y_raw, lead_days),
                            rhs_per_day=(
                                dTdt * SECONDS_PER_DAY
                                - residual_per_day
                                - (
                                    torch.zeros_like(output.residual_pred)
                                    if getattr(args, "zero_rres_physics", False)
                                    else output.residual_pred
                                ) * SECONDS_PER_DAY
                            ),
                        )
                core = model.module if hasattr(model, "module") else model
                rres_l2, we_tie, diag = identifiability_terms(
                    we_pred=output.we_pred,
                    rres_pred=output.residual_pred,
                    pred_raw=pred_raw,
                    batch=batch,
                    mask=mask,
                    lead_days=lead_days,
                    args=args,
                    we_scale=_require_scale(core, "we_scale"),
                    rres_scale=_require_scale(core, "rres_scale"),
                )
                total_loss = (
                    data_loss
                    + lambda_eff * phys_loss
                    + args.lambda_rres * rres_l2
                    + args.lambda_we_tie * we_tie
                )
                rres_sum += float(rres_l2.item())
                we_tie_sum += float(we_tie.item())
                for key, value in diag.items():
                    diag_sums[key] = diag_sums.get(key, 0.0) + value
                with torch.no_grad():
                    for key, value in physics_budget_diagnostics(
                        pred_raw=pred_raw,
                        Tm=Tm,
                        mask=mask,
                        batch=batch,
                        args=args,
                        lead_days=lead_days,
                        we_pred=output.we_pred,
                        rres_pred=output.residual_pred,
                    ).items():
                        budget_sums[key] = budget_sums.get(key, 0.0) + value

            rmse = masked_rmse_raw(pred, y_raw, mask, target_mean=target_mean, target_std=target_std)
            if collect_outputs:
                pred_raw = (pred * target_std + target_mean).detach().cpu().numpy()
                pred_raw_list.append(pred_raw.astype(np.float32))
                target_raw_list.append(y_raw.detach().cpu().numpy().astype(np.float32))
                mask_list.append(mask.detach().cpu().numpy().astype(np.float32))
                if model_name == "hybrid_mt_pinn":
                    rres_raw_list.append(output.residual_pred.detach().cpu().numpy().astype(np.float32))
                    we_raw_list.append(output.we_pred.detach().cpu().numpy().astype(np.float32))
            if is_train:
                total_loss.backward()
                grad_clip = float(getattr(args, "grad_clip_norm", 0.0))
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        total_sum += float(total_loss.item())
        data_sum += float(data_loss.item())
        phys_sum += float(phys_loss.item())
        rmse_sum += float(rmse.item())
        batch_count += 1

    denom = max(batch_count, 1)
    metrics = {
        "total_loss": total_sum / denom,
        "data_loss": data_sum / denom,
        "physics_loss": phys_sum / denom,
        "lambda_physics_effective": lambda_eff,
        "rres_l2": rres_sum / denom,
        "we_tie_loss": we_tie_sum / denom,
        "rmse": rmse_sum / denom,
    }
    for key, value in diag_sums.items():
        metrics[key] = value / denom
    reduced = {key: maybe_reduce_mean(value, device) for key, value in metrics.items()}
    # Budget diagnostics: pool per-cell SUMS across ranks, then form ratios once.
    # (Reducing them as batch-averaged means would reintroduce the Jensen gap.)
    pooled_budget = {key: maybe_reduce_sum(value, device) for key, value in budget_sums.items()}
    reduced.update(finalize_budget_metrics(pooled_budget))
    if not collect_outputs:
        return reduced
    outputs = {
        "pred_raw": np.concatenate(pred_raw_list, axis=0) if pred_raw_list else np.empty((0,), dtype=np.float32),
        "target_raw": np.concatenate(target_raw_list, axis=0) if target_raw_list else np.empty((0,), dtype=np.float32),
        "mask": np.concatenate(mask_list, axis=0) if mask_list else np.empty((0,), dtype=np.float32),
    }
    if rres_raw_list:
        outputs["rres_raw"] = np.concatenate(rres_raw_list, axis=0)
        outputs["rres_kday"] = outputs["rres_raw"] * SECONDS_PER_DAY
    if we_raw_list:
        outputs["we_raw"] = np.concatenate(we_raw_list, axis=0)
    return reduced, outputs


def make_dataloaders(
    ds: xr.Dataset,
    residual_maps: np.ndarray | None,
    history: int,
    horizon: int,
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, tuple[float, float]], int, list[str]]:
    splits = make_time_splits(time_size=int(ds.sizes["time"]), history=history, lead=horizon)
    requested_dynamic = [c.strip() for c in getattr(args, "dynamic_vars", "").split(",") if c.strip()]
    dynamic_vars = requested_dynamic if requested_dynamic else list(DEFAULT_DYNAMIC_VARS)
    missing_dynamic = [name for name in dynamic_vars if name not in ds.variables]
    if missing_dynamic:
        raise KeyError(f"Requested --dynamic-vars are missing from dataset: {missing_dynamic}")
    # optional extra input channels (e.g. subsurface thetao_z*_anom); only those
    # actually present in the dataset are added, keeping this backward compatible.
    for name in [c.strip() for c in getattr(args, "extra_dynamic_vars", "").split(",") if c.strip()]:
        if name in ds.variables and name not in dynamic_vars:
            dynamic_vars.append(name)
    static_vars = list(DEFAULT_STATIC_VARS)
    stats = compute_train_stats(
        ds=ds,
        dynamic_vars=dynamic_vars,
        static_vars=static_vars,
        history=history,
        lead=horizon,
        train_indices=splits.train,
        residual_maps=residual_maps,
    )
    if getattr(args, "physics_normalize", "none") == "tendency" and getattr(args, "physics_tendency_scale", None) is None:
        args.physics_tendency_scale = float(stats.get("observed_tendency_mse_kday2", (1.0, 1.0))[0])
    forcing_mode = getattr(args, "forcing_mode", "perfect")
    datasets = {
        "train": JejuForecastDataset(ds, history, horizon, splits.train, dynamic_vars, static_vars, stats, residual_maps, forcing_mode=forcing_mode),
        "val": JejuForecastDataset(ds, history, horizon, splits.val, dynamic_vars, static_vars, stats, residual_maps, forcing_mode=forcing_mode),
        "test": JejuForecastDataset(ds, history, horizon, splits.test, dynamic_vars, static_vars, stats, residual_maps, forcing_mode=forcing_mode),
    }
    samplers = {}
    for name, dataset in datasets.items():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            samplers[name] = DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=False,
                drop_last=False,
            )
        else:
            samplers[name] = None

    loaders = {}
    for name, dataset in datasets.items():
        loaders[name] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=samplers[name],
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
    )
    # Fix the RHS normalization scale ONCE from the train split (removes the
    # batch-adaptive drift; the online path then uses this constant every batch).
    if getattr(args, "physics_normalize", "none") == "rhs" and getattr(args, "physics_rhs_scale", None) is None:
        args.physics_rhs_scale = compute_fixed_rhs_scale(
            loaders["train"], args, lead_days=horizon, device=torch.device(args.device)
        )
    in_channels = len(dynamic_vars) + len(static_vars)
    return loaders["train"], loaders["val"], loaders["test"], stats, in_channels, dynamic_vars


def _parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value not in values:
            values.append(value)
    return values


def _bo_score_key(objective: str) -> str:
    if objective == "rmse":
        return "rmse"
    if objective == "data":
        return "data_loss"
    if objective == "total":
        return "total_loss"
    raise ValueError(f"Unknown BO objective: {objective}")


def tune_lambda_physics(
    model_name: str,
    ds: xr.Dataset,
    residual_maps: np.ndarray | None,
    horizon: int,
    history: int,
    args: argparse.Namespace,
) -> float:
    if model_name != "hybrid_mt_pinn":
        return 0.0

    init_values = _parse_float_list(args.bo_init_values)
    candidate_values = _parse_float_list(args.bo_candidates)
    train_loader, val_loader, _, stats, in_channels, _ = make_dataloaders(
        ds=ds,
        residual_maps=residual_maps,
        history=history,
        horizon=horizon,
        args=args,
    )
    hidden_channels = [int(x) for x in args.hidden_channels.split(",") if x]
    device = torch.device(args.device)
    target_mean, target_std = stats["target"]

    tried: list[float] = []
    scores: list[float] = []
    trial_details: list[dict[str, float | int | str]] = []
    score_key = _bo_score_key(args.bo_objective)

    def evaluate(lambda_value: float) -> dict[str, float | int]:
        model = build_model(
            model_name,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
        we_nonneg=args.we_nonneg,
        entrainment_dt=args.entrainment_dt,
        hm_min=args.hm_min,
        residual_feature_channels=0 if getattr(args, "disable_residual_map", False) else 1,
    ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best: dict[str, float | int] = {
            "score": float("inf"),
            "epoch": 0,
            "val_rmse": float("inf"),
            "val_data_loss": float("inf"),
            "val_total_loss": float("inf"),
            "val_physics_loss": float("inf"),
        }
        for local_epoch in range(1, min(args.bo_epochs, args.epochs) + 1):
            run_epoch(
                model_name=model_name,
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                target_mean=target_mean,
                target_std=target_std,
                lambda_physics=lambda_value,
                lead_days=horizon,
                epoch=local_epoch,
                max_batches=args.max_train_batches,
                args=args,
            )
            val_metrics = run_epoch(
                model_name=model_name,
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                target_mean=target_mean,
                target_std=target_std,
                lambda_physics=lambda_value,
                lead_days=horizon,
                epoch=local_epoch,
                max_batches=args.max_val_batches,
                args=args,
            )
            score = float(val_metrics[score_key])
            if score < float(best["score"]):
                best = {
                    "score": score,
                    "epoch": local_epoch,
                    "val_rmse": float(val_metrics["rmse"]),
                    "val_data_loss": float(val_metrics["data_loss"]),
                    "val_total_loss": float(val_metrics["total_loss"]),
                    "val_physics_loss": float(val_metrics["physics_loss"]),
                }
        return best

    for lambda_value in init_values:
        result = evaluate(lambda_value)
        score = float(result["score"])
        tried.append(lambda_value)
        scores.append(score)
        trial_details.append({"lambda_physics": lambda_value, "phase": "init", **result})
        if is_main_process():
            print(
                f"[{model_name} lead={horizon} history={history}] "
                f"BO init lambda={lambda_value:.6g} objective={args.bo_objective} "
                f"score={score:.6f} val_rmse={float(result['val_rmse']):.4f} "
                f"val_data={float(result['val_data_loss']):.6f} "
                f"val_total={float(result['val_total_loss']):.6f}",
                flush=True,
            )

    for _ in range(args.bo_iters):
        gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True)
        x_train = torch.tensor(tried, dtype=torch.float64).view(-1, 1).numpy()
        y_train = torch.tensor(scores, dtype=torch.float64).numpy()
        gp.fit(x_train, y_train)

        remaining = [v for v in candidate_values if v not in tried]
        if not remaining:
            break
        x_cand = torch.tensor(remaining, dtype=torch.float64).view(-1, 1).numpy()
        mean_pred, std_pred = gp.predict(x_cand, return_std=True)
        acquisition = mean_pred - 0.5 * std_pred
        next_lambda = remaining[int(acquisition.argmin())]
        result = evaluate(next_lambda)
        score = float(result["score"])
        tried.append(next_lambda)
        scores.append(score)
        trial_details.append({"lambda_physics": next_lambda, "phase": "step", **result})
        if is_main_process():
            print(
                f"[{model_name} lead={horizon} history={history}] "
                f"BO step lambda={next_lambda:.6g} objective={args.bo_objective} "
                f"score={score:.6f} val_rmse={float(result['val_rmse']):.4f} "
                f"val_data={float(result['val_data_loss']):.6f} "
                f"val_total={float(result['val_total_loss']):.6f}",
                flush=True,
            )

    best_idx = min(range(len(scores)), key=lambda i: scores[i])
    best_lambda = float(tried[best_idx])
    args.bo_trials = trial_details
    args.bo_selected_score = float(scores[best_idx])
    if is_main_process():
        print(
            f"[{model_name} lead={horizon} history={history}] "
            f"BO selected lambda_physics={best_lambda:.6g} "
            f"objective={args.bo_objective} score={scores[best_idx]:.6f}",
            flush=True,
        )
    return best_lambda


def train_single_model(
    model_name: str,
    ds: xr.Dataset,
    residual_maps: np.ndarray | None,
    output_dir: Path,
    horizon: int,
    history: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    train_loader, val_loader, test_loader, stats, in_channels, dynamic_vars = make_dataloaders(
        ds=ds,
        residual_maps=residual_maps,
        history=history,
        horizon=horizon,
        args=args,
    )
    hidden_channels = [int(x) for x in args.hidden_channels.split(",") if x]
    residual_feature_map_enabled = not bool(getattr(args, "disable_residual_map", False))
    residual_feature_channels = 1 if residual_feature_map_enabled else 0
    base_input_channels = in_channels
    model_input_channels = base_input_channels + residual_feature_channels
    dist_cfg = setup_distributed()
    device = dist_cfg.device
    model = build_model(
        model_name,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        we_nonneg=args.we_nonneg,
        entrainment_dt=args.entrainment_dt,
        hm_min=args.hm_min,
        we_softplus_shift=args.we_softplus_shift,
        residual_feature_channels=residual_feature_channels,
    ).to(device)
    if dist_cfg.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_cfg.local_rank] if device.type == "cuda" else None,
            output_device=dist_cfg.local_rank if device.type == "cuda" else None,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    initial_lambda_physics = args.lambda_physics
    lambda_physics = initial_lambda_physics
    uses_pinn_bound = model_name == "hybrid_mt_pinn"
    if args.bayes_opt_lambda and model_name == "hybrid_mt_pinn":
        lambda_physics = tune_lambda_physics(model_name, ds, residual_maps, horizon, history, args)
        initial_lambda_physics = lambda_physics

    target_mean, target_std = stats["target"]
    selection_key = _bo_score_key(args.selection_metric)
    best_val_score = float("inf")
    best_val_total_loss_at_selection = float("inf")
    best_state = None
    best_epoch = 0
    best_lambda_physics = lambda_physics
    epochs_without_improve = 0
    history_log: list[dict[str, float]] = []
    target_model = model.module if hasattr(model, "module") else model

    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{model_name}_lead_{horizon:02d}_history_{history:02d}"
    csv_log_path = output_dir / f"{run_name}_epoch_log.csv"
    csv_fieldnames = [
        "epoch",
        "train_total_loss",
        "train_data_loss",
        "train_physics_loss",
        "lambda_physics_current",
        "train_lambda_physics_effective",
        "train_rmse",
        "val_total_loss",
        "val_data_loss",
        "val_physics_loss",
        "val_lambda_physics_effective",
        "val_rmse",
        "selection_metric",
        "val_selection_score",
        "best_val_selection_score_so_far",
        "bayes_lambda_suggested",
        "bayes_lambda_next",
        "bayes_lambda_loss_ratio",
        "best_val_total_loss_so_far",
        "epochs_without_improve",
    ]
    with csv_log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()

    for epoch in range(1, args.epochs + 1):
        if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        train_metrics = run_epoch(
            model_name=model_name,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
            lambda_physics=lambda_physics,
            lead_days=horizon,
            epoch=epoch,
            max_batches=args.max_train_batches,
            args=args,
        )
        val_metrics = run_epoch(
            model_name=model_name,
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
            lambda_physics=lambda_physics,
            lead_days=horizon,
            epoch=epoch,
            max_batches=args.max_val_batches,
            args=args,
        )

        lambda_next = lambda_physics
        bayes_diag = {
            "bayes_lambda_suggested": lambda_physics,
            "bayes_lambda_next": lambda_physics,
            "bayes_lambda_loss_ratio": 0.0,
        }
        bayes_start_epoch = (
            args.physics_warmup_epochs
            if args.bayes_lambda_start_epoch is None
            else args.bayes_lambda_start_epoch
        )
        if (
            model_name == "hybrid_mt_pinn"
            and args.adaptive_lambda_mode == "bayes"
            and epoch >= bayes_start_epoch
        ):
            source_metrics = train_metrics if args.bayes_lambda_source == "train" else val_metrics
            lambda_next, bayes_diag = bayesian_lambda_update(
                current_lambda=lambda_physics,
                data_loss=source_metrics["data_loss"],
                physics_loss=source_metrics["physics_loss"],
                args=args,
            )

        log_row = {
            "epoch": float(epoch),
            "train_total_loss": train_metrics["total_loss"],
            "train_data_loss": train_metrics["data_loss"],
            "train_physics_loss": train_metrics["physics_loss"],
            "lambda_physics_current": lambda_physics,
            "train_lambda_physics_effective": train_metrics["lambda_physics_effective"],
            "train_rmse": train_metrics["rmse"],
            "val_total_loss": val_metrics["total_loss"],
            "val_data_loss": val_metrics["data_loss"],
            "val_physics_loss": val_metrics["physics_loss"],
            "val_lambda_physics_effective": val_metrics["lambda_physics_effective"],
            "val_rmse": val_metrics["rmse"],
            "selection_metric": args.selection_metric,
            "val_selection_score": val_metrics[selection_key],
            "bayes_lambda_suggested": bayes_diag["bayes_lambda_suggested"],
            "bayes_lambda_next": bayes_diag["bayes_lambda_next"],
            "bayes_lambda_loss_ratio": bayes_diag["bayes_lambda_loss_ratio"],
        }
        history_log.append(log_row)

        if val_metrics[selection_key] < best_val_score:
            best_val_score = val_metrics[selection_key]
            best_val_total_loss_at_selection = val_metrics["total_loss"]
            best_state = {k: v.detach().cpu() for k, v in target_model.state_dict().items()}
            best_epoch = epoch
            best_lambda_physics = lambda_physics
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        log_row["best_val_selection_score_so_far"] = best_val_score
        log_row["best_val_total_loss_so_far"] = best_val_total_loss_at_selection
        log_row["epochs_without_improve"] = float(epochs_without_improve)

        if is_main_process():
            with csv_log_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                writer.writerow(log_row)
            print(
                f"[{model_name} lead={horizon} history={history}] "
                f"epoch={epoch}/{args.epochs} "
                f"train_total={train_metrics['total_loss']:.6f} "
                f"train_data={train_metrics['data_loss']:.6f} "
                f"train_phys={train_metrics['physics_loss']:.6f} "
                f"lambda={lambda_physics:.4f} "
                f"lambda_eff={train_metrics['lambda_physics_effective']:.4f} "
                f"lambda_next={lambda_next:.4f} "
                f"train_rmse={train_metrics['rmse']:.4f} "
                f"val_total={val_metrics['total_loss']:.6f} "
                f"val_data={val_metrics['data_loss']:.6f} "
                f"val_phys={val_metrics['physics_loss']:.6f} "
                f"val_rmse={val_metrics['rmse']:.4f} "
                f"sel_{args.selection_metric}={val_metrics[selection_key]:.6f} "
                f"rres_l2={train_metrics.get('rres_l2', 0.0):.3e} "
                f"we_tie={train_metrics.get('we_tie_loss', 0.0):.3e} "
                f"|Rres|={train_metrics.get('rres_abs_mean', 0.0):.3e} "
                f"we_mean={train_metrics.get('we_mean', 0.0):.3e} "
                f"best_sel={best_val_score:.6f} "
                f"patience_count={epochs_without_improve}/{args.patience}",
                flush=True,
            )

        if epochs_without_improve >= args.patience:
            if is_main_process():
                print(
                    f"[{model_name} lead={horizon} history={history}] "
                    f"early stopping at epoch={epoch} best_epoch={best_epoch}",
                    flush=True,
                )
            break

        lambda_physics = lambda_next

    final_lambda_physics = lambda_physics
    if best_state is not None:
        target_model.load_state_dict(strip_module_prefix(best_state))
    lambda_physics = best_lambda_physics

    test_metrics, test_outputs = run_epoch(
        model_name=model_name,
        model=model,
        loader=test_loader,
        optimizer=None,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
        lambda_physics=lambda_physics,
        lead_days=horizon,
        epoch=args.physics_warmup_epochs,
        max_batches=args.max_test_batches,
        args=args,
        collect_outputs=True,
    )

    model_path = output_dir / f"{run_name}.pt"
    metrics_path = output_dir / f"{run_name}_metrics.json"
    state_dict = strip_module_prefix(target_model.state_dict())
    if is_main_process():
        torch.save(
            {
                "model_name": model_name,
                "model_state_dict": state_dict,
                "stats": stats,
                "history_days": history,
                "lead_days": horizon,
                "best_epoch": best_epoch,
                "selection_metric": args.selection_metric,
                "best_val_selection_score": best_val_score,
                "best_val_total_loss_at_selection": best_val_total_loss_at_selection,
                "lambda_physics": lambda_physics,
                "lambda_physics_initial": initial_lambda_physics,
                "lambda_physics_best": best_lambda_physics,
                "lambda_physics_final": final_lambda_physics,
                "lambda_physics_effective": test_metrics["lambda_physics_effective"],
                "pinn_prediction_bound": uses_pinn_bound,
                "pinn_max_delta": args.pinn_max_delta,
                "physics_loss": args.physics_loss,
                "physics_time_mode": args.physics_time_mode,
                "physics_preset": args.physics_preset,
                "physics_components": args.physics_components,
                "adaptive_weighting": args.adaptive_weighting,
                "adaptive_sampling": args.adaptive_sampling,
                "adaptive_temperature": args.adaptive_temperature,
                "adaptive_samples": args.adaptive_samples,
                "adaptive_weight_min": args.adaptive_weight_min,
                "adaptive_weight_max": args.adaptive_weight_max,
                "forcing_mode": args.forcing_mode,
                "forecast_mode": args.forecast_mode,
                "physics_huber_beta": args.physics_huber_beta,
                "physics_clip": args.physics_clip,
                "physics_normalize": args.physics_normalize,
                "physics_normalize_eps": args.physics_normalize_eps,
                "physics_tendency_scale": args.physics_tendency_scale,
                "physics_rhs_scale": args.physics_rhs_scale,
                "observed_tendency_rms_kday": float(stats.get("observed_tendency_rms_kday", (0.0, 1.0))[0]),
                "physics_warmup_epochs": args.physics_warmup_epochs,
                "residual_path": str(args.residual_path) if args.residual_path is not None else None,
                "entrainment_dt": args.entrainment_dt,
                "we_nonneg": args.we_nonneg,
                "lambda_rres": args.lambda_rres,
                "rres_penalty_normalize": args.rres_penalty_normalize,
                "lambda_we_tie": args.lambda_we_tie,
                "zero_rres_physics": args.zero_rres_physics,
                "hm_min": args.hm_min,
                "dynamic_vars": dynamic_vars,
                "dynamic_var_count": len(dynamic_vars),
                "static_vars": list(DEFAULT_STATIC_VARS),
                "static_var_count": len(DEFAULT_STATIC_VARS),
                "base_input_channels": base_input_channels,
                "residual_feature_map_enabled": residual_feature_map_enabled,
                "residual_feature_channels": residual_feature_channels,
                "model_input_channels": model_input_channels,
                "disable_residual_map": args.disable_residual_map,
                "extra_dynamic_vars": args.extra_dynamic_vars,
                "adaptive_lambda_mode": args.adaptive_lambda_mode,
                "bayes_lambda_target_ratio": args.bayes_lambda_target_ratio,
                "bayes_lambda_min": args.bayes_lambda_min,
                "bayes_lambda_max": args.bayes_lambda_max,
                "bayes_lambda_prior_strength": args.bayes_lambda_prior_strength,
                "bayes_lambda_evidence_strength": args.bayes_lambda_evidence_strength,
                "bayes_lambda_source": args.bayes_lambda_source,
                "bayes_lambda_start_epoch": (
                    args.physics_warmup_epochs
                    if args.bayes_lambda_start_epoch is None
                    else args.bayes_lambda_start_epoch
                ),
                "bayes_opt_lambda": args.bayes_opt_lambda,
                "bo_objective": args.bo_objective,
                "bo_init_values": args.bo_init_values,
                "bo_candidates": args.bo_candidates,
                "bo_iters": args.bo_iters,
                "bo_epochs": args.bo_epochs,
                "bo_trials": getattr(args, "bo_trials", []),
                "bo_selected_score": getattr(args, "bo_selected_score", None),
            },
            model_path,
        )

    metrics = {
        "model_name": model_name,
        "lead_days": horizon,
        "history_days": history,
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "best_val_selection_score": best_val_score,
        "best_val_total_loss_at_selection": best_val_total_loss_at_selection,
        "lambda_physics": lambda_physics,
        "lambda_physics_initial": initial_lambda_physics,
        "lambda_physics_best": best_lambda_physics,
        "lambda_physics_final": final_lambda_physics,
        "lambda_physics_effective": test_metrics["lambda_physics_effective"],
        "pinn_prediction_bound": uses_pinn_bound,
        "pinn_max_delta": args.pinn_max_delta,
        "physics_loss": args.physics_loss,
        "physics_time_mode": args.physics_time_mode,
        "physics_preset": args.physics_preset,
        "physics_components": args.physics_components,
        "adaptive_weighting": args.adaptive_weighting,
        "adaptive_sampling": args.adaptive_sampling,
        "adaptive_temperature": args.adaptive_temperature,
        "adaptive_samples": args.adaptive_samples,
        "adaptive_weight_min": args.adaptive_weight_min,
        "adaptive_weight_max": args.adaptive_weight_max,
        "forcing_mode": args.forcing_mode,
        "forecast_mode": args.forecast_mode,
        "budget_substeps": args.budget_substeps,
        "physics_hdiff": args.physics_hdiff,
        "k_diff": args.k_diff,
        "physics_vertical": args.physics_vertical,
        "dynamic_vars": dynamic_vars,
        "dynamic_var_count": len(dynamic_vars),
        "static_vars": list(DEFAULT_STATIC_VARS),
        "static_var_count": len(DEFAULT_STATIC_VARS),
        "base_input_channels": base_input_channels,
        "residual_feature_map_enabled": residual_feature_map_enabled,
        "residual_feature_channels": residual_feature_channels,
        "model_input_channels": model_input_channels,
        "disable_residual_map": args.disable_residual_map,
        "extra_dynamic_vars": args.extra_dynamic_vars,
        "adaptive_lambda_mode": args.adaptive_lambda_mode,
        "bayes_lambda_target_ratio": args.bayes_lambda_target_ratio,
        "bayes_lambda_min": args.bayes_lambda_min,
        "bayes_lambda_max": args.bayes_lambda_max,
        "bayes_lambda_prior_strength": args.bayes_lambda_prior_strength,
        "bayes_lambda_evidence_strength": args.bayes_lambda_evidence_strength,
        "bayes_lambda_source": args.bayes_lambda_source,
        "bayes_lambda_start_epoch": (
            args.physics_warmup_epochs
            if args.bayes_lambda_start_epoch is None
            else args.bayes_lambda_start_epoch
        ),
        "bayes_opt_lambda": args.bayes_opt_lambda,
        "bo_objective": args.bo_objective,
        "bo_init_values": args.bo_init_values,
        "bo_candidates": args.bo_candidates,
        "bo_iters": args.bo_iters,
        "bo_epochs": args.bo_epochs,
        "bo_trials": getattr(args, "bo_trials", []),
        "bo_selected_score": getattr(args, "bo_selected_score", None),
        "is_hindcast": args.forcing_mode == "perfect",
        "physics_huber_beta": args.physics_huber_beta,
        "physics_clip": args.physics_clip,
        "physics_normalize": args.physics_normalize,
        "physics_normalize_eps": args.physics_normalize_eps,
        "physics_tendency_scale": args.physics_tendency_scale,
        "physics_rhs_scale": args.physics_rhs_scale,
        "observed_tendency_rms_kday": float(stats.get("observed_tendency_rms_kday", (0.0, 1.0))[0]),
        "physics_warmup_epochs": args.physics_warmup_epochs,
        "best_val_total_loss": best_val_total_loss_at_selection,
        "test_total_loss": test_metrics["total_loss"],
        "test_data_loss": test_metrics["data_loss"],
        "test_physics_loss": test_metrics["physics_loss"],
        "test_rmse": test_metrics["rmse"],
        "entrainment_dt": args.entrainment_dt,
        "we_nonneg": args.we_nonneg,
        "lambda_rres": args.lambda_rres,
        "rres_penalty_normalize": args.rres_penalty_normalize,
        "lambda_we_tie": args.lambda_we_tie,
        "zero_feature_map": args.zero_feature_map,
        "zero_rres_physics": args.zero_rres_physics,
        "hm_min": args.hm_min,
        "we_softplus_shift": args.we_softplus_shift,
        "test_rres_l2": test_metrics.get("rres_l2"),
        "test_we_tie_loss": test_metrics.get("we_tie_loss"),
        "test_rres_abs_mean": test_metrics.get("rres_abs_mean"),
        "test_we_mean": test_metrics.get("we_mean"),
        "test_we_abs_mean": test_metrics.get("we_abs_mean"),
        "test_entrainment_abs_mean": test_metrics.get("entrainment_abs_mean"),
        "test_dhdt_deepen_mean": test_metrics.get("dhdt_deepen_mean"),
        "test_budget_flux_std_kday": test_metrics.get("budget_flux_std_kday"),
        "test_budget_adv_std_kday": test_metrics.get("budget_adv_std_kday"),
        "test_budget_entrain_std_kday": test_metrics.get("budget_entrain_std_kday"),
        "test_budget_rres_std_kday": test_metrics.get("budget_rres_std_kday"),
        "test_budget_flux_absmean_kday": test_metrics.get("budget_flux_absmean_kday"),
        "test_budget_adv_absmean_kday": test_metrics.get("budget_adv_absmean_kday"),
        "test_budget_entrain_absmean_kday": test_metrics.get("budget_entrain_absmean_kday"),
        "test_budget_rres_absmean_kday": test_metrics.get("budget_rres_absmean_kday"),
        "test_budget_dTdt_absmean_kday": test_metrics.get("budget_dTdt_absmean_kday"),
        "test_budget_rres_l2_kday": test_metrics.get("budget_rres_l2_kday"),
        "test_budget_dTdt_l2_kday": test_metrics.get("budget_dTdt_l2_kday"),
        "test_rres_frac_of_tendency": test_metrics.get("rres_frac_of_tendency"),
        "test_rres_frac_of_physics": test_metrics.get("rres_frac_of_physics"),
        "test_rres_l2_frac_of_tendency": test_metrics.get("rres_l2_frac_of_tendency"),
        "test_rres_l2_frac_of_physics": test_metrics.get("rres_l2_frac_of_physics"),
        "test_rres_crest_factor": test_metrics.get("rres_crest_factor"),
        "test_dtdt_crest_factor": test_metrics.get("dtdt_crest_factor"),
        "test_budget_hdiff_absmean_kday": test_metrics.get("budget_hdiff_absmean_kday"),
        "test_budget_vadv_absmean_kday": test_metrics.get("budget_vadv_absmean_kday"),
        "test_share_flux": test_metrics.get("share_flux"),
        "test_share_adv": test_metrics.get("share_adv"),
        "test_share_entrain": test_metrics.get("share_entrain"),
        "test_share_hdiff": test_metrics.get("share_hdiff"),
        "test_share_vadv": test_metrics.get("share_vadv"),
        "test_share_rres": test_metrics.get("share_rres"),
        "history_log": history_log,
        "model_path": str(model_path),
        "epoch_log_csv": str(csv_log_path),
        "residual_path": str(args.residual_path) if args.residual_path is not None else None,
    }
    forecast_metrics = compute_forecast_metrics(
        pred=test_outputs["pred_raw"],
        target=test_outputs["target_raw"],
        mask=test_outputs["mask"],
        quantile=0.90,
        event_min_duration=5,
    )
    metrics.update(forecast_metrics)
    if is_main_process():
        fields_path = output_dir / f"{run_name}_test_fields.npz"
        np.savez_compressed(
            fields_path,
            pred_raw=test_outputs["pred_raw"],
            target_raw=test_outputs["target_raw"],
            mask=test_outputs["mask"],
            rres_raw=test_outputs.get("rres_raw", np.empty((0,), dtype=np.float32)),
            rres_kday=test_outputs.get("rres_kday", np.empty((0,), dtype=np.float32)),
            we_raw=test_outputs.get("we_raw", np.empty((0,), dtype=np.float32)),
        )
        metrics["test_fields_npz"] = str(fields_path)
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print(f"saved model: {model_path}", flush=True)
        print(f"saved metrics: {metrics_path}", flush=True)
        print(f"saved epoch log: {csv_log_path}", flush=True)
        print(f"saved test fields: {fields_path}", flush=True)
    return metrics


def main() -> None:
    args = parse_args()
    configure_physics_components(args)
    set_process_title()
    set_seed(args.seed)
    if args.disable_residual_map:
        residual_maps = None
    else:
        if args.residual_path is None:
            raise ValueError("--residual-path is required unless --disable-residual-map is set")
        residual_maps = np.load(args.residual_path).astype(np.float32)
    try:
        ds = xr.open_dataset(args.data_path)

        if args.run_all:
            configs = [(lead, history) for lead, history in PRESET_CONFIGS.items()]
        else:
            if args.horizon is None:
                raise ValueError("--horizon is required unless --run-all is used")
            history = args.history if args.history is not None else PRESET_CONFIGS[args.horizon]
            configs = [(args.horizon, history)]

        summary: list[dict[str, float]] = []
        for horizon, history in configs:
            metrics = train_single_model(
                model_name=args.model,
                ds=ds,
                residual_maps=residual_maps,
                output_dir=args.output_dir,
                horizon=horizon,
                history=history,
                args=args,
            )
            summary.append(metrics)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        if is_main_process():
            summary_path = args.output_dir / f"{args.model}_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2))
            print(f"saved summary: {summary_path}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
