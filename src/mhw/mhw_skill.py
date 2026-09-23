"""Hobday-style marine-heatwave (MHW) detection skill metrics.

This module scores *event detection* skill, not just SST regression error.
It follows Hobday et al. (2016):

  - A day-of-year seasonally varying 90th-percentile threshold is built from a
    reference (train-only) climatology using a +/- window-day pooling window.
  - A grid cell is in an MHW on day t when SST(t) > threshold(doy(t)) for at
    least ``min_duration`` consecutive days.

From the observed and predicted MHW binary fields we compute a contingency
table (hits / misses / false alarms / correct negatives) and the standard
categorical forecast scores: POD (hit rate), FAR (false alarm ratio), CSI,
F1, frequency bias, and accuracy. We also detect events on the domain-mean
MHW-area index and match them to report onset lead-time and duration error.

All functions are numpy-only (pandas only for day-of-year) so they can be unit
tested offline and reused from any training / evaluation script.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


DAYS_PER_YEAR = 366


def day_of_year(time_values: np.ndarray) -> np.ndarray:
    """Return the 1..366 day-of-year for a datetime64 array."""
    return pd.to_datetime(time_values).dayofyear.to_numpy().astype(np.int64)


def _circular_doy_distance(a: np.ndarray, b: int) -> np.ndarray:
    """Minimum wrap-around distance (in days) between doy arrays and a doy."""
    raw = np.abs(a - b)
    return np.minimum(raw, DAYS_PER_YEAR - raw)


def compute_climatology_threshold(
    sst: np.ndarray,
    doy: np.ndarray,
    reference_time_mask: np.ndarray,
    window: int = 5,
    quantile: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    """Build seasonally varying climatology mean and percentile threshold.

    Parameters
    ----------
    sst : array [T, H, W]
        Daily SST field.
    doy : array [T]
        Day-of-year (1..366) for each timestep.
    reference_time_mask : bool array [T]
        Timesteps allowed into the climatology (use train-only to avoid leakage).
    window : int
        Half-width of the day-of-year pooling window (Hobday default 5).
    quantile : float
        Percentile for the MHW threshold (Hobday default 0.90).

    Returns
    -------
    clim : array [DAYS_PER_YEAR + 1, H, W]
        Climatological mean SST indexed by doy (index 0 unused).
    thresh : array [DAYS_PER_YEAR + 1, H, W]
        Climatological ``quantile`` threshold indexed by doy (index 0 unused).
    """
    sst = np.asarray(sst, dtype=np.float32)
    if sst.ndim != 3:
        raise ValueError("sst must be [T, H, W]")
    _, h, w = sst.shape
    ref_doy = doy[reference_time_mask]
    ref_sst = sst[reference_time_mask]

    clim = np.full((DAYS_PER_YEAR + 1, h, w), np.nan, dtype=np.float32)
    thresh = np.full((DAYS_PER_YEAR + 1, h, w), np.nan, dtype=np.float32)
    for j in range(1, DAYS_PER_YEAR + 1):
        sel = _circular_doy_distance(ref_doy, j) <= window
        if not np.any(sel):
            continue
        pooled = ref_sst[sel]  # [n, H, W]
        with warnings.catch_warnings():
            # land columns are all-NaN by construction; that is expected
            warnings.simplefilter("ignore", RuntimeWarning)
            clim[j] = np.nanmean(pooled, axis=0)
            thresh[j] = np.nanquantile(pooled, quantile, axis=0)
    return clim, thresh


def threshold_for_times(thresh_by_doy: np.ndarray, doy: np.ndarray) -> np.ndarray:
    """Gather the per-doy threshold field for an arbitrary sequence of days."""
    return thresh_by_doy[doy]


def enforce_min_duration(binary_txy: np.ndarray, min_duration: int) -> np.ndarray:
    """Keep only MHW spells lasting >= ``min_duration`` consecutive days.

    Operates independently per grid cell along the time axis (axis 0).
    """
    binary = np.asarray(binary_txy, dtype=bool)
    if min_duration <= 1:
        return binary
    t = binary.shape[0]
    flat = binary.reshape(t, -1)
    out = np.zeros_like(flat)
    for p in range(flat.shape[1]):
        col = flat[:, p]
        if not col.any():
            continue
        start = None
        for i in range(t):
            if col[i] and start is None:
                start = i
            elif not col[i] and start is not None:
                if i - start >= min_duration:
                    out[start:i, p] = True
                start = None
        if start is not None and t - start >= min_duration:
            out[start:t, p] = True
    return out.reshape(binary.shape)


def mhw_binary_field(
    sst: np.ndarray,
    doy: np.ndarray,
    thresh_by_doy: np.ndarray,
    min_duration: int = 5,
) -> np.ndarray:
    """Return the boolean MHW-day field for an SST series [T, H, W]."""
    exceed = sst > threshold_for_times(thresh_by_doy, doy)
    exceed = np.nan_to_num(exceed.astype(np.float32)).astype(bool)
    return enforce_min_duration(exceed, min_duration)


@dataclass
class ContingencySkill:
    hits: int
    misses: int
    false_alarms: int
    correct_negatives: int
    pod: float            # probability of detection (hit rate) = H / (H + M)
    far: float            # false alarm ratio = F / (H + F)
    csi: float            # critical success index = H / (H + M + F)
    f1: float             # 2H / (2H + M + F)
    frequency_bias: float  # (H + F) / (H + M)
    accuracy: float       # (H + CN) / total
    base_rate_obs: float  # observed MHW frequency
    base_rate_pred: float  # predicted MHW frequency


def _safe(n: float, d: float) -> float:
    return float(n / d) if d > 0 else 0.0


def contingency_skill(
    pred_binary: np.ndarray,
    obs_binary: np.ndarray,
    mask: np.ndarray | None = None,
) -> ContingencySkill:
    """Categorical detection scores from two boolean fields of equal shape."""
    pred = np.asarray(pred_binary, dtype=bool)
    obs = np.asarray(obs_binary, dtype=bool)
    if pred.shape != obs.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {obs.shape}")
    if mask is not None:
        valid = np.asarray(mask).astype(bool)
        valid = np.broadcast_to(valid, pred.shape)
        pred = pred & valid
        obs = obs & valid
    else:
        valid = np.ones_like(pred, dtype=bool)

    hits = int(np.sum(pred & obs & valid))
    false_alarms = int(np.sum(pred & ~obs & valid))
    misses = int(np.sum(~pred & obs & valid))
    correct_negatives = int(np.sum(~pred & ~obs & valid))
    total = hits + misses + false_alarms + correct_negatives
    return ContingencySkill(
        hits=hits,
        misses=misses,
        false_alarms=false_alarms,
        correct_negatives=correct_negatives,
        pod=_safe(hits, hits + misses),
        far=_safe(false_alarms, hits + false_alarms),
        csi=_safe(hits, hits + misses + false_alarms),
        f1=_safe(2 * hits, 2 * hits + misses + false_alarms),
        frequency_bias=_safe(hits + false_alarms, hits + misses),
        accuracy=_safe(hits + correct_negatives, total),
        base_rate_obs=_safe(hits + misses, total),
        base_rate_pred=_safe(hits + false_alarms, total),
    )


def fixed_threshold_peak_metrics(
    pred_sst: np.ndarray,
    obs_sst: np.ndarray,
    target_doy: np.ndarray,
    thresh_by_doy: np.ndarray,
    mask: np.ndarray,
    min_duration: int = 5,
) -> dict[str, float]:
    """Peak/bias diagnostics on the same fixed-threshold MHW event support.

    These differ from legacy global peak bias. All ``fixed_mhw_obs_event_*``
    values use the observed MHW grid-time support, so they answer whether the
    forecast reproduces the observed event tail under the same climatological
    threshold. Predicted-event and union-event supports are reported separately.
    """
    pred = np.squeeze(np.asarray(pred_sst, dtype=np.float32))
    obs = np.squeeze(np.asarray(obs_sst, dtype=np.float32))
    mask2d = np.squeeze(np.asarray(mask)).astype(bool)
    if mask2d.ndim == 3:
        mask2d = mask2d[0]
    valid = np.broadcast_to(mask2d[None, ...], obs.shape)
    pred_mhw = mhw_binary_field(pred, target_doy, thresh_by_doy, min_duration=min_duration)
    obs_mhw = mhw_binary_field(obs, target_doy, thresh_by_doy, min_duration=min_duration)

    def _support_stats(support: np.ndarray, prefix: str) -> dict[str, float]:
        support = support & valid
        count = int(np.sum(support))
        if count == 0:
            return {
                f"{prefix}_count": 0.0,
                f"{prefix}_mean_bias": 0.0,
                f"{prefix}_mae": 0.0,
                f"{prefix}_rmse": 0.0,
                f"{prefix}_peak_obs": 0.0,
                f"{prefix}_peak_pred": 0.0,
                f"{prefix}_peak_bias": 0.0,
            }
        diff = pred[support] - obs[support]
        peak_obs = float(np.max(obs[support]))
        peak_pred = float(np.max(pred[support]))
        return {
            f"{prefix}_count": float(count),
            f"{prefix}_mean_bias": float(np.mean(diff)),
            f"{prefix}_mae": float(np.mean(np.abs(diff))),
            f"{prefix}_rmse": float(np.sqrt(np.mean(diff ** 2))),
            f"{prefix}_peak_obs": peak_obs,
            f"{prefix}_peak_pred": peak_pred,
            f"{prefix}_peak_bias": peak_pred - peak_obs,
        }

    out = {}
    out.update(_support_stats(obs_mhw, "fixed_mhw_obs_event"))
    out.update(_support_stats(pred_mhw, "fixed_mhw_pred_event"))
    out.update(_support_stats(obs_mhw | pred_mhw, "fixed_mhw_union_event"))
    return out


def _spells(binary_1d: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index pairs of True runs in a 1-D boolean series."""
    b = np.asarray(binary_1d, dtype=bool)
    spells = []
    start = None
    for i, flag in enumerate(b):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spells.append((start, i))
            start = None
    if start is not None:
        spells.append((start, len(b)))
    return spells


@dataclass
class OnsetSkill:
    observed_events: int
    predicted_events: int
    matched_events: int
    onset_mae_days: float          # |predicted onset - observed onset|
    onset_bias_days: float         # mean(predicted onset - observed onset), + = late
    duration_mae_days: float
    duration_bias_days: float
    event_pod: float               # matched / observed
    event_far: float               # unmatched predicted / predicted
    onset_tolerance_days: int      # tolerance used for the "onset hit" test below
    onset_within_tol_frac: float   # matched events with |onset err| <= tolerance, over observed


def onset_duration_skill(
    pred_index: np.ndarray,
    obs_index: np.ndarray,
    min_duration: int = 5,
    onset_tolerance_days: int = 3,
) -> OnsetSkill:
    """Match predicted vs observed events on a 1-D MHW index and score onset.

    An event is a run of >= ``min_duration`` True days in the boolean series.
    A predicted event matches an observed one when their day ranges overlap;
    each observed event is matched at most once (earliest overlapping pred).
    Onset is noisy, so ``onset_within_tol_frac`` reports the fraction of
    observed events whose matched-onset error is within ``onset_tolerance_days``.
    """
    pred = enforce_min_duration(np.asarray(pred_index, dtype=bool)[:, None, None], min_duration)[:, 0, 0]
    obs = enforce_min_duration(np.asarray(obs_index, dtype=bool)[:, None, None], min_duration)[:, 0, 0]
    obs_spells = _spells(obs)
    pred_spells = _spells(pred)

    used = set()
    onset_err = []
    dur_err = []
    matched = 0
    within_tol = 0
    for (os, oe) in obs_spells:
        best = None
        for k, (ps, pe) in enumerate(pred_spells):
            if k in used:
                continue
            if ps < oe and os < pe:  # overlap
                best = k
                break
        if best is not None:
            used.add(best)
            ps, pe = pred_spells[best]
            err = ps - os
            onset_err.append(err)
            dur_err.append((pe - ps) - (oe - os))
            matched += 1
            if abs(err) <= onset_tolerance_days:
                within_tol += 1

    return OnsetSkill(
        observed_events=len(obs_spells),
        predicted_events=len(pred_spells),
        matched_events=matched,
        onset_mae_days=float(np.mean(np.abs(onset_err))) if onset_err else 0.0,
        onset_bias_days=float(np.mean(onset_err)) if onset_err else 0.0,
        duration_mae_days=float(np.mean(np.abs(dur_err))) if dur_err else 0.0,
        duration_bias_days=float(np.mean(dur_err)) if dur_err else 0.0,
        event_pod=_safe(matched, len(obs_spells)),
        event_far=_safe(len(pred_spells) - matched, len(pred_spells)),
        onset_tolerance_days=int(onset_tolerance_days),
        onset_within_tol_frac=_safe(within_tol, len(obs_spells)),
    )


def domain_mhw_index(binary_txy: np.ndarray, mask: np.ndarray, area_frac: float = 0.15) -> np.ndarray:
    """Collapse a boolean [T, H, W] field to a 1-D domain MHW state series.

    A day is a domain MHW day when the fraction of valid ocean cells in MHW
    exceeds ``area_frac``.
    """
    binary = np.asarray(binary_txy, dtype=bool)
    valid = np.broadcast_to(np.asarray(mask).astype(bool), binary.shape)
    num = (binary & valid).reshape(binary.shape[0], -1).sum(axis=1)
    den = valid.reshape(binary.shape[0], -1).sum(axis=1).clip(min=1)
    return (num / den) > area_frac


def compute_mhw_skill_metrics(
    pred_sst: np.ndarray,
    obs_sst: np.ndarray,
    target_doy: np.ndarray,
    thresh_by_doy: np.ndarray,
    mask: np.ndarray,
    min_duration: int = 5,
    area_frac: float = 0.15,
    onset_tolerance_days: int = 3,
) -> dict[str, float]:
    """Full MHW detection skill for one predictor over the test period.

    Parameters
    ----------
    pred_sst, obs_sst : array [T, H, W] (a leading singleton channel is squeezed)
        Predicted and observed SST at the forecast target days, in chronological
        order (consecutive daily targets so duration filtering is valid).
    target_doy : array [T]
        Day-of-year of each target day.
    thresh_by_doy : array [DAYS_PER_YEAR + 1, H, W]
        Reference threshold from :func:`compute_climatology_threshold`.
    mask : array [H, W] or [T, H, W]
        Ocean mask (True/1 = valid).
    """
    pred = np.squeeze(np.asarray(pred_sst, dtype=np.float32))
    obs = np.squeeze(np.asarray(obs_sst, dtype=np.float32))
    if pred.ndim != 3:
        raise ValueError(f"expected [T, H, W] after squeeze, got {pred.shape}")
    mask2d = np.squeeze(np.asarray(mask))
    if mask2d.ndim == 3:
        mask2d = mask2d[0]
    mask_bool = mask2d.astype(bool)

    pred_mhw = mhw_binary_field(pred, target_doy, thresh_by_doy, min_duration=min_duration)
    obs_mhw = mhw_binary_field(obs, target_doy, thresh_by_doy, min_duration=min_duration)

    grid = contingency_skill(pred_mhw, obs_mhw, mask=mask_bool[None, ...])

    pred_domain = domain_mhw_index(pred_mhw, mask_bool, area_frac=area_frac)
    obs_domain = domain_mhw_index(obs_mhw, mask_bool, area_frac=area_frac)
    onset = onset_duration_skill(
        pred_domain, obs_domain, min_duration=min_duration, onset_tolerance_days=onset_tolerance_days
    )

    out = {f"mhw_grid_{k}": v for k, v in asdict(grid).items()}
    out.update({f"mhw_event_{k}": v for k, v in asdict(onset).items()})
    out.update(
        fixed_threshold_peak_metrics(
            pred_sst=pred,
            obs_sst=obs,
            target_doy=target_doy,
            thresh_by_doy=thresh_by_doy,
            mask=mask_bool,
            min_duration=min_duration,
        )
    )
    out["mhw_grid_obs_positive_count"] = float(grid.hits + grid.misses)
    out["mhw_grid_pred_positive_count"] = float(grid.hits + grid.false_alarms)
    out["mhw_grid_pooled_counting"] = "pooled_time_space"
    out["mhw_min_duration"] = int(min_duration)
    out["mhw_area_frac"] = float(area_frac)
    return out
