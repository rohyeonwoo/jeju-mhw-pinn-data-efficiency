from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    diff2 = ((pred - target) ** 2) * mask
    return float(np.sqrt(_safe_div(float(diff2.sum()), float(mask.sum()))))


def mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    diff = np.abs(pred - target) * mask
    return _safe_div(float(diff.sum()), float(mask.sum()))


def bias(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    diff = (pred - target) * mask
    return _safe_div(float(diff.sum()), float(mask.sum()))


def build_top_quantile_mask(target: np.ndarray, mask: np.ndarray, quantile: float = 0.90) -> np.ndarray:
    valid = target[mask > 0]
    threshold = float(np.quantile(valid, quantile))
    return ((target >= threshold) & (mask > 0)).astype(np.float32)


def peak_amplitude_bias(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    pred_peak = np.max(np.where(mask > 0, pred, -np.inf))
    true_peak = np.max(np.where(mask > 0, target, -np.inf))
    return float(pred_peak - true_peak)


def spatial_mean_series(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weighted = (field * mask).sum(axis=(-2, -1))
    denom = np.clip(mask.sum(axis=(-2, -1)), 1.0, None)
    return np.asarray(weighted / denom, dtype=np.float32).reshape(-1)


@dataclass
class EventSummary:
    count: int
    duration_mean: float
    mean_intensity_mean: float
    max_intensity_mean: float
    cumulative_intensity_mean: float


def detect_mhw_events(
    series: np.ndarray,
    threshold: float,
    min_duration: int = 5,
) -> list[dict[str, float]]:
    exceed = series > threshold
    events: list[dict[str, float]] = []
    start = None
    for idx, flag in enumerate(exceed):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            end = idx
            if end - start >= min_duration:
                segment = series[start:end]
                intensity = segment - threshold
                events.append(
                    {
                        "duration": float(end - start),
                        "mean_intensity": float(intensity.mean()),
                        "max_intensity": float(intensity.max()),
                        "cumulative_intensity": float(intensity.sum()),
                    }
                )
            start = None
    if start is not None:
        end = len(series)
        if end - start >= min_duration:
            segment = series[start:end]
            intensity = segment - threshold
            events.append(
                {
                    "duration": float(end - start),
                    "mean_intensity": float(intensity.mean()),
                    "max_intensity": float(intensity.max()),
                    "cumulative_intensity": float(intensity.sum()),
                }
            )
    return events


def summarize_events(events: list[dict[str, float]]) -> EventSummary:
    if not events:
        return EventSummary(0, 0.0, 0.0, 0.0, 0.0)
    return EventSummary(
        count=len(events),
        duration_mean=float(np.mean([e["duration"] for e in events])),
        mean_intensity_mean=float(np.mean([e["mean_intensity"] for e in events])),
        max_intensity_mean=float(np.mean([e["max_intensity"] for e in events])),
        cumulative_intensity_mean=float(np.mean([e["cumulative_intensity"] for e in events])),
    )


def compute_forecast_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    quantile: float = 0.90,
    event_min_duration: int = 5,
) -> dict[str, float]:
    top_mask = build_top_quantile_mask(target, mask, quantile=quantile)
    threshold_series = spatial_mean_series(target, mask)
    threshold = float(np.quantile(threshold_series, quantile))
    pred_series = spatial_mean_series(pred, mask)
    target_series = threshold_series
    pred_events = summarize_events(detect_mhw_events(pred_series, threshold, min_duration=event_min_duration))
    target_events = summarize_events(detect_mhw_events(target_series, threshold, min_duration=event_min_duration))

    return {
        "test_rmse": rmse(pred, target, mask),
        "test_mae": mae(pred, target, mask),
        "test_bias": bias(pred, target, mask),
        "heatwave_rmse": rmse(pred, target, top_mask),
        "top10_error": rmse(pred, target, top_mask),
        "peak_amplitude_bias": peak_amplitude_bias(pred, target, mask),
        "mhw_threshold_quantile": quantile,
        "mhw_event_min_duration": event_min_duration,
        "observed_event_count": float(target_events.count),
        "predicted_event_count": float(pred_events.count),
        "duration_observed": target_events.duration_mean,
        "duration_predicted": pred_events.duration_mean,
        "duration_bias": pred_events.duration_mean - target_events.duration_mean,
        "mean_intensity_observed": target_events.mean_intensity_mean,
        "mean_intensity_predicted": pred_events.mean_intensity_mean,
        "mean_intensity_bias": pred_events.mean_intensity_mean - target_events.mean_intensity_mean,
        "max_intensity_observed": target_events.max_intensity_mean,
        "max_intensity_predicted": pred_events.max_intensity_mean,
        "max_intensity_bias": pred_events.max_intensity_mean - target_events.max_intensity_mean,
        "cumulative_intensity_observed": target_events.cumulative_intensity_mean,
        "cumulative_intensity_predicted": pred_events.cumulative_intensity_mean,
        "cumulative_intensity_bias": pred_events.cumulative_intensity_mean - target_events.cumulative_intensity_mean,
    }
