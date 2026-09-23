from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


DEFAULT_DYNAMIC_VARS = [
    "sst_mean",
    "sst_anom",
    "u_mld",
    "v_mld",
    "t2m",
    "d2m",
    "u10",
    "v10",
    "mslp",
    "tcc",
    "slhf",
    "sshf",
    "ssr",
    "str",
    "mlotst",
    "wind10",
    "qnet",
    "ocean_mask",
    "u_mld_filled_mask",
    "v_mld_filled_mask",
    "mlotst_filled_mask",
]
DEFAULT_STATIC_VARS = ["ocean_mask_static", "lat2d", "lon2d"]
EARTH_RADIUS_M = 6_371_000.0


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def build_lat_lon_2d(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    lat = ds["latitude"].values.astype(np.float32)
    lon = ds["longitude"].values.astype(np.float32)
    lon2d, lat2d = np.meshgrid(lon, lat)
    return lat2d, lon2d


def load_time_grid_array(ds: xr.Dataset, var_name: str) -> np.ndarray:
    """Load a variable as [time, latitude, longitude], repeating 2D grids over time."""
    if var_name not in ds:
        raise KeyError(f"Dataset variable not found: {var_name}")
    arr = ds[var_name]
    if "time" in arr.dims:
        spatial_dims = [dim for dim in arr.dims if dim != "time"]
        if len(spatial_dims) != 2:
            raise ValueError(f"Expected {var_name} to have two spatial dims, got {arr.dims}")
        return arr.transpose("time", *spatial_dims).values.astype(np.float32)
    if len(arr.dims) == 2:
        values = arr.values.astype(np.float32)
        return np.repeat(values[None, ...], int(ds.sizes["time"]), axis=0)
    raise ValueError(f"Expected {var_name} to be a 2D grid or 3D time grid, got {arr.dims}")


def compute_spatial_gradients_k_per_m(
    field_2d: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    field = np.asarray(field_2d, dtype=np.float32)
    lat_rad = np.deg2rad(lat2d.astype(np.float64))
    lon_rad = np.deg2rad(lon2d.astype(np.float64))

    dlat = np.gradient(lat_rad, axis=0)
    dlon = np.gradient(lon_rad, axis=1)

    dy = EARTH_RADIUS_M * dlat
    dx = EARTH_RADIUS_M * np.cos(lat_rad) * dlon

    dTdy_num = np.gradient(field.astype(np.float64), axis=0)
    dTdx_num = np.gradient(field.astype(np.float64), axis=1)

    dTdy = np.divide(
        dTdy_num,
        dy,
        out=np.zeros_like(dTdy_num, dtype=np.float64),
        where=np.abs(dy) > 1e-12,
    )
    dTdx = np.divide(
        dTdx_num,
        dx,
        out=np.zeros_like(dTdx_num, dtype=np.float64),
        where=np.abs(dx) > 1e-12,
    )
    return dTdx.astype(np.float32), dTdy.astype(np.float32)


def compute_grid_spacing_m(lat2d: np.ndarray, lon2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat_rad = np.deg2rad(lat2d.astype(np.float64))
    lon_rad = np.deg2rad(lon2d.astype(np.float64))
    dlat = np.gradient(lat_rad, axis=0)
    dlon = np.gradient(lon_rad, axis=1)
    dy = EARTH_RADIUS_M * dlat
    dx = EARTH_RADIUS_M * np.cos(lat_rad) * dlon
    return dx.astype(np.float32), dy.astype(np.float32)


def compute_sample_count(time_size: int, history: int, lead: int) -> int:
    count = time_size - history - lead + 1
    if count <= 0:
        raise ValueError(
            f"Not enough timesteps for history={history}, lead={lead}, time_size={time_size}"
        )
    return count


def make_time_splits(time_size: int, history: int, lead: int) -> SplitIndices:
    sample_count = compute_sample_count(time_size=time_size, history=history, lead=lead)
    train_end = int(sample_count * 0.7)
    val_end = train_end + int(sample_count * 0.15)
    all_indices = np.arange(sample_count, dtype=np.int64)
    return SplitIndices(
        train=all_indices[:train_end],
        val=all_indices[train_end:val_end],
        test=all_indices[val_end:],
    )


def compute_train_stats(
    ds: xr.Dataset,
    dynamic_vars: list[str],
    static_vars: list[str],
    history: int,
    lead: int,
    train_indices: np.ndarray,
    residual_maps: np.ndarray | None = None,
) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    train_last_target = int(train_indices[-1] + history - 1 + lead)
    train_time_slice = slice(0, train_last_target + 1)
    ocean_mask = np.isfinite(ds["ocean_mask_static"].values)

    for var_name in dynamic_vars:
        values = load_time_grid_array(ds, var_name)[train_time_slice]
        selected = values[:, ocean_mask]
        mean = float(np.mean(selected))
        std = float(np.std(selected))
        stats[var_name] = (mean, std if std > 1e-6 else 1.0)

    if "ocean_mask_static" in static_vars:
        stats["ocean_mask_static"] = (0.0, 1.0)

    if "lat2d" in static_vars:
        lat2d, _ = build_lat_lon_2d(ds)
        stats["lat2d"] = (float(lat2d.mean()), float(lat2d.std()) or 1.0)

    if "lon2d" in static_vars:
        _, lon2d = build_lat_lon_2d(ds)
        stats["lon2d"] = (float(lon2d.mean()), float(lon2d.std()) or 1.0)

    target_values = ds["sst_mean"].isel(time=train_time_slice).values.astype(np.float32)
    selected_target = target_values[:, ocean_mask]
    target_mean = float(np.mean(selected_target))
    target_std = float(np.std(selected_target))
    stats["target"] = (target_mean, target_std if target_std > 1e-6 else 1.0)

    current_indices = train_indices + history - 1
    target_indices = current_indices + lead
    obs_tendency = (
        ds["sst_mean"].values.astype(np.float32)[target_indices]
        - ds["sst_mean"].values.astype(np.float32)[current_indices]
    ) / max(float(lead), 1.0)
    selected_tendency = obs_tendency[:, ocean_mask]
    selected_tendency = selected_tendency[np.isfinite(selected_tendency)]
    if selected_tendency.size:
        tendency_mse = float(np.mean(selected_tendency ** 2))
        tendency_rms = float(np.sqrt(tendency_mse))
    else:
        tendency_mse = 1.0
        tendency_rms = 1.0
    stats["observed_tendency_mse_kday2"] = (tendency_mse if tendency_mse > 1e-12 else 1e-12, 1.0)
    stats["observed_tendency_rms_kday"] = (tendency_rms if tendency_rms > 1e-6 else 1e-6, 1.0)

    if residual_maps is not None:
        train_last_input = int(train_indices[-1] + history)
        residual_values = residual_maps[:train_last_input].astype(np.float32)
        selected_residual = residual_values[:, ocean_mask]
        selected_residual = selected_residual[np.isfinite(selected_residual)]
        if selected_residual.size:
            residual_mean = float(np.mean(selected_residual))
            residual_std = float(np.std(selected_residual))
            stats["residual"] = (residual_mean, residual_std if residual_std > 1e-12 else 1.0)
        else:
            stats["residual"] = (0.0, 1.0)
    return stats


class JejuForecastDataset(Dataset):
    def __init__(
        self,
        ds: xr.Dataset,
        history: int,
        lead: int,
        indices: np.ndarray,
        dynamic_vars: list[str] | None = None,
        static_vars: list[str] | None = None,
        stats: dict[str, tuple[float, float]] | None = None,
        residual_maps: np.ndarray | None = None,
        forcing_mode: str = "perfect",
    ) -> None:
        self.ds = ds
        self.history = history
        self.lead = lead
        self.indices = indices.astype(np.int64)
        self.dynamic_vars = dynamic_vars or list(DEFAULT_DYNAMIC_VARS)
        self.static_vars = static_vars or list(DEFAULT_STATIC_VARS)
        self.stats = stats or {}
        self.residual_maps = residual_maps.astype(np.float32) if residual_maps is not None else None
        if forcing_mode not in {"perfect", "persist"}:
            raise ValueError(f"forcing_mode must be 'perfect' or 'persist', got {forcing_mode!r}")
        self.forcing_mode = forcing_mode

        self.ocean_mask = np.isfinite(ds["ocean_mask_static"].values).astype(np.float32)
        self.dynamic_arrays = {
            name: load_time_grid_array(ds, name) for name in self.dynamic_vars
        }
        # Improved-physics forcing from the GLORYS-derived dataset (optional; the
        # old dataset lacks these, so we fall back to legacy sources per-field).
        self.physics_arrays = {
            name: load_time_grid_array(ds, name)
            for name in (
                "qnet",
                "mlotst",
                "u_ml",
                "v_ml",
                "u_mld",
                "v_mld",
                "Td_glorys",
                "vadv_kday",
            )
            if name in ds.variables
        }
        self.target_array = ds["sst_mean"].values.astype(np.float32)
        lat2d, lon2d = build_lat_lon_2d(ds)
        self.static_arrays = {
            "ocean_mask_static": self.ocean_mask,
            "lat2d": lat2d.astype(np.float32),
            "lon2d": lon2d.astype(np.float32),
        }
        self.lat2d = lat2d.astype(np.float32)
        self.lon2d = lon2d.astype(np.float32)
        self.dx, self.dy = compute_grid_spacing_m(self.lat2d, self.lon2d)

    def __len__(self) -> int:
        return len(self.indices)

    def _normalize(self, values: np.ndarray, key: str) -> np.ndarray:
        mean, std = self.stats.get(key, (0.0, 1.0))
        normalized = (values - mean) / std
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    def _forcing_window(self, arr: np.ndarray, current_idx: int) -> np.ndarray:
        if self.forcing_mode == "persist":
            return np.repeat(arr[current_idx][None, ...], self.lead, axis=0)
        return arr[current_idx:current_idx + self.lead]

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        start_idx = int(self.indices[item])
        input_end = start_idx + self.history
        target_idx = start_idx + self.history - 1 + self.lead

        dynamic_seq = []
        for name in self.dynamic_vars:
            values = self.dynamic_arrays[name][start_idx:input_end]
            dynamic_seq.append(self._normalize(values, name))
        x_dynamic = np.stack(dynamic_seq, axis=1)

        static_seq = []
        for name in self.static_vars:
            static_map = self._normalize(self.static_arrays[name], name)
            static_seq.append(np.repeat(static_map[None, ...], self.history, axis=0))
        x_static = np.stack(static_seq, axis=1) if static_seq else None

        x = x_dynamic if x_static is None else np.concatenate([x_dynamic, x_static], axis=1)

        if self.residual_maps is not None:
            residual_seq = self._normalize(self.residual_maps[start_idx:input_end], "residual")
            f_seq = residual_seq[:, None, ...]
        else:
            f_seq = np.zeros((self.history, 1, self.ocean_mask.shape[0], self.ocean_mask.shape[1]), dtype=np.float32)

        target = self.target_array[target_idx][None, ...]
        target_norm = self._normalize(target, "target")
        mask = self.ocean_mask[None, ...]
        target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        target_norm = np.nan_to_num(target_norm, nan=0.0, posinf=0.0, neginf=0.0)

        current_idx = input_end - 1
        current_sst = np.nan_to_num(self.target_array[current_idx], nan=0.0, posinf=0.0, neginf=0.0)
        # physics forcing: prefer GLORYS-consistent fields (u_ml/v_ml/Td_glorys)
        # when the improved dataset provides them, else fall back to legacy.
        u_src = self.physics_arrays.get("u_ml", self.physics_arrays.get("u_mld"))
        v_src = self.physics_arrays.get("v_ml", self.physics_arrays.get("v_mld"))
        qnet_src = self.physics_arrays.get("qnet")
        hm_src = self.physics_arrays.get("mlotst")
        if u_src is None or v_src is None or qnet_src is None or hm_src is None:
            missing = [
                name
                for name, arr in (("u/v", u_src if u_src is not None and v_src is not None else None),
                                  ("qnet", qnet_src), ("mlotst", hm_src))
                if arr is None
            ]
            raise KeyError(f"Missing physics forcing arrays needed for the heat budget: {missing}")
        qnet = np.nan_to_num(qnet_src[current_idx], nan=0.0, posinf=0.0, neginf=0.0)
        hm = np.nan_to_num(hm_src[current_idx], nan=0.0, posinf=0.0, neginf=0.0)
        u = np.nan_to_num(u_src[current_idx], nan=0.0, posinf=0.0, neginf=0.0)
        v = np.nan_to_num(v_src[current_idx], nan=0.0, posinf=0.0, neginf=0.0)
        qnet_seq = np.nan_to_num(self._forcing_window(qnet_src, current_idx), nan=0.0, posinf=0.0, neginf=0.0)
        hm_seq = np.nan_to_num(self._forcing_window(hm_src, current_idx), nan=0.0, posinf=0.0, neginf=0.0)
        u_seq = np.nan_to_num(self._forcing_window(u_src, current_idx), nan=0.0, posinf=0.0, neginf=0.0)
        v_seq = np.nan_to_num(self._forcing_window(v_src, current_idx), nan=0.0, posinf=0.0, neginf=0.0)
        dTdx, dTdy = compute_spatial_gradients_k_per_m(current_sst, self.lat2d, self.lon2d)
        zeros = np.zeros_like(current_sst, dtype=np.float32)
        # Td = temp of water just below the ML base (GLORYS) if available, else Tm
        if "Td_glorys" in self.physics_arrays:
            Td_field = np.nan_to_num(self.physics_arrays["Td_glorys"][current_idx], nan=0.0, posinf=0.0, neginf=0.0)
        else:
            Td_field = current_sst

        batch = {
            "x": torch.from_numpy(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)).float(),
            "f_seq": torch.from_numpy(np.nan_to_num(f_seq, nan=0.0, posinf=0.0, neginf=0.0)).float(),
            "y": torch.from_numpy(target_norm).float(),
            "y_raw": torch.from_numpy(target).float(),
            "mask": torch.from_numpy(mask).float(),
            "qnet": torch.from_numpy(qnet[None, ...]).float(),
            "hm": torch.from_numpy(hm[None, ...]).float(),
            "u": torch.from_numpy(u[None, ...]).float(),
            "v": torch.from_numpy(v[None, ...]).float(),
            "qnet_seq": torch.from_numpy(qnet_seq[:, None, ...]).float(),
            "hm_seq": torch.from_numpy(hm_seq[:, None, ...]).float(),
            "u_seq": torch.from_numpy(u_seq[:, None, ...]).float(),
            "v_seq": torch.from_numpy(v_seq[:, None, ...]).float(),
            "Tm": torch.from_numpy(current_sst[None, ...]).float(),
            "Td": torch.from_numpy(Td_field[None, ...]).float(),
            "we": torch.from_numpy(zeros[None, ...]).float(),
            "Rres": torch.from_numpy(zeros[None, ...]).float(),
            "dTdx": torch.from_numpy(dTdx[None, ...]).float(),
            "dTdy": torch.from_numpy(dTdy[None, ...]).float(),
            "dx": torch.from_numpy(self.dx[None, ...]).float(),
            "dy": torch.from_numpy(self.dy[None, ...]).float(),
        }
        if "vadv_kday" in self.physics_arrays:
            vadv_seq = np.nan_to_num(self._forcing_window(self.physics_arrays["vadv_kday"], current_idx),
                                     nan=0.0, posinf=0.0, neginf=0.0)
            batch["vadv_seq"] = torch.from_numpy(vadv_seq[:, None, ...]).float()
        return batch
