"""Dataset wrapper for the Jeju-adapted HiLAM baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from .datasets import JejuForecastDataset, compute_train_stats, make_time_splits
from .jeju_hilam_graph import JejuHiLAMGraph, build_jeju_hilam_graph


BASE_DYNAMIC_VARS = [
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

STATIC_VARS_FOR_BASE_DATASET = ["ocean_mask_static", "lat2d", "lon2d"]
RESIDUAL_CHANNEL_NAME = "f_theta_residual_map"


@dataclass(frozen=True)
class JejuHiLAMDataBundle:
    ds: xr.Dataset
    graph: JejuHiLAMGraph
    train: "JejuHiLAMDataset"
    val: "JejuHiLAMDataset"
    test: "JejuHiLAMDataset"
    stats: dict
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray


class JejuHiLAMDataset(Dataset):
    """Ocean-node view of the existing gridded Jeju forecast dataset.

    The wrapped base dataset still handles splits, normalization and lead-day
    indexing.  This class only converts [T, C, H, W] grids to [N, T, C_dyn]
    ocean-node tensors for the graph model.
    """

    def __init__(
        self,
        base: JejuForecastDataset,
        graph: JejuHiLAMGraph,
        include_residual_feature: bool = True,
    ) -> None:
        self.base = base
        self.graph = graph
        self.include_residual_feature = bool(include_residual_feature)
        self.dynamic_channel_indices = list(range(len(BASE_DYNAMIC_VARS)))
        self.input_feature_names = list(BASE_DYNAMIC_VARS)
        if self.include_residual_feature:
            self.dynamic_channel_indices.append(len(BASE_DYNAMIC_VARS) + len(STATIC_VARS_FOR_BASE_DATASET))
            self.input_feature_names.append(RESIDUAL_CHANNEL_NAME)

        self._ocean_flat = torch.as_tensor(graph.ocean_flat_indices, dtype=torch.long)
        self.target_mean = float(base.stats["target"][0])
        self.target_std = float(base.stats["target"][1])
        self.history = int(base.history)
        self.lead_days = int(base.lead)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.base[idx]
        x_aug = torch.cat([sample["x"], sample["f_seq"]], dim=1)
        x_dyn = x_aug[:, self.dynamic_channel_indices, :, :]
        time_steps, channels, height, width = x_dyn.shape
        x_nodes = x_dyn.reshape(time_steps, channels, height * width)[:, :, self._ocean_flat]
        x_nodes = x_nodes.permute(2, 0, 1).contiguous()

        y_nodes = sample["y"].reshape(-1)[self._ocean_flat].contiguous()
        y_raw_nodes = sample["y_raw"].reshape(-1)[self._ocean_flat].contiguous()
        tm_raw_nodes = sample["Tm"].reshape(-1)[self._ocean_flat].contiguous()
        tm_norm_nodes = ((tm_raw_nodes - self.target_mean) / self.target_std).contiguous()

        start_index = int(self.base.indices[idx])
        current_index = start_index + self.history - 1
        target_index = current_index + self.lead_days

        out = {
            "x_nodes": x_nodes.float(),
            "y_nodes": y_nodes.float(),
            "y_raw_nodes": y_raw_nodes.float(),
            "tm_norm_nodes": tm_norm_nodes.float(),
            "tm_raw_nodes": tm_raw_nodes.float(),
            "y_raw": sample["y_raw"].float(),
            "mask": sample["mask"].float(),
            "Tm": sample["Tm"].float(),
            "y_raw_grid": sample["y_raw"].float(),
            "mask_grid": sample["mask"].float(),
            "start_index": torch.tensor(start_index, dtype=torch.long),
            "current_index": torch.tensor(current_index, dtype=torch.long),
            "target_index": torch.tensor(target_index, dtype=torch.long),
        }
        for key in (
            "qnet_seq",
            "hm_seq",
            "u_seq",
            "v_seq",
            "vadv_seq",
            "dx",
            "dy",
            "qnet",
            "hm",
            "u",
            "v",
            "Td",
            "dTdx",
            "dTdy",
        ):
            if key in sample:
                out[key] = sample[key].float()
        return out


def _load_residual_map(path: str | Path | None) -> np.ndarray | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    arr = np.load(path)
    if arr.shape[-2:] != (29, 33):
        raise ValueError(f"Expected residual map trailing shape (29, 33), got {arr.shape}")
    return arr.astype(np.float32)


def make_jeju_hilam_bundle(
    data_path: str | Path,
    residual_map_path: str | Path | None,
    history: int,
    lead_days: int,
    include_residual_feature: bool = True,
) -> JejuHiLAMDataBundle:
    ds = xr.open_dataset(data_path)
    graph = build_jeju_hilam_graph(ds, levels=3)
    residual_map = _load_residual_map(residual_map_path) if include_residual_feature else None

    splits = make_time_splits(time_size=int(ds.sizes["time"]), history=history, lead=lead_days)
    train_idx, val_idx, test_idx = splits.train, splits.val, splits.test
    stats = compute_train_stats(
        ds,
        dynamic_vars=BASE_DYNAMIC_VARS,
        static_vars=STATIC_VARS_FOR_BASE_DATASET,
        history=history,
        lead=lead_days,
        train_indices=train_idx,
        residual_maps=residual_map,
    )

    if include_residual_feature and residual_map is None:
        raise FileNotFoundError(
            "include_residual_feature=True but the residual map was not found. "
            f"Checked: {residual_map_path}"
        )

    dataset_kwargs = {
        "ds": ds,
        "indices": None,
        "stats": stats,
        "history": history,
        "lead": lead_days,
        "dynamic_vars": BASE_DYNAMIC_VARS,
        "static_vars": STATIC_VARS_FOR_BASE_DATASET,
        "residual_maps": residual_map,
    }

    train_base = JejuForecastDataset(**{**dataset_kwargs, "indices": train_idx})
    val_base = JejuForecastDataset(**{**dataset_kwargs, "indices": val_idx})
    test_base = JejuForecastDataset(**{**dataset_kwargs, "indices": test_idx})

    return JejuHiLAMDataBundle(
        ds=ds,
        graph=graph,
        train=JejuHiLAMDataset(train_base, graph, include_residual_feature=include_residual_feature),
        val=JejuHiLAMDataset(val_base, graph, include_residual_feature=include_residual_feature),
        test=JejuHiLAMDataset(test_base, graph, include_residual_feature=include_residual_feature),
        stats=stats,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
    )
