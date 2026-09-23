"""Graph utilities for the Jeju-adapted HiLAM baseline.

The graph is intentionally deterministic and ocean-only.  Level 0 is the
native ocean grid.  Higher levels are simple geographic coarsenings of that
grid, which keeps the baseline reproducible without depending on external mesh
generation tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json

import numpy as np
import torch


DEFAULT_SW_BOUNDS = {
    "lat_min": 33.5,
    "lat_max": 35.0,
    "lon_min": 125.5,
    "lon_max": 127.5,
}


@dataclass(frozen=True)
class JejuHiLAMGraph:
    height: int
    width: int
    ocean_mask: np.ndarray
    grid_to_node: np.ndarray
    ocean_flat_indices: np.ndarray
    node_to_grid: np.ndarray
    sw_node_indices: np.ndarray
    level_node_features: list[torch.Tensor]
    level_node_rc: list[np.ndarray]
    level_bin_keys: list[list[tuple[int, int]]]
    same_edge_index: list[torch.Tensor]
    same_edge_attr: list[torch.Tensor]
    up_edge_index: list[torch.Tensor]
    up_edge_attr: list[torch.Tensor]
    down_edge_index: list[torch.Tensor]
    down_edge_attr: list[torch.Tensor]
    g2m_edge_index: torch.Tensor
    g2m_edge_attr: torch.Tensor
    m2g_edge_index: torch.Tensor
    m2g_edge_attr: torch.Tensor

    @property
    def num_levels(self) -> int:
        return len(self.level_node_features)

    @property
    def num_ocean_nodes(self) -> int:
        return int(self.ocean_flat_indices.size)

    @property
    def edge_feature_dim(self) -> int:
        return int(self.g2m_edge_attr.shape[-1])

    @property
    def static_feature_dim(self) -> int:
        return int(self.level_node_features[0].shape[-1])

    def summary(self) -> dict:
        return {
            "height": self.height,
            "width": self.width,
            "grid_cells": self.height * self.width,
            "ocean_nodes": self.num_ocean_nodes,
            "land_or_masked_cells": int(self.height * self.width - self.num_ocean_nodes),
            "sw_nodes": int(self.sw_node_indices.size),
            "levels": [
                {
                    "level": i,
                    "nodes": int(feat.shape[0]),
                    "same_directed_edges": int(self.same_edge_index[i].shape[1]),
                }
                for i, feat in enumerate(self.level_node_features)
            ],
            "cross_level_edges": [
                {
                    "from_level": i,
                    "to_level": i + 1,
                    "up_directed_edges": int(self.up_edge_index[i].shape[1]),
                    "down_directed_edges": int(self.down_edge_index[i].shape[1]),
                }
                for i in range(len(self.up_edge_index))
            ],
            "g2m_directed_edges": int(self.g2m_edge_index.shape[1]),
            "m2g_directed_edges": int(self.m2g_edge_index.shape[1]),
            "static_feature_dim": self.static_feature_dim,
            "edge_feature_dim": self.edge_feature_dim,
        }


def _as_2d_lat_lon(ds) -> tuple[np.ndarray, np.ndarray]:
    lat_name = "lat" if "lat" in ds else "latitude"
    lon_name = "lon" if "lon" in ds else "longitude"
    lat = np.asarray(ds[lat_name].values, dtype=np.float32)
    lon = np.asarray(ds[lon_name].values, dtype=np.float32)
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
        return lat2d.astype(np.float32), lon2d.astype(np.float32)
    if lat.ndim == 2 and lon.ndim == 2:
        return lat.astype(np.float32), lon.astype(np.float32)
    raise ValueError(f"Unsupported lat/lon dimensions: lat={lat.shape}, lon={lon.shape}")


def ocean_mask_from_dataset(ds) -> np.ndarray:
    """Return the static ocean mask used by the existing Jeju dataset."""

    if "ocean_mask_static" not in ds:
        raise KeyError("Dataset is missing ocean_mask_static")
    mask = np.asarray(ds["ocean_mask_static"].values)
    if mask.ndim != 2:
        raise ValueError(f"ocean_mask_static must be 2-D, got {mask.shape}")
    return np.isfinite(mask) & (mask > 0.5)


def assert_expected_jeju_grid(ocean_mask: np.ndarray, sw_node_count: int) -> None:
    h, w = ocean_mask.shape
    ocean_nodes = int(ocean_mask.sum())
    land_nodes = int(ocean_mask.size - ocean_nodes)
    assert (h, w) == (29, 33), f"Expected grid 29x33, got {h}x{w}"
    assert h * w == 957, f"Expected 957 grid cells, got {h * w}"
    assert ocean_nodes == 661, f"Expected 661 ocean nodes, got {ocean_nodes}"
    assert land_nodes == 296, f"Expected 296 land/masked cells, got {land_nodes}"
    assert sw_node_count == 36, f"Expected 36 SW ocean nodes, got {sw_node_count}"


def _build_mapping(ocean_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = ocean_mask.shape
    ocean_flat = np.flatnonzero(ocean_mask.reshape(-1)).astype(np.int64)
    node_to_grid = np.column_stack(np.unravel_index(ocean_flat, (h, w))).astype(np.int64)
    grid_to_node = np.full((h, w), -1, dtype=np.int64)
    grid_to_node.reshape(-1)[ocean_flat] = np.arange(ocean_flat.size, dtype=np.int64)
    return grid_to_node, ocean_flat, node_to_grid


def _normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    denom = float(np.nanmax(values) - np.nanmin(values))
    if denom == 0:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - np.nanmin(values)) / denom).astype(np.float32)


def _node_features(
    rc: np.ndarray,
    lat_values: np.ndarray,
    lon_values: np.ndarray,
    height: int,
    width: int,
    level_index: int,
    num_levels: int,
) -> torch.Tensor:
    row_norm = (rc[:, 0] / max(height - 1, 1)).astype(np.float32)
    col_norm = (rc[:, 1] / max(width - 1, 1)).astype(np.float32)
    lat_norm = _normalize(lat_values)
    lon_norm = _normalize(lon_values)
    level_norm = np.full_like(row_norm, level_index / max(num_levels - 1, 1), dtype=np.float32)
    features = np.stack([row_norm, col_norm, lat_norm, lon_norm, level_norm], axis=-1)
    return torch.from_numpy(features.astype(np.float32))


def _edge_features(
    src_rc: np.ndarray,
    dst_rc: np.ndarray,
    src_latlon: np.ndarray,
    dst_latlon: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    drow = (dst_rc[:, 0] - src_rc[:, 0]) / max(height - 1, 1)
    dcol = (dst_rc[:, 1] - src_rc[:, 1]) / max(width - 1, 1)
    dlat = dst_latlon[:, 0] - src_latlon[:, 0]
    dlon = dst_latlon[:, 1] - src_latlon[:, 1]
    lat_scale = max(float(np.ptp(np.concatenate([src_latlon[:, 0], dst_latlon[:, 0]]))), 1e-6)
    lon_scale = max(float(np.ptp(np.concatenate([src_latlon[:, 1], dst_latlon[:, 1]]))), 1e-6)
    dlat = dlat / lat_scale
    dlon = dlon / lon_scale
    dist = np.sqrt(drow**2 + dcol**2).astype(np.float32)
    return np.stack([dist, drow, dcol, dlat, dlon], axis=-1).astype(np.float32)


def _same_edges_from_grid_mapping(
    grid_to_node: np.ndarray,
    node_to_grid: np.ndarray,
    latlon: np.ndarray,
    height: int,
    width: int,
    use_diagonals: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if use_diagonals:
        deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    src: list[int] = []
    dst: list[int] = []
    for node_idx, (row, col) in enumerate(node_to_grid):
        for dr, dc in deltas:
            nr = int(row + dr)
            nc = int(col + dc)
            if 0 <= nr < height and 0 <= nc < width:
                neighbor = int(grid_to_node[nr, nc])
                if neighbor >= 0:
                    src.append(node_idx)
                    dst.append(neighbor)

    edge_index_np = np.asarray([src, dst], dtype=np.int64)
    src_rc = node_to_grid[edge_index_np[0]].astype(np.float32)
    dst_rc = node_to_grid[edge_index_np[1]].astype(np.float32)
    edge_attr_np = _edge_features(
        src_rc,
        dst_rc,
        latlon[edge_index_np[0]],
        latlon[edge_index_np[1]],
        height,
        width,
    )
    return torch.from_numpy(edge_index_np), torch.from_numpy(edge_attr_np)


def _coarsen_from_level(
    fine_rc: np.ndarray,
    fine_latlon: np.ndarray,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], np.ndarray]:
    keys: list[tuple[int, int]] = [
        (int(row // stride), int(col // stride)) for row, col in fine_rc
    ]
    unique_keys = sorted(set(keys))
    key_to_node = {key: idx for idx, key in enumerate(unique_keys)}
    parent = np.asarray([key_to_node[key] for key in keys], dtype=np.int64)

    coarse_rc = np.zeros((len(unique_keys), 2), dtype=np.float32)
    coarse_latlon = np.zeros((len(unique_keys), 2), dtype=np.float32)
    counts = np.zeros(len(unique_keys), dtype=np.float32)
    for fine_idx, coarse_idx in enumerate(parent):
        coarse_rc[coarse_idx] += fine_rc[fine_idx]
        coarse_latlon[coarse_idx] += fine_latlon[fine_idx]
        counts[coarse_idx] += 1.0
    coarse_rc /= counts[:, None]
    coarse_latlon /= counts[:, None]

    return coarse_rc, coarse_latlon, unique_keys, parent


def _coarsen_from_keys(
    fine_rc: np.ndarray,
    fine_latlon: np.ndarray,
    fine_keys: list[tuple[int, int]],
    key_divisor: int = 2,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], np.ndarray]:
    keys = [(key[0] // key_divisor, key[1] // key_divisor) for key in fine_keys]
    unique_keys = sorted(set(keys))
    key_to_node = {key: idx for idx, key in enumerate(unique_keys)}
    parent = np.asarray([key_to_node[key] for key in keys], dtype=np.int64)

    coarse_rc = np.zeros((len(unique_keys), 2), dtype=np.float32)
    coarse_latlon = np.zeros((len(unique_keys), 2), dtype=np.float32)
    counts = np.zeros(len(unique_keys), dtype=np.float32)
    for fine_idx, coarse_idx in enumerate(parent):
        coarse_rc[coarse_idx] += fine_rc[fine_idx]
        coarse_latlon[coarse_idx] += fine_latlon[fine_idx]
        counts[coarse_idx] += 1.0
    coarse_rc /= counts[:, None]
    coarse_latlon /= counts[:, None]
    return coarse_rc, coarse_latlon, unique_keys, parent


def _same_edges_from_bin_keys(
    keys: list[tuple[int, int]],
    rc: np.ndarray,
    latlon: np.ndarray,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_to_node = {key: idx for idx, key in enumerate(keys)}
    src: list[int] = []
    dst: list[int] = []
    for node_idx, (bi, bj) in enumerate(keys):
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                neighbor = key_to_node.get((bi + di, bj + dj))
                if neighbor is not None:
                    src.append(node_idx)
                    dst.append(neighbor)

    edge_index_np = np.asarray([src, dst], dtype=np.int64)
    edge_attr_np = _edge_features(
        rc[edge_index_np[0]],
        rc[edge_index_np[1]],
        latlon[edge_index_np[0]],
        latlon[edge_index_np[1]],
        height,
        width,
    )
    return torch.from_numpy(edge_index_np), torch.from_numpy(edge_attr_np)


def _cross_edges(
    parent: np.ndarray,
    fine_rc: np.ndarray,
    coarse_rc: np.ndarray,
    fine_latlon: np.ndarray,
    coarse_latlon: np.ndarray,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fine_ids = np.arange(parent.size, dtype=np.int64)
    coarse_ids = parent.astype(np.int64)

    up_index = np.stack([fine_ids, coarse_ids], axis=0)
    up_attr = _edge_features(
        fine_rc[fine_ids],
        coarse_rc[coarse_ids],
        fine_latlon[fine_ids],
        coarse_latlon[coarse_ids],
        height,
        width,
    )
    down_index = np.stack([coarse_ids, fine_ids], axis=0)
    down_attr = _edge_features(
        coarse_rc[coarse_ids],
        fine_rc[fine_ids],
        coarse_latlon[coarse_ids],
        fine_latlon[fine_ids],
        height,
        width,
    )
    return (
        torch.from_numpy(up_index),
        torch.from_numpy(up_attr),
        torch.from_numpy(down_index),
        torch.from_numpy(down_attr),
    )


def _identity_grid_mesh_edges(num_nodes: int, edge_feature_dim: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    ids = np.arange(num_nodes, dtype=np.int64)
    edge_index = torch.from_numpy(np.stack([ids, ids], axis=0))
    edge_attr = torch.zeros((num_nodes, edge_feature_dim), dtype=torch.float32)
    return edge_index, edge_attr


def sw_mask_from_lat_lon(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    ocean_mask: np.ndarray,
    bounds: dict | None = None,
) -> np.ndarray:
    bounds = DEFAULT_SW_BOUNDS if bounds is None else bounds
    return (
        ocean_mask
        & (lat2d >= bounds["lat_min"])
        & (lat2d <= bounds["lat_max"])
        & (lon2d >= bounds["lon_min"])
        & (lon2d <= bounds["lon_max"])
    )


def build_jeju_hilam_graph(ds, levels: int = 3, sw_bounds: dict | None = None) -> JejuHiLAMGraph:
    if levels != 3:
        raise ValueError("This baseline currently expects exactly 3 hierarchy levels")

    ocean_mask = ocean_mask_from_dataset(ds)
    height, width = ocean_mask.shape
    lat2d, lon2d = _as_2d_lat_lon(ds)
    grid_to_node, ocean_flat_indices, node_to_grid = _build_mapping(ocean_mask)

    sw_mask = sw_mask_from_lat_lon(lat2d, lon2d, ocean_mask, sw_bounds)
    sw_node_indices = grid_to_node[sw_mask].astype(np.int64)
    assert_expected_jeju_grid(ocean_mask, int(sw_node_indices.size))

    level_rc: list[np.ndarray] = [node_to_grid.astype(np.float32)]
    level_latlon: list[np.ndarray] = [
        np.column_stack(
            [
                lat2d.reshape(-1)[ocean_flat_indices],
                lon2d.reshape(-1)[ocean_flat_indices],
            ]
        ).astype(np.float32)
    ]
    level_keys: list[list[tuple[int, int]]] = [
        [(int(row), int(col)) for row, col in node_to_grid]
    ]

    # Level 1 uses 2x2 native grid bins. Level 2 groups level-1 bins in 2x2
    # blocks, equivalent to a 4x4 native-grid coarsening.
    rc1, latlon1, keys1, parent0 = _coarsen_from_level(level_rc[0], level_latlon[0], stride=2)
    rc2, latlon2, keys2, parent1 = _coarsen_from_keys(rc1, latlon1, keys1, key_divisor=2)
    level_rc.extend([rc1, rc2])
    level_latlon.extend([latlon1, latlon2])
    level_keys.extend([keys1, keys2])

    level_features = [
        _node_features(level_rc[i], level_latlon[i][:, 0], level_latlon[i][:, 1], height, width, i, levels)
        for i in range(levels)
    ]

    same_edge_index: list[torch.Tensor] = []
    same_edge_attr: list[torch.Tensor] = []
    edge_i, edge_a = _same_edges_from_grid_mapping(
        grid_to_node, node_to_grid, level_latlon[0], height, width
    )
    same_edge_index.append(edge_i)
    same_edge_attr.append(edge_a)
    for i in (1, 2):
        edge_i, edge_a = _same_edges_from_bin_keys(level_keys[i], level_rc[i], level_latlon[i], height, width)
        same_edge_index.append(edge_i)
        same_edge_attr.append(edge_a)

    up_edge_index: list[torch.Tensor] = []
    up_edge_attr: list[torch.Tensor] = []
    down_edge_index: list[torch.Tensor] = []
    down_edge_attr: list[torch.Tensor] = []
    for parent, fine_level in ((parent0, 0), (parent1, 1)):
        up_i, up_a, down_i, down_a = _cross_edges(
            parent,
            level_rc[fine_level],
            level_rc[fine_level + 1],
            level_latlon[fine_level],
            level_latlon[fine_level + 1],
            height,
            width,
        )
        up_edge_index.append(up_i)
        up_edge_attr.append(up_a)
        down_edge_index.append(down_i)
        down_edge_attr.append(down_a)

    g2m_edge_index, g2m_edge_attr = _identity_grid_mesh_edges(ocean_flat_indices.size)
    m2g_edge_index, m2g_edge_attr = _identity_grid_mesh_edges(ocean_flat_indices.size)

    return JejuHiLAMGraph(
        height=height,
        width=width,
        ocean_mask=ocean_mask,
        grid_to_node=grid_to_node,
        ocean_flat_indices=ocean_flat_indices,
        node_to_grid=node_to_grid,
        sw_node_indices=sw_node_indices,
        level_node_features=level_features,
        level_node_rc=level_rc,
        level_bin_keys=level_keys,
        same_edge_index=same_edge_index,
        same_edge_attr=same_edge_attr,
        up_edge_index=up_edge_index,
        up_edge_attr=up_edge_attr,
        down_edge_index=down_edge_index,
        down_edge_attr=down_edge_attr,
        g2m_edge_index=g2m_edge_index,
        g2m_edge_attr=g2m_edge_attr,
        m2g_edge_index=m2g_edge_index,
        m2g_edge_attr=m2g_edge_attr,
    )


def nodes_to_grid(
    node_values: np.ndarray,
    graph: JejuHiLAMGraph,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Convert ocean node values to full grid arrays.

    Accepts either [N] or [B, N].  Returns [H, W] or [B, 1, H, W].
    """

    arr = np.asarray(node_values)
    if arr.ndim == 1:
        out = np.full((graph.height * graph.width,), fill_value, dtype=arr.dtype)
        out[graph.ocean_flat_indices] = arr
        return out.reshape(graph.height, graph.width)
    if arr.ndim == 2:
        out = np.full((arr.shape[0], graph.height * graph.width), fill_value, dtype=arr.dtype)
        out[:, graph.ocean_flat_indices] = arr
        return out.reshape(arr.shape[0], 1, graph.height, graph.width)
    raise ValueError(f"Expected node values with 1 or 2 dims, got {arr.shape}")


def save_graph_artifacts(graph: JejuHiLAMGraph, out_dir: str | Path) -> tuple[Path, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    npz_path = out_path / "jeju_hilam_graph.npz"
    json_path = out_path / "jeju_hilam_graph_summary.json"

    arrays: dict[str, np.ndarray] = {
        "ocean_mask": graph.ocean_mask.astype(np.uint8),
        "grid_to_node": graph.grid_to_node,
        "ocean_flat_indices": graph.ocean_flat_indices,
        "node_to_grid": graph.node_to_grid,
        "sw_node_indices": graph.sw_node_indices,
        "g2m_edge_index": graph.g2m_edge_index.numpy(),
        "g2m_edge_attr": graph.g2m_edge_attr.numpy(),
        "m2g_edge_index": graph.m2g_edge_index.numpy(),
        "m2g_edge_attr": graph.m2g_edge_attr.numpy(),
    }
    for i in range(graph.num_levels):
        arrays[f"level_{i}_node_features"] = graph.level_node_features[i].numpy()
        arrays[f"level_{i}_node_rc"] = graph.level_node_rc[i]
        arrays[f"level_{i}_same_edge_index"] = graph.same_edge_index[i].numpy()
        arrays[f"level_{i}_same_edge_attr"] = graph.same_edge_attr[i].numpy()
    for i in range(graph.num_levels - 1):
        arrays[f"level_{i}_up_edge_index"] = graph.up_edge_index[i].numpy()
        arrays[f"level_{i}_up_edge_attr"] = graph.up_edge_attr[i].numpy()
        arrays[f"level_{i}_down_edge_index"] = graph.down_edge_index[i].numpy()
        arrays[f"level_{i}_down_edge_attr"] = graph.down_edge_attr[i].numpy()

    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(graph.summary(), indent=2), encoding="utf-8")
    return npz_path, json_path


def maybe_plot_graph(graph: JejuHiLAMGraph, out_path: str | Path) -> Path | None:
    """Save a lightweight hierarchy plot if matplotlib is available."""

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=160)
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    sizes = [5, 16, 36]
    labels = ["G1 ocean grid", "G2 coarse", "G3 coarse"]
    for i, rc in enumerate(graph.level_node_rc):
        ax.scatter(rc[:, 1], rc[:, 0], s=sizes[i], c=colors[i], alpha=0.78, label=labels[i])
    sw_rc = graph.node_to_grid[graph.sw_node_indices]
    ax.scatter(sw_rc[:, 1], sw_rc[:, 0], s=20, facecolors="none", edgecolors="#111111", linewidths=0.8)
    ax.set_xlim(-1, graph.width)
    ax.set_ylim(graph.height, -1)
    ax.set_xlabel("lon/grid column")
    ax.set_ylabel("lat/grid row")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("JejuHiLAM ocean-only hierarchy")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def iter_graph_tensors(graph: JejuHiLAMGraph) -> Iterable[torch.Tensor]:
    yield from graph.level_node_features
    yield from graph.same_edge_index
    yield from graph.same_edge_attr
    yield from graph.up_edge_index
    yield from graph.up_edge_attr
    yield from graph.down_edge_index
    yield from graph.down_edge_attr
    yield graph.g2m_edge_index
    yield graph.g2m_edge_attr
    yield graph.m2g_edge_index
    yield graph.m2g_edge_attr
