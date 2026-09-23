"""Pure PyTorch Jeju-adapted HiLAM baseline.

This module mirrors the encode-process-decode shape of Hi-LAM/Neural-LAM while
keeping the implementation local and dependency-light for the Jeju MHW project.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from ..jeju_hilam_graph import JejuHiLAMGraph


class BufferList(nn.Module):
    def __init__(self, tensors: list[torch.Tensor]) -> None:
        super().__init__()
        for idx, tensor in enumerate(tensors):
            self.register_buffer(str(idx), tensor)

    def __len__(self) -> int:
        return len(self._buffers)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._buffers[str(idx)]

    def as_list(self) -> list[torch.Tensor]:
        return [self[i] for i in range(len(self))]


def make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    hidden_layers: int = 1,
    layer_norm: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(hidden_layers):
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.SiLU())
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    if layer_norm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


class InteractionLayer(nn.Module):
    """Residual message passing layer inspired by Neural-LAM InteractionNet."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        hidden_layers: int = 1,
        update_edges: bool = True,
        aggregation: str = "sum",
    ) -> None:
        super().__init__()
        if aggregation not in {"sum", "mean"}:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        self.update_edges = update_edges
        self.aggregation = aggregation
        self.edge_encoder = make_mlp(edge_dim, hidden_dim, hidden_dim, hidden_layers, layer_norm=True)
        self.edge_mlp = make_mlp(hidden_dim * 3, hidden_dim, hidden_dim, hidden_layers, layer_norm=True)
        self.node_mlp = make_mlp(hidden_dim * 2, hidden_dim, hidden_dim, hidden_layers, layer_norm=True)

    def forward(
        self,
        sender_nodes: torch.Tensor,
        receiver_nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src = edge_index[0].long()
        dst = edge_index[1].long()

        if edge_state is None:
            edge_state = self.edge_encoder(edge_attr).unsqueeze(0).expand(sender_nodes.shape[0], -1, -1)

        sender = sender_nodes[:, src, :]
        receiver = receiver_nodes[:, dst, :]
        edge_delta = self.edge_mlp(torch.cat([edge_state, sender, receiver], dim=-1))
        new_edge = edge_state + edge_delta if self.update_edges else edge_state

        aggr = receiver_nodes.new_zeros(receiver_nodes.shape)
        aggr.index_add_(1, dst, new_edge)
        if self.aggregation == "mean":
            degree = torch.bincount(dst, minlength=receiver_nodes.shape[1]).to(receiver_nodes.device)
            degree = degree.clamp_min(1).to(receiver_nodes.dtype).view(1, -1, 1)
            aggr = aggr / degree

        node_delta = self.node_mlp(torch.cat([receiver_nodes, aggr], dim=-1))
        return receiver_nodes + node_delta, new_edge


@dataclass(frozen=True)
class JejuHiLAMConfig:
    input_dim: int
    hidden_dim: int = 256
    temporal_hidden_dim: int = 256
    processor_layers: int = 3
    mlp_hidden_layers: int = 1
    prediction_mode: str = "direct"
    physics_heads: bool = False


class JejuHiLAM(nn.Module):
    """SeaCast-inspired hierarchical GNN for direct Jeju SST forecasting."""

    def __init__(
        self,
        graph: JejuHiLAMGraph,
        input_dim: int,
        hidden_dim: int = 256,
        temporal_hidden_dim: int = 256,
        processor_layers: int = 3,
        mlp_hidden_layers: int = 1,
        prediction_mode: str = "direct",
        physics_heads: bool = False,
        we_nonneg: bool = True,
        we_scale: float = 1.0e-5,
        rres_scale: float = 1.0e-4,
        we_softplus_shift: float = 1.0,
    ) -> None:
        super().__init__()
        if prediction_mode not in {"direct", "residual"}:
            raise ValueError(f"prediction_mode must be 'direct' or 'residual', got {prediction_mode!r}")
        if graph.num_levels != 3:
            raise ValueError("JejuHiLAM expects a 3-level graph")

        self.config = JejuHiLAMConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            temporal_hidden_dim=temporal_hidden_dim,
            processor_layers=processor_layers,
            mlp_hidden_layers=mlp_hidden_layers,
            prediction_mode=prediction_mode,
            physics_heads=physics_heads,
        )
        self.prediction_mode = prediction_mode
        self.hidden_dim = hidden_dim
        self.physics_heads = bool(physics_heads)
        self.we_nonneg = bool(we_nonneg)
        self.we_scale = float(we_scale)
        self.rres_scale = float(rres_scale)
        self.we_softplus_shift = float(we_softplus_shift)

        self.level_node_features = BufferList(graph.level_node_features)
        self.same_edge_index = BufferList(graph.same_edge_index)
        self.same_edge_attr = BufferList(graph.same_edge_attr)
        self.up_edge_index = BufferList(graph.up_edge_index)
        self.up_edge_attr = BufferList(graph.up_edge_attr)
        self.down_edge_index = BufferList(graph.down_edge_index)
        self.down_edge_attr = BufferList(graph.down_edge_attr)
        self.register_buffer("g2m_edge_index", graph.g2m_edge_index)
        self.register_buffer("g2m_edge_attr", graph.g2m_edge_attr)
        self.register_buffer("m2g_edge_index", graph.m2g_edge_index)
        self.register_buffer("m2g_edge_attr", graph.m2g_edge_attr)

        self.temporal_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=temporal_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.temporal_projection = make_mlp(
            temporal_hidden_dim, hidden_dim, hidden_dim, hidden_layers=mlp_hidden_layers, layer_norm=True
        )
        self.static_embedders = nn.ModuleList(
            [
                make_mlp(
                    graph.static_feature_dim,
                    hidden_dim,
                    hidden_dim,
                    hidden_layers=mlp_hidden_layers,
                    layer_norm=True,
                )
                for _ in range(graph.num_levels)
            ]
        )

        edge_dim = graph.edge_feature_dim
        self.g2m = InteractionLayer(hidden_dim, edge_dim, mlp_hidden_layers, update_edges=False)
        self.init_up = nn.ModuleList(
            [InteractionLayer(hidden_dim, edge_dim, mlp_hidden_layers, update_edges=False) for _ in range(2)]
        )

        self.same_processors = nn.ModuleList()
        self.down_processors = nn.ModuleList()
        self.up_processors = nn.ModuleList()
        for _ in range(processor_layers):
            self.same_processors.append(
                nn.ModuleList(
                    [InteractionLayer(hidden_dim, edge_dim, mlp_hidden_layers, update_edges=True) for _ in range(3)]
                )
            )
            self.down_processors.append(
                nn.ModuleList(
                    [InteractionLayer(hidden_dim, edge_dim, mlp_hidden_layers, update_edges=True) for _ in range(2)]
                )
            )
            self.up_processors.append(
                nn.ModuleList(
                    [InteractionLayer(hidden_dim, edge_dim, mlp_hidden_layers, update_edges=True) for _ in range(2)]
                )
            )

        self.read_down = nn.ModuleList(
            [InteractionLayer(hidden_dim, edge_dim, mlp_hidden_layers, update_edges=False) for _ in range(2)]
        )
        self.m2g = InteractionLayer(hidden_dim, edge_dim, mlp_hidden_layers, update_edges=False)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        if self.physics_heads:
            self.we_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.rres_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

    def _embed_static(self, batch_size: int) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for i, embedder in enumerate(self.static_embedders):
            emb = embedder(self.level_node_features[i].float())
            out.append(emb.unsqueeze(0).expand(batch_size, -1, -1))
        return out

    def _encode_temporal(self, x_nodes: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, time_steps, channels = x_nodes.shape
        flat = x_nodes.reshape(batch_size * num_nodes, time_steps, channels)
        _, hidden = self.temporal_encoder(flat)
        encoded = hidden[-1].reshape(batch_size, num_nodes, -1)
        return self.temporal_projection(encoded)

    def _processor_step(self, levels: list[torch.Tensor], layer_idx: int) -> list[torch.Tensor]:
        same = self.same_processors[layer_idx]
        down = self.down_processors[layer_idx]
        up = self.up_processors[layer_idx]

        levels[2], edge_22 = same[2](
            levels[2], levels[2], self.same_edge_index[2], self.same_edge_attr[2]
        )
        levels[1], edge_21 = down[1](
            levels[2], levels[1], self.down_edge_index[1], self.down_edge_attr[1]
        )
        levels[1], edge_11 = same[1](
            levels[1], levels[1], self.same_edge_index[1], self.same_edge_attr[1]
        )
        levels[0], edge_10 = down[0](
            levels[1], levels[0], self.down_edge_index[0], self.down_edge_attr[0]
        )
        levels[0], edge_00 = same[0](
            levels[0], levels[0], self.same_edge_index[0], self.same_edge_attr[0]
        )
        levels[1], edge_01 = up[0](
            levels[0], levels[1], self.up_edge_index[0], self.up_edge_attr[0]
        )
        levels[1], edge_11_b = same[1](
            levels[1], levels[1], self.same_edge_index[1], self.same_edge_attr[1], edge_11
        )
        levels[2], edge_12 = up[1](
            levels[1], levels[2], self.up_edge_index[1], self.up_edge_attr[1]
        )
        levels[2], _ = same[2](
            levels[2], levels[2], self.same_edge_index[2], self.same_edge_attr[2], edge_22
        )
        return levels

    def forward(
        self,
        x_nodes: torch.Tensor,
        base_nodes: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Run direct lead-specific forecasting.

        Args:
            x_nodes: [B, ocean_nodes, history, input_dim] normalized inputs.
            base_nodes: Optional [B, ocean_nodes] current SST in target-normalized
                units. Required only for prediction_mode='residual'.
        """

        batch_size = x_nodes.shape[0]
        grid_rep = self._encode_temporal(x_nodes)
        static_levels = self._embed_static(batch_size)

        mesh0, _ = self.g2m(
            grid_rep,
            static_levels[0],
            self.g2m_edge_index,
            self.g2m_edge_attr,
        )
        levels = [mesh0, static_levels[1], static_levels[2]]

        levels[1], _ = self.init_up[0](
            levels[0], levels[1], self.up_edge_index[0], self.up_edge_attr[0]
        )
        levels[2], _ = self.init_up[1](
            levels[1], levels[2], self.up_edge_index[1], self.up_edge_attr[1]
        )

        for layer_idx in range(self.config.processor_layers):
            levels = self._processor_step(levels, layer_idx)

        levels[1], _ = self.read_down[1](
            levels[2], levels[1], self.down_edge_index[1], self.down_edge_attr[1]
        )
        levels[0], _ = self.read_down[0](
            levels[1], levels[0], self.down_edge_index[0], self.down_edge_attr[0]
        )
        grid_out, _ = self.m2g(levels[0], grid_rep, self.m2g_edge_index, self.m2g_edge_attr)
        delta_or_pred = self.output_head(grid_out).squeeze(-1)

        if self.prediction_mode == "residual":
            if base_nodes is None:
                raise ValueError("base_nodes is required when prediction_mode='residual'")
            pred = base_nodes + delta_or_pred
        else:
            pred = delta_or_pred

        if not return_aux:
            return pred
        if not self.physics_heads:
            raise ValueError("return_aux=True requires physics_heads=True")
        if self.we_nonneg:
            we = self.we_scale * F.softplus(self.we_head(grid_out).squeeze(-1) - self.we_softplus_shift)
        else:
            we = self.we_scale * torch.tanh(self.we_head(grid_out).squeeze(-1))
        rres = self.rres_scale * torch.tanh(self.rres_head(grid_out).squeeze(-1))
        return {"pred": pred, "we": we, "rres": rres}
