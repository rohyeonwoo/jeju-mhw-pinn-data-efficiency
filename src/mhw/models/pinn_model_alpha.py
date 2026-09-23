from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .convlstm import ConvLSTMForecaster


def data_loss_torch(T_pred: torch.Tensor, T_true: torch.Tensor) -> torch.Tensor:
    """
    L_data = (1/N) * sum_i (T_pred(i) - T_true(i))^2
    """
    return torch.mean((T_pred - T_true) ** 2)


def physics_loss_torch(residual: torch.Tensor) -> torch.Tensor:
    """
    L_physics = (1/M) * sum_j R(x_j, y_j, t_j)^2
    """
    return torch.mean(residual ** 2)


def masked_physics_loss_torch(residual: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff2 = residual ** 2
    return (diff2 * mask).sum() / mask.sum().clamp_min(1.0)


def _boltzmann_weights(
    values: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    masked_values = values.abs() * mask
    batch = masked_values.shape[0]
    flat_vals = masked_values.view(batch, -1)
    flat_mask = mask.view(batch, -1)
    logits = flat_vals / max(temperature, 1e-6)
    logits = logits.masked_fill(flat_mask <= 0, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return probs / denom


def adaptive_weighted_physics_loss_torch(
    residual: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    weights = _boltzmann_weights(residual.detach(), mask, temperature=temperature)
    flat_res2 = residual.view(residual.shape[0], -1) ** 2
    weighted = (flat_res2 * weights).sum(dim=-1)
    return weighted.mean()


def adaptive_sampled_physics_loss_torch(
    residual: torch.Tensor,
    mask: torch.Tensor,
    num_samples: int = 256,
    temperature: float = 1.0,
) -> torch.Tensor:
    weights = _boltzmann_weights(residual.detach(), mask, temperature=temperature)
    flat_res2 = residual.view(residual.shape[0], -1) ** 2
    losses = []
    for batch_idx in range(residual.shape[0]):
        probs = weights[batch_idx]
        if torch.count_nonzero(probs).item() == 0:
            losses.append(torch.zeros((), device=residual.device, dtype=residual.dtype))
            continue
        sample_count = min(num_samples, probs.numel())
        sampled_idx = torch.multinomial(probs, num_samples=sample_count, replacement=True)
        losses.append(flat_res2[batch_idx, sampled_idx].mean())
    return torch.stack(losses).mean()


def total_loss_torch(
    T_pred: torch.Tensor,
    T_true: torch.Tensor,
    residual: torch.Tensor,
    lambda_data: float = 1.0,
    lambda_physics: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    L_total = lambda_data * L_data + lambda_physics * L_physics
    """
    L_data = data_loss_torch(T_pred, T_true)
    L_phys = physics_loss_torch(residual)
    L_total = lambda_data * L_data + lambda_physics * L_phys
    return L_total, L_data, L_phys


def compute_residual(
    dTdt: torch.Tensor,
    qnet: torch.Tensor,
    rho: float,
    Cp: float,
    hm: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dTdx: torch.Tensor,
    dTdy: torch.Tensor,
    we: torch.Tensor,
    Tm: torch.Tensor,
    Td: torch.Tensor,
    Rres: torch.Tensor,
    hm_min: float = 5.0,
) -> torch.Tensor:
    """
    Conventions:
    - qnet > 0 warms the ocean
    - dTdx, dTdy are in K m^-1
    - u, v are in m s^-1
    - we > 0 represents upward/entrainment velocity
    """
    hm_safe = torch.clamp(hm, min=hm_min)
    rhs = qnet / (rho * Cp * hm_safe)
    rhs = rhs - u * dTdx - v * dTdy - we * (Tm - Td) / hm_safe + Rres
    return dTdt - rhs


@dataclass
class PINNForwardOutput:
    T_pred: torch.Tensor
    residual: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class SimplePINNConvLSTM(nn.Module):
    """
    ConvLSTM backbone plus a simple heat-budget residual.

    The network predicts the next SST-like target field. Physics terms are
    supplied externally so the same module can be used with observed forcings
    or with differentiable preprocessing pipelines.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int] | None = None,
        out_channels: int = 1,
        kernel_size: int = 3,
        rho: float = 1025.0,
        Cp: float = 3990.0,
        hm_min: float = 5.0,
    ) -> None:
        super().__init__()
        self.backbone = ConvLSTMForecaster(
            in_channels=in_channels,
            hidden_channels=hidden_channels or [32, 32],
            out_channels=out_channels,
            kernel_size=kernel_size,
        )
        self.rho = rho
        self.Cp = Cp
        self.hm_min = hm_min

    def forward(
        self,
        x: torch.Tensor,
        qnet: torch.Tensor,
        hm: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        dTdx: torch.Tensor,
        dTdy: torch.Tensor,
        we: torch.Tensor,
        Tm: torch.Tensor,
        Td: torch.Tensor,
        Rres: torch.Tensor,
        dTdt: torch.Tensor | None = None,
    ) -> PINNForwardOutput:
        backbone_out = self.backbone(x)
        T_pred = backbone_out.prediction

        if dTdt is None:
            dTdt = T_pred - Tm

        residual = compute_residual(
            dTdt=dTdt,
            qnet=qnet,
            rho=self.rho,
            Cp=self.Cp,
            hm=hm,
            u=u,
            v=v,
            dTdx=dTdx,
            dTdy=dTdy,
            we=we,
            Tm=Tm,
            Td=Td,
            Rres=Rres,
            hm_min=self.hm_min,
        )

        diagnostics = {
            "dTdt": dTdt,
            "surface_flux": qnet / (self.rho * self.Cp * torch.clamp(hm, min=self.hm_min)),
            "adv_x": -u * dTdx,
            "adv_y": -v * dTdy,
            "entrainment": -we * (Tm - Td) / torch.clamp(hm, min=self.hm_min),
            "residual_term": Rres,
        }
        return PINNForwardOutput(T_pred=T_pred, residual=residual, diagnostics=diagnostics)
