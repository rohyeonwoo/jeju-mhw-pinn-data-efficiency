from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convlstm import ConvLSTMForecaster
from .pinn_model_alpha import compute_residual


@dataclass
class MultiTaskPINNOutput:
    T_pred: torch.Tensor
    we_pred: torch.Tensor
    residual_pred: torch.Tensor
    physics_residual: torch.Tensor


class ConvLSTMMultiTaskPINN(nn.Module):
    """
    Multi-task ConvLSTM PINN.

    Shared ConvLSTM features are used to predict:
    - target temperature field
    - entrainment velocity we
    - unresolved residual term Rres
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int] | None = None,
        kernel_size: int = 3,
        rho: float = 1025.0,
        Cp: float = 3990.0,
        hm_min: float = 5.0,
        we_scale: float = 1.0e-5,
        rres_scale: float = 1.0e-4,
        we_nonneg: bool = False,
        entrainment_dt: float = 0.0,
        we_softplus_shift: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_channels = hidden_channels or [32, 32]
        self.backbone = ConvLSTMForecaster(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels[-1],
            kernel_size=kernel_size,
        )
        feature_channels = hidden_channels[-1]
        self.temperature_head = nn.Conv2d(feature_channels, 1, kernel_size=1)
        self.we_head = nn.Conv2d(feature_channels, 1, kernel_size=1)
        self.rres_head = nn.Conv2d(feature_channels, 1, kernel_size=1)
        self.rho = rho
        self.Cp = Cp
        self.hm_min = hm_min
        self.we_scale = we_scale
        self.rres_scale = rres_scale
        # Identifiability controls (default off preserves legacy behavior):
        #   we_nonneg      -> entrainment velocity we >= 0 via softplus (else signed tanh)
        #   entrainment_dt -> stratification offset so Td = Tm - entrainment_dt (K),
        #                     giving the we*(Tm-Td)/h term a non-zero coefficient so
        #                     we is identifiable rather than receiving zero gradient.
        self.we_nonneg = bool(we_nonneg)
        self.entrainment_dt = float(entrainment_dt)
        # softplus(0)=ln2 leaves a ~0.69*we_scale floor that biases we away from 0
        # (needed: dh/dt median is 0, i.e. many cells should have we~0). A positive
        # shift lowers the init/floor: we = we_scale*softplus(head - shift).
        self.we_softplus_shift = float(we_softplus_shift)

    def forward(
        self,
        x: torch.Tensor,
        qnet: torch.Tensor,
        hm: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        dTdx: torch.Tensor,
        dTdy: torch.Tensor,
        Tm: torch.Tensor,
        Td: torch.Tensor,
        dTdt: torch.Tensor | None = None,
    ) -> MultiTaskPINNOutput:
        backbone_out = self.backbone(x)
        shared = backbone_out.prediction

        T_pred = self.temperature_head(shared)
        if self.we_nonneg:
            we_pred = self.we_scale * F.softplus(self.we_head(shared) - self.we_softplus_shift)
        else:
            we_pred = self.we_scale * torch.tanh(self.we_head(shared))
        residual_pred = self.rres_scale * torch.tanh(self.rres_head(shared))

        if dTdt is None:
            dTdt = T_pred - Tm

        # Apply the stratification offset so entrainment has a real coefficient.
        Td_eff = Tm - self.entrainment_dt if self.entrainment_dt > 0.0 else Td

        physics_residual = compute_residual(
            dTdt=dTdt,
            qnet=qnet,
            rho=self.rho,
            Cp=self.Cp,
            hm=hm,
            u=u,
            v=v,
            dTdx=dTdx,
            dTdy=dTdy,
            we=we_pred,
            Tm=Tm,
            Td=Td_eff,
            Rres=residual_pred,
            hm_min=self.hm_min,
        )

        return MultiTaskPINNOutput(
            T_pred=T_pred,
            we_pred=we_pred,
            residual_pred=residual_pred,
            physics_residual=physics_residual,
        )
