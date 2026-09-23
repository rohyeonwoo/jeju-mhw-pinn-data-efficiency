from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.gates(combined)
        i, f, o, g = torch.chunk(gates, chunks=4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(
        self,
        batch_size: int,
        spatial_size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = spatial_size
        shape = (batch_size, self.hidden_channels, height, width)
        h = torch.zeros(shape, device=device, dtype=dtype)
        c = torch.zeros(shape, device=device, dtype=dtype)
        return h, c


class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: list[int],
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if not hidden_channels:
            raise ValueError("hidden_channels must contain at least one layer size")

        cells = []
        in_channels = input_channels
        for hidden_dim in hidden_channels:
            cells.append(
                ConvLSTMCell(
                    input_channels=in_channels,
                    hidden_channels=hidden_dim,
                    kernel_size=kernel_size,
                )
            )
            in_channels = hidden_dim
        self.cells = nn.ModuleList(cells)

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if x.ndim != 5:
            raise ValueError("Expected input shape (batch, seq, channels, height, width)")

        batch_size, seq_len, _, height, width = x.shape
        if hidden_state is None:
            hidden_state = [
                cell.init_state(
                    batch_size=batch_size,
                    spatial_size=(height, width),
                    device=x.device,
                    dtype=x.dtype,
                )
                for cell in self.cells
            ]

        outputs = []
        current_input = x
        next_states: list[tuple[torch.Tensor, torch.Tensor]] = []

        for layer_idx, cell in enumerate(self.cells):
            h, c = hidden_state[layer_idx]
            layer_outputs = []
            for t in range(seq_len):
                h, c = cell(current_input[:, t], (h, c))
                layer_outputs.append(h)
            current_input = torch.stack(layer_outputs, dim=1)
            outputs = layer_outputs
            next_states.append((h, c))

        return torch.stack(outputs, dim=1), next_states


@dataclass
class ConvLSTMOutput:
    prediction: torch.Tensor
    sequence_features: torch.Tensor
    hidden_state: list[tuple[torch.Tensor, torch.Tensor]]


class ConvLSTMForecaster(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int] | None = None,
        out_channels: int = 1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        hidden_channels = hidden_channels or [32, 32]
        self.encoder = ConvLSTM(
            input_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels[-1], hidden_channels[-1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels[-1], out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> ConvLSTMOutput:
        features, hidden_state = self.encoder(x)
        last_hidden = features[:, -1]
        prediction = self.head(last_hidden)
        return ConvLSTMOutput(
            prediction=prediction,
            sequence_features=features,
            hidden_state=hidden_state,
        )
