"""Squeeze-and-excitation residual encoder for one-dimensional PXRD profiles."""

from __future__ import annotations

import torch
import torch.nn as nn


class SEBlock1D(nn.Module):
    def __init__(self, channels: int, ratio: int = 16) -> None:
        super().__init__()
        reduced = max(4, int(channels) // int(ratio))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        weights = self.fc(self.pool(values).squeeze(-1)).unsqueeze(-1)
        return values * weights


class SEResBlock1D(nn.Module):
    def __init__(self, channels: int, dropout: float, ratio: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.se = SEBlock1D(channels, ratio=ratio)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.activation(self.bn1(self.conv1(values)))
        values = self.dropout(values)
        values = self.bn2(self.conv2(values))
        values = self.se(values)
        return self.activation(values + residual)


class PxrdEncoder(nn.Module):
    """Formal four-stage SERes PXRD encoder used by the screening model."""

    def __init__(self, input_length: int = 3400, output_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        if input_length < 60:
            raise ValueError("input_length must be at least 60")
        self.input_length = int(input_length)
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.stage1 = self._stage(64, dropout * 0.5, pool=4)
        self.transition1 = self._transition(64, 128)
        self.stage2 = self._stage(128, dropout * 0.5, pool=5)
        self.transition2 = self._transition(128, 256)
        self.stage3 = self._stage(256, dropout, pool=3)
        self.transition3 = self._transition(256, 512)
        self.stage4 = nn.Sequential(
            SEResBlock1D(512, dropout),
            SEResBlock1D(512, dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, output_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

    @staticmethod
    def _stage(channels: int, dropout: float, pool: int) -> nn.Sequential:
        return nn.Sequential(
            SEResBlock1D(channels, dropout),
            SEResBlock1D(channels, dropout),
            nn.AvgPool1d(kernel_size=pool),
        )

    @staticmethod
    def _transition(input_channels: int, output_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 2:
            values = values.unsqueeze(1)
        if values.ndim != 3 or values.shape[1] != 1:
            raise ValueError("PXRD input must have shape [batch, length] or [batch, 1, length]")
        if values.shape[-1] != self.input_length:
            raise ValueError(
                f"PXRD length mismatch: got {values.shape[-1]}, expected {self.input_length}"
            )
        values = self.stage1(self.stem(values))
        values = self.stage2(self.transition1(values))
        values = self.stage3(self.transition2(values))
        values = self.stage4(self.transition3(values))
        return self.head(self.pool(values).squeeze(-1))
