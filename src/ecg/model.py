"""1D convolutional network for single-beat ECG classification."""
from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU, twice, with a residual (skip) connection.

    The skip connection lets gradients flow straight through, which fixes the
    underfitting I saw with deep plain CNNs in my thesis.
    """

    def __init__(self, c_in: int, c_out: int, kernel: int):
        super().__init__()
        pad = kernel // 2
        self.body = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel, padding=pad), nn.BatchNorm1d(c_out), nn.ReLU(),
            nn.Conv1d(c_out, c_out, kernel, padding=pad), nn.BatchNorm1d(c_out),
        )
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.body(x) + self.skip(x)))


class ECGNet(nn.Module):
    def __init__(self, n_classes: int = 5, channels: tuple[int, ...] = (32, 64, 128),
                 kernel: int = 7, dropout: float = 0.2):
        super().__init__()
        blocks, c_in = [], 1
        for c in channels:
            blocks.append(ConvBlock(c_in, c, kernel))
            c_in = c
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window) -> raw logits (batch, n_classes). No softmax here."""
        return self.head(self.features(x.unsqueeze(1)))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
