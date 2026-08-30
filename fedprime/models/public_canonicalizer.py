from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = next(group for group in range(min(8, channels), 0, -1) if channels % group == 0)
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.silu(inputs + self.layers(inputs))


class PublicNuisanceCanonicalizer(nn.Module):
    """Small input-space canonicalizer for 32x32 public/private images.

    The module predicts a bounded residual around the input. It contains no
    task classifier, corruption-type head, prompt bank, or client-specific
    parameters. Inputs and outputs are expected in the [0, 1] range.
    """

    def __init__(self, base_channels: int = 24, residual_scale: float = 0.5) -> None:
        super().__init__()
        base = int(base_channels)
        if base < 8:
            raise ValueError("base_channels must be at least 8")
        self.base_channels = base
        self.residual_scale = float(residual_scale)

        self.stem = nn.Conv2d(3, base, kernel_size=3, padding=1)
        self.enc1 = _ResidualBlock(base)
        self.down1 = nn.Conv2d(base, base * 2, kernel_size=4, stride=2, padding=1)
        self.enc2 = _ResidualBlock(base * 2)
        self.down2 = nn.Conv2d(base * 2, base * 4, kernel_size=4, stride=2, padding=1)
        self.bottleneck = nn.Sequential(
            _ResidualBlock(base * 4),
            _ResidualBlock(base * 4),
        )
        self.up2 = nn.Conv2d(base * 4, base * 2, kernel_size=3, padding=1)
        self.dec2 = _ResidualBlock(base * 4)
        self.reduce2 = nn.Conv2d(base * 4, base * 2, kernel_size=1)
        self.up1 = nn.Conv2d(base * 2, base, kernel_size=3, padding=1)
        self.dec1 = _ResidualBlock(base * 2)
        self.reduce1 = nn.Conv2d(base * 2, base, kernel_size=1)
        self.head = nn.Conv2d(base, 3, kernel_size=3, padding=1)

    def config(self) -> dict[str, float | int]:
        return {
            "base_channels": self.base_channels,
            "residual_scale": self.residual_scale,
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError("canonicalizer expects BCHW RGB inputs")
        skip1 = self.enc1(self.stem(inputs))
        skip2 = self.enc2(self.down1(skip1))
        hidden = self.bottleneck(self.down2(skip2))

        hidden = F.interpolate(hidden, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.up2(hidden)
        hidden = self.reduce2(self.dec2(torch.cat([hidden, skip2], dim=1)))
        hidden = F.interpolate(hidden, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.up1(hidden)
        hidden = self.reduce1(self.dec1(torch.cat([hidden, skip1], dim=1)))
        residual = torch.tanh(self.head(hidden)) * self.residual_scale
        return torch.clamp(inputs + residual, 0.0, 1.0)


def load_public_canonicalizer_checkpoint(
    path: str,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[PublicNuisanceCanonicalizer, dict[str, object]]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # Older OpenI PyTorch.
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise TypeError("canonicalizer checkpoint must contain state_dict")
    config = dict(payload.get("model_config", {}))
    model = PublicNuisanceCanonicalizer(
        base_channels=int(config.get("base_channels", 24)),
        residual_scale=float(config.get("residual_scale", 0.5)),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload
