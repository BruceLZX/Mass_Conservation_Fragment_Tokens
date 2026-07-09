from __future__ import annotations

import inspect
import torch
from torch import nn

from eventclock.models.transformer import TinyTransformerClassifier


class CNNBaseline(nn.Module):
    def __init__(self, channels: int, n_classes: int, dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, dim, 7, padding=3),
            nn.GELU(),
            nn.BatchNorm1d(dim),
            nn.Conv1d(dim, dim, 7, padding=3),
            nn.GELU(),
            nn.BatchNorm1d(dim),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.net(x).squeeze(-1)
        return {"logits": self.head(h)}


class FixedPatchTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        n_classes: int,
        dim: int = 64,
        patch_size: int = 16,
        stride: int | None = None,
        depth: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        stride = stride or patch_size
        self.patch = nn.Conv1d(channels, dim, kernel_size=patch_size, stride=stride)
        self.classifier = TinyTransformerClassifier(dim, n_classes, depth=depth, heads=heads, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.patch(x).transpose(1, 2)
        return {"logits": self.classifier(tokens), "tokens": tokens}


class RandomTokenTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        n_classes: int,
        dim: int = 64,
        k_tokens: int = 16,
        depth: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.k_tokens = k_tokens
        self.proj = nn.Conv1d(channels, dim, kernel_size=9, padding=4)
        self.classifier = TinyTransformerClassifier(dim, n_classes, depth=depth, heads=heads, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.proj(x).transpose(1, 2)
        b, t, _ = feats.shape
        idx = torch.stack([torch.randperm(t, device=x.device)[: self.k_tokens].sort().values for _ in range(b)])
        tokens = torch.gather(feats, 1, idx.unsqueeze(-1).expand(-1, -1, feats.shape[-1]))
        return {"logits": self.classifier(tokens), "tokens": tokens}


class ComplexityTokenTransformer(nn.Module):
    """TimeSqueeze-style local-complexity baseline without evidence calibration."""

    def __init__(
        self,
        channels: int,
        n_classes: int,
        dim: int = 64,
        k_tokens: int = 16,
        depth: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        sigma: float = 0.08,
    ) -> None:
        super().__init__()
        self.k_tokens = k_tokens
        self.sigma = sigma
        self.proj = nn.Conv1d(channels, dim, kernel_size=9, padding=4)
        self.classifier = TinyTransformerClassifier(dim, n_classes, depth=depth, heads=heads, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        diff = torch.abs(x[..., 1:] - x[..., :-1]).mean(dim=1)
        complexity = torch.nn.functional.pad(diff, (1, 0)) + 1e-4
        velocity = complexity / complexity.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        tau = torch.cumsum(velocity, dim=-1)
        grid = (torch.arange(self.k_tokens, device=x.device, dtype=x.dtype) + 0.5) / self.k_tokens
        dist = torch.abs(tau.unsqueeze(1) - grid.view(1, self.k_tokens, 1))
        weights = torch.softmax(-dist / self.sigma, dim=-1)
        feats = self.proj(x).transpose(1, 2)
        tokens = torch.einsum("bkt,btd->bkd", weights, feats)
        return {"logits": self.classifier(tokens), "velocity": velocity, "assign": weights, "tokens": tokens}


def build_model(cfg: dict, input_channels: int, n_classes: int) -> nn.Module:
    model_cfg = dict(cfg["model"])
    name = model_cfg.pop("name")
    def kwargs_for(cls):
        allowed = set(inspect.signature(cls.__init__).parameters) - {"self", "channels", "n_classes"}
        return {key: value for key, value in model_cfg.items() if key in allowed}

    if name == "cnn":
        return CNNBaseline(input_channels, n_classes, **kwargs_for(CNNBaseline))
    if name == "fixed_patch":
        return FixedPatchTransformer(input_channels, n_classes, **kwargs_for(FixedPatchTransformer))
    if name == "random_token":
        return RandomTokenTransformer(input_channels, n_classes, **kwargs_for(RandomTokenTransformer))
    if name == "complexity_token":
        return ComplexityTokenTransformer(input_channels, n_classes, **kwargs_for(ComplexityTokenTransformer))
    if name == "event_clock":
        from eventclock.models.event_clock import EventClockTransformer

        return EventClockTransformer(input_channels, n_classes, **kwargs_for(EventClockTransformer))
    raise ValueError(f"Unknown model {name!r}")
