from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from eventclock.models.transformer import TinyTransformerClassifier


class ScoutNet(nn.Module):
    def __init__(self, channels: int, hidden: int = 64, depth: int = 3, kernel_size: int = 7) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = channels
        pad = kernel_size // 2
        for _ in range(depth):
            layers.extend(
                [
                    nn.Conv1d(in_ch, in_ch, kernel_size, padding=pad, groups=in_ch),
                    nn.Conv1d(in_ch, hidden, 1),
                    nn.GELU(),
                    nn.BatchNorm1d(hidden),
                ]
            )
            in_ch = hidden
        self.net = nn.Sequential(*layers)
        self.to_velocity = nn.Conv1d(hidden, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_velocity(self.net(x)).squeeze(1)


class WindowEncoder(nn.Module):
    def __init__(self, channels: int, dim: int, window: int = 9) -> None:
        super().__init__()
        self.window = window
        self.proj = nn.Sequential(
            nn.Conv1d(channels, dim, kernel_size=window, padding=window // 2),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).transpose(1, 2)


class EventClockTokenizer(nn.Module):
    def __init__(
        self,
        channels: int,
        dim: int,
        k_tokens: int,
        scout_hidden: int = 64,
        scout_depth: int = 3,
        sigma: float = 0.08,
        uniform_tokens: int = 0,
        uniform_anchor_sigma: float | None = None,
        uniform_prior_mix: float = 0.0,
        epsilon: float = 1e-4,
        budget: float = 1.0,
        mask_quantile: float = 0.75,
        mask_temperature_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if uniform_tokens < 0 or uniform_tokens >= k_tokens:
            raise ValueError("uniform_tokens must be in [0, k_tokens).")
        self.k_tokens = k_tokens
        self.uniform_tokens = uniform_tokens
        self.clock_tokens = k_tokens - uniform_tokens
        self.sigma = sigma
        self.uniform_anchor_sigma = uniform_anchor_sigma if uniform_anchor_sigma is not None else sigma
        self.uniform_prior_mix = uniform_prior_mix
        self.epsilon = epsilon
        self.budget = budget
        self.mask_quantile = mask_quantile
        self.mask_temperature_scale = mask_temperature_scale
        self.scout = ScoutNet(channels, scout_hidden, scout_depth)
        self.local = WindowEncoder(channels, dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw = self.scout(x)
        velocity = F.softplus(raw) + self.epsilon
        velocity = self.budget * velocity / velocity.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if self.uniform_prior_mix > 0:
            mix = min(max(float(self.uniform_prior_mix), 0.0), 1.0)
            uniform_velocity = torch.full_like(velocity, self.budget / velocity.shape[-1])
            velocity = (1.0 - mix) * velocity + mix * uniform_velocity
            velocity = self.budget * velocity / velocity.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        tau = torch.cumsum(velocity, dim=-1)
        grid = (torch.arange(self.clock_tokens, device=x.device, dtype=x.dtype) + 0.5) / self.clock_tokens
        grid = grid.view(1, self.clock_tokens, 1) * self.budget
        dist = torch.abs(tau.unsqueeze(1) - grid)
        weights = torch.softmax(-dist / self.sigma, dim=-1)
        features = self.local(x)
        tokens = torch.einsum("bkt,btd->bkd", weights, features)
        if self.uniform_tokens > 0:
            anchor_weights = uniform_anchor_weights(
                x.shape[-1],
                self.uniform_tokens,
                x.device,
                x.dtype,
                sigma=self.uniform_anchor_sigma,
            )
            anchor_tokens = torch.einsum("kt,btd->bkd", anchor_weights, features)
            tokens = torch.cat([tokens, anchor_tokens], dim=1)
        mask = clock_mask(velocity, quantile=self.mask_quantile, temperature_scale=self.mask_temperature_scale)
        aux = {"velocity": velocity, "tau": tau, "assign": weights, "mask": mask}
        return tokens, aux


def uniform_anchor_weights(
    length: int,
    k_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    sigma: float,
) -> torch.Tensor:
    positions = (torch.arange(length, device=device, dtype=dtype) + 0.5) / length
    grid = (torch.arange(k_tokens, device=device, dtype=dtype) + 0.5) / k_tokens
    dist = torch.abs(grid.view(k_tokens, 1) - positions.view(1, length))
    return torch.softmax(-dist / sigma, dim=-1)


def clock_mask(velocity: torch.Tensor, quantile: float = 0.75, temperature_scale: float = 0.5) -> torch.Tensor:
    q = torch.quantile(velocity.detach(), quantile, dim=-1, keepdim=True)
    temperature = temperature_scale * velocity.detach().std(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.sigmoid((velocity - q) / temperature)


class EventClockTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        n_classes: int,
        dim: int = 64,
        k_tokens: int = 16,
        uniform_tokens: int = 0,
        scout_hidden: int = 64,
        scout_depth: int = 3,
        sigma: float = 0.08,
        uniform_anchor_sigma: float | None = None,
        uniform_prior_mix: float = 0.0,
        mask_quantile: float = 0.75,
        mask_temperature_scale: float = 0.5,
        depth: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tokenizer = EventClockTokenizer(
            channels=channels,
            dim=dim,
            k_tokens=k_tokens,
            uniform_tokens=uniform_tokens,
            scout_hidden=scout_hidden,
            scout_depth=scout_depth,
            sigma=sigma,
            uniform_anchor_sigma=uniform_anchor_sigma,
            uniform_prior_mix=uniform_prior_mix,
            mask_quantile=mask_quantile,
            mask_temperature_scale=mask_temperature_scale,
        )
        self.classifier = TinyTransformerClassifier(dim, n_classes, depth=depth, heads=heads, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens, aux = self.tokenizer(x)
        logits = self.classifier(tokens)
        aux["tokens"] = tokens
        aux["logits"] = logits
        return aux
