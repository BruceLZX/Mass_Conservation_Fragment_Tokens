from __future__ import annotations

import torch


def add_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    return x + torch.randn_like(x) * std


def time_shift(x: torch.Tensor, max_shift: int) -> torch.Tensor:
    if max_shift <= 0:
        return x
    shifts = torch.randint(-max_shift, max_shift + 1, (x.shape[0],), device=x.device)
    out = torch.empty_like(x)
    for i, shift in enumerate(shifts.tolist()):
        out[i] = torch.roll(x[i], shifts=shift, dims=-1)
    return out


def channel_dropout(x: torch.Tensor, prob: float) -> torch.Tensor:
    if prob <= 0:
        return x
    keep = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > prob).float()
    return x * keep


def amplitude_scale(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    if low == 1.0 and high == 1.0:
        return x
    scale = torch.empty(x.shape[0], 1, 1, device=x.device).uniform_(low, high)
    return x * scale


def random_mask_span(x: torch.Tensor, prob: float, span: int) -> torch.Tensor:
    if prob <= 0 or span <= 0:
        return x
    out = x.clone()
    for i in range(x.shape[0]):
        if torch.rand((), device=x.device) < prob:
            start = int(torch.randint(0, max(1, x.shape[-1] - span), (), device=x.device))
            out[i, :, start : start + span] = 0
    return out


def apply_robustness(x: torch.Tensor, cfg: dict | None) -> torch.Tensor:
    cfg = cfg or {}
    if cfg.get("noise_std", 0) > 0:
        x = add_noise(x, float(cfg["noise_std"]))
    if cfg.get("max_shift", 0) > 0:
        x = time_shift(x, int(cfg["max_shift"]))
    if cfg.get("channel_dropout", 0) > 0:
        x = channel_dropout(x, float(cfg["channel_dropout"]))
    if cfg.get("amp_low", 1.0) != 1.0 or cfg.get("amp_high", 1.0) != 1.0:
        x = amplitude_scale(x, float(cfg.get("amp_low", 1.0)), float(cfg.get("amp_high", 1.0)))
    if cfg.get("mask_prob", 0) > 0:
        x = random_mask_span(x, float(cfg["mask_prob"]), int(cfg.get("mask_span", 16)))
    return x

