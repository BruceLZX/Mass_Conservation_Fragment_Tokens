from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticSpec:
    n_samples: int = 2000
    length: int = 256
    channels: int = 1
    n_classes: int = 2
    motif_len: int = 24
    noise_std: float = 0.4
    motif_amp: float = 1.5
    distractor_prob: float = 0.25
    discriminative_motif_all_classes: bool = False
    decoy_prob: float = 0.0
    decoy_len: int = 48
    decoy_amp: float = 2.0
    decoy_cycles: int = 9
    seed: int = 0


class SyntheticSparseMotif(Dataset):
    """Sparse motif classification with ground-truth evidence masks.

    Class 1 contains a localized motif. Optionally, both classes contain
    class-specific motifs, and every sample can contain a label-independent
    high-complexity decoy region. The decoy makes local-complexity routing a
    deliberately bad shortcut for evidence localization.
    """

    def __init__(self, spec: SyntheticSpec, split: str = "train") -> None:
        self.spec = spec
        split_offsets = {"train": 0, "val": 10_000, "test": 20_000}
        rng = np.random.default_rng(spec.seed + split_offsets.get(split, 30_000))
        self.x = np.zeros((spec.n_samples, spec.channels, spec.length), dtype=np.float32)
        self.y = np.zeros((spec.n_samples,), dtype=np.int64)
        self.mask = np.zeros((spec.n_samples, spec.length), dtype=np.float32)
        self.decoy_mask = np.zeros((spec.n_samples, spec.length), dtype=np.float32)

        t = np.linspace(0, 1, spec.motif_len, dtype=np.float32)
        motif_pos = np.sin(2 * np.pi * 3 * t) * np.hanning(spec.motif_len)
        motif_neg = np.cos(2 * np.pi * 2 * t + 0.7) * np.hanning(spec.motif_len)
        td = np.linspace(0, 1, spec.decoy_len, dtype=np.float32)
        decoy_wave = np.sign(np.sin(2 * np.pi * spec.decoy_cycles * td)) * np.hanning(spec.decoy_len)

        for i in range(spec.n_samples):
            label = int(rng.integers(0, spec.n_classes))
            signal = rng.normal(0, spec.noise_std, size=(spec.channels, spec.length)).astype(np.float32)
            start = int(rng.integers(8, spec.length - spec.motif_len - 8))
            if label == 1:
                for c in range(spec.channels):
                    scale = spec.motif_amp * (1.0 + 0.15 * c)
                    signal[c, start : start + spec.motif_len] += scale * motif_pos
                self.mask[i, start : start + spec.motif_len] = 1.0
            elif spec.discriminative_motif_all_classes:
                for c in range(spec.channels):
                    scale = spec.motif_amp * (1.0 + 0.15 * c)
                    signal[c, start : start + spec.motif_len] += scale * motif_neg
                self.mask[i, start : start + spec.motif_len] = 1.0
            elif rng.random() < spec.distractor_prob:
                for c in range(spec.channels):
                    signal[c, start : start + spec.motif_len] += 0.65 * spec.motif_amp * motif_neg
            if spec.decoy_prob > 0 and rng.random() < spec.decoy_prob:
                decoy_start = _sample_nonoverlapping_start(
                    rng,
                    spec.length,
                    spec.decoy_len,
                    avoid_start=start,
                    avoid_len=spec.motif_len,
                    margin=4,
                )
                jitter = rng.normal(0, 0.15, size=spec.decoy_len).astype(np.float32)
                for c in range(spec.channels):
                    scale = spec.decoy_amp * (1.0 + 0.1 * c)
                    signal[c, decoy_start : decoy_start + spec.decoy_len] += scale * (decoy_wave + jitter)
                self.decoy_mask[i, decoy_start : decoy_start + spec.decoy_len] = 1.0
            self.x[i] = signal
            self.y[i] = label

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[idx]),
            "y": torch.tensor(self.y[idx], dtype=torch.long),
            "evidence_mask": torch.from_numpy(self.mask[idx]),
            "decoy_mask": torch.from_numpy(self.decoy_mask[idx]),
        }


def build_synthetic(cfg: dict, split: str) -> SyntheticSparseMotif:
    split_cfg = dict(cfg)
    if "splits" in cfg:
        split_cfg["n_samples"] = cfg["splits"].get(split, cfg.get("n_samples", 2000))
    split_cfg.pop("splits", None)
    return SyntheticSparseMotif(SyntheticSpec(**split_cfg), split=split)


def _sample_nonoverlapping_start(
    rng: np.random.Generator,
    length: int,
    span: int,
    avoid_start: int,
    avoid_len: int,
    margin: int,
) -> int:
    candidates = []
    avoid_end = avoid_start + avoid_len
    for start in range(8, length - span - 8):
        end = start + span
        if end + margin < avoid_start or start - margin > avoid_end:
            candidates.append(start)
    if not candidates:
        return int(rng.integers(8, length - span - 8))
    return int(rng.choice(candidates))
