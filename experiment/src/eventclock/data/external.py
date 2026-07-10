from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NPZSignalDataset(Dataset):
    """Generic preprocessed time-series dataset.

    Expected arrays:
    - x: float array [N, C, T] or [N, T, C]
    - y: integer labels [N]
    - split: optional string array [N] with train/val/test
    - evidence_mask: optional float array [N, T]
    """

    def __init__(self, path: str | Path, split: str = "train", channels_first: bool = True) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Preprocessed dataset not found: {path}. "
                "Create an NPZ with arrays x, y, optional split/evidence_mask."
            )
        data = np.load(path, allow_pickle=True)
        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.int64)
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [N, C, T] or [N, T, C], got {x.shape}.")
        if y.ndim != 1:
            raise ValueError(f"Expected y to have shape [N], got {y.shape}.")
        if len(x) != len(y):
            raise ValueError(f"x/y length mismatch: {len(x)} vs {len(y)}.")
        if not channels_first:
            x = np.transpose(x, (0, 2, 1))
        if "split" in data:
            keep = np.asarray(data["split"]).astype(str) == split
            x, y = x[keep], y[keep]
            mask = data["evidence_mask"][keep].astype(np.float32) if "evidence_mask" in data else None
            decoy_mask = data["decoy_mask"][keep].astype(np.float32) if "decoy_mask" in data else None
        else:
            idx = _deterministic_split_indices(len(y), split)
            x, y = x[idx], y[idx]
            mask = data["evidence_mask"][idx].astype(np.float32) if "evidence_mask" in data else None
            decoy_mask = data["decoy_mask"][idx].astype(np.float32) if "decoy_mask" in data else None
        if len(y) == 0:
            raise ValueError(f"No samples found for split {split!r} in {path}.")
        if mask is not None and mask.shape != (len(y), x.shape[-1]):
            raise ValueError(f"Expected evidence_mask shape {(len(y), x.shape[-1])}, got {mask.shape}.")
        if decoy_mask is not None and decoy_mask.shape != (len(y), x.shape[-1]):
            raise ValueError(f"Expected decoy_mask shape {(len(y), x.shape[-1])}, got {decoy_mask.shape}.")
        self.x = x
        self.y = y
        self.mask = mask
        self.decoy_mask = decoy_mask

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {"x": torch.from_numpy(self.x[idx]), "y": torch.tensor(self.y[idx], dtype=torch.long)}
        if self.mask is not None:
            item["evidence_mask"] = torch.from_numpy(self.mask[idx])
        if self.decoy_mask is not None:
            item["decoy_mask"] = torch.from_numpy(self.decoy_mask[idx])
        return item


def _deterministic_split_indices(n: int, split: str) -> np.ndarray:
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    if split == "train":
        return np.arange(0, train_end)
    if split == "val":
        return np.arange(train_end, val_end)
    if split == "test":
        return np.arange(val_end, n)
    raise ValueError(f"Unknown split {split!r}")


def build_external(cfg: dict, split: str) -> NPZSignalDataset:
    return NPZSignalDataset(
        cfg["path"],
        split=split,
        channels_first=bool(cfg.get("channels_first", True)),
    )
