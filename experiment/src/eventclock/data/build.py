from __future__ import annotations

from torch.utils.data import DataLoader

from eventclock.data.external import build_external
from eventclock.data.synthetic_sparse import build_synthetic


def build_dataset(cfg: dict, split: str):
    name = cfg["name"]
    if name == "synthetic_sparse":
        return build_synthetic(cfg["params"], split)
    if name in {"ptbxl", "sleep_edf", "wesad", "ppg_dalia", "npz"}:
        return build_external(cfg["params"], split)
    raise ValueError(f"Unknown dataset {name!r}")


def build_loader(cfg: dict, split: str, shuffle: bool | None = None) -> DataLoader:
    ds = build_dataset(cfg["dataset"], split)
    batch_size = int(cfg["train"].get("batch_size", 64))
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=bool(cfg["train"].get("pin_memory", False)),
    )

