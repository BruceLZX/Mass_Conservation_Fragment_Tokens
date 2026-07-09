from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate preprocessed NPZ dataset format.")
    parser.add_argument("path")
    parser.add_argument("--channels-last", action="store_true", help="Use when x is [N, T, C].")
    args = parser.parse_args()

    path = Path(args.path)
    data = np.load(path, allow_pickle=True)
    required = {"x", "y"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Missing required arrays: {sorted(missing)}")

    x = data["x"]
    y = data["y"]
    if x.ndim != 3:
        raise ValueError(f"x must be 3D, got {x.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got {y.shape}")
    if len(x) != len(y):
        raise ValueError(f"x/y length mismatch: {len(x)} vs {len(y)}")

    n, a, b = x.shape
    channels = b if args.channels_last else a
    length = a if args.channels_last else b
    print(f"samples={n} channels={channels} length={length} classes={sorted(set(y.astype(int).tolist()))}")

    if "split" in data:
        split = data["split"].astype(str)
        for name in ["train", "val", "test"]:
            print(f"{name}={int((split == name).sum())}")
    else:
        print("split=absent; loader will use deterministic 70/15/15 split")

    if "evidence_mask" in data:
        mask = data["evidence_mask"]
        expected = (len(y), length)
        if mask.shape != expected:
            raise ValueError(f"evidence_mask must have shape {expected}, got {mask.shape}")
        print("evidence_mask=present")
    else:
        print("evidence_mask=absent")


if __name__ == "__main__":
    main()

