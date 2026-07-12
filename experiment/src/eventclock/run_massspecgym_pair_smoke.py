from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def parse_array(text: str) -> np.ndarray:
    if not isinstance(text, str) or not text:
        return np.asarray([], dtype=np.float32)
    return np.asarray([float(x) for x in text.split(",") if x], dtype=np.float32)


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train - mean) / std, (test - mean) / std


def ridge_binary(x: np.ndarray, y: np.ndarray, ridge: float = 1e-2) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1).astype(np.float64)
    y = y.astype(np.float64)
    eye = np.eye(x_aug.shape[1])
    eye[-1, -1] = 0.0
    xtx = np.einsum("ni,nj->ij", x_aug, x_aug)
    xty = np.einsum("ni,n->i", x_aug, y)
    return np.linalg.solve(xtx + ridge * eye, xty)


def predict_binary(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1).astype(np.float64)
    return (np.einsum("ni,i->n", x_aug, coef) >= 0.5).astype(np.int64)


def load_rows(path: Path) -> list[dict]:
    df = pd.read_csv(path, sep="\t", on_bad_lines="skip")
    rows = []
    for _, r in df.iterrows():
        mzs = parse_array(r["mzs"])
        ints = parse_array(r["intensities"])
        if len(mzs) < 3 or len(mzs) != len(ints):
            continue
        rows.append(
            {
                "identifier": r["identifier"],
                "inchikey": r["inchikey"],
                "mzs": mzs,
                "intensities": ints / max(float(np.max(ints)), 1e-6),
                "precursor_mz": float(r["precursor_mz"]),
                "instrument_type": str(r["instrument_type"]),
                "adduct": str(r["adduct"]),
                "fold": str(r["fold"]),
            }
        )
    return rows


def make_pairs(rows: list[dict], seed: int, max_pairs: int) -> tuple[list[tuple[dict, dict]], np.ndarray]:
    rng = np.random.default_rng(seed)
    by_key: dict[str, list[dict]] = {}
    for row in rows:
        by_key.setdefault(row["inchikey"], []).append(row)

    positives = []
    for group in by_key.values():
        if len(group) < 2:
            continue
        for a, b in combinations(group[:12], 2):
            positives.append((a, b))
    rng.shuffle(positives)
    positives = positives[:max_pairs]

    all_rows = rows[:]
    negatives = []
    tries = 0
    while len(negatives) < len(positives) and tries < len(positives) * 100:
        a, b = rng.choice(all_rows, size=2, replace=False)
        tries += 1
        if a["inchikey"] == b["inchikey"]:
            continue
        if a["adduct"] != b["adduct"]:
            continue
        if abs(a["precursor_mz"] - b["precursor_mz"]) > 80.0:
            continue
        negatives.append((a, b))
    pairs = positives + negatives
    y = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int64)
    idx = rng.permutation(len(pairs))
    return [pairs[i] for i in idx], y[idx]


def split_rows_by_inchikey(rows: list[dict], seed: int, train_frac: float = 0.7) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    keys = np.asarray(sorted({row["inchikey"] for row in rows}))
    rng.shuffle(keys)
    cut = max(1, int(train_frac * len(keys)))
    train_keys = set(keys[:cut])
    train_rows = [row for row in rows if row["inchikey"] in train_keys]
    test_rows = [row for row in rows if row["inchikey"] not in train_keys]
    return train_rows, test_rows


def binned_pair_features(pairs: list[tuple[dict, dict]], bins: int = 256, max_mz: float = 1200.0) -> np.ndarray:
    edges = np.linspace(0.0, max_mz, bins + 1)
    feats = []
    for a, b in pairs:
        ha, _ = np.histogram(a["mzs"], bins=edges, weights=a["intensities"])
        hb, _ = np.histogram(b["mzs"], bins=edges, weights=b["intensities"])
        denom = max(float(np.linalg.norm(ha) * np.linalg.norm(hb)), 1e-8)
        cosine = float(np.dot(ha, hb) / denom)
        feats.append(np.concatenate([np.abs(ha - hb), [cosine, abs(a["precursor_mz"] - b["precursor_mz"]) / max_mz]]).astype(np.float32))
    return np.stack(feats)


def peakset_embedding(row: dict, dims: int = 96, max_mz: float = 1200.0) -> np.ndarray:
    rng = np.random.default_rng(1729)
    freqs = rng.lognormal(mean=0.0, sigma=1.0, size=dims).astype(np.float32)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=dims).astype(np.float32)
    mz = (row["mzs"].astype(np.float32) / max_mz)[:, None]
    intensities = row["intensities"].astype(np.float32)[:, None]
    angles = mz * freqs[None, :] * (2.0 * np.pi) + phases[None, :]
    weighted_cos = (intensities * np.cos(angles)).sum(axis=0)
    weighted_sin = (intensities * np.sin(angles)).sum(axis=0)
    norm = max(float(intensities.sum()), 1e-6)
    pooled = np.concatenate([weighted_cos / norm, weighted_sin / norm])
    stats = np.asarray(
        [
            len(row["mzs"]) / 256.0,
            float(row["mzs"].min()) / max_mz,
            float(row["mzs"].max()) / max_mz,
            float(np.average(row["mzs"], weights=np.maximum(row["intensities"], 1e-6))) / max_mz,
            float(row["precursor_mz"]) / max_mz,
        ],
        dtype=np.float32,
    )
    return np.concatenate([pooled.astype(np.float32), stats])


def peakset_pair_features(pairs: list[tuple[dict, dict]]) -> np.ndarray:
    feats = []
    cache: dict[str, np.ndarray] = {}
    for a, b in pairs:
        if a["identifier"] not in cache:
            cache[a["identifier"]] = peakset_embedding(a)
        if b["identifier"] not in cache:
            cache[b["identifier"]] = peakset_embedding(b)
        za = cache[a["identifier"]]
        zb = cache[b["identifier"]]
        denom = max(float(np.linalg.norm(za) * np.linalg.norm(zb)), 1e-8)
        cosine = float(np.dot(za, zb) / denom)
        row = np.concatenate(
            [
                np.abs(za - zb),
                za * zb,
                np.asarray(
                    [
                        cosine,
                        abs(a["precursor_mz"] - b["precursor_mz"]) / 1200.0,
                        float(a["adduct"] == b["adduct"]),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        feats.append(row.astype(np.float32))
    return np.stack(feats)


def conservation_pair_features(
    pairs: list[tuple[dict, dict]], tolerances=(0.01, 0.03, 0.08), include_dynamic: bool = False
) -> np.ndarray:
    fixed_shifts = np.asarray([0.0, 1.0034, -1.0034, 14.0157, 15.9949, -18.0106], dtype=np.float32)
    feats = []
    for a, b in pairs:
        diff = b["mzs"][:, None] - a["mzs"][None, :]
        intensity_outer = b["intensities"][:, None] * a["intensities"][None, :]
        row = [abs(a["precursor_mz"] - b["precursor_mz"]) / 1200.0]
        shifts = fixed_shifts
        if include_dynamic:
            precursor_shift = float(b["precursor_mz"] - a["precursor_mz"])
            shifts = np.concatenate([fixed_shifts, np.asarray([precursor_shift, -precursor_shift], dtype=np.float32)])
        for tol in tolerances:
            for shift in shifts:
                close = np.abs(diff - float(shift)) <= tol
                row.extend(
                    [
                        float(close.sum()) / max(len(a["mzs"]) + len(b["mzs"]), 1),
                        float(intensity_outer[close].sum()) if close.any() else 0.0,
                        float(close.any(axis=0).mean()),
                        float(close.any(axis=1).mean()),
                    ]
                )
        feats.append(np.asarray(row, dtype=np.float32))
    return np.stack(feats)


def run(seed: int, rows: list[dict], max_pairs: int, group_split: bool) -> dict[str, float]:
    if group_split:
        train_rows, test_rows = split_rows_by_inchikey(rows, seed)
        train_pairs, train_y = make_pairs(train_rows, seed + 17, max_pairs)
        test_pairs, test_y = make_pairs(test_rows, seed + 31, max_pairs)
    else:
        pairs, y = make_pairs(rows, seed, max_pairs)
        split = int(0.7 * len(pairs))
        train_pairs, test_pairs = pairs[:split], pairs[split:]
        train_y, test_y = y[:split], y[split:]
    out = {
        "num_train_pairs": float(len(train_pairs)),
        "num_test_pairs": float(len(test_pairs)),
        "train_positive_rate": float(train_y.mean()),
        "test_positive_rate": float(test_y.mean()),
    }
    for name, featurizer in [
        ("binned_pair_linear", binned_pair_features),
        ("peakset_pair_linear", peakset_pair_features),
        ("mcft_pair_linear", conservation_pair_features),
    ]:
        xtr, xte = featurizer(train_pairs), featurizer(test_pairs)
        xtr, xte = standardize(xtr, xte)
        coef = ridge_binary(xtr, train_y)
        out[name] = float((predict_binary(xte, coef) == test_y).mean())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", default="experiment/data/massspecgym/MassSpecGym.tsv")
    parser.add_argument("--out-dir", default="experiment/outputs/massspecgym_pair_smoke")
    parser.add_argument("--max-pairs", type=int, default=1000)
    parser.add_argument("--group-split", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args()
    rows = load_rows(Path(args.tsv))
    results = []
    for seed in [int(s) for s in args.seeds.split(",") if s]:
        row = run(seed, rows, args.max_pairs, args.group_split)
        row["seed"] = seed
        results.append(row)
        print(json.dumps(row), flush=True)
    mean = {k: float(np.mean([r[k] for r in results])) for k in results[0] if k != "seed"}
    payload = {"metadata": {**vars(args), "num_rows": len(rows)}, "rows": results, "mean": mean}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
