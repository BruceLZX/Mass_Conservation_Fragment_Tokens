from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DELTAS = np.asarray([14.0157, 15.9949, 31.9898, -2.0157], dtype=np.float32)


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train - mean) / std, (test - mean) / std


def ridge_multiclass(x: np.ndarray, y: np.ndarray, num_classes: int, ridge: float = 1e-2) -> np.ndarray:
    x_aug = np.ascontiguousarray(np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1).astype(np.float64))
    y_onehot = np.eye(num_classes, dtype=np.float64)[y]
    eye = np.eye(x_aug.shape[1])
    eye[-1, -1] = 0.0
    xtx = np.einsum("ni,nj->ij", x_aug, x_aug)
    xty = np.einsum("ni,nc->ic", x_aug, y_onehot)
    return np.linalg.solve(xtx + ridge * eye, xty)


def predict(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    x_aug = np.ascontiguousarray(np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1).astype(np.float64))
    return np.argmax(np.einsum("ni,ic->nc", x_aug, coef), axis=1)


def make_pair(
    rng: np.random.Generator,
    low_mass: float,
    high_mass: float,
    noise_peaks: int,
    site_prob_low: float,
    site_prob_high: float,
    mass_noise: float,
    distractor_shift_prob: float,
) -> tuple[dict[str, np.ndarray | float], int]:
    label = int(rng.integers(0, len(DELTAS)))
    delta = float(DELTAS[label])
    precursor = float(rng.uniform(low_mass, high_mass))
    n_frag = int(rng.integers(12, 25))
    parent_core = np.sort(rng.uniform(45.0, precursor - 25.0, size=n_frag))
    contains_site = rng.random(n_frag) < rng.uniform(site_prob_low, site_prob_high)
    if not np.any(contains_site):
        contains_site[int(rng.integers(0, n_frag))] = True

    product_core = parent_core.copy()
    product_core[contains_site] += delta
    parent_int_core = rng.uniform(0.25, 1.0, size=n_frag)
    product_int_core = parent_int_core * rng.uniform(0.75, 1.15, size=n_frag)

    parent_noise = rng.uniform(40.0, precursor + 45.0, size=noise_peaks)
    product_noise = rng.uniform(40.0, precursor + 45.0, size=noise_peaks)
    parent_noise_int = rng.uniform(0.01, 0.18, size=noise_peaks)
    product_noise_int = rng.uniform(0.01, 0.18, size=noise_peaks)
    if distractor_shift_prob > 0.0:
        mask = rng.random(noise_peaks) < distractor_shift_prob
        wrong_deltas = rng.choice(DELTAS[DELTAS != delta], size=int(mask.sum()), replace=True)
        product_noise[mask] = parent_noise[mask] + wrong_deltas + rng.normal(0.0, mass_noise, size=int(mask.sum()))
        product_noise_int[mask] *= rng.uniform(0.1, 0.5, size=int(mask.sum()))

    parent = np.concatenate([parent_core, [precursor], parent_noise])
    product = np.concatenate([product_core, [precursor + delta], product_noise])
    parent_int = np.concatenate([parent_int_core, [0.8], parent_noise_int])
    product_int = np.concatenate([product_int_core, [0.8], product_noise_int])
    parent = parent + rng.normal(0.0, mass_noise, size=len(parent))
    product = product + rng.normal(0.0, mass_noise, size=len(product))
    parent_order = np.argsort(parent)
    product_order = np.argsort(product)
    return (
        {
            "parent_mz": parent[parent_order].astype(np.float32),
            "product_mz": product[product_order].astype(np.float32),
            "parent_intensity": parent_int[parent_order].astype(np.float32),
            "product_intensity": product_int[product_order].astype(np.float32),
        },
        label,
    )


def make_dataset(
    n: int,
    seed: int,
    low_mass: float,
    high_mass: float,
    noise_peaks: int,
    site_prob_low: float,
    site_prob_high: float,
    mass_noise: float,
    distractor_shift_prob: float,
) -> tuple[list[dict[str, np.ndarray | float]], np.ndarray]:
    rng = np.random.default_rng(seed)
    pairs, labels = [], []
    for _ in range(n):
        pair, label = make_pair(
            rng,
            low_mass,
            high_mass,
            noise_peaks,
            site_prob_low,
            site_prob_high,
            mass_noise,
            distractor_shift_prob,
        )
        pairs.append(pair)
        labels.append(label)
    return pairs, np.asarray(labels, dtype=np.int64)


def binned_features(pairs: list[dict[str, np.ndarray | float]], max_mz: float, bins: int) -> np.ndarray:
    feats = []
    edges = np.linspace(0.0, max_mz, bins + 1)
    for pair in pairs:
        parent = pair["parent_mz"]
        product = pair["product_mz"]
        parent_intensity = pair["parent_intensity"]
        product_intensity = pair["product_intensity"]
        hp, _ = np.histogram(parent, bins=edges, weights=parent_intensity)
        hq, _ = np.histogram(product, bins=edges, weights=product_intensity)
        feats.append(np.concatenate([hp, hq, hq - hp]).astype(np.float32))
    return np.stack(feats)


def conservation_features(pairs: list[dict[str, np.ndarray | float]], tolerance: float = 0.08) -> np.ndarray:
    feats = []
    for pair in pairs:
        parent = pair["parent_mz"]
        product = pair["product_mz"]
        parent_intensity = pair["parent_intensity"]
        product_intensity = pair["product_intensity"]
        diff = product[:, None] - parent[None, :]
        intensity_outer = product_intensity[:, None] * parent_intensity[None, :]
        row = []
        for delta in DELTAS:
            close = np.abs(diff - float(delta)) <= tolerance
            weighted = float(intensity_outer[close].sum()) if close.any() else 0.0
            top_parent = parent_intensity >= np.quantile(parent_intensity, 0.7)
            top_product = product_intensity >= np.quantile(product_intensity, 0.7)
            top_close = close & top_product[:, None] & top_parent[None, :]
            row.extend(
                [
                    float(close.sum()),
                    weighted,
                    float(top_close.sum()),
                    float(intensity_outer[top_close].sum()) if top_close.any() else 0.0,
                    float(close.any(axis=0).mean()),
                    float(close.any(axis=1).mean()),
                    float(np.min(np.abs(diff - float(delta)))),
                ]
            )
        # Also include precursor shift as a separate conservation certificate.
        precursor_shift = float(product.max() - parent.max())
        row.extend([abs(precursor_shift - float(delta)) for delta in DELTAS])
        feats.append(np.asarray(row, dtype=np.float32))
    return np.stack(feats)


def run_once(
    seed: int,
    train_n: int,
    test_n: int,
    noise_peaks: int,
    site_prob_low: float,
    site_prob_high: float,
    mass_noise: float,
    tolerance: float,
    distractor_shift_prob: float,
) -> dict[str, float]:
    train_pairs, train_y = make_dataset(
        train_n,
        seed,
        low_mass=300.0,
        high_mass=650.0,
        noise_peaks=noise_peaks,
        site_prob_low=site_prob_low,
        site_prob_high=site_prob_high,
        mass_noise=mass_noise,
        distractor_shift_prob=distractor_shift_prob,
    )
    test_pairs, test_y = make_dataset(
        test_n,
        10_000 + seed,
        low_mass=700.0,
        high_mass=1100.0,
        noise_peaks=noise_peaks,
        site_prob_low=site_prob_low,
        site_prob_high=site_prob_high,
        mass_noise=mass_noise,
        distractor_shift_prob=distractor_shift_prob,
    )

    results = {}
    for name, featurizer in [
        ("binned_linear", lambda pairs: binned_features(pairs, max_mz=1150.0, bins=128)),
        ("mcft_linear", lambda pairs: conservation_features(pairs, tolerance=tolerance)),
    ]:
        xtr = featurizer(train_pairs)
        xte = featurizer(test_pairs)
        xtr, xte = standardize(xtr, xte)
        coef = ridge_multiclass(xtr, train_y, len(DELTAS))
        results[name] = float((predict(xte, coef) == test_y).mean())

    nearest_precursor = []
    for pair in test_pairs:
        parent = pair["parent_mz"]
        product = pair["product_mz"]
        shift = float(product.max() - parent.max())
        nearest_precursor.append(int(np.argmin(np.abs(DELTAS - shift))))
    results["precursor_rule"] = float((np.asarray(nearest_precursor) == test_y).mean())
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="experiment/outputs/mass_conservation_tokens")
    parser.add_argument("--train-n", type=int, default=256)
    parser.add_argument("--test-n", type=int, default=1000)
    parser.add_argument("--noise-peaks", type=int, default=18)
    parser.add_argument("--site-prob-low", type=float, default=0.25)
    parser.add_argument("--site-prob-high", type=float, default=0.55)
    parser.add_argument("--mass-noise", type=float, default=0.015)
    parser.add_argument("--tolerance", type=float, default=0.08)
    parser.add_argument("--distractor-shift-prob", type=float, default=0.0)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args()

    rows = []
    for seed in [int(s) for s in args.seeds.split(",") if s]:
        row = run_once(
            seed,
            args.train_n,
            args.test_n,
            args.noise_peaks,
            args.site_prob_low,
            args.site_prob_high,
            args.mass_noise,
            args.tolerance,
            args.distractor_shift_prob,
        )
        row["seed"] = seed
        rows.append(row)
        print(json.dumps(row), flush=True)
    mean = {k: float(np.mean([row[k] for row in rows])) for k in rows[0] if k != "seed"}
    payload = {"metadata": vars(args), "rows": rows, "mean": mean}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
