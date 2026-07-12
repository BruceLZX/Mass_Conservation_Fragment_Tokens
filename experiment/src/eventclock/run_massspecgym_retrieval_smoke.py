from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eventclock.run_massspecgym_pair_smoke import (
    binned_pair_features,
    conservation_pair_features,
    load_rows,
    make_pairs,
    peakset_pair_features,
    ridge_binary,
)


def binned_cosine(a: dict, b: dict, bins: int = 256, max_mz: float = 1200.0) -> float:
    edges = np.linspace(0.0, max_mz, bins + 1)
    ha, _ = np.histogram(a["mzs"], bins=edges, weights=a["intensities"])
    hb, _ = np.histogram(b["mzs"], bins=edges, weights=b["intensities"])
    denom = max(float(np.linalg.norm(ha) * np.linalg.norm(hb)), 1e-8)
    return float(np.dot(ha, hb) / denom)


def shift_components(a: dict, b: dict, shift: float, tolerance: float = 0.03) -> tuple[float, float, float]:
    diff = b["mzs"][:, None] - a["mzs"][None, :]
    close = np.abs(diff - shift) <= tolerance
    if not close.any():
        return 0.0, 0.0, 0.0
    intensity_outer = b["intensities"][:, None] * a["intensities"][None, :]
    coverage = 0.5 * (close.any(axis=0).mean() + close.any(axis=1).mean())
    count = float(close.sum()) / max(len(a["mzs"]) + len(b["mzs"]), 1)
    intensity = float(intensity_outer[close].sum())
    return count, intensity, float(intensity * coverage)


def zero_shift_components(a: dict, b: dict, tolerance: float = 0.03) -> tuple[float, float, float]:
    return shift_components(a, b, 0.0, tolerance)


def mcft_zero_shift_count(a: dict, b: dict, tolerance: float = 0.03) -> float:
    return zero_shift_components(a, b, tolerance)[0]


def mcft_zero_shift_intensity(a: dict, b: dict, tolerance: float = 0.03) -> float:
    return zero_shift_components(a, b, tolerance)[1]


def mcft_zero_shift_score(a: dict, b: dict, tolerance: float = 0.03) -> float:
    return zero_shift_components(a, b, tolerance)[2]


def mcft_precursor_shift_score(a: dict, b: dict, tolerance: float = 0.03) -> float:
    return shift_components(a, b, b["precursor_mz"] - a["precursor_mz"], tolerance)[2]


def mcft_best_shift_score(a: dict, b: dict, tolerance: float = 0.03) -> float:
    return max(mcft_zero_shift_score(a, b, tolerance), mcft_precursor_shift_score(a, b, tolerance))


def modified_cosine_score(a: dict, b: dict, tolerance: float = 0.03) -> float:
    diff = b["mzs"][:, None] - a["mzs"][None, :]
    precursor_shift = b["precursor_mz"] - a["precursor_mz"]
    close = (np.abs(diff) <= tolerance) | (np.abs(diff - precursor_shift) <= tolerance)
    if not close.any():
        return 0.0
    intensity_outer = b["intensities"][:, None] * a["intensities"][None, :]
    denom = max(float(np.linalg.norm(a["intensities"]) * np.linalg.norm(b["intensities"])), 1e-8)
    return float(intensity_outer[close].sum() / denom)


def make_cross_adduct_pairs(rows: list[dict], seed: int, max_pairs: int) -> tuple[list[tuple[dict, dict]], np.ndarray]:
    rng = np.random.default_rng(seed)
    by_key: dict[str, list[dict]] = {}
    for row in rows:
        by_key.setdefault(row["inchikey"], []).append(row)

    positives = []
    for group in by_key.values():
        for a in group[:12]:
            for b in group[:12]:
                if a["identifier"] != b["identifier"] and a["adduct"] != b["adduct"]:
                    positives.append((a, b))
    rng.shuffle(positives)
    positives = positives[:max_pairs]

    all_rows = rows[:]
    negatives = []
    tries = 0
    while len(negatives) < len(positives) and tries < max(len(positives) * 200, 1):
        tries += 1
        a = all_rows[int(rng.integers(0, len(all_rows)))]
        b = all_rows[int(rng.integers(0, len(all_rows)))]
        if a["identifier"] == b["identifier"] or a["inchikey"] == b["inchikey"]:
            continue
        if a["adduct"] == b["adduct"]:
            continue
        if abs(a["precursor_mz"] - b["precursor_mz"]) > 180.0:
            continue
        negatives.append((a, b))

    pairs = positives + negatives
    y = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int64)
    idx = rng.permutation(len(pairs))
    return [pairs[i] for i in idx], y[idx]


def fit_learned_mcft(
    rows: list[dict], seed: int, max_pairs: int, positive_adduct: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool] | None:
    train_rows = [row for row in rows if row.get("fold") == "train"]
    if len({row["inchikey"] for row in train_rows}) < 4:
        return None
    include_dynamic = positive_adduct == "different"
    if positive_adduct == "different":
        pairs, y = make_cross_adduct_pairs(train_rows, seed + 10_000, max_pairs)
    else:
        pairs, y = make_pairs(train_rows, seed + 10_000, max_pairs)
    if len(pairs) < 100 or len(np.unique(y)) < 2:
        return None
    x = conservation_pair_features(pairs, include_dynamic=include_dynamic)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    xz = (x - mean) / std
    coef = ridge_binary(xz, y)
    return coef, mean, std, include_dynamic


def fit_learned_binned(rows: list[dict], seed: int, max_pairs: int, positive_adduct: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    train_rows = [row for row in rows if row.get("fold") == "train"]
    if len({row["inchikey"] for row in train_rows}) < 4:
        return None
    if positive_adduct == "different":
        pairs, y = make_cross_adduct_pairs(train_rows, seed + 20_000, max_pairs)
    else:
        pairs, y = make_pairs(train_rows, seed + 20_000, max_pairs)
    if len(pairs) < 100 or len(np.unique(y)) < 2:
        return None
    x = binned_pair_features(pairs)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    xz = (x - mean) / std
    coef = ridge_binary(xz, y)
    return coef, mean, std


def fit_learned_peakset(rows: list[dict], seed: int, max_pairs: int, positive_adduct: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    train_rows = [row for row in rows if row.get("fold") == "train"]
    if len({row["inchikey"] for row in train_rows}) < 4:
        return None
    if positive_adduct == "different":
        pairs, y = make_cross_adduct_pairs(train_rows, seed + 30_000, max_pairs)
    else:
        pairs, y = make_pairs(train_rows, seed + 30_000, max_pairs)
    if len(pairs) < 100 or len(np.unique(y)) < 2:
        return None
    x = peakset_pair_features(pairs)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    xz = (x - mean) / std
    coef = ridge_binary(xz, y)
    return coef, mean, std


def learned_binned_scores(q: dict, candidates: list[dict], model: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> list[float]:
    if model is None:
        return [0.0 for _ in candidates]
    coef, mean, std = model
    pairs = [(q, c) for c in candidates]
    x = binned_pair_features(pairs)
    xz = (x - mean) / std
    x_aug = np.concatenate([xz, np.ones((len(xz), 1), dtype=xz.dtype)], axis=1).astype(np.float64)
    return np.einsum("ni,i->n", x_aug, coef).astype(float).tolist()


def learned_peakset_scores(q: dict, candidates: list[dict], model: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> list[float]:
    if model is None:
        return [0.0 for _ in candidates]
    coef, mean, std = model
    pairs = [(q, c) for c in candidates]
    x = peakset_pair_features(pairs)
    xz = (x - mean) / std
    x_aug = np.concatenate([xz, np.ones((len(xz), 1), dtype=xz.dtype)], axis=1).astype(np.float64)
    return np.einsum("ni,i->n", x_aug, coef).astype(float).tolist()


def learned_mcft_scores(
    q: dict, candidates: list[dict], model: tuple[np.ndarray, np.ndarray, np.ndarray, bool] | None
) -> list[float]:
    if model is None:
        return [0.0 for _ in candidates]
    coef, mean, std, include_dynamic = model
    pairs = [(q, c) for c in candidates]
    x = conservation_pair_features(pairs, include_dynamic=include_dynamic)
    xz = (x - mean) / std
    x_aug = np.concatenate([xz, np.ones((len(xz), 1), dtype=xz.dtype)], axis=1).astype(np.float64)
    return np.einsum("ni,i->n", x_aug, coef).astype(float).tolist()


def build_queries(
    rows: list[dict],
    seed: int,
    num_queries: int,
    num_negatives: int,
    query_folds: set[str] | None = None,
    candidate_folds: set[str] | None = None,
    positive_adduct: str = "any",
    negative_strategy: str = "random",
    negative_window: float = 120.0,
) -> list[tuple[dict, list[dict], int]]:
    rng = np.random.default_rng(seed)
    query_pool = [row for row in rows if query_folds is None or row.get("fold") in query_folds]
    candidate_pool = [row for row in rows if candidate_folds is None or row.get("fold") in candidate_folds]
    by_key: dict[str, list[dict]] = {}
    for row in query_pool:
        by_key.setdefault(row["inchikey"], []).append(row)
    def is_positive(q: dict, c: dict) -> bool:
        if c["inchikey"] != q["inchikey"] or c["identifier"] == q["identifier"]:
            return False
        if positive_adduct == "same" and c["adduct"] != q["adduct"]:
            return False
        if positive_adduct == "different" and c["adduct"] == q["adduct"]:
            return False
        return True

    eligible = [group for group in by_key.values() if any(any(is_positive(q, c) for c in candidate_pool) for q in group)]
    if not eligible:
        return []
    queries = []
    attempts = 0
    while len(queries) < num_queries and attempts < num_queries * 200:
        attempts += 1
        group = eligible[int(rng.integers(0, len(eligible)))]
        q = rng.choice(group)
        positives = [row for row in candidate_pool if is_positive(q, row)]
        if not positives:
            continue
        pos = rng.choice(positives)
        negative_adduct = pos["adduct"] if positive_adduct == "different" else q["adduct"]
        negative_precursor = pos["precursor_mz"] if positive_adduct == "different" else q["precursor_mz"]
        negatives = [
            row
            for row in candidate_pool
            if row["inchikey"] != q["inchikey"]
            and row["adduct"] == negative_adduct
            and abs(row["precursor_mz"] - negative_precursor) <= negative_window
        ]
        if len(negatives) < num_negatives:
            continue
        if negative_strategy == "closest":
            negatives.sort(key=lambda row: abs(row["precursor_mz"] - negative_precursor))
            neg = negatives[:num_negatives]
        else:
            neg = list(rng.choice(negatives, size=num_negatives, replace=False))
        candidates = [pos] + neg
        order = rng.permutation(len(candidates))
        label = int(np.where(order == 0)[0][0])
        queries.append((q, [candidates[i] for i in order], label))
    return queries


def parse_folds(text: str) -> set[str] | None:
    if text.lower() in {"", "all", "none"}:
        return None
    return {x.strip() for x in text.split(",") if x.strip()}


def run(
    seed: int,
    rows: list[dict],
    num_queries: int,
    num_negatives: int,
    query_folds: set[str] | None,
    candidate_folds: set[str] | None,
    tolerance: float,
    learned_pairs: int,
    positive_adduct: str,
    negative_strategy: str,
    negative_window: float,
) -> dict[str, float]:
    queries = build_queries(
        rows,
        seed,
        num_queries,
        num_negatives,
        query_folds,
        candidate_folds,
        positive_adduct,
        negative_strategy,
        negative_window,
    )
    learned_model = fit_learned_mcft(rows, seed, learned_pairs, positive_adduct) if learned_pairs > 0 else None
    learned_binned_model = fit_learned_binned(rows, seed, learned_pairs, positive_adduct) if learned_pairs > 0 else None
    learned_peakset_model = fit_learned_peakset(rows, seed, learned_pairs, positive_adduct) if learned_pairs > 0 else None
    hits: dict[str, int] = {}
    ranks: dict[str, list[int]] = {}
    for q, candidates, label in queries:
        scores = {
            "binned_cosine": [binned_cosine(q, c) for c in candidates],
            "modified_cosine": [modified_cosine_score(q, c, tolerance=tolerance) for c in candidates],
            "mcft_count": [mcft_zero_shift_count(q, c, tolerance=tolerance) for c in candidates],
            "mcft_intensity": [mcft_zero_shift_intensity(q, c, tolerance=tolerance) for c in candidates],
            "mcft_zero_shift": [mcft_zero_shift_score(q, c, tolerance=tolerance) for c in candidates],
            "mcft_precursor_shift": [mcft_precursor_shift_score(q, c, tolerance=tolerance) for c in candidates],
            "mcft_best_shift": [mcft_best_shift_score(q, c, tolerance=tolerance) for c in candidates],
        }
        if learned_model is not None:
            scores["learned_mcft_pair"] = learned_mcft_scores(q, candidates, learned_model)
        if learned_binned_model is not None:
            scores["learned_binned_pair"] = learned_binned_scores(q, candidates, learned_binned_model)
        if learned_peakset_model is not None:
            scores["learned_peakset_pair"] = learned_peakset_scores(q, candidates, learned_peakset_model)
        if not hits:
            hits = {name: 0 for name in scores}
            ranks = {name: [] for name in scores}
        for name, vals in scores.items():
            order = np.argsort(vals)[::-1]
            rank = int(np.where(order == label)[0][0]) + 1
            hits[name] += int(rank == 1)
            ranks[name].append(rank)
    out = {"num_queries": float(len(queries)), "num_candidates": float(num_negatives + 1)}
    for name in hits:
        out[f"{name}_hit1"] = float(hits[name] / max(len(queries), 1))
        out[f"{name}_mrr"] = float(np.mean([1.0 / r for r in ranks[name]])) if ranks[name] else float("nan")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", default="experiment/data/massspecgym/MassSpecGym.tsv")
    parser.add_argument("--out-dir", default="experiment/outputs/massspecgym_retrieval_smoke")
    parser.add_argument("--num-queries", type=int, default=500)
    parser.add_argument("--num-negatives", type=int, default=20)
    parser.add_argument("--query-folds", default="all")
    parser.add_argument("--candidate-folds", default="all")
    parser.add_argument("--positive-adduct", choices=["any", "same", "different"], default="any")
    parser.add_argument("--negative-strategy", choices=["random", "closest"], default="random")
    parser.add_argument("--negative-window", type=float, default=120.0)
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--learned-pairs", type=int, default=0)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args()
    rows = load_rows(Path(args.tsv))
    query_folds = parse_folds(args.query_folds)
    candidate_folds = parse_folds(args.candidate_folds)
    results = []
    for seed in [int(s) for s in args.seeds.split(",") if s]:
        row = run(
            seed,
            rows,
            args.num_queries,
            args.num_negatives,
            query_folds,
            candidate_folds,
            args.tolerance,
            args.learned_pairs,
            args.positive_adduct,
            args.negative_strategy,
            args.negative_window,
        )
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
