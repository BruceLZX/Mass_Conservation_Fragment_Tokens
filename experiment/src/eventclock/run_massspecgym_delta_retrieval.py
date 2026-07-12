from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from eventclock.run_massspecgym_pair_smoke import (
    binned_pair_features,
    conservation_pair_features,
    load_rows,
    peakset_pair_features,
    ridge_binary,
)
from eventclock.run_massspecgym_retrieval_smoke import binned_cosine, modified_cosine_score, shift_components


ELEMENT_RE = re.compile(r"([A-Z][a-z]*)(\d*)")
TRANSFORMS = {
    "+O": ({"O": 1}, 15.99491462),
    "+CH2": ({"C": 1, "H": 2}, 14.01565006),
    "+H2": ({"H": 2}, 2.01565006),
    "+CO": ({"C": 1, "O": 1}, 27.99491462),
    "+CO2": ({"C": 1, "O": 2}, 43.98982924),
    "+C2H2": ({"C": 2, "H": 2}, 26.01565006),
}


def parse_formula(text: str) -> dict[str, int]:
    return {elem: int(count or 1) for elem, count in ELEMENT_RE.findall(str(text))}


def formula_diff(a: str, b: str) -> dict[str, int]:
    aa = parse_formula(a)
    bb = parse_formula(b)
    out = {}
    for elem in set(aa) | set(bb):
        val = bb.get(elem, 0) - aa.get(elem, 0)
        if val:
            out[elem] = val
    return out


def formula_key(counts: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((elem, val) for elem, val in counts.items() if val))


def add_formula_delta(counts: dict[str, int], delta: dict[str, int]) -> tuple[tuple[str, int], ...]:
    out = dict(counts)
    for elem, val in delta.items():
        out[elem] = out.get(elem, 0) + val
        if out[elem] == 0:
            del out[elem]
    return formula_key(out)


def attach_formula(rows: list[dict], tsv: Path) -> list[dict]:
    df = pd.read_csv(tsv, sep="\t", on_bad_lines="skip")
    meta = {r.identifier: str(r.formula) for r in df.itertuples()}
    out = []
    for row in rows:
        if row["identifier"] not in meta:
            continue
        copied = dict(row)
        copied["formula"] = meta[row["identifier"]]
        copied["formula_counts"] = parse_formula(copied["formula"])
        copied["formula_key"] = formula_key(copied["formula_counts"])
        out.append(copied)
    return out


def is_delta_pair(a: dict, b: dict, delta_formula: dict[str, int]) -> bool:
    return (
        a["inchikey"] != b["inchikey"]
        and a["adduct"] == b["adduct"]
        and b["formula_key"] == add_formula_delta(a["formula_counts"], delta_formula)
    )


def make_delta_pairs(
    rows: list[dict],
    seed: int,
    max_pairs: int,
    delta_formula: dict[str, int],
    delta_mass: float,
    negative_window: float,
) -> tuple[list[tuple[dict, dict]], np.ndarray]:
    rng = np.random.default_rng(seed)
    by_adduct_formula: dict[tuple[str, tuple[tuple[str, int], ...]], list[dict]] = defaultdict(list)
    for row in rows:
        by_adduct_formula[(row["adduct"], row["formula_key"])].append(row)
    positives = []
    for a in rows:
        target = add_formula_delta(a["formula_counts"], delta_formula)
        for b in by_adduct_formula.get((a["adduct"], target), []):
            if a["inchikey"] != b["inchikey"]:
                positives.append((a, b))
    rng.shuffle(positives)
    positives = positives[:max_pairs]

    negative_pool = []
    for a, _ in positives:
        local = [
            (abs((b["precursor_mz"] - a["precursor_mz"]) - delta_mass), a, b)
            for b in rows
            if a["inchikey"] != b["inchikey"]
            and a["adduct"] == b["adduct"]
            and not is_delta_pair(a, b, delta_formula)
            and abs((b["precursor_mz"] - a["precursor_mz"]) - delta_mass) <= negative_window
        ]
        local.sort(key=lambda item: item[0])
        negative_pool.extend((a, b) for _, a, b in local[:20])
    rng.shuffle(negative_pool)
    negatives = negative_pool[: len(positives)]

    pairs = positives + negatives
    y = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int64)
    if not pairs:
        return [], y
    order = rng.permutation(len(pairs))
    return [pairs[i] for i in order], y[order]


def fit_linear_pair_model(
    rows: list[dict],
    seed: int,
    max_pairs: int,
    delta_formula: dict[str, int],
    delta_mass: float,
    negative_window: float,
    featurizer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    pairs, y = make_delta_pairs(rows, seed, max_pairs, delta_formula, delta_mass, negative_window)
    if len(pairs) < 100 or len(np.unique(y)) < 2:
        return None
    x = featurizer(pairs)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    coef = ridge_binary((x - mean) / std, y)
    return coef, mean, std


def model_scores(q: dict, candidates: list[dict], model, featurizer) -> list[float]:
    if model is None:
        return [0.0 for _ in candidates]
    coef, mean, std = model
    x = featurizer([(q, c) for c in candidates])
    xz = (x - mean) / std
    x_aug = np.concatenate([xz, np.ones((len(xz), 1), dtype=xz.dtype)], axis=1).astype(np.float64)
    return np.einsum("ni,i->n", x_aug, coef).astype(float).tolist()


def build_queries(
    rows: list[dict],
    seed: int,
    num_queries: int,
    num_negatives: int,
    delta_formula: dict[str, int],
    delta_mass: float,
    negative_window: float,
) -> list[tuple[dict, list[dict], int]]:
    rng = np.random.default_rng(seed)
    by_adduct_formula: dict[tuple[str, tuple[tuple[str, int], ...]], list[dict]] = defaultdict(list)
    for row in rows:
        by_adduct_formula[(row["adduct"], row["formula_key"])].append(row)
    eligible = []
    for q in rows:
        target = add_formula_delta(q["formula_counts"], delta_formula)
        positives = [r for r in by_adduct_formula.get((q["adduct"], target), []) if q["inchikey"] != r["inchikey"]]
        negatives = [
            r
            for r in rows
            if q["inchikey"] != r["inchikey"]
            and q["adduct"] == r["adduct"]
            and not is_delta_pair(q, r, delta_formula)
            and abs((r["precursor_mz"] - q["precursor_mz"]) - delta_mass) <= negative_window
        ]
        if positives and len(negatives) >= num_negatives:
            eligible.append((q, positives, negatives))
    if not eligible:
        return []

    queries = []
    for _ in range(num_queries):
        q, positives, negatives = eligible[int(rng.integers(0, len(eligible)))]
        pos = positives[int(rng.integers(0, len(positives)))]
        negatives.sort(key=lambda r: abs((r["precursor_mz"] - q["precursor_mz"]) - delta_mass))
        hard_pool = negatives[: max(num_negatives * 3, num_negatives)]
        neg = list(rng.choice(hard_pool, size=num_negatives, replace=False))
        candidates = [pos] + neg
        order = rng.permutation(len(candidates))
        label = int(np.where(order == 0)[0][0])
        queries.append((q, [candidates[i] for i in order], label))
    return queries


def run(seed: int, rows: list[dict], args: argparse.Namespace) -> dict[str, float]:
    delta_formula, delta_mass = TRANSFORMS[args.transform]
    train_rows = [row for row in rows if row["fold"] == "train"]
    eval_rows = [row for row in rows if row["fold"] in set(args.eval_folds.split(","))]
    models = {
        "learned_delta_binned": fit_linear_pair_model(
            train_rows, seed + 1000, args.learned_pairs, delta_formula, delta_mass, args.negative_window, binned_pair_features
        ),
        "learned_delta_peakset": fit_linear_pair_model(
            train_rows, seed + 2000, args.learned_pairs, delta_formula, delta_mass, args.negative_window, peakset_pair_features
        ),
        "learned_delta_mcft": fit_linear_pair_model(
            train_rows, seed + 3000, args.learned_pairs, delta_formula, delta_mass, args.negative_window, conservation_pair_features
        ),
    }
    queries = build_queries(
        eval_rows, seed, args.num_queries, args.num_negatives, delta_formula, delta_mass, args.negative_window
    )
    hits: dict[str, int] = {}
    ranks: dict[str, list[int]] = {}
    for q, candidates, label in queries:
        scores = {
            "precursor_delta": [-abs((c["precursor_mz"] - q["precursor_mz"]) - delta_mass) for c in candidates],
            "binned_cosine": [binned_cosine(q, c) for c in candidates],
            "modified_cosine": [modified_cosine_score(q, c, tolerance=args.tolerance) for c in candidates],
            "mcft_delta": [shift_components(q, c, delta_mass, tolerance=args.tolerance)[2] for c in candidates],
            "mcft_zero": [shift_components(q, c, 0.0, tolerance=args.tolerance)[2] for c in candidates],
        }
        scores["learned_delta_binned"] = model_scores(q, candidates, models["learned_delta_binned"], binned_pair_features)
        scores["learned_delta_peakset"] = model_scores(q, candidates, models["learned_delta_peakset"], peakset_pair_features)
        scores["learned_delta_mcft"] = model_scores(q, candidates, models["learned_delta_mcft"], conservation_pair_features)
        if not hits:
            hits = {name: 0 for name in scores}
            ranks = {name: [] for name in scores}
        for name, values in scores.items():
            order = np.argsort(values)[::-1]
            rank = int(np.where(order == label)[0][0]) + 1
            hits[name] += int(rank == 1)
            ranks[name].append(rank)
    out = {
        "num_queries": float(len(queries)),
        "num_candidates": float(args.num_negatives + 1),
        "transform_mass": float(delta_mass),
    }
    for name in hits:
        out[f"{name}_hit1"] = float(hits[name] / max(len(queries), 1))
        out[f"{name}_mrr"] = float(np.mean([1.0 / r for r in ranks[name]])) if ranks[name] else float("nan")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", default="experiment/data/massspecgym/MassSpecGym_rows_10k.tsv")
    parser.add_argument("--out-dir", default="experiment/outputs/massspecgym_delta_retrieval")
    parser.add_argument("--transform", choices=sorted(TRANSFORMS), default="+O")
    parser.add_argument("--eval-folds", default="val,test")
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--num-negatives", type=int, default=20)
    parser.add_argument("--negative-window", type=float, default=50.0)
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--learned-pairs", type=int, default=1000)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args()

    rows = attach_formula(load_rows(Path(args.tsv)), Path(args.tsv))
    results = []
    for seed in [int(s) for s in args.seeds.split(",") if s]:
        row = run(seed, rows, args)
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
