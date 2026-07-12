from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from eventclock.run_massspecgym_pair_smoke import load_rows
from eventclock.run_massspecgym_retrieval_smoke import (
    binned_cosine,
    build_queries,
    fit_learned_binned,
    fit_learned_mcft,
    fit_learned_peakset,
    learned_binned_scores,
    learned_mcft_scores,
    learned_peakset_scores,
    mcft_precursor_shift_score,
    mcft_zero_shift_score,
    modified_cosine_score,
    parse_folds,
)


def attach_formula(rows: list[dict], tsv: Path) -> list[dict]:
    df = pd.read_csv(tsv, sep="\t", on_bad_lines="skip")
    formulas = {r.identifier: str(r.formula) for r in df.itertuples()}
    out = []
    for row in rows:
        if row["identifier"] in formulas:
            copied = dict(row)
            copied["formula"] = formulas[row["identifier"]]
            out.append(copied)
    return out


def build_formula_queries(
    rows: list[dict],
    seed: int,
    num_queries: int,
    num_negatives: int,
    query_folds: set[str] | None,
    candidate_folds: set[str] | None,
) -> list[tuple[dict, list[dict], int]]:
    rng = np.random.default_rng(seed)
    query_pool = [r for r in rows if query_folds is None or r["fold"] in query_folds]
    candidate_pool = [r for r in rows if candidate_folds is None or r["fold"] in candidate_folds]
    by_formula_adduct: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in candidate_pool:
        by_formula_adduct[(row["formula"], row["adduct"])].append(row)

    eligible = []
    for q in query_pool:
        positives = [
            r
            for r in candidate_pool
            if r["inchikey"] == q["inchikey"] and r["identifier"] != q["identifier"]
        ]
        negatives = [
            r
            for r in by_formula_adduct[(q["formula"], q["adduct"])]
            if r["inchikey"] != q["inchikey"]
        ]
        if positives and len(negatives) >= num_negatives:
            eligible.append((q, positives, negatives))
    if not eligible:
        return []

    queries = []
    for _ in range(num_queries):
        q, positives, negatives = eligible[int(rng.integers(0, len(eligible)))]
        pos = positives[int(rng.integers(0, len(positives)))]
        neg = list(rng.choice(negatives, size=num_negatives, replace=False))
        candidates = [pos] + neg
        order = rng.permutation(len(candidates))
        label = int(np.where(order == 0)[0][0])
        queries.append((q, [candidates[i] for i in order], label))
    return queries


def run(
    seed: int,
    rows: list[dict],
    num_queries: int,
    num_negatives: int,
    query_folds: set[str] | None,
    candidate_folds: set[str] | None,
    tolerance: float,
    learned_pairs: int,
) -> dict[str, float]:
    queries = build_formula_queries(rows, seed, num_queries, num_negatives, query_folds, candidate_folds)
    learned_mcft = fit_learned_mcft(rows, seed, learned_pairs, "any") if learned_pairs > 0 else None
    learned_binned = fit_learned_binned(rows, seed, learned_pairs, "any") if learned_pairs > 0 else None
    learned_peakset = fit_learned_peakset(rows, seed, learned_pairs, "any") if learned_pairs > 0 else None
    hits: dict[str, int] = {}
    ranks: dict[str, list[int]] = {}
    for q, candidates, label in queries:
        scores = {
            "binned_cosine": [binned_cosine(q, c) for c in candidates],
            "modified_cosine": [modified_cosine_score(q, c, tolerance=tolerance) for c in candidates],
            "mcft_zero_shift": [mcft_zero_shift_score(q, c, tolerance=tolerance) for c in candidates],
            "mcft_precursor_shift": [mcft_precursor_shift_score(q, c, tolerance=tolerance) for c in candidates],
        }
        if learned_binned is not None:
            scores["learned_binned_pair"] = learned_binned_scores(q, candidates, learned_binned)
        if learned_peakset is not None:
            scores["learned_peakset_pair"] = learned_peakset_scores(q, candidates, learned_peakset)
        if learned_mcft is not None:
            scores["learned_mcft_pair"] = learned_mcft_scores(q, candidates, learned_mcft)
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
    parser.add_argument("--tsv", default="experiment/data/massspecgym/MassSpecGym_rows_10k.tsv")
    parser.add_argument("--out-dir", default="experiment/outputs/massspecgym_formula_conditioned_retrieval")
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--num-negatives", type=int, default=5)
    parser.add_argument("--query-folds", default="val,test")
    parser.add_argument("--candidate-folds", default="all")
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--learned-pairs", type=int, default=12000)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args()
    rows = attach_formula(load_rows(Path(args.tsv)), Path(args.tsv))
    query_folds = parse_folds(args.query_folds)
    candidate_folds = parse_folds(args.candidate_folds)
    results = []
    for seed in [int(s) for s in args.seeds.split(",") if s]:
        row = run(seed, rows, args.num_queries, args.num_negatives, query_folds, candidate_folds, args.tolerance, args.learned_pairs)
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
