from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from eventclock.run_massspecgym_pair_smoke import load_rows
from eventclock.run_massspecgym_retrieval_smoke import (
    build_queries,
    mcft_zero_shift_count,
    mcft_zero_shift_score,
    modified_cosine_score,
    parse_folds,
)


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(arr.mean()),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def audit_seed(args: argparse.Namespace, rows: list[dict[str, Any]], seed: int) -> dict[str, float]:
    queries = build_queries(
        rows,
        seed,
        args.num_queries,
        args.num_negatives,
        parse_folds(args.query_folds),
        parse_folds(args.candidate_folds),
        args.positive_adduct,
        args.negative_strategy,
        args.negative_window,
    )
    precursor_gaps: list[float] = []
    neg_zero_scores: list[float] = []
    neg_zero_counts: list[float] = []
    neg_modcos: list[float] = []
    max_neg_zero_scores: list[float] = []
    max_neg_modcos: list[float] = []
    pos_zero_scores: list[float] = []
    pos_modcos: list[float] = []

    for query, candidates, label in queries:
        pos = candidates[label]
        negs = [candidate for idx, candidate in enumerate(candidates) if idx != label]
        pos_zero_scores.append(mcft_zero_shift_score(query, pos, args.tolerance))
        pos_modcos.append(modified_cosine_score(query, pos, args.tolerance))
        seed_neg_zero_scores = []
        seed_neg_modcos = []
        for neg in negs:
            precursor_gaps.append(abs(float(query["precursor_mz"]) - float(neg["precursor_mz"])))
            zero_score = mcft_zero_shift_score(query, neg, args.tolerance)
            modcos = modified_cosine_score(query, neg, args.tolerance)
            neg_zero_scores.append(zero_score)
            neg_zero_counts.append(mcft_zero_shift_count(query, neg, args.tolerance))
            neg_modcos.append(modcos)
            seed_neg_zero_scores.append(zero_score)
            seed_neg_modcos.append(modcos)
        if seed_neg_zero_scores:
            max_neg_zero_scores.append(float(max(seed_neg_zero_scores)))
            max_neg_modcos.append(float(max(seed_neg_modcos)))

    row: dict[str, float] = {
        "seed": float(seed),
        "num_queries": float(len(queries)),
        "num_negatives": float(args.num_negatives),
    }
    for prefix, values in [
        ("negative_precursor_gap", precursor_gaps),
        ("negative_zero_score", neg_zero_scores),
        ("negative_zero_count", neg_zero_counts),
        ("negative_modified_cosine", neg_modcos),
        ("max_negative_zero_score", max_neg_zero_scores),
        ("max_negative_modified_cosine", max_neg_modcos),
        ("positive_zero_score", pos_zero_scores),
        ("positive_modified_cosine", pos_modcos),
    ]:
        for key, value in summarize(values).items():
            row[f"{prefix}_{key}"] = value
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", default="experiment/data/massspecgym/MassSpecGym_rows_25k.tsv")
    parser.add_argument("--out-dir", default="experiment/outputs/massspecgym_hardness_audit")
    parser.add_argument("--num-queries", type=int, default=300)
    parser.add_argument("--num-negatives", type=int, default=500)
    parser.add_argument("--query-folds", default="val,test")
    parser.add_argument("--candidate-folds", default="val,test")
    parser.add_argument("--positive-adduct", choices=["any", "same", "different"], default="any")
    parser.add_argument("--negative-strategy", choices=["random", "closest", "overlap"], default="random")
    parser.add_argument("--negative-window", type=float, default=120.0)
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()

    rows = load_rows(Path(args.tsv))
    results = [audit_seed(args, rows, int(seed)) for seed in args.seeds.split(",") if seed]
    scalar_keys = [key for key in results[0] if key != "seed"]
    mean = {key: float(np.mean([row[key] for row in results])) for key in scalar_keys}
    stderr95 = {
        key: float(1.96 * np.std([row[key] for row in results], ddof=1) / np.sqrt(len(results)))
        for key in scalar_keys
        if len(results) > 1
    }
    payload = {"metadata": {**vars(args), "num_rows": len(rows)}, "rows": results, "mean": mean, "stderr95": stderr95}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
