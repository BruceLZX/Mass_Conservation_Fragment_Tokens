from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eventclock.run_massspecgym_pair_smoke import load_rows
from eventclock.run_massspecgym_retrieval_smoke import (
    build_queries,
    fit_learned_mcft,
    learned_mcft_scores,
    mcft_precursor_shift_score,
    mcft_zero_shift_score,
    modified_cosine_score,
)


def top_matches(q: dict, c: dict, shift: float, tolerance: float, limit: int) -> list[dict]:
    diff = c["mzs"][:, None] - q["mzs"][None, :]
    close = np.abs(diff - shift) <= tolerance
    matches = []
    for j, i in np.argwhere(close):
        matches.append(
            {
                "query_mz": float(q["mzs"][i]),
                "candidate_mz": float(c["mzs"][j]),
                "delta": float(c["mzs"][j] - q["mzs"][i]),
                "query_intensity": float(q["intensities"][i]),
                "candidate_intensity": float(c["intensities"][j]),
                "weight": float(q["intensities"][i] * c["intensities"][j]),
            }
        )
    matches.sort(key=lambda x: x["weight"], reverse=True)
    return matches[:limit]


def rank_of(scores: list[float], label: int) -> int:
    order = np.argsort(scores)[::-1]
    return int(np.where(order == label)[0][0]) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", default="experiment/data/massspecgym/MassSpecGym_rows_10k.tsv")
    parser.add_argument("--out-dir", default="experiment/outputs/massspecgym_10k_evidence_examples")
    parser.add_argument("--num-queries", type=int, default=300)
    parser.add_argument("--num-negatives", type=int, default=500)
    parser.add_argument("--query-folds", default="val,test")
    parser.add_argument("--candidate-folds", default="val,test")
    parser.add_argument("--learned-pairs", type=int, default=12000)
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    rows = load_rows(Path(args.tsv))
    query_folds = {x for x in args.query_folds.split(",") if x}
    candidate_folds = {x for x in args.candidate_folds.split(",") if x}
    queries = build_queries(rows, args.seed, args.num_queries, args.num_negatives, query_folds, candidate_folds, "any")
    model = fit_learned_mcft(rows, args.seed, args.learned_pairs, "any")

    examples = []
    summary = {
        "queries": len(queries),
        "learned_mcft_top1_modified_not_top1": 0,
        "zero_or_precursor_top1_modified_not_top1": 0,
    }
    for q, candidates, label in queries:
        modified = [modified_cosine_score(q, c, args.tolerance) for c in candidates]
        learned = learned_mcft_scores(q, candidates, model)
        zero = [mcft_zero_shift_score(q, c, args.tolerance) for c in candidates]
        precursor = [mcft_precursor_shift_score(q, c, args.tolerance) for c in candidates]
        best_structural = [max(a, b) for a, b in zip(zero, precursor)]
        modified_rank = rank_of(modified, label)
        learned_rank = rank_of(learned, label)
        structural_rank = rank_of(best_structural, label)
        if learned_rank == 1 and modified_rank > 1:
            summary["learned_mcft_top1_modified_not_top1"] += 1
        if structural_rank == 1 and modified_rank > 1:
            summary["zero_or_precursor_top1_modified_not_top1"] += 1
        if learned_rank == 1 and modified_rank > 1 and len(examples) < args.limit:
            pos = candidates[label]
            best_neg_idx = int(np.argmax([s if i != label else -np.inf for i, s in enumerate(learned)]))
            neg = candidates[best_neg_idx]
            shift = pos["precursor_mz"] - q["precursor_mz"]
            examples.append(
                {
                    "query": q["identifier"],
                    "positive": pos["identifier"],
                    "hard_negative_by_learned_mcft": neg["identifier"],
                    "adduct": q["adduct"],
                    "query_fold": q["fold"],
                    "positive_fold": pos["fold"],
                    "modified_rank": modified_rank,
                    "learned_mcft_rank": learned_rank,
                    "structural_rank": structural_rank,
                    "positive_scores": {
                        "modified_cosine": float(modified[label]),
                        "learned_mcft": float(learned[label]),
                        "mcft_zero": float(zero[label]),
                        "mcft_precursor_shift": float(precursor[label]),
                        "precursor_shift": float(shift),
                    },
                    "hard_negative_scores": {
                        "modified_cosine": float(modified[best_neg_idx]),
                        "learned_mcft": float(learned[best_neg_idx]),
                        "mcft_zero": float(zero[best_neg_idx]),
                        "mcft_precursor_shift": float(precursor[best_neg_idx]),
                        "precursor_shift": float(neg["precursor_mz"] - q["precursor_mz"]),
                    },
                    "top_zero_shift_matches": top_matches(q, pos, 0.0, args.tolerance, 8),
                    "top_precursor_shift_matches": top_matches(q, pos, shift, args.tolerance, 8),
                }
            )

    payload = {"metadata": vars(args), "summary": summary, "examples": examples}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "examples.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
