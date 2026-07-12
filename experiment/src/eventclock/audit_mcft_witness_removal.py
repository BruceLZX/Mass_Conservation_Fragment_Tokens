from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eventclock.audit_mcft_evidence_examples import rank_of
from eventclock.run_massspecgym_pair_smoke import load_rows
from eventclock.run_massspecgym_retrieval_smoke import build_queries, fit_learned_mcft, learned_mcft_scores


def top_zero_shift_candidate_peak_indices(q: dict, c: dict, tolerance: float, top_k: int) -> list[int]:
    diff = c["mzs"][:, None] - q["mzs"][None, :]
    close = np.abs(diff) <= tolerance
    matches = []
    for candidate_idx, query_idx in np.argwhere(close):
        matches.append(
            (
                float(c["intensities"][candidate_idx] * q["intensities"][query_idx]),
                int(candidate_idx),
            )
        )
    matches.sort(reverse=True)

    out = []
    seen = set()
    for _, candidate_idx in matches:
        if candidate_idx in seen:
            continue
        seen.add(candidate_idx)
        out.append(candidate_idx)
        if len(out) >= top_k:
            break
    return out


def drop_candidate_peaks(candidate: dict, indices: list[int]) -> dict:
    if not indices:
        return candidate
    mask = np.ones(len(candidate["mzs"]), dtype=bool)
    mask[np.asarray(indices, dtype=np.int64)] = False
    out = dict(candidate)
    out["mzs"] = candidate["mzs"][mask]
    out["intensities"] = candidate["intensities"][mask]
    return out


def score_with_replaced_positive(
    q: dict, baseline_scores: list[float], label: int, replacement: dict, model: tuple
) -> list[float]:
    edited_scores = list(baseline_scores)
    edited_scores[label] = learned_mcft_scores(q, [replacement], model)[0]
    return edited_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", default="experiment/data/massspecgym/MassSpecGym_rows_10k.tsv")
    parser.add_argument("--out-dir", default="experiment/outputs/massspecgym_10k_witness_removal_audit")
    parser.add_argument("--num-queries", type=int, default=300)
    parser.add_argument("--num-negatives", type=int, default=500)
    parser.add_argument("--query-folds", default="val,test")
    parser.add_argument("--candidate-folds", default="val,test")
    parser.add_argument("--learned-pairs", type=int, default=12000)
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--random-trials", type=int, default=10)
    parser.add_argument("--limit-examples", type=int, default=12)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed + 70_000)
    rows = load_rows(Path(args.tsv))
    query_folds = {x for x in args.query_folds.split(",") if x}
    candidate_folds = {x for x in args.candidate_folds.split(",") if x}
    queries = build_queries(rows, args.seed, args.num_queries, args.num_negatives, query_folds, candidate_folds, "any")
    model = fit_learned_mcft(rows, args.seed, args.learned_pairs, "any")
    if model is None:
        raise RuntimeError("Could not fit learned MCFT model; check train-fold rows and --learned-pairs.")

    top_removed_hit1 = 0
    random_removed_hit1 = 0
    evaluated_cases = 0
    random_case_trials = 0
    baseline_positive_scores = []
    top_removed_positive_scores = []
    random_removed_positive_scores = []
    top_removed_ranks = []
    random_removed_ranks = []
    examples = []

    for q, candidates, label in queries:
        baseline_scores = learned_mcft_scores(q, candidates, model)
        baseline_rank = rank_of(baseline_scores, label)
        if baseline_rank != 1:
            continue

        positive = candidates[label]
        top_indices = top_zero_shift_candidate_peak_indices(q, positive, args.tolerance, args.top_k)
        if len(positive["mzs"]) <= 1:
            continue
        top_indices = top_indices[: max(len(positive["mzs"]) - 1, 1)]
        if not top_indices:
            continue

        evaluated_cases += 1
        top_removed_positive = drop_candidate_peaks(positive, top_indices)
        top_scores = score_with_replaced_positive(q, baseline_scores, label, top_removed_positive, model)
        top_rank = rank_of(top_scores, label)
        top_removed_hit1 += int(top_rank == 1)

        baseline_positive_scores.append(float(baseline_scores[label]))
        top_removed_positive_scores.append(float(top_scores[label]))
        top_removed_ranks.append(top_rank)

        random_trial_rows = []
        removable = np.arange(len(positive["mzs"]))
        remove_count = min(len(top_indices), max(len(positive["mzs"]) - 1, 1))
        for _ in range(args.random_trials):
            random_indices = rng.choice(removable, size=remove_count, replace=False).astype(int).tolist()
            random_positive = drop_candidate_peaks(positive, random_indices)
            random_scores = score_with_replaced_positive(q, baseline_scores, label, random_positive, model)
            random_rank = rank_of(random_scores, label)
            random_removed_hit1 += int(random_rank == 1)
            random_case_trials += 1
            random_removed_positive_scores.append(float(random_scores[label]))
            random_removed_ranks.append(random_rank)
            random_trial_rows.append(
                {
                    "removed_indices": random_indices,
                    "rank": random_rank,
                    "positive_score": float(random_scores[label]),
                }
            )

        if len(examples) < args.limit_examples:
            examples.append(
                {
                    "query": q["identifier"],
                    "positive": positive["identifier"],
                    "removed_top_candidate_peak_indices": top_indices,
                    "baseline_rank": baseline_rank,
                    "top_removed_rank": top_rank,
                    "baseline_positive_score": float(baseline_scores[label]),
                    "top_removed_positive_score": float(top_scores[label]),
                    "random_trials": random_trial_rows[:3],
                }
            )

    baseline_scores_arr = np.asarray(baseline_positive_scores, dtype=np.float64)
    top_scores_arr = np.asarray(top_removed_positive_scores, dtype=np.float64)
    random_scores_arr = np.asarray(random_removed_positive_scores, dtype=np.float64)
    top_drops = baseline_scores_arr - top_scores_arr
    random_drops = np.repeat(baseline_scores_arr, args.random_trials)[: len(random_scores_arr)] - random_scores_arr

    summary = {
        "queries": len(queries),
        "evaluated_cases_learned_mcft_initially_hit1_with_witness": evaluated_cases,
        "top_k": args.top_k,
        "random_trials_per_case": args.random_trials,
        "baseline_hit1_on_evaluated_cases": 1.0 if evaluated_cases else float("nan"),
        "top_witness_removed_hit1": float(top_removed_hit1 / max(evaluated_cases, 1)),
        "random_removed_hit1": float(random_removed_hit1 / max(random_case_trials, 1)),
        "mean_top_witness_removed_rank": float(np.mean(top_removed_ranks)) if top_removed_ranks else float("nan"),
        "mean_random_removed_rank": float(np.mean(random_removed_ranks)) if random_removed_ranks else float("nan"),
        "mean_top_witness_score_drop": float(np.mean(top_drops)) if len(top_drops) else float("nan"),
        "mean_random_score_drop": float(np.mean(random_drops)) if len(random_drops) else float("nan"),
        "median_top_witness_score_drop": float(np.median(top_drops)) if len(top_drops) else float("nan"),
        "median_random_score_drop": float(np.median(random_drops)) if len(random_drops) else float("nan"),
    }
    payload = {"metadata": vars(args) | {"num_rows": len(rows)}, "summary": summary, "examples": examples}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
