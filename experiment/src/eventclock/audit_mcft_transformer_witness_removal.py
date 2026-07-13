from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from eventclock.audit_mcft_evidence_examples import rank_of
from eventclock.run_massspecgym_mcft_transformer import (
    ConservationTokenTransformer,
    TOKEN_DIM,
    parse_folds,
    require_torch,
    score_query,
)
from eventclock.run_massspecgym_pair_smoke import load_rows
from eventclock.run_massspecgym_retrieval_smoke import build_queries

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def top_zero_shift_candidate_indices(q: dict, c: dict, top_k: int, tolerance: float) -> list[int]:
    diff = c["mzs"][:, None] - q["mzs"][None, :]
    close = np.abs(diff) <= tolerance
    intensity_outer = c["intensities"][:, None] * q["intensities"][None, :]
    witnesses: list[tuple[float, int]] = []
    for cand_idx, query_idx in np.argwhere(close):
        residual = abs(float(diff[cand_idx, query_idx]))
        evidence = float(intensity_outer[cand_idx, query_idx]) - 0.05 * residual
        witnesses.append((evidence, int(cand_idx)))
    witnesses.sort(reverse=True)
    chosen: list[int] = []
    for _, cand_idx in witnesses:
        if cand_idx not in chosen:
            chosen.append(cand_idx)
        if len(chosen) >= top_k:
            break
    return chosen


def remove_candidate_peaks(candidate: dict, indices: list[int]) -> dict:
    if not indices:
        return candidate
    keep = np.ones(len(candidate["mzs"]), dtype=bool)
    keep[np.asarray(indices, dtype=int)] = False
    mutated = dict(candidate)
    mutated["mzs"] = candidate["mzs"][keep]
    mutated["intensities"] = candidate["intensities"][keep]
    return mutated


def metrics_from_ranks(ranks: list[int]) -> dict[str, float]:
    return {
        "hit1": float(np.mean([rank == 1 for rank in ranks])) if ranks else float("nan"),
        "mean_rank": float(np.mean(ranks)) if ranks else float("nan"),
        "mrr": float(np.mean([1.0 / rank for rank in ranks])) if ranks else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tsv", default="experiment/data/massspecgym/MassSpecGym_rows_10k.tsv")
    parser.add_argument("--out-dir", default="experiment/outputs/mcft_transformer_witness_removal_audit")
    parser.add_argument("--eval-queries", type=int, default=300)
    parser.add_argument("--eval-negatives", type=int, default=500)
    parser.add_argument("--query-folds", default="val,test")
    parser.add_argument("--candidate-folds", default="val,test")
    parser.add_argument("--positive-adduct", choices=["any", "same", "different"], default="any")
    parser.add_argument("--negative-strategy", choices=["random", "closest", "overlap"], default="random")
    parser.add_argument("--negative-window", type=float, default=120.0)
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--tolerances", default="0.01,0.03,0.08")
    parser.add_argument("--include-precursor-shift", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--random-trials", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    require_torch()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    metadata = checkpoint.get("metadata", {})
    for key in ["max_tokens", "width", "depth", "heads", "dropout", "include_precursor_shift", "tolerances"]:
        if key in metadata and getattr(args, key) == parser.get_default(key):
            setattr(args, key, metadata[key])
    args.tolerances_tuple = tuple(float(x) for x in str(args.tolerances).split(",") if x)

    rows = load_rows(Path(args.tsv))
    queries = build_queries(
        rows,
        args.seed,
        args.eval_queries,
        args.eval_negatives,
        parse_folds(args.query_folds),
        parse_folds(args.candidate_folds),
        args.positive_adduct,
        args.negative_strategy,
        args.negative_window,
    )

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = ConservationTokenTransformer(TOKEN_DIM, int(args.width), int(args.depth), int(args.heads), float(args.dropout))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    rng = np.random.default_rng(args.seed + 17_003)
    original_ranks: list[int] = []
    top_ranks: list[int] = []
    random_ranks: list[int] = []
    score_drops_top: list[float] = []
    score_drops_random: list[float] = []
    evaluated = 0
    skipped_no_witness = 0

    for q, candidates, label in queries:
        original_scores = score_query(model, q, candidates, args, device)
        original_rank = rank_of(original_scores, label)
        if original_rank != 1:
            continue
        positive = candidates[label]
        top_indices = top_zero_shift_candidate_indices(q, positive, args.top_k, args.tolerance)
        if not top_indices:
            skipped_no_witness += 1
            continue

        top_candidates = list(candidates)
        top_candidates[label] = remove_candidate_peaks(positive, top_indices)
        top_scores = score_query(model, q, top_candidates, args, device)
        original_ranks.append(original_rank)
        top_ranks.append(rank_of(top_scores, label))
        score_drops_top.append(float(original_scores[label] - top_scores[label]))

        removable = list(range(len(positive["mzs"])))
        trial_ranks = []
        trial_drops = []
        for _ in range(args.random_trials):
            if len(removable) <= len(top_indices):
                sampled = removable
            else:
                sampled = rng.choice(removable, size=len(top_indices), replace=False).astype(int).tolist()
            random_candidates = list(candidates)
            random_candidates[label] = remove_candidate_peaks(positive, sampled)
            random_scores = score_query(model, q, random_candidates, args, device)
            trial_ranks.append(rank_of(random_scores, label))
            trial_drops.append(float(original_scores[label] - random_scores[label]))
        random_ranks.extend(trial_ranks)
        score_drops_random.extend(trial_drops)
        evaluated += 1

    payload: dict[str, Any] = {
        "metadata": {**vars(args), "num_rows": len(rows), "num_queries_built": len(queries), "device_used": str(device)},
        "metrics": {
            "evaluated_initial_hit1_cases": evaluated,
            "random_deletion_trials": len(random_ranks),
            "skipped_no_zero_shift_witness": skipped_no_witness,
            "original": metrics_from_ranks(original_ranks),
            "top_witness_deletion": metrics_from_ranks(top_ranks),
            "random_deletion": metrics_from_ranks(random_ranks),
            "mean_score_drop_top": float(np.mean(score_drops_top)) if score_drops_top else float("nan"),
            "mean_score_drop_random": float(np.mean(score_drops_random)) if score_drops_random else float("nan"),
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
