from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from eventclock.audit_mcft_evidence_examples import rank_of
from eventclock.run_massspecgym_mcft_transformer import conservation_tokens
from eventclock.run_massspecgym_mlp_stats_control import fit_mlp_stats, forward
from eventclock.run_massspecgym_pair_smoke import load_rows
from eventclock.run_massspecgym_retrieval_smoke import (
    build_queries,
    mcft_zero_shift_score,
    modified_cosine_score,
    parse_folds,
)


def pooled_token_features(
    q: dict,
    c: dict,
    max_tokens: int,
    tolerances: tuple[float, ...],
    include_precursor_shift: bool,
    drop_precursor_features: bool,
) -> np.ndarray:
    tokens, mask = conservation_tokens(
        q,
        c,
        max_tokens,
        tolerances,
        include_precursor_shift,
        drop_precursor_features,
    )
    active = tokens[mask]
    if len(active) == 0:
        return np.zeros(4 * tokens.shape[-1] + 1, dtype=np.float32)
    sums = active.sum(axis=0)
    means = active.mean(axis=0)
    maxima = active.max(axis=0)
    minima = active.min(axis=0)
    count = np.asarray([float(len(active)) / max(float(max_tokens), 1.0)], dtype=np.float32)
    return np.concatenate([sums, means, maxima, minima, count]).astype(np.float32)


def query_features(
    queries: list[tuple[dict, list[dict], int]],
    max_tokens: int,
    tolerances: tuple[float, ...],
    include_precursor_shift: bool,
    drop_precursor_features: bool,
) -> tuple[np.ndarray, np.ndarray]:
    feature_rows = []
    labels = []
    for q, candidates, label in queries:
        feature_rows.append(
            [
                pooled_token_features(q, c, max_tokens, tolerances, include_precursor_shift, drop_precursor_features)
                for c in candidates
            ]
        )
        labels.append(label)
    return np.asarray(feature_rows, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def standardize_3d(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat = train.reshape(-1, train.shape[-1])
    mean = flat.mean(axis=0, keepdims=True).astype(np.float32)
    std = flat.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return (train - mean) / std, (test - mean) / std, mean, std


def metrics_from_scores(queries: list[tuple[dict, list[dict], int]], scores_by_query: list[list[float]]) -> dict[str, float]:
    ranks = [rank_of(scores, label) for (_, _, label), scores in zip(queries, scores_by_query)]
    return {
        "hit1": float(np.mean([rank == 1 for rank in ranks])) if ranks else float("nan"),
        "mrr": float(np.mean([1.0 / rank for rank in ranks])) if ranks else float("nan"),
    }


def run_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    rows = load_rows(Path(args.tsv))
    tolerances = tuple(float(x) for x in args.tolerances.split(",") if x)
    train_queries = build_queries(
        rows,
        seed + 101,
        args.train_queries,
        args.train_negatives,
        {"train"},
        {"train"},
        args.positive_adduct,
        "random",
        args.negative_window,
    )
    eval_queries = build_queries(
        rows,
        seed,
        args.eval_queries,
        args.eval_negatives,
        parse_folds(args.query_folds),
        parse_folds(args.candidate_folds),
        args.positive_adduct,
        args.negative_strategy,
        args.negative_window,
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(train_queries)
    split = max(1, int(0.9 * len(train_queries)))
    train_split, val_split = train_queries[:split], train_queries[split:]

    train_x, train_y = query_features(
        train_split,
        args.max_tokens,
        tolerances,
        args.include_precursor_shift,
        args.drop_precursor_features,
    )
    val_x, val_y = query_features(
        val_split,
        args.max_tokens,
        tolerances,
        args.include_precursor_shift,
        args.drop_precursor_features,
    )
    eval_x, _ = query_features(
        eval_queries,
        args.max_tokens,
        tolerances,
        args.include_precursor_shift,
        args.drop_precursor_features,
    )
    train_x, eval_x, mean, std = standardize_3d(train_x, eval_x)
    val_x = (val_x - mean) / std

    params, history = fit_mlp_stats(
        train_x,
        train_y,
        val_x,
        val_y,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.width,
        seed,
    )
    mlp_scores = forward(eval_x, params)[0].astype(float).tolist()
    modified_scores = [[modified_cosine_score(q, c, args.tolerance) for c in candidates] for q, candidates, _ in eval_queries]
    zero_scores = [[mcft_zero_shift_score(q, c, args.tolerance) for c in candidates] for q, candidates, _ in eval_queries]
    out = {
        "seed": seed,
        "num_train_queries_built": len(train_queries),
        "num_eval_queries_built": len(eval_queries),
        "sum_pool_mlp_mcft_hit1": metrics_from_scores(eval_queries, mlp_scores)["hit1"],
        "sum_pool_mlp_mcft_mrr": metrics_from_scores(eval_queries, mlp_scores)["mrr"],
        "modified_cosine_hit1": metrics_from_scores(eval_queries, modified_scores)["hit1"],
        "modified_cosine_mrr": metrics_from_scores(eval_queries, modified_scores)["mrr"],
        "mcft_zero_shift_hit1": metrics_from_scores(eval_queries, zero_scores)["hit1"],
        "mcft_zero_shift_mrr": metrics_from_scores(eval_queries, zero_scores)["mrr"],
        "history": history,
    }
    print(json.dumps(out), flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", default="experiment/data/massspecgym/MassSpecGym_rows_25k.tsv")
    parser.add_argument("--out-dir", default="experiment/outputs/massspecgym_sum_pool_mlp_mcft")
    parser.add_argument("--train-queries", type=int, default=3000)
    parser.add_argument("--eval-queries", type=int, default=300)
    parser.add_argument("--train-negatives", type=int, default=63)
    parser.add_argument("--eval-negatives", type=int, default=500)
    parser.add_argument("--query-folds", default="val,test")
    parser.add_argument("--candidate-folds", default="val,test")
    parser.add_argument("--positive-adduct", choices=["any", "same", "different"], default="any")
    parser.add_argument("--negative-strategy", choices=["random", "closest"], default="closest")
    parser.add_argument("--negative-window", type=float, default=20.0)
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--tolerances", default="0.01,0.03,0.08")
    parser.add_argument("--include-precursor-shift", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--drop-precursor-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()

    rows = [run_seed(args, int(seed)) for seed in args.seeds.split(",") if seed]
    scalar_keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k != "seed"]
    mean = {k: float(np.mean([row[k] for row in rows])) for k in scalar_keys}
    stderr95 = {
        k: float(1.96 * np.std([row[k] for row in rows], ddof=1) / np.sqrt(len(rows)))
        for k in scalar_keys
        if len(rows) > 1
    }
    payload = {"metadata": vars(args), "rows": rows, "mean": mean, "stderr95": stderr95}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
