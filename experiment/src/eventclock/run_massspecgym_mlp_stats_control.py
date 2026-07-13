from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from eventclock.audit_mcft_evidence_examples import rank_of
from eventclock.run_massspecgym_listwise_linear import query_features, standardize_3d
from eventclock.run_massspecgym_pair_smoke import load_rows
from eventclock.run_massspecgym_retrieval_smoke import (
    build_queries,
    mcft_zero_shift_score,
    modified_cosine_score,
    parse_folds,
)


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), 1e-8)


def init_params(dim: int, width: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    scale1 = np.sqrt(2.0 / max(dim + width, 1))
    scale2 = np.sqrt(2.0 / max(width + 1, 1))
    return {
        "w1": rng.normal(0.0, scale1, size=(dim, width)).astype(np.float32),
        "b1": np.zeros(width, dtype=np.float32),
        "w2": rng.normal(0.0, scale2, size=(width,)).astype(np.float32),
        "b2": np.zeros((), dtype=np.float32),
    }


def forward(x: np.ndarray, params: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.tanh(np.einsum("nkd,dh->nkh", x, params["w1"]) + params["b1"])
    scores = np.einsum("nkh,h->nk", hidden, params["w2"]) + params["b2"]
    return scores.astype(np.float32), hidden.astype(np.float32)


def fit_mlp_stats(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    epochs: int,
    lr: float,
    weight_decay: float,
    width: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    params = init_params(train_x.shape[-1], width, seed)
    moments = {k: np.zeros_like(v) for k, v in params.items()}
    velocities = {k: np.zeros_like(v) for k, v in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    onehot = np.zeros((len(train_x), train_x.shape[1]), dtype=np.float32)
    onehot[np.arange(len(train_y)), train_y] = 1.0
    best_params = {k: v.copy() for k, v in params.items()}
    best_val = -1.0
    history: list[dict[str, float]] = []
    n = max(len(train_x), 1)

    for epoch in range(1, epochs + 1):
        scores, hidden = forward(train_x, params)
        probs = softmax(scores).astype(np.float32)
        loss = -float(np.log(np.maximum(probs[np.arange(len(train_y)), train_y], 1e-8)).mean())
        residual = (probs - onehot) / n
        grad_w2 = np.einsum("nk,nkh->h", residual, hidden) + weight_decay * params["w2"]
        grad_b2 = np.asarray(residual.sum(), dtype=np.float32)
        grad_hidden = residual[:, :, None] * params["w2"][None, None, :]
        grad_pre = grad_hidden * (1.0 - hidden * hidden)
        grad_w1 = np.einsum("nkd,nkh->dh", train_x, grad_pre) + weight_decay * params["w1"]
        grad_b1 = grad_pre.sum(axis=(0, 1))
        grads = {
            "w1": grad_w1.astype(np.float32),
            "b1": grad_b1.astype(np.float32),
            "w2": grad_w2.astype(np.float32),
            "b2": grad_b2.astype(np.float32),
        }

        for key in params:
            moments[key] = beta1 * moments[key] + (1.0 - beta1) * grads[key]
            velocities[key] = beta2 * velocities[key] + (1.0 - beta2) * (grads[key] * grads[key])
            mhat = moments[key] / (1.0 - beta1**epoch)
            vhat = velocities[key] / (1.0 - beta2**epoch)
            params[key] = (params[key] - lr * mhat / (np.sqrt(vhat) + eps)).astype(np.float32)

        val_scores, _ = forward(val_x, params)
        val_hit1 = float((val_scores.argmax(axis=1) == val_y).mean()) if len(val_y) else 0.0
        row = {"epoch": float(epoch), "train_loss": loss, "val_hit1": val_hit1}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_hit1 >= best_val:
            best_val = val_hit1
            best_params = {k: v.copy() for k, v in params.items()}
    return best_params, history


def metrics_from_scores(queries: list[tuple[dict, list[dict], int]], scores_by_query: list[list[float]]) -> dict[str, float]:
    ranks = [rank_of(scores, label) for (_, _, label), scores in zip(queries, scores_by_query)]
    return {
        "hit1": float(np.mean([rank == 1 for rank in ranks])) if ranks else float("nan"),
        "mrr": float(np.mean([1.0 / rank for rank in ranks])) if ranks else float("nan"),
    }


def run_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    rows = load_rows(Path(args.tsv))
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
    train_x, train_y = query_features(train_split)
    val_x, val_y = query_features(val_split)
    eval_x, _ = query_features(eval_queries)
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
        "mlp_stats_mcft_hit1": metrics_from_scores(eval_queries, mlp_scores)["hit1"],
        "mlp_stats_mcft_mrr": metrics_from_scores(eval_queries, mlp_scores)["mrr"],
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
    parser.add_argument("--out-dir", default="experiment/outputs/massspecgym_mlp_stats_control")
    parser.add_argument("--train-queries", type=int, default=3000)
    parser.add_argument("--eval-queries", type=int, default=300)
    parser.add_argument("--train-negatives", type=int, default=63)
    parser.add_argument("--eval-negatives", type=int, default=500)
    parser.add_argument("--query-folds", default="val,test")
    parser.add_argument("--candidate-folds", default="val,test")
    parser.add_argument("--positive-adduct", choices=["any", "same", "different"], default="any")
    parser.add_argument("--negative-strategy", choices=["random", "closest", "overlap"], default="closest")
    parser.add_argument("--negative-window", type=float, default=20.0)
    parser.add_argument("--tolerance", type=float, default=0.03)
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
