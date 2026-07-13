from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eventclock.audit_mcft_evidence_examples import rank_of
from eventclock.run_massspecgym_pair_smoke import conservation_pair_features, load_rows
from eventclock.run_massspecgym_retrieval_smoke import (
    build_queries,
    mcft_zero_shift_score,
    modified_cosine_score,
    parse_folds,
)


def query_features(queries: list[tuple[dict, list[dict], int]]) -> tuple[np.ndarray, np.ndarray]:
    feature_rows = []
    labels = []
    for q, candidates, label in queries:
        feature_rows.append(conservation_pair_features([(q, c) for c in candidates], include_dynamic=False))
        labels.append(label)
    return np.stack(feature_rows).astype(np.float32), np.asarray(labels, dtype=np.int64)


def standardize_3d(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat = train.reshape(-1, train.shape[-1])
    mean = flat.mean(axis=0, keepdims=True).astype(np.float32)
    std = flat.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return (train - mean) / std, (test - mean) / std, mean, std


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), 1e-8)


def fit_listwise_linear(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    dim = train_x.shape[-1]
    weights = rng.normal(0.0, 0.01, size=dim).astype(np.float32)
    bias = np.float32(0.0)
    m_w = np.zeros_like(weights)
    v_w = np.zeros_like(weights)
    m_b = np.float32(0.0)
    v_b = np.float32(0.0)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    best_weights = weights.copy()
    best_bias = float(bias)
    best_val = -1.0
    history: list[dict[str, float]] = []
    n = max(len(train_x), 1)
    onehot = np.zeros((len(train_x), train_x.shape[1]), dtype=np.float32)
    onehot[np.arange(len(train_y)), train_y] = 1.0

    for epoch in range(1, epochs + 1):
        scores = np.einsum("nkd,d->nk", train_x, weights) + bias
        probs = softmax(scores).astype(np.float32)
        loss = -float(np.log(np.maximum(probs[np.arange(len(train_y)), train_y], 1e-8)).mean())
        residual = (probs - onehot) / n
        grad_w = np.einsum("nk,nkd->d", residual, train_x) + weight_decay * weights
        grad_b = np.float32(residual.sum())

        m_w = beta1 * m_w + (1.0 - beta1) * grad_w
        v_w = beta2 * v_w + (1.0 - beta2) * (grad_w * grad_w)
        m_b = beta1 * m_b + (1.0 - beta1) * grad_b
        v_b = beta2 * v_b + (1.0 - beta2) * (grad_b * grad_b)
        m_w_hat = m_w / (1.0 - beta1**epoch)
        v_w_hat = v_w / (1.0 - beta2**epoch)
        m_b_hat = m_b / (1.0 - beta1**epoch)
        v_b_hat = v_b / (1.0 - beta2**epoch)
        weights = (weights - lr * m_w_hat / (np.sqrt(v_w_hat) + eps)).astype(np.float32)
        bias = np.float32(bias - lr * m_b_hat / (np.sqrt(v_b_hat) + eps))

        val_scores = np.einsum("nkd,d->nk", val_x, weights) + bias
        val_hit1 = float((val_scores.argmax(axis=1) == val_y).mean()) if len(val_y) else 0.0
        history.append({"epoch": float(epoch), "train_loss": loss, "val_hit1": val_hit1})
        print(json.dumps(history[-1]), flush=True)
        if val_hit1 >= best_val:
            best_val = val_hit1
            best_weights = weights.copy()
            best_bias = float(bias)

    return best_weights, best_bias, history


def metrics_from_scores(queries: list[tuple[dict, list[dict], int]], scores_by_query: list[list[float]]) -> dict[str, float]:
    ranks = [rank_of(scores, label) for (_, _, label), scores in zip(queries, scores_by_query)]
    return {
        "hit1": float(np.mean([rank == 1 for rank in ranks])) if ranks else float("nan"),
        "mrr": float(np.mean([1.0 / rank for rank in ranks])) if ranks else float("nan"),
    }


def run_seed(args: argparse.Namespace, seed: int) -> dict:
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
    eval_x, eval_y = query_features(eval_queries)
    train_x, eval_x, mean, std = standardize_3d(train_x, eval_x)
    val_x = (val_x - mean) / std

    weights, bias, history = fit_listwise_linear(
        train_x,
        train_y,
        val_x,
        val_y,
        args.epochs,
        args.lr,
        args.weight_decay,
        seed,
    )
    listwise_scores = (np.einsum("nkd,d->nk", eval_x, weights) + bias).astype(float).tolist()
    modified_scores = [[modified_cosine_score(q, c, args.tolerance) for c in candidates] for q, candidates, _ in eval_queries]
    zero_scores = [[mcft_zero_shift_score(q, c, args.tolerance) for c in candidates] for q, candidates, _ in eval_queries]

    out = {
        "seed": seed,
        "num_train_queries_built": len(train_queries),
        "num_eval_queries_built": len(eval_queries),
        "listwise_linear_mcft_hit1": metrics_from_scores(eval_queries, listwise_scores)["hit1"],
        "listwise_linear_mcft_mrr": metrics_from_scores(eval_queries, listwise_scores)["mrr"],
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
    parser.add_argument("--out-dir", default="experiment/outputs/massspecgym_listwise_linear_mcft")
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
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()

    rows = []
    for seed in [int(s) for s in args.seeds.split(",") if s]:
        rows.append(run_seed(args, seed))
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
