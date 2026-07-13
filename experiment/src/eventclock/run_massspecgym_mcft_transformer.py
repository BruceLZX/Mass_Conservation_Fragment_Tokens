from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:  # pragma: no cover - exercised only in missing-dependency environments.
    torch = None
    nn = None
    DataLoader = None
    Dataset = object

from eventclock.audit_mcft_evidence_examples import rank_of
from eventclock.run_massspecgym_pair_smoke import load_rows
from eventclock.run_massspecgym_retrieval_smoke import (
    build_queries,
    fit_learned_mcft,
    learned_mcft_scores,
    mcft_zero_shift_score,
    modified_cosine_score,
)


FIXED_SHIFTS = np.asarray([0.0, 1.0034, -1.0034, 14.0157, 15.9949, -18.0106], dtype=np.float32)
TOKEN_DIM = 12


def require_torch() -> None:
    if torch is None or nn is None or DataLoader is None:
        raise RuntimeError(
            "PyTorch is required for the MCFT transformer experiment. "
            "Install experiment/requirements.txt in a Python version supported by torch."
        )


def conservation_tokens(
    q: dict,
    c: dict,
    max_tokens: int,
    tolerances: tuple[float, ...],
    include_precursor_shift: bool,
) -> tuple[np.ndarray, np.ndarray]:
    shifts = FIXED_SHIFTS
    precursor_shift = float(c["precursor_mz"] - q["precursor_mz"])
    if include_precursor_shift:
        shifts = np.concatenate([FIXED_SHIFTS, np.asarray([precursor_shift, -precursor_shift], dtype=np.float32)])

    diff = c["mzs"][:, None] - q["mzs"][None, :]
    intensity_outer = c["intensities"][:, None] * q["intensities"][None, :]
    candidates: list[tuple[float, np.ndarray]] = []
    for tol in tolerances:
        for shift_index, shift in enumerate(shifts):
            residual = diff - float(shift)
            close = np.abs(residual) <= tol
            for cand_idx, query_idx in np.argwhere(close):
                qi = float(q["intensities"][query_idx])
                ci = float(c["intensities"][cand_idx])
                product = float(intensity_outer[cand_idx, query_idx])
                abs_residual = abs(float(residual[cand_idx, query_idx]))
                token = np.asarray(
                    [
                        float(q["mzs"][query_idx]) / 1200.0,
                        float(c["mzs"][cand_idx]) / 1200.0,
                        qi,
                        ci,
                        product,
                        abs_residual / max(float(tol), 1e-6),
                        float(residual[cand_idx, query_idx]) / max(float(tol), 1e-6),
                        float(shift) / 100.0,
                        precursor_shift / 1200.0,
                        float(tol) / 0.08,
                        float(shift_index == 0),
                        float(abs(float(shift) - precursor_shift) <= 1e-4),
                    ],
                    dtype=np.float32,
                )
                candidates.append((product - 0.05 * abs_residual, token))
    candidates.sort(key=lambda x: x[0], reverse=True)

    tokens = np.zeros((max_tokens, TOKEN_DIM), dtype=np.float32)
    mask = np.zeros(max_tokens, dtype=bool)
    for idx, (_, token) in enumerate(candidates[:max_tokens]):
        tokens[idx] = token
        mask[idx] = True
    return tokens, mask


class QueryListDataset(Dataset):
    def __init__(
        self,
        queries: list[tuple[dict, list[dict], int]],
        max_tokens: int,
        tolerances: tuple[float, ...],
        include_precursor_shift: bool,
    ) -> None:
        self.queries = queries
        self.max_tokens = max_tokens
        self.tolerances = tolerances
        self.include_precursor_shift = include_precursor_shift

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        q, candidates, label = self.queries[idx]
        tokens, masks = [], []
        for c in candidates:
            token, mask = conservation_tokens(q, c, self.max_tokens, self.tolerances, self.include_precursor_shift)
            tokens.append(token)
            masks.append(mask)
        return {
            "tokens": np.stack(tokens),
            "mask": np.stack(masks),
            "label": np.asarray(label, dtype=np.int64),
        }


def collate_query_lists(batch: list[dict[str, Any]]) -> dict[str, Any]:
    require_torch()
    tokens = torch.as_tensor(np.stack([row["tokens"] for row in batch]), dtype=torch.float32)
    mask = torch.as_tensor(np.stack([row["mask"] for row in batch]), dtype=torch.bool)
    labels = torch.as_tensor(np.stack([row["label"] for row in batch]), dtype=torch.long)
    return {"tokens": tokens, "mask": mask, "label": labels}


class ConservationTokenTransformer(nn.Module if nn is not None else object):
    def __init__(self, token_dim: int, width: int, depth: int, heads: int, dropout: float) -> None:
        require_torch()
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.in_proj = nn.Sequential(nn.Linear(token_dim, width), nn.GELU(), nn.LayerNorm(width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=4 * width,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, tokens: Any, mask: Any) -> Any:
        batch = tokens.shape[0]
        h = self.in_proj(tokens)
        cls = self.cls.expand(batch, -1, -1)
        h = torch.cat([cls, h], dim=1)
        cls_mask = torch.ones((batch, 1), dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)
        encoded = self.encoder(h, src_key_padding_mask=~full_mask)
        return self.head(encoded[:, 0]).squeeze(-1)


def parse_folds(text: str) -> set[str] | None:
    if text.lower() in {"", "all", "none"}:
        return None
    return {x.strip() for x in text.split(",") if x.strip()}


def score_query(
    model: Any,
    q: dict,
    candidates: list[dict],
    args: argparse.Namespace,
    device: Any,
    chunk_size: int = 256,
) -> list[float]:
    require_torch()
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(candidates), chunk_size):
            chunk = candidates[start : start + chunk_size]
            token_rows, mask_rows = [], []
            for c in chunk:
                token, mask = conservation_tokens(
                    q, c, args.max_tokens, args.tolerances_tuple, args.include_precursor_shift
                )
                token_rows.append(token)
                mask_rows.append(mask)
            tokens = torch.as_tensor(np.stack(token_rows), dtype=torch.float32, device=device)
            masks = torch.as_tensor(np.stack(mask_rows), dtype=torch.bool, device=device)
            scores.extend(model(tokens, masks).detach().cpu().numpy().astype(float).tolist())
    return scores


def retrieval_metrics(rows: list[tuple[dict, list[dict], int]], scores_by_query: list[list[float]]) -> dict[str, float]:
    hits = 0
    ranks = []
    for (_, _, label), scores in zip(rows, scores_by_query):
        rank = rank_of(scores, label)
        hits += int(rank == 1)
        ranks.append(rank)
    return {
        "hit1": float(hits / max(len(rows), 1)),
        "mrr": float(np.mean([1.0 / rank for rank in ranks])) if ranks else float("nan"),
    }


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


def rank_summary(ranks: list[int]) -> dict[str, float]:
    return {
        "hit1": float(np.mean([rank == 1 for rank in ranks])) if ranks else float("nan"),
        "mean_rank": float(np.mean(ranks)) if ranks else float("nan"),
        "mrr": float(np.mean([1.0 / rank for rank in ranks])) if ranks else float("nan"),
    }


def transformer_witness_removal_audit(
    model: Any,
    queries: list[tuple[dict, list[dict], int]],
    args: argparse.Namespace,
    device: Any,
) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed + 17_003)
    original_ranks: list[int] = []
    top_ranks: list[int] = []
    random_ranks: list[int] = []
    score_drops_top: list[float] = []
    score_drops_random: list[float] = []
    skipped_no_witness = 0

    for q, candidates, label in queries:
        original_scores = score_query(model, q, candidates, args, device)
        original_rank = rank_of(original_scores, label)
        if original_rank != 1:
            continue
        positive = candidates[label]
        top_indices = top_zero_shift_candidate_indices(q, positive, args.audit_top_k, args.tolerance)
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
        for _ in range(args.audit_random_trials):
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

    return {
        "evaluated_initial_hit1_cases": len(original_ranks),
        "random_deletion_trials": len(random_ranks),
        "skipped_no_zero_shift_witness": skipped_no_witness,
        "original": rank_summary(original_ranks),
        "top_witness_deletion": rank_summary(top_ranks),
        "random_deletion": rank_summary(random_ranks),
        "mean_score_drop_top": float(np.mean(score_drops_top)) if score_drops_top else float("nan"),
        "mean_score_drop_random": float(np.mean(score_drops_random)) if score_drops_random else float("nan"),
    }


def evaluate_model(model: Any, queries: list[tuple[dict, list[dict], int]], args: argparse.Namespace, device: Any) -> dict[str, Any]:
    transformer_scores = [score_query(model, q, candidates, args, device) for q, candidates, _ in queries]
    out = {f"mcft_transformer_{k}": v for k, v in retrieval_metrics(queries, transformer_scores).items()}

    modified_scores = [[modified_cosine_score(q, c, args.tolerance) for c in candidates] for q, candidates, _ in queries]
    zero_scores = [[mcft_zero_shift_score(q, c, args.tolerance) for c in candidates] for q, candidates, _ in queries]
    out.update({f"modified_cosine_{k}": v for k, v in retrieval_metrics(queries, modified_scores).items()})
    out.update({f"mcft_zero_shift_{k}": v for k, v in retrieval_metrics(queries, zero_scores).items()})

    ridge_model = fit_learned_mcft(args.all_rows, args.seed, args.learned_pairs, args.positive_adduct)
    if ridge_model is not None:
        ridge_scores = [learned_mcft_scores(q, candidates, ridge_model) for q, candidates, _ in queries]
        out.update({f"ridge_mcft_{k}": v for k, v in retrieval_metrics(queries, ridge_scores).items()})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", default="experiment/data/massspecgym/MassSpecGym_rows_10k.tsv")
    parser.add_argument("--out-dir", default="experiment/outputs/massspecgym_mcft_transformer")
    parser.add_argument("--train-queries", type=int, default=1200)
    parser.add_argument("--eval-queries", type=int, default=300)
    parser.add_argument("--train-negatives", type=int, default=63)
    parser.add_argument("--eval-negatives", type=int, default=500)
    parser.add_argument("--query-folds", default="val,test")
    parser.add_argument("--candidate-folds", default="val,test")
    parser.add_argument("--positive-adduct", choices=["any", "same", "different"], default="any")
    parser.add_argument("--negative-strategy", choices=["random", "closest"], default="random")
    parser.add_argument("--negative-window", type=float, default=120.0)
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--tolerances", default="0.01,0.03,0.08")
    parser.add_argument("--include-precursor-shift", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--learned-pairs", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--witness-removal-audit",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("MCFT_WITNESS_REMOVAL_AUDIT", "0") == "1",
    )
    parser.add_argument("--audit-top-k", type=int, default=int(os.environ.get("MCFT_AUDIT_TOP_K", "3")))
    parser.add_argument("--audit-random-trials", type=int, default=int(os.environ.get("MCFT_AUDIT_RANDOM_TRIALS", "3")))
    args = parser.parse_args()
    require_torch()

    args.tolerances_tuple = tuple(float(x) for x in args.tolerances.split(",") if x)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    rows = load_rows(Path(args.tsv))
    args.all_rows = rows

    train_queries = build_queries(
        rows,
        args.seed + 101,
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
        args.seed,
        args.eval_queries,
        args.eval_negatives,
        parse_folds(args.query_folds),
        parse_folds(args.candidate_folds),
        args.positive_adduct,
        args.negative_strategy,
        args.negative_window,
    )
    rng.shuffle(train_queries)
    split = max(1, int(0.9 * len(train_queries)))
    train_split, val_split = train_queries[:split], train_queries[split:]

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = ConservationTokenTransformer(TOKEN_DIM, args.width, args.depth, args.heads, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        QueryListDataset(train_split, args.max_tokens, args.tolerances_tuple, args.include_precursor_shift),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_query_lists,
    )
    val_loader = DataLoader(
        QueryListDataset(val_split, args.max_tokens, args.tolerances_tuple, args.include_precursor_shift),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_query_lists,
    )

    history = []
    best_val = -float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            tokens = batch["tokens"].to(device)
            masks = batch["mask"].to(device)
            labels = batch["label"].to(device)
            bsz, cand_count, max_tokens, token_dim = tokens.shape
            scores = model(tokens.reshape(bsz * cand_count, max_tokens, token_dim), masks.reshape(bsz * cand_count, max_tokens))
            scores = scores.reshape(bsz, cand_count)
            loss = torch.nn.functional.cross_entropy(scores, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        val_hits, val_count = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch["tokens"].to(device)
                masks = batch["mask"].to(device)
                labels = batch["label"].to(device)
                bsz, cand_count, max_tokens, token_dim = tokens.shape
                scores = model(
                    tokens.reshape(bsz * cand_count, max_tokens, token_dim),
                    masks.reshape(bsz * cand_count, max_tokens),
                ).reshape(bsz, cand_count)
                val_hits += int((scores.argmax(dim=1) == labels).sum().item())
                val_count += bsz
        val_hit1 = float(val_hits / max(val_count, 1))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_hit1": val_hit1}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_hit1 > best_val:
            best_val = val_hit1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = evaluate_model(model, eval_queries, args, device)
    witness_audit = None
    if args.witness_removal_audit:
        witness_audit = transformer_witness_removal_audit(model, eval_queries, args, device)
    payload = {
        "metadata": {
            **{k: v for k, v in vars(args).items() if k not in {"all_rows", "tolerances_tuple"}},
            "num_rows": len(rows),
            "num_train_queries_built": len(train_queries),
            "num_eval_queries_built": len(eval_queries),
            "device_used": str(device),
        },
        "history": history,
        "metrics": metrics,
    }
    if witness_audit is not None:
        payload["transformer_witness_removal_audit"] = witness_audit
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    if best_state is not None:
        torch.save({"model": best_state, "metadata": payload["metadata"]}, out_dir / "best.pt")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
