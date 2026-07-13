from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eventclock.run_massspecgym_pair_smoke import load_rows
from eventclock.run_massspecgym_retrieval_smoke import (
    build_queries,
    mcft_zero_shift_score,
    modified_cosine_score,
    parse_folds,
)


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def collect_strategy(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    strategy: str,
    window: float,
    seed: int,
) -> list[dict[str, float | str | int]]:
    queries = build_queries(
        rows,
        seed,
        args.num_queries,
        args.num_negatives,
        parse_folds(args.query_folds),
        parse_folds(args.candidate_folds),
        args.positive_adduct,
        strategy,
        window,
    )
    out: list[dict[str, float | str | int]] = []
    for query_idx, (query, candidates, label) in enumerate(queries):
        pos = candidates[label]
        negatives = [candidate for idx, candidate in enumerate(candidates) if idx != label]
        neg_zero = [mcft_zero_shift_score(query, neg, args.tolerance) for neg in negatives]
        neg_modcos = [modified_cosine_score(query, neg, args.tolerance) for neg in negatives]
        precursor_gaps = [abs(float(query["precursor_mz"]) - float(neg["precursor_mz"])) for neg in negatives]
        out.append(
            {
                "strategy": strategy,
                "window": window,
                "seed": seed,
                "query_idx": query_idx,
                "max_negative_zero_score": float(max(neg_zero)),
                "max_negative_modified_cosine": float(max(neg_modcos)),
                "mean_negative_precursor_gap": float(np.mean(precursor_gaps)),
                "positive_zero_score": float(mcft_zero_shift_score(query, pos, args.tolerance)),
                "positive_modified_cosine": float(modified_cosine_score(query, pos, args.tolerance)),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    fieldnames = [
        "strategy",
        "window",
        "seed",
        "query_idx",
        "max_negative_zero_score",
        "max_negative_modified_cosine",
        "mean_negative_precursor_gap",
        "positive_zero_score",
        "positive_modified_cosine",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, float | str | int]], out_path: Path) -> None:
    labels = [("random", "Random120"), ("closest", "Closest20"), ("overlap", "Overlap120")]
    colors = {"random": "#4C78A8", "closest": "#F58518", "overlap": "#54A24B"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    bins = np.linspace(0.0, 1.0, 26)
    for ax, key, title in [
        (axes[0], "max_negative_zero_score", "Max negative MCFT"),
        (axes[1], "max_negative_modified_cosine", "Max negative modified cosine"),
    ]:
        for strategy, label in labels:
            vals = [float(row[key]) for row in rows if row["strategy"] == strategy]
            ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.6, label=label, color=colors[strategy])
        ax.set_title(title)
        ax.set_xlabel("query-level maximum score")
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("density")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", default="experiment/data/massspecgym/MassSpecGym_rows_25k.tsv")
    parser.add_argument("--out-dir", default="experiment/outputs/massspecgym_25k_hardness_distributions")
    parser.add_argument("--num-queries", type=int, default=300)
    parser.add_argument("--num-negatives", type=int, default=500)
    parser.add_argument("--query-folds", default="val,test")
    parser.add_argument("--candidate-folds", default="val,test")
    parser.add_argument("--positive-adduct", choices=["any", "same", "different"], default="any")
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--figure", default="")
    args = parser.parse_args()

    rows = load_rows(Path(args.tsv))
    all_rows: list[dict[str, float | str | int]] = []
    for seed in [int(text) for text in args.seeds.split(",") if text]:
        all_rows.extend(collect_strategy(rows, args, "random", 120.0, seed))
        all_rows.extend(collect_strategy(rows, args, "closest", 20.0, seed))
        all_rows.extend(collect_strategy(rows, args, "overlap", 120.0, seed))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "query_hardness.csv", all_rows)
    summary = {
        strategy: {
            key: summarize([float(row[key]) for row in all_rows if row["strategy"] == strategy])
            for key in [
                "max_negative_zero_score",
                "max_negative_modified_cosine",
                "mean_negative_precursor_gap",
                "positive_zero_score",
                "positive_modified_cosine",
            ]
        }
        for strategy in ["random", "closest", "overlap"]
    }
    payload = {"metadata": vars(args), "summary": summary}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    figure_path = Path(args.figure) if args.figure else out_dir / "hardness_distributions.pdf"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plot(all_rows, figure_path)
    print(json.dumps({**payload, "figure": str(figure_path)}, indent=2))


if __name__ == "__main__":
    main()
