from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def flatten(prefix: str, obj: dict, out: dict) -> None:
    for key, value in obj.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flatten(name, value, out)
        elif isinstance(value, (int, float, str, bool)):
            out[name] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize metrics.json files into reviewer tables.")
    parser.add_argument("root", help="Output directory containing run subdirectories.")
    parser.add_argument("--out", default="experiment/outputs/summary.csv")
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.root).rglob("metrics.json")):
        obj = json.loads(path.read_text())
        row = {"run": str(path.parent)}
        flatten("", obj, row)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No metrics.json files found under {args.root}")
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"saved {out}")
    metric_cols = [c for c in df.columns if c.startswith("test.") or c.startswith("robustness.")]
    if metric_cols:
        print(df[["run", *metric_cols]].to_string(index=False))


if __name__ == "__main__":
    main()

