from __future__ import annotations

import argparse
import csv
import json
import ssl
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


FIELDS = [
    "identifier",
    "mzs",
    "intensities",
    "smiles",
    "inchikey",
    "formula",
    "precursor_formula",
    "parent_mass",
    "precursor_mz",
    "adduct",
    "instrument_type",
    "collision_energy",
    "fold",
    "simulation_challenge",
]


def fetch_page(offset: int, length: int, timeout: int = 90, retries: int = 4) -> list[dict]:
    params = urllib.parse.urlencode(
        {
            "dataset": "roman-bushuiev/MassSpecGym",
            "config": "main",
            "split": "val",
            "offset": offset,
            "length": length,
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{params}"
    ctx = ssl._create_unverified_context()
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout, context=ctx) as response:
                payload = json.load(response)
            return [item["row"] for item in payload["rows"]]
        except Exception as exc:  # network endpoint is occasionally flaky.
            last_error = exc
            wait = min(2.0 * (attempt + 1), 10.0)
            print(f"[massspecgym-fetch] retry offset={offset} attempt={attempt + 1} error={type(exc).__name__}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed offset={offset} length={length}: {last_error}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-tsv", default="experiment/data/massspecgym/MassSpecGym_rows_10k.tsv")
    parser.add_argument("--rows-per-offset", type=int, default=2000)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--offsets", default="0,50000,100000,150000,200000")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    offsets = [int(x) for x in args.offsets.split(",") if x]
    pages = []
    for base in offsets:
        fetched_for_base = 0
        while fetched_for_base < args.rows_per_offset:
            length = min(args.page_size, args.rows_per_offset - fetched_for_base)
            pages.append((base + fetched_for_base, length))
            fetched_for_base += length

    def record_page(offset: int, length: int) -> list[dict]:
        try:
            rows = fetch_page(offset, length)
        except Exception as exc:
            print(f"[massspecgym-fetch] failed offset={offset} requested={length} error={exc}", flush=True)
            rows = []
        return rows

    page_rows: dict[int, list[dict]] = {}
    total = 0
    if args.workers <= 1:
        for offset, length in pages:
            rows = record_page(offset, length)
            page_rows[offset] = rows
            total += len(rows)
            print(f"[massspecgym-fetch] offset={offset} requested={length} rows={len(rows)} total={total}", flush=True)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(fetch_page, offset, length): (offset, length) for offset, length in pages}
            for future in as_completed(futures):
                offset, length = futures[future]
                try:
                    rows = future.result()
                except Exception as exc:
                    print(f"[massspecgym-fetch] failed offset={offset} requested={length} error={exc}", flush=True)
                    rows = []
                page_rows[offset] = rows
                total += len(rows)
                print(f"[massspecgym-fetch] offset={offset} requested={length} rows={len(rows)} total={total}", flush=True)
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for offset in sorted(page_rows):
            writer.writerows(page_rows[offset])
    print(json.dumps({"out_tsv": str(out_path), "rows": total, "offsets": offsets}, indent=2))


if __name__ == "__main__":
    main()
