from __future__ import annotations

import itertools
from copy import deepcopy
from pathlib import Path

from eventclock.config import build_arg_parser, deep_update, load_config, parse_overrides
from eventclock.train import run


def set_dotted(cfg: dict, dotted: str, value) -> None:
    cursor = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def flatten_grid(grid: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in grid.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_grid(value, dotted))
        else:
            flat[dotted] = value
    return flat


def main() -> None:
    parser = build_arg_parser("Run config grid.")
    args = parser.parse_args()
    cfg = deep_update(load_config(args.config), parse_overrides(args.set))
    grid = flatten_grid(cfg.pop("grid", {}))
    keys = list(grid)
    values = [grid[key] for key in keys]
    base_out = Path(cfg.get("output_dir", "experiment/outputs/grid"))
    for i, combo in enumerate(itertools.product(*values)):
        run_cfg = deepcopy(cfg)
        tags = []
        for key, value in zip(keys, combo):
            set_dotted(run_cfg, key, value)
            tags.append(f"{key.replace('.', '-')}_{value}")
        run_cfg["output_dir"] = str(base_out / f"{i:03d}_{'_'.join(tags)}")
        run(run_cfg)


if __name__ == "__main__":
    main()
