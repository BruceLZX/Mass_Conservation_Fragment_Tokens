from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML configs require PyYAML. Install experiment/requirements.txt.") from exc
    return yaml.safe_load(text)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        try:
            value: Any = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        cursor = result
        parts = dotted.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return result


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config value with dotted key syntax, e.g. --set train.epochs=5.",
    )
    return parser

