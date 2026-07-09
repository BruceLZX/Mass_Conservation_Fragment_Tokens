from __future__ import annotations

from pathlib import Path

import torch

from eventclock.config import build_arg_parser, deep_update, load_config, parse_overrides
from eventclock.data import build_loader
from eventclock.metrics import save_json
from eventclock.models import build_model
from eventclock.train import evaluate


def main() -> None:
    parser = build_arg_parser("Evaluate a trained EventClock checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = deep_update(load_config(args.config), parse_overrides(args.set))
    device = torch.device(cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(cfg, int(ckpt["channels"]), int(ckpt["n_classes"])).to(device)
    model.load_state_dict(ckpt["model"])
    metrics = evaluate(model, build_loader(cfg, args.split, shuffle=False), device, cfg)
    print(metrics)
    if args.out:
        save_json(metrics, Path(args.out))


if __name__ == "__main__":
    main()
