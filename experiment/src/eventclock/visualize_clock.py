from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from eventclock.config import build_arg_parser, deep_update, load_config, parse_overrides
from eventclock.data import build_loader
from eventclock.models import build_model


@torch.no_grad()
def main() -> None:
    parser = build_arg_parser("Visualize learned clock density.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="experiment/outputs/clock_examples.png")
    parser.add_argument("--n", type=int, default=4)
    args = parser.parse_args()
    cfg = deep_update(load_config(args.config), parse_overrides(args.set))
    device = torch.device(cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(cfg, int(ckpt["channels"]), int(ckpt["n_classes"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    batch = next(iter(build_loader(cfg, "test", shuffle=False)))
    x = batch["x"][: args.n].to(device)
    out = model(x)
    velocity = out.get("velocity")
    if velocity is None:
        raise RuntimeError("Selected model does not expose a learned/derived velocity.")
    fig, axes = plt.subplots(args.n, 1, figsize=(10, 2.2 * args.n), squeeze=False)
    for i in range(args.n):
        ax = axes[i, 0]
        sig = x[i, 0].detach().cpu()
        vel = velocity[i].detach().cpu()
        ax.plot(sig, label="signal", color="black", linewidth=1)
        ax2 = ax.twinx()
        ax2.plot(vel, label="clock density", color="tab:red", alpha=0.75)
        if "evidence_mask" in batch:
            mask = batch["evidence_mask"][i].detach().cpu()
            ax.fill_between(range(len(mask)), sig.min(), sig.max(), where=mask > 0.5, color="tab:green", alpha=0.15)
        ax.set_title(f"label={int(batch['y'][i])}")
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
