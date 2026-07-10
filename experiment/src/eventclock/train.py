from __future__ import annotations

import time
from pathlib import Path

import torch

from eventclock.config import build_arg_parser, deep_update, load_config, parse_overrides
from eventclock.data import build_loader
from eventclock.data.robustness import apply_robustness
from eventclock.losses import total_loss
from eventclock.metrics import classification_metrics, deletion_insertion_auc, importance_localization, mask_overlap, save_json
from eventclock.models import build_model
from eventclock.seed import seed_everything


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def infer_shape_and_classes(loader, cfg: dict) -> tuple[int, int]:
    batch = next(iter(loader))
    channels = int(batch["x"].shape[1])
    if hasattr(loader.dataset, "y"):
        observed_max = int(max(loader.dataset.y))
    else:
        observed_max = int(batch["y"].max().item())
    if "n_classes" in cfg["task"]:
        n_classes = int(cfg["task"]["n_classes"])
    else:
        n_classes = observed_max + 1
    if observed_max >= n_classes:
        raise ValueError(f"Observed label {observed_max}, but task.n_classes={n_classes}.")
    return channels, n_classes


def train_one_epoch(model, loader, optimizer, device, cfg: dict) -> dict[str, float]:
    model.train()
    running: dict[str, float] = {}
    n = 0
    for batch in loader:
        batch = _move(batch, device)
        batch["x"] = apply_robustness(batch["x"], cfg.get("robustness_train"))
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["x"])
        loss, values = total_loss(model, batch, output, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 1.0)))
        optimizer.step()
        bs = batch["x"].shape[0]
        n += bs
        for key, value in values.items():
            running[key] = running.get(key, 0.0) + value * bs
    return {key: value / max(1, n) for key, value in running.items()}


@torch.no_grad()
def evaluate(model, loader, device, cfg: dict, robustness: dict | None = None) -> dict[str, float]:
    model.eval()
    logits_all, y_all = [], []
    overlap_values: list[dict[str, float]] = []
    evidence_values: list[dict[str, float]] = []
    max_evidence_batches = int(cfg.get("eval", {}).get("max_evidence_batches", 2))
    for bi, batch in enumerate(loader):
        batch = _move(batch, device)
        x = apply_robustness(batch["x"], robustness)
        out = model(x)
        logits_all.append(out["logits"].detach().cpu())
        y_all.append(batch["y"].detach().cpu())
        if robustness is None and "mask" in out and "evidence_mask" in batch:
            overlap_values.append(mask_overlap(out["mask"], batch["evidence_mask"]))
        if robustness is None and "velocity" in out and "evidence_mask" in batch:
            overlap_values.append(importance_localization(out["velocity"], batch["evidence_mask"], batch.get("decoy_mask")))
        if "velocity" in out and bi < max_evidence_batches:
            evidence_values.append(deletion_insertion_auc(model, x, batch["y"], out["velocity"]))
    logits = torch.cat(logits_all)
    y = torch.cat(y_all)
    metrics = classification_metrics(logits, y)
    for values in (overlap_values, evidence_values):
        if values:
            keys = set().union(*(v.keys() for v in values))
            metrics.update({key: float(sum(v.get(key, 0.0) for v in values) / len(values)) for key in keys})
    if robustness is None and "importance_top_iou" in metrics:
        lift_gap = max(0.0, metrics.get("importance_lift_gap", 0.0))
        top_iou = metrics.get("importance_top_iou", 0.0)
        top_decoy = metrics.get("importance_top_decoy_fraction", 0.0)
        metrics["evidence_score"] = metrics["macro_f1"] + 0.5 * top_iou + 0.2 * lift_gap - 0.1 * top_decoy
    return metrics


def run(cfg: dict) -> dict:
    seed_everything(int(cfg.get("seed", 0)))
    out_dir = Path(cfg.get("output_dir", "experiment/outputs/run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    train_loader = build_loader(cfg, "train", shuffle=True)
    val_loader = build_loader(cfg, "val", shuffle=False)
    test_loader = build_loader(cfg, "test", shuffle=False)
    channels, n_classes = infer_shape_and_classes(train_loader, cfg)
    model = build_model(cfg, channels, n_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 1e-3)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    best_metric = -1.0
    best_path = out_dir / "best.pt"
    history = []
    monitor = cfg["train"].get("monitor", "macro_f1")
    for epoch in range(1, int(cfg["train"].get("epochs", 10)) + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg)
        val_metrics = evaluate(model, val_loader, device, cfg)
        score = val_metrics.get(monitor, val_metrics.get("accuracy", 0.0))
        row = {"epoch": epoch, "seconds": time.time() - start, "train": train_loss, "val": val_metrics}
        history.append(row)
        print(row, flush=True)
        if score > best_metric:
            best_metric = score
            torch.save({"model": model.state_dict(), "cfg": cfg, "channels": channels, "n_classes": n_classes}, best_path)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    result = {"best_val": best_metric, "test": evaluate(model, test_loader, device, cfg), "history": history}
    robust_results = {}
    for item in cfg.get("robustness_eval", []):
        name = item.get("name", "robust")
        robust_results[name] = evaluate(model, test_loader, device, cfg, robustness=item)
    if robust_results:
        result["robustness"] = robust_results
    save_json(result, out_dir / "metrics.json")
    print(f"saved {best_path}", flush=True)
    return result


def main() -> None:
    parser = build_arg_parser("Train EventClock experiments.")
    args = parser.parse_args()
    cfg = deep_update(load_config(args.config), parse_overrides(args.set))
    run(cfg)


if __name__ == "__main__":
    main()
