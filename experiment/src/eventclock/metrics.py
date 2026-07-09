from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def classification_metrics(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
    pred = probs.argmax(axis=-1)
    target = y.detach().cpu().numpy()
    out = {"accuracy": float((pred == target).mean())}
    try:
        from sklearn.metrics import f1_score, roc_auc_score

        out["macro_f1"] = float(f1_score(target, pred, average="macro", zero_division=0))
        if probs.shape[1] == 2:
            out["auroc"] = float(roc_auc_score(target, probs[:, 1]))
        else:
            out["macro_auroc"] = float(roc_auc_score(target, probs, multi_class="ovr", average="macro"))
    except Exception:
        pass
    return out


def mask_overlap(pred_mask: torch.Tensor, true_mask: torch.Tensor | None) -> dict[str, float]:
    if true_mask is None:
        return {}
    pred = (pred_mask.detach().cpu() > pred_mask.detach().cpu().median(dim=-1, keepdim=True).values).float()
    true = (true_mask.detach().cpu() > 0.5).float()
    inter = (pred * true).sum(dim=-1)
    union = ((pred + true) > 0).float().sum(dim=-1).clamp_min(1)
    precision = inter / pred.sum(dim=-1).clamp_min(1)
    recall = inter / true.sum(dim=-1).clamp_min(1)
    return {
        "mask_iou": float((inter / union).mean()),
        "mask_precision": float(precision.mean()),
        "mask_recall": float(recall.mean()),
    }


@torch.no_grad()
def deletion_insertion_auc(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    importance: torch.Tensor,
    steps: int = 10,
) -> dict[str, float]:
    b, _, t = x.shape
    order = torch.argsort(importance, dim=-1, descending=True)
    fractions = torch.linspace(0, 1, steps + 1, device=x.device)
    del_scores = []
    ins_scores = []
    baseline = torch.zeros_like(x)
    for frac in fractions:
        k = int(float(frac) * t)
        del_x = x.clone()
        ins_x = baseline.clone()
        if k > 0:
            idx = order[:, :k]
            del_x.scatter_(2, idx.unsqueeze(1).expand(-1, x.shape[1], -1), 0)
            ins_x.scatter_(2, idx.unsqueeze(1).expand(-1, x.shape[1], -1), torch.gather(x, 2, idx.unsqueeze(1).expand(-1, x.shape[1], -1)))
        del_prob = F.softmax(model(del_x)["logits"], dim=-1).gather(1, y.view(-1, 1)).mean()
        ins_prob = F.softmax(model(ins_x)["logits"], dim=-1).gather(1, y.view(-1, 1)).mean()
        del_scores.append(float(del_prob.cpu()))
        ins_scores.append(float(ins_prob.cpu()))
    xs = fractions.detach().cpu().numpy()
    return {
        "deletion_auc": float(np.trapz(np.asarray(del_scores), xs)),
        "insertion_auc": float(np.trapz(np.asarray(ins_scores), xs)),
    }


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))

