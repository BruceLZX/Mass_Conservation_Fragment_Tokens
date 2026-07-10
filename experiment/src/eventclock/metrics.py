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


def importance_localization(
    importance: torch.Tensor,
    evidence_mask: torch.Tensor | None,
    decoy_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    if evidence_mask is None:
        return {}
    imp = importance.detach().cpu().float().clamp_min(0)
    evidence = (evidence_mask.detach().cpu().float() > 0.5).float()
    decoy = (decoy_mask.detach().cpu().float() > 0.5).float() if decoy_mask is not None else torch.zeros_like(evidence)
    prob = imp / imp.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    evidence_mass = (prob * evidence).sum(dim=-1)
    decoy_mass = (prob * decoy).sum(dim=-1)
    evidence_fraction = evidence.mean(dim=-1).clamp_min(1e-8)
    decoy_fraction = decoy.mean(dim=-1).clamp_min(1e-8)
    top = torch.zeros_like(evidence)
    for i in range(imp.shape[0]):
        k = max(1, int(evidence[i].sum().item()))
        idx = torch.topk(imp[i], k=min(k, imp.shape[-1])).indices
        top[i, idx] = 1.0
    inter = (top * evidence).sum(dim=-1)
    union = ((top + evidence) > 0).float().sum(dim=-1).clamp_min(1)
    precision = inter / top.sum(dim=-1).clamp_min(1)
    recall = inter / evidence.sum(dim=-1).clamp_min(1)
    top_decoy = (top * decoy).sum(dim=-1) / top.sum(dim=-1).clamp_min(1)
    return {
        "importance_evidence_mass": float(evidence_mass.mean()),
        "importance_decoy_mass": float(decoy_mass.mean()),
        "importance_evidence_lift": float((evidence_mass / evidence_fraction).mean()),
        "importance_decoy_lift": float((decoy_mass / decoy_fraction).mean()),
        "importance_lift_gap": float(((evidence_mass / evidence_fraction) - (decoy_mass / decoy_fraction)).mean()),
        "importance_evidence_minus_decoy": float((evidence_mass - decoy_mass).mean()),
        "importance_top_iou": float((inter / union).mean()),
        "importance_top_precision": float(precision.mean()),
        "importance_top_recall": float(recall.mean()),
        "importance_top_decoy_fraction": float(top_decoy.mean()),
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
        "deletion_auc": float(np.trapezoid(np.asarray(del_scores), xs)),
        "insertion_auc": float(np.trapezoid(np.asarray(ins_scores), xs)),
    }


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))
