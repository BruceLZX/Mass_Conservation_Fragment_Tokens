from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_divergence_from_logits(reference_logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    p_ref = F.softmax(reference_logits.detach(), dim=-1)
    log_target = F.log_softmax(target_logits, dim=-1)
    return F.kl_div(log_target, p_ref, reduction="batchmean")


def smoothness_loss(velocity: torch.Tensor | None) -> torch.Tensor:
    if velocity is None:
        return torch.tensor(0.0)
    return torch.abs(velocity[..., 1:] - velocity[..., :-1]).mean()


def entropy_loss(velocity: torch.Tensor | None) -> torch.Tensor:
    if velocity is None:
        return torch.tensor(0.0)
    p = velocity / velocity.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    entropy = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=-1).mean()
    return -entropy


def evidence_losses(
    model: torch.nn.Module,
    x: torch.Tensor,
    logits: torch.Tensor,
    mask: torch.Tensor | None,
    gamma: float = 0.5,
) -> dict[str, torch.Tensor]:
    if mask is None:
        zero = logits.new_tensor(0.0)
        return {"suff": zero, "nec": zero}
    mask = mask.unsqueeze(1)
    suff_logits = model(x * mask)["logits"]
    nec_logits = model(x * (1.0 - mask))["logits"]
    suff = kl_divergence_from_logits(logits, suff_logits)
    nec_div = kl_divergence_from_logits(logits, nec_logits)
    nec = torch.relu(logits.new_tensor(gamma) - nec_div)
    return {"suff": suff, "nec": nec}


def total_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    output: dict[str, torch.Tensor],
    cfg: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    y = batch["y"]
    logits = output["logits"]
    ce = F.cross_entropy(logits, y)
    coeffs = cfg.get("loss", {})
    loss = ce
    values = {"task": float(ce.detach().cpu())}

    velocity = output.get("velocity")
    if coeffs.get("lambda_smooth", 0) > 0 and velocity is not None:
        smooth = smoothness_loss(velocity).to(logits.device)
        loss = loss + float(coeffs["lambda_smooth"]) * smooth
        values["smooth"] = float(smooth.detach().cpu())
    if coeffs.get("lambda_entropy", 0) > 0 and velocity is not None:
        ent = entropy_loss(velocity).to(logits.device)
        loss = loss + float(coeffs["lambda_entropy"]) * ent
        values["entropy"] = float(ent.detach().cpu())

    if (coeffs.get("lambda_suff", 0) > 0 or coeffs.get("lambda_nec", 0) > 0) and "mask" in output:
        ev = evidence_losses(model, batch["x"], logits, output.get("mask"), gamma=float(coeffs.get("gamma", 0.5)))
        loss = loss + float(coeffs.get("lambda_suff", 0)) * ev["suff"]
        loss = loss + float(coeffs.get("lambda_nec", 0)) * ev["nec"]
        values["suff"] = float(ev["suff"].detach().cpu())
        values["nec"] = float(ev["nec"].detach().cpu())
    values["total"] = float(loss.detach().cpu())
    return loss, values

