from __future__ import annotations

import math

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
    y: torch.Tensor,
    logits: torch.Tensor,
    mask: torch.Tensor | None,
    gamma: float = 0.5,
    ce_margin: float = 1.0,
) -> dict[str, torch.Tensor]:
    if mask is None:
        zero = logits.new_tensor(0.0)
        return {"suff": zero, "nec": zero, "suff_ce": zero, "nec_ce": zero}
    mask = mask.unsqueeze(1)
    suff_logits = model(x * mask)["logits"]
    nec_logits = model(x * (1.0 - mask))["logits"]
    suff = kl_divergence_from_logits(logits, suff_logits)
    nec_div = kl_divergence_from_logits(logits, nec_logits)
    nec = torch.relu(logits.new_tensor(gamma) - nec_div)
    suff_ce = F.cross_entropy(suff_logits, y)
    inverse_ce = F.cross_entropy(nec_logits, y)
    nec_ce = torch.relu(logits.new_tensor(ce_margin) - inverse_ce)
    return {"suff": suff, "nec": nec, "suff_ce": suff_ce, "nec_ce": nec_ce}


def oracle_localization_loss(
    velocity: torch.Tensor | None,
    evidence_mask: torch.Tensor | None,
    decoy_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if velocity is None or evidence_mask is None:
        zero = torch.tensor(0.0)
        return {"oracle_evidence": zero, "oracle_decoy": zero}
    evidence = (evidence_mask.to(velocity.device).float() > 0.5).float()
    decoy = (decoy_mask.to(velocity.device).float() > 0.5).float() if decoy_mask is not None else torch.zeros_like(evidence)
    prob = velocity / velocity.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    evidence_mass = (prob * evidence).sum(dim=-1)
    decoy_mass = (prob * decoy).sum(dim=-1)
    evidence_loss = -torch.log(evidence_mass.clamp_min(1e-8)).mean()
    decoy_loss = decoy_mass.mean()
    return {"oracle_evidence": evidence_loss, "oracle_decoy": decoy_loss}


def nuisance_intervention_losses(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    logits: torch.Tensor,
    span: int = 56,
    amp: float = 2.5,
    cycles: int = 11,
) -> dict[str, torch.Tensor]:
    """Inject a label-preserving high-frequency nuisance and discourage clock mass on it."""
    b, _, t = x.shape
    if span <= 0 or span >= t:
        zero = logits.new_tensor(0.0)
        return {"nuisance_consistency": zero, "nuisance_ce": zero, "nuisance_decoy": zero}
    device = x.device
    starts = torch.randint(8, max(9, t - span - 8), (b,), device=device)
    mask = torch.zeros((b, t), device=device, dtype=x.dtype)
    pos = torch.arange(span, device=device)
    for i, start in enumerate(starts):
        mask[i, start : start + span] = 1.0
    phase = (pos.float() / max(1, span - 1)).view(1, 1, span)
    window = torch.hann_window(span, periodic=False, device=device, dtype=x.dtype).view(1, 1, span)
    wave = torch.sign(torch.sin(2 * math.pi * cycles * phase)).to(dtype=x.dtype) * window
    noise = 0.15 * torch.randn((b, x.shape[1], span), device=device, dtype=x.dtype)
    decoy = amp * (wave + noise)
    x_aug = x.clone()
    for i, start in enumerate(starts):
        x_aug[i, :, start : start + span] = x_aug[i, :, start : start + span] + decoy[i]
    aug = model(x_aug)
    consistency = kl_divergence_from_logits(logits, aug["logits"])
    ce = F.cross_entropy(aug["logits"], y)
    velocity = aug.get("velocity")
    if velocity is None:
        decoy_mass = logits.new_tensor(0.0)
    else:
        prob = velocity / velocity.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        decoy_mass = (prob * mask).sum(dim=-1).mean()
    return {"nuisance_consistency": consistency, "nuisance_ce": ce, "nuisance_decoy": decoy_mass}


def saliency_alignment_loss(
    velocity: torch.Tensor | None,
    x: torch.Tensor,
    logits: torch.Tensor,
    y: torch.Tensor,
    blur: int = 7,
) -> torch.Tensor:
    if velocity is None or not x.requires_grad:
        return logits.new_tensor(0.0)
    selected = logits.gather(1, y.view(-1, 1)).sum()
    grad = torch.autograd.grad(selected, x, retain_graph=True, create_graph=False, allow_unused=True)[0]
    if grad is None:
        return logits.new_tensor(0.0)
    saliency = grad.detach().abs().mean(dim=1)
    if blur > 1:
        pad = blur // 2
        saliency = F.avg_pool1d(saliency.unsqueeze(1), kernel_size=blur, stride=1, padding=pad).squeeze(1)
        saliency = saliency[..., : velocity.shape[-1]]
    target = saliency + 1e-8
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    prob = velocity / velocity.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return -(target * torch.log(prob.clamp_min(1e-8))).sum(dim=-1).mean()


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

    if (
        coeffs.get("lambda_suff", 0) > 0
        or coeffs.get("lambda_nec", 0) > 0
        or coeffs.get("lambda_suff_ce", 0) > 0
        or coeffs.get("lambda_nec_ce", 0) > 0
    ) and "mask" in output:
        ev = evidence_losses(
            model,
            batch["x"],
            y,
            logits,
            output.get("mask"),
            gamma=float(coeffs.get("gamma", 0.5)),
            ce_margin=float(coeffs.get("ce_margin", 1.0)),
        )
        loss = loss + float(coeffs.get("lambda_suff", 0)) * ev["suff"]
        loss = loss + float(coeffs.get("lambda_nec", 0)) * ev["nec"]
        loss = loss + float(coeffs.get("lambda_suff_ce", 0)) * ev["suff_ce"]
        loss = loss + float(coeffs.get("lambda_nec_ce", 0)) * ev["nec_ce"]
        values["suff"] = float(ev["suff"].detach().cpu())
        values["nec"] = float(ev["nec"].detach().cpu())
        values["suff_ce"] = float(ev["suff_ce"].detach().cpu())
        values["nec_ce"] = float(ev["nec_ce"].detach().cpu())
    if (
        coeffs.get("lambda_nuisance_consistency", 0) > 0
        or coeffs.get("lambda_nuisance_ce", 0) > 0
        or coeffs.get("lambda_nuisance_decoy", 0) > 0
    ):
        ni = nuisance_intervention_losses(
            model,
            batch["x"],
            y,
            logits,
            span=int(coeffs.get("nuisance_span", 56)),
            amp=float(coeffs.get("nuisance_amp", 2.5)),
            cycles=int(coeffs.get("nuisance_cycles", 11)),
        )
        loss = loss + float(coeffs.get("lambda_nuisance_consistency", 0)) * ni["nuisance_consistency"]
        loss = loss + float(coeffs.get("lambda_nuisance_ce", 0)) * ni["nuisance_ce"]
        loss = loss + float(coeffs.get("lambda_nuisance_decoy", 0)) * ni["nuisance_decoy"]
        values["nuisance_consistency"] = float(ni["nuisance_consistency"].detach().cpu())
        values["nuisance_ce"] = float(ni["nuisance_ce"].detach().cpu())
        values["nuisance_decoy"] = float(ni["nuisance_decoy"].detach().cpu())
    if coeffs.get("lambda_saliency_clock", 0) > 0:
        saliency = saliency_alignment_loss(
            velocity,
            batch["x"],
            logits,
            y,
            blur=int(coeffs.get("saliency_blur", 7)),
        ).to(logits.device)
        loss = loss + float(coeffs.get("lambda_saliency_clock", 0)) * saliency
        values["saliency_clock"] = float(saliency.detach().cpu())
    if (
        coeffs.get("lambda_oracle_evidence", 0) > 0
        or coeffs.get("lambda_oracle_decoy", 0) > 0
    ) and "evidence_mask" in batch:
        loc = oracle_localization_loss(velocity, batch["evidence_mask"], batch.get("decoy_mask"))
        loc = {key: value.to(logits.device) for key, value in loc.items()}
        loss = loss + float(coeffs.get("lambda_oracle_evidence", 0)) * loc["oracle_evidence"]
        loss = loss + float(coeffs.get("lambda_oracle_decoy", 0)) * loc["oracle_decoy"]
        values["oracle_evidence"] = float(loc["oracle_evidence"].detach().cpu())
        values["oracle_decoy"] = float(loc["oracle_decoy"].detach().cpu())
    values["total"] = float(loss.detach().cpu())
    return loss, values
