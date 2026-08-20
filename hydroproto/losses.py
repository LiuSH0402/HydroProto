from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("features must have shape [batch, dim]")
    if len(features) != len(labels):
        raise ValueError("features and labels have inconsistent lengths")
    features = F.normalize(features, dim=1)
    logits = features @ features.T / temperature
    batch_size = len(labels)
    self_mask = torch.eye(batch_size, device=features.device, dtype=torch.bool)
    logits = logits.masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    positive_count = positive_mask.sum(dim=1)
    if torch.any(positive_count == 0):
        raise ValueError("The batch contains an anchor without a same-class positive")
    positive_log_prob = torch.where(positive_mask, log_prob, torch.zeros_like(log_prob)).sum(dim=1)
    return -(positive_log_prob / positive_count).mean()


def local_supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Contrast local same-class neighbors while ignoring distant same-class pairs."""

    if features.ndim != 2 or positive_mask.shape != (len(features), len(features)):
        raise ValueError("Invalid feature or positive-mask shape")
    features = F.normalize(features, dim=1)
    positive_mask = positive_mask.to(device=features.device, dtype=torch.bool)
    self_mask = torch.eye(len(features), device=features.device, dtype=torch.bool)
    positive_mask = positive_mask & ~self_mask
    negative_mask = labels[:, None].ne(labels[None, :])
    denominator_mask = (positive_mask | negative_mask) & ~self_mask
    positive_count = positive_mask.sum(dim=1)
    valid_anchor = positive_count > 0
    if not torch.any(valid_anchor):
        raise ValueError("The batch contains no local positive pairs")

    logits = features @ features.T / float(temperature)
    logits = logits.masked_fill(~denominator_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_prob = torch.where(positive_mask, log_prob, torch.zeros_like(log_prob)).sum(dim=1)
    per_anchor = -positive_log_prob / positive_count.clamp_min(1)
    return per_anchor[valid_anchor].mean()
