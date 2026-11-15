"""Shared supervised replay objectives with explicit reductions."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def er_loss(current_logits: torch.Tensor, current_labels: torch.Tensor,
            replay_logits: torch.Tensor | None = None,
            replay_labels: torch.Tensor | None = None,
            replay_weight: float = 1.0) -> torch.Tensor:
    loss = F.cross_entropy(current_logits, current_labels)
    if (replay_logits is None) != (replay_labels is None):
        raise ValueError("Replay logits and labels must be supplied together")
    if replay_logits is not None:
        loss = loss + replay_weight * F.cross_entropy(replay_logits, replay_labels)
    return loss


def derpp_loss(current_logits: torch.Tensor, current_labels: torch.Tensor,
               distill_logits: torch.Tensor | None = None,
               stored_logits: torch.Tensor | None = None,
               replay_logits: torch.Tensor | None = None,
               replay_labels: torch.Tensor | None = None,
               alpha: float = 0.5, beta: float = 0.5) -> torch.Tensor:
    if alpha < 0 or beta < 0:
        raise ValueError("DER coefficients must be nonnegative")
    loss = er_loss(current_logits, current_labels, replay_logits, replay_labels, beta)
    if (distill_logits is None) != (stored_logits is None):
        raise ValueError("Distillation predictions and historical logits must be supplied together")
    if distill_logits is not None:
        if distill_logits.shape != stored_logits.shape:
            raise ValueError("Historical and current logit dimensions differ")
        # Mammoth's DER++ implementation uses elementwise mean-squared error.
        # Historical logits remain fixed targets; no softmax or temperature.
        loss = loss + alpha * F.mse_loss(distill_logits, stored_logits.detach())
    return loss
