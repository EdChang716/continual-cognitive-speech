"""Query-local supervised updates that leave global model state unchanged."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F


@contextmanager
def temporary_parameters(model: nn.Module) -> Iterator[dict[str, torch.Tensor]]:
    """Snapshot only trainable weights, plus buffers, gradients, and module modes."""
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    anchors = {name: parameter.detach().clone() for name, parameter in named}
    gradients = {name: None if p.grad is None else p.grad.detach().clone() for name, p in named}
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    modes = [(module, module.training) for module in model.modules()]
    try:
        yield anchors
    finally:
        with torch.no_grad():
            for name, parameter in named:
                parameter.copy_(anchors[name])
                parameter.grad = gradients[name]
            for name, value in model.named_buffers():
                value.copy_(buffers[name])
        for module, mode in modes:
            module.training = mode


def local_adapt_predict(model: nn.Module, query_inputs: dict[str, torch.Tensor],
                        neighbor_inputs: dict[str, torch.Tensor] | None,
                        neighbor_labels: torch.Tensor | None, *,
                        steps: int = 30, lr: float = 1e-3,
                        anchor_strength: float = 1e-3,
                        max_grad_norm: float | None = 1.0,
                        optimizer_name: str = "adam",
                        microbatch_size: int = 4) -> torch.Tensor:
    """Adapt to equally weighted neighbors, predict one query, and restore state.

    Query labels are intentionally absent from this API. A fresh local optimizer
    is discarded after prediction, so its moments cannot contaminate later
    queries or the global training optimizer. One call corresponds to one query;
    callers must not share a temporary adaptation across unrelated queries.
    """
    if steps < 0 or lr <= 0 or anchor_strength < 0 or microbatch_size < 1:
        raise ValueError("Invalid local-adaptation hyperparameters")
    if not query_inputs or any(value.shape[0] != 1 for value in query_inputs.values()):
        raise ValueError("Query-local adaptation requires exactly one query")
    if (neighbor_inputs is None) != (neighbor_labels is None):
        raise ValueError("Neighbor inputs and labels must be supplied together")
    if neighbor_labels is not None and not neighbor_labels.numel():
        neighbor_inputs = neighbor_labels = None
    if neighbor_inputs is not None and any(value.shape[0] != len(neighbor_labels) for value in neighbor_inputs.values()):
        raise ValueError("Neighbor input and label counts differ")
    parameters = [p for p in model.parameters() if p.requires_grad]
    with temporary_parameters(model) as anchors:
        if steps and neighbor_labels is not None and parameters:
            if optimizer_name == "adam":
                optimizer = torch.optim.Adam(parameters, lr=lr)
            elif optimizer_name == "sgd":
                optimizer = torch.optim.SGD(parameters, lr=lr)
            else:
                raise ValueError("Local optimizer must be adam or sgd")
            model.train()
            # Override any outer inference no_grad context. Do not call this
            # from torch.inference_mode(), which disables autograd tensors.
            with torch.enable_grad():
                for _ in range(steps):
                    optimizer.zero_grad(set_to_none=True)
                    count = len(neighbor_labels)
                    for start in range(0, count, microbatch_size):
                        stop = min(start + microbatch_size, count)
                        inputs = {key: value[start:stop] for key, value in neighbor_inputs.items()}
                        logits = model(inputs)
                        loss = F.cross_entropy(logits, neighbor_labels[start:stop]) * ((stop-start)/count)
                        loss.backward()
                    # Exact gradient of lambda * SUM ||p-anchor||^2, applied
                    # once per local step without retaining a second full-model
                    # regularization graph. Microbatching changes no CE weights.
                    if anchor_strength:
                        with torch.no_grad():
                            for name, parameter in model.named_parameters():
                                if parameter.requires_grad:
                                    penalty_gradient = 2 * anchor_strength * (parameter - anchors[name])
                                    if parameter.grad is None:
                                        parameter.grad = penalty_gradient
                                    else:
                                        parameter.grad.add_(penalty_gradient)
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
                    optimizer.step()
        model.eval()
        with torch.no_grad():
            return model(query_inputs).detach().clone()
