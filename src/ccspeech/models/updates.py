"""Select trainable modules and inject native low-rank linear adapters."""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn

from .backbones import Classifier

POLICIES = {"full", "classifier_only", "encoder_lora_only", "lora_classifier"}


class LoRALinear(nn.Module):
    """Frozen W plus alpha/r * B(A(dropout(x))), with zero-initialized B."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")
        if not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid LoRA alpha or dropout")
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.base.requires_grad_(False)

    @property
    def weight(self) -> Tensor:
        return self.base.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base.bias

    def forward(self, inputs: Tensor) -> Tensor:
        residual = torch.nn.functional.linear(self.dropout(inputs), self.lora_A)
        residual = torch.nn.functional.linear(residual, self.lora_B)
        return self.base(inputs) + residual * self.scaling


def _layers(model: Classifier) -> nn.ModuleList:
    if model.modality == "text":
        return model.encoder.encoder.layer
    return model.encoder.layers


def _layer_indices(total: int, placement: str, count: int | None) -> list[int]:
    if placement not in {"all", "early", "late"}:
        raise ValueError("LoRA placement must be all, early, or late")
    if count is None:
        count = total
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= total:
        raise ValueError(f"num_layers must be in 1..{total}")
    if placement == "all":
        if count != total:
            raise ValueError("placement=all must select all layers; use early/late for a subset")
        return list(range(total))
    return list(range(count)) if placement == "early" else list(range(total - count, total))


def _matches(name: str, targets: Iterable[str]) -> bool:
    return any(name == target or name.endswith("." + target) for target in targets)


def configure_updates(model: Classifier, config: dict) -> dict:
    """Apply a policy to a common trained checkpoint and report exact modules.

    Reapplying the same LoRA placement/rank is supported for checkpoint restore;
    changing adapter layouts on an already adapted model is an explicit error.
    """
    policy = config.get("policy", config.get("name", "full"))
    if policy not in POLICIES:
        raise ValueError(f"Unknown update policy: {policy}")
    model.requires_grad_(False)
    selected = []
    adapted_names = []
    existing = {name: module for name, module in model.encoder.named_modules() if isinstance(module, LoRALinear)}
    if policy == "full":
        if existing:
            raise ValueError("Full updating requires a model without injected adapters")
        model.requires_grad_(True)
    elif policy == "classifier_only":
        model.classifier.requires_grad_(True)
    else:
        layers = _layers(model)
        selected = _layer_indices(len(layers), config.get("placement", "all"), config.get("num_layers"))
        targets = config.get("targets") or (["query", "value"] if model.modality == "text" else
                                             ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])
        if not isinstance(targets, list) or not targets or any(not isinstance(target, str) for target in targets):
            raise ValueError("LoRA targets must be a nonempty list of module names")
        rank = config.get("rank", 8)
        alpha = float(config.get("alpha", 2 * rank))
        dropout = float(config.get("dropout", 0.05))
        prefix = "encoder.layer" if model.modality == "text" else "layers"
        desired = []
        for index in selected:
            layer = layers[index]
            for relative, module in list(layer.named_modules()):
                if relative and _matches(relative, targets) and isinstance(module, (nn.Linear, LoRALinear)):
                    name = f"{prefix}.{index}.{relative}"
                    desired.append((layer, relative, name, module))
        if not desired or any(not any(name.startswith(f"{prefix}.{index}.") for _, _, name, _ in desired)
                              for index in selected):
            raise ValueError("No requested linear targets found in one or more selected layers")
        if existing and set(existing) != {name for _, _, name, _ in desired}:
            raise ValueError("Existing adapters use a different layer or target layout")
        for layer, relative, name, module in desired:
            if isinstance(module, LoRALinear):
                if (module.rank != rank or module.alpha != alpha
                        or module.dropout.p != dropout):
                    raise ValueError("Existing adapter hyperparameters differ")
                adapter = module
            else:
                adapter = LoRALinear(module, rank, alpha, dropout)
                parent_name, _, leaf = relative.rpartition(".")
                parent = layer.get_submodule(parent_name) if parent_name else layer
                setattr(parent, leaf, adapter)
            adapter.lora_A.requires_grad_(True)
            adapter.lora_B.requires_grad_(True)
            adapted_names.append(name)
        model.classifier.requires_grad_(policy == "lora_classifier")
    groups = parameter_groups(model)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if not trainable:
        raise ValueError("Update policy selected no trainable parameters")
    return {
        "policy": policy,
        "selected_layers": selected,
        "adapter_modules": adapted_names,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": trainable,
        "encoder_trainable_parameters": sum(parameter.numel() for parameter in groups["encoder"]),
        "head_trainable_parameters": sum(parameter.numel() for parameter in groups["head"]),
    }


def parameter_groups(model: Classifier) -> dict[str, list[nn.Parameter]]:
    """Disjoint trainable groups used by uniform or multiple-timescale updates."""
    return {
        "encoder": [parameter for parameter in model.encoder.parameters() if parameter.requires_grad],
        "head": [parameter for parameter in model.classifier.parameters() if parameter.requires_grad],
    }
