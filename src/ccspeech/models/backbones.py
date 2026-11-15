"""BERT and Whisper encoders with the same classification interface."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import Tensor, nn


class Classifier(nn.Module):
    """Expose pooled representations independently from the classification head.

    The audio model uses the Whisper encoder only. Transcript matching is a
    preprocessing deliverable, not an implicit decoder or text-model input.
    """

    def __init__(self, encoder: nn.Module, num_classes: int, modality: str,
                 model_name: str, feature_dim: int, model_config: dict[str, Any]):
        super().__init__()
        if modality not in {"text", "audio"} or num_classes < 2:
            raise ValueError("Expected text/audio modality and at least two classes")
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.modality = modality
        self.model_name = model_name
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.model_config = deepcopy(model_config)
        self.resolved_revision = getattr(getattr(encoder, "config", None), "_commit_hash", None)

    def train(self, mode: bool = True) -> "Classifier":
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.encoder.parameters()):
            self.encoder.eval()
        return self

    def input_keys(self) -> tuple[str, ...]:
        if self.modality == "text":
            return "input_ids", "attention_mask", "token_type_ids"
        return "input_features", "attention_mask"

    def metadata(self) -> dict[str, Any]:
        return {"name": self.model_name, "modality": self.modality, "num_classes": self.num_classes,
                "feature_dim": self.feature_dim, "pretrained": self.model_config.get("pretrained", True),
                "requested_revision": self.model_config.get("revision"),
                "resolved_revision": self.resolved_revision}

    def encode(self, batch: dict[str, Tensor]) -> Tensor:
        if self.modality == "text":
            if "input_ids" not in batch:
                raise ValueError("Text input_ids are required")
            inputs = {name: batch[name] for name in self.input_keys() if name in batch}
            return self.encoder(**inputs, return_dict=True).last_hidden_state[:, 0]

        if "input_features" not in batch:
            raise ValueError("Audio input_features are required")
        features = batch["input_features"]
        if features.ndim != 3:
            raise ValueError("Whisper input_features must have shape [batch, mel, frames]")
        hidden = self.encoder(input_features=features, return_dict=True).last_hidden_state
        mask = batch.get("attention_mask")
        if mask is None:
            raise ValueError("Audio attention_mask is required to avoid pooling padded frames")
        if mask.ndim != 2 or mask.shape != (features.shape[0], features.shape[2]):
            raise ValueError("Audio attention_mask must match input feature frames")
        # Whisper's first convolution has stride 1 and its second has stride 2.
        # The feature extractor mask is at the pre-convolution frame rate.
        downsampled = mask[:, ::2][:, :hidden.shape[1]].to(hidden.dtype)
        if downsampled.shape[1] != hidden.shape[1] or (downsampled.sum(1) == 0).any():
            raise ValueError("Invalid Whisper frame mask")
        return (hidden * downsampled.unsqueeze(-1)).sum(1) / downsampled.sum(1, keepdim=True)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        return self.classifier(self.encode(batch))


def build_model(config: dict[str, Any], num_classes: int) -> Classifier:
    """Load real pretrained encoders, or explicit tiny random models for tests.

    ``pretrained=False`` is never inferred from a failed model download. It must
    be explicitly selected and accompanied by a configuration dictionary.
    """
    from transformers import BertConfig, BertModel, WhisperConfig, WhisperModel
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    modality = config.get("modality")
    if modality not in {"text", "audio"}:
        raise ValueError("model.modality must be text or audio")
    name = config.get("name", "bert-base-uncased" if modality == "text" else "openai/whisper-medium")
    pretrained = config.get("pretrained", True)
    kwargs = {"local_files_only": bool(config.get("local_files_only", False))}
    if config.get("revision"):
        kwargs["revision"] = config["revision"]
    if config.get("cache_dir"):
        kwargs["cache_dir"] = config["cache_dir"]
    if pretrained:
        if modality == "text":
            encoder = BertModel.from_pretrained(name, add_pooling_layer=False, **kwargs)
            feature_dim = encoder.config.hidden_size
        else:
            # Loading the complete pretrained container guarantees the published
            # encoder key mapping is correct. The unused decoder is then freed.
            container = WhisperModel.from_pretrained(name, **kwargs)
            encoder = container.encoder
            feature_dim = encoder.config.d_model
            del container
    else:
        if not isinstance(config.get("random_config"), dict) or not config["random_config"]:
            raise ValueError("pretrained=False requires an explicit random_config")
        if modality == "text":
            hf_config = BertConfig(**config["random_config"])
            encoder = BertModel(hf_config, add_pooling_layer=False)
            feature_dim = hf_config.hidden_size
        else:
            hf_config = WhisperConfig(**config["random_config"])
            encoder = WhisperEncoder(hf_config)
            feature_dim = hf_config.d_model
    if config.get("gradient_checkpointing", False):
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return Classifier(encoder, num_classes, modality, name, feature_dim, config)
