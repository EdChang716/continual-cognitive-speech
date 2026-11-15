"""Validated, explicit experiment configuration with stable run fingerprints."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

METHODS = {"sequential", "er", "derpp", "sparse_replay", "mbpa", "mbpa_random", "joint", "separate"}


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def semantic_config(config: dict) -> dict:
    """Identify scientific settings independently of machine paths/devices."""
    value = deepcopy(config)
    value.get("data", {}).pop("root", None)
    for key in ("cache_dir", "local_files_only"):
        value.get("model", {}).pop(key, None)
    for key in ("device", "num_threads"):
        value.get("runtime", {}).pop(key, None)
    value.pop("experiment_family", None)
    value.pop("description", None)
    return value


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Expected experiment schema_version=1")
    phase = config.get("phase")
    modality = {"text": "text", "speech": "audio"}.get(phase)
    if not modality or config.get("model", {}).get("modality") != modality:
        raise ValueError("phase must agree with model.modality")
    if config.get("data", {}).get("modality") != modality:
        raise ValueError("data.modality must agree with model.modality")
    domains = config.get("domains", [])
    expected = {"amazon", "yelp"} if phase == "text" else {"en", "zh"}
    if len(domains) != 2 or set(domains) != expected:
        raise ValueError(f"domains must specify both domains exactly once: {expected}")
    if config.get("method") not in METHODS:
        raise ValueError("Unknown method")
    policy = config.get("updates", {}).get("policy")
    if policy not in {"full", "classifier_only", "encoder_lora_only", "lora_classifier"}:
        raise ValueError("Unknown parameter-update policy")
    if config["method"] in {"joint", "separate"} and policy != "full":
        raise ValueError("Joint/separate references use full updating from the pretrained model")
    if isinstance(config.get("seed"), bool) or not isinstance(config.get("seed"), int):
        raise ValueError("seed must be an integer")
    training = config.get("training", {})
    for key in ("source_epochs", "target_epochs", "batch_size"):
        if type(training.get(key)) is not int or training[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if training.get("lr", 0) <= 0:
        raise ValueError("training.lr must be positive")
    if training.get("precision", "float32") not in {"float32", "bfloat16"}:
        raise ValueError("Supported training precisions are float32 and bfloat16")
    for key in ("head_interval", "encoder_interval"):
        value = config.get("timescales", {}).get(key, 1)
        if type(value) is not int or value < 1:
            raise ValueError(f"timescales.{key} must be a positive integer")
    memory = config.get("memory", {})
    if type(memory.get("capacity", 0)) is not int or memory.get("capacity", 0) < 0:
        raise ValueError("memory.capacity must be a nonnegative integer")
    if config["method"] in {"er", "derpp", "sparse_replay", "mbpa", "mbpa_random"} and memory.get("capacity", 0) < 1:
        raise ValueError("Replay methods require positive memory capacity")
    if config["method"] in {"mbpa", "mbpa_random"}:
        local = config.get("local_adaptation", {})
        for key in ("neighbors", "steps", "microbatch_size"):
            if type(local.get(key)) is not int or local[key] < 1:
                raise ValueError(f"local_adaptation.{key} must be positive")
        if local.get("lr", 0) <= 0 or local.get("anchor_strength", -1) < 0:
            raise ValueError("Invalid local-adaptation learning rate or anchor penalty")
    if config.get("evaluation", {}).get("split", "validation") != "validation":
        raise ValueError("Training selects settings on validation only; use the separate final-test command")
    runtime = config.get("runtime", {})
    if any(runtime.get(key) is not None for key in ("max_steps_per_stage", "train_limit", "evaluation_limit")):
        if runtime.get("smoke") is not True:
            raise ValueError("Truncated runs require runtime.smoke=true and cannot be used as final results")
    if not config["model"].get("pretrained", True) and runtime.get("smoke") is not True:
        raise ValueError("Random backbones are allowed only in explicit smoke tests")


def base_config(phase: str) -> dict:
    speech = phase == "speech"
    if phase not in {"text", "speech"}:
        raise ValueError("Unknown phase")
    return {
        "schema_version": 1, "phase": phase, "method": "derpp", "seed": 42,
        "domains": ["en", "zh"] if speech else ["amazon", "yelp"],
        "model": {"modality": "audio" if speech else "text",
                  "name": "openai/whisper-medium" if speech else "google-bert/bert-base-uncased",
                  "revision": "abdf7c39ab9d0397620ccaea8974cc764cd0953e" if speech else "86b5e0934494bd15c9632b12f734a8a67f723594",
                  "pretrained": True, "gradient_checkpointing": True},
        "data": {"modality": "audio" if speech else "text", "root": "data",
                 "text_dir": "processed/text", "speech_dir": "processed/speech_v2",
                 "test_labels": "locked/taukadial/test_labels.csv"},
        "training": {"source_epochs": 5 if speech else 1, "target_epochs": 5 if speech else 1,
                     "batch_size": 2 if speech else 16, "lr": 3e-5,
                     "weight_decay": .01, "max_grad_norm": 1., "max_length": 256,
                     "precision": "float32", "checkpoint_every": 500,
                     "source_selection": "source_validation_accuracy",
                     "target_selection": "domain_averaged_validation_accuracy"},
        "updates": {"policy": "full", "rank": 8, "alpha": 16, "dropout": .05,
                    "placement": "all"},
        "memory": {"capacity": 128 if speech else 1024, "replay_batch_size": 2 if speech else 16,
                   "alpha": .5, "beta": .5, "sparse_frequency": 100 if speech else 10000,
                   "sparse_batch_size": 100, "sparse_microbatch_size": 2 if speech else 16},
        "local_adaptation": {"neighbors": 32, "steps": 30, "lr": .001,
                             "anchor_strength": .001, "microbatch_size": 2 if speech else 8},
        "timescales": {"head_interval": 1, "encoder_interval": 1},
        "evaluation": {"split": "validation", "batch_size": 4 if speech else 32,
                       "bootstrap_samples": 1000, "confidence": .95},
        "runtime": {"device": "cuda", "num_threads": 4},
        "protocol": {"common_supervised_source_checkpoint": True,
                     "memory_admission": "once_per_unique_training_example_on_first_epoch",
                     "speech_aggregation": "window_mean_then_equal_recording_mean_per_participant"},
    }


def merge_config(base: dict, changes: dict) -> dict:
    result = deepcopy(base)
    for key, value in changes.items():
        result[key] = merge_config(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result
