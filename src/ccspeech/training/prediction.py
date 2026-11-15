"""Label-blind prediction, temporary query adaptation, and domain evaluation."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import time

import numpy as np
import torch

from ccspeech.evaluation.metrics import (
    aggregate_speech_predictions, bootstrap_intervals, classification_metrics,
)
from ccspeech.methods.local_adaptation import local_adapt_predict
from .state import isolated_rng, portable_records


def _device(model) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return buffer.device if buffer is not None else torch.device("cpu")


def _sync(*devices) -> None:
    for device in set(devices):
        if device.type == "cuda":
            torch.cuda.synchronize(device)


def _move(inputs: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in inputs.items()}


def _unlabeled(records: list[dict]) -> list[dict]:
    # Labels never reach query preprocessing, retrieval, or model inputs. Keep
    # their original copies exclusively for the downstream scoring step.
    return [{key: value for key, value in row.items() if key != "label"} for row in records]


def _precision(device: torch.device, bfloat16: bool):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bfloat16)
    if device.type == "cpu":
        return torch.autocast(device_type="cpu", enabled=False)
    return nullcontext()


@contextmanager
def _evaluation_state(model, retrieval_model, memory):
    models = [model] + ([retrieval_model] if retrieval_model is not None and retrieval_model is not model else [])
    modes = [(module, module.training) for owner in models for module in owner.modules()]
    local_rng = memory.local_rng.getstate() if memory is not None else None
    with isolated_rng():
        try:
            for owner in models:
                owner.eval()
            yield
        finally:
            for module, training in modes:
                module.training = training
            if memory is not None:
                memory.local_rng.setstate(local_rng)


def _probabilities(logits: torch.Tensor, size: int, classes: int) -> np.ndarray:
    if logits.ndim != 2 or logits.shape != (size, classes) or not torch.isfinite(logits).all():
        raise ValueError("Model produced invalid classification logits")
    return logits.detach().float().softmax(-1).cpu().numpy()


def predict_records(model, retrieval_model, collator, records: list[dict], memory,
                    config: dict, *, method: str | None = None,
                    include_intervals: bool = False) -> dict:
    """Return window/sample predictions and metrics without changing training state.

    Models are already placed on their intended devices. Direct CUDA inference
    can use bfloat16 autocast; fixed-key encoding and temporary adaptation use
    float32, matching the stored-key geometry and global parameter storage.
    Speech intervals resample aggregated participants. Text bootstrap requires
    the additional explicit ``evaluation.bootstrap_text`` option because full
    review test sets are large.
    """
    if not records:
        raise ValueError("Prediction requires at least one record")
    selected_method = method or config.get("method", "sequential")
    if selected_method not in {"sequential", "er", "derpp", "sparse_replay", "mbpa", "mbpa_random", "joint", "separate"}:
        raise ValueError("Unknown prediction method")
    local = selected_method in {"mbpa", "mbpa_random"}
    if local and memory is None:
        raise ValueError("Local adaptation requires a training memory")
    if local and any(entry.record.get("split") != "train" or entry.record.get("label") is None
                     for entry in memory.entries):
        raise ValueError("Local adaptation memory must contain labeled training records only")
    if selected_method == "mbpa" and len(memory):
        if retrieval_model is None or retrieval_model is model:
            raise ValueError("Nearest retrieval requires a separate fixed retrieval model")
    modality = config.get("model", {}).get("modality", getattr(model, "modality", None))
    if modality not in {"text", "audio"}:
        raise ValueError("Evaluation modality must be text or audio")
    classes = model.num_classes
    evaluation = config.get("evaluation", {})
    batch_size = evaluation.get("batch_size", 4 if modality == "audio" else 32)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Evaluation batch_size must be a positive integer")
    labeled = [row.get("label") is not None for row in records]
    if any(labeled) and not all(labeled):
        raise ValueError("Evaluation records must be entirely labeled or entirely unlabeled")
    device = _device(model)
    retrieval_device = _device(retrieval_model) if retrieval_model is not None else device
    requested_precision = config.get("training", {}).get("precision", "float32")
    use_bfloat16 = requested_precision == "bfloat16" and device.type == "cuda" and not local
    settings = config.get("local_adaptation", {})
    neighbors = settings.get("neighbors", 32)
    local_steps = settings.get("steps", 30)
    if local and (type(neighbors) is not int or neighbors < 1 or type(local_steps) is not int or local_steps < 0):
        raise ValueError("Invalid local neighbor or step count")
    trainable = any(parameter.requires_grad for parameter in model.parameters())
    cost = {"queries": len(records), "direct_queries": 0, "local_queries": 0,
            "adapted_queries": 0, "local_optimizer_steps": 0,
            "retrieval_encoder_examples": 0, "local_neighbor_forward_examples": 0,
            "direct_wall_seconds": 0.0, "retrieval_wall_seconds": 0.0,
            "local_wall_seconds": 0.0,
            "requested_precision": requested_precision,
            "prediction_precision": "bfloat16" if use_bfloat16 else "float32",
            "local_precision": "float32" if local else None,
            "retrieval_precision": "float32" if selected_method == "mbpa" else None}
    all_probabilities = []
    neighbor_counts = []
    _sync(device, retrieval_device)
    started = time.perf_counter()
    with _evaluation_state(model, retrieval_model, memory):
        if not local:
            for start in range(0, len(records), batch_size):
                _sync(device)
                batch_started = time.perf_counter()
                rows = records[start:start + batch_size]
                batch = collator(_unlabeled(rows))
                with torch.no_grad(), _precision(device, use_bfloat16):
                    logits = model(_move(batch["inputs"], device))
                all_probabilities.append(_probabilities(logits, len(rows), classes))
                _sync(device)
                cost["direct_wall_seconds"] += time.perf_counter() - batch_started
                cost["direct_queries"] += len(rows)
        else:
            for query in records:
                _sync(device, retrieval_device)
                retrieval_started = time.perf_counter()
                query_batch = collator(_unlabeled([query]))
                query_inputs = _move(query_batch["inputs"], device)
                if not len(memory):
                    selected = []
                elif selected_method == "mbpa_random":
                    selected = memory.sample(neighbors, purpose="local")
                else:
                    with torch.no_grad(), _precision(retrieval_device, False):
                        key = retrieval_model.encode(_move(query_batch["inputs"], retrieval_device))[0]
                    selected = memory.nearest(key, neighbors)
                    cost["retrieval_encoder_examples"] += 1
                neighbor_inputs = neighbor_labels = None
                if selected:
                    query_id = query.get("example_id", query.get("segment_id"))
                    if query.get("split") != "train" and query_id is not None and any(
                            entry.example_id == str(query_id) for entry in selected):
                        raise ValueError("Evaluation query identity overlaps training memory")
                    neighbor_batch = collator([entry.record for entry in selected])
                    neighbor_inputs = _move(neighbor_batch["inputs"], device)
                    neighbor_labels = neighbor_batch["labels"]
                    if neighbor_labels is None:
                        raise ValueError("Training neighbors require labels")
                    neighbor_labels = neighbor_labels.to(device)
                _sync(device, retrieval_device)
                cost["retrieval_wall_seconds"] += time.perf_counter() - retrieval_started
                local_started = time.perf_counter()
                with _precision(device, False):
                    logits = local_adapt_predict(
                        model, query_inputs, neighbor_inputs, neighbor_labels,
                        steps=local_steps, lr=settings.get("lr", .001),
                        anchor_strength=settings.get("anchor_strength", .001),
                        max_grad_norm=settings.get("max_grad_norm", config.get("training", {}).get("max_grad_norm", 1.)),
                        optimizer_name=settings.get("optimizer", "adam"),
                        microbatch_size=settings.get("microbatch_size", 4),
                    )
                all_probabilities.append(_probabilities(logits, 1, classes))
                _sync(device)
                cost["local_wall_seconds"] += time.perf_counter() - local_started
                count = len(selected)
                neighbor_counts.append(count)
                cost["local_queries"] += 1
                if count and local_steps and trainable:
                    cost["adapted_queries"] += 1
                    cost["local_optimizer_steps"] += local_steps
                    cost["local_neighbor_forward_examples"] += count * local_steps
    _sync(device, retrieval_device)
    cost["prediction_wall_seconds"] = time.perf_counter() - started
    cost["prediction_seconds_per_query"] = cost["prediction_wall_seconds"] / len(records)
    cost["neighbors_per_query"] = {
        "minimum": min(neighbor_counts) if neighbor_counts else 0,
        "maximum": max(neighbor_counts) if neighbor_counts else 0,
        "mean": float(np.mean(neighbor_counts)) if neighbor_counts else 0.0,
        "total": sum(neighbor_counts),
    }
    probabilities = np.concatenate(all_probabilities)
    output_records = portable_records(records)
    report = {"records": output_records, "probabilities": probabilities, "cost": cost}
    scoring_records, scoring_probabilities = output_records, probabilities
    if modality == "audio":
        scoring_records, scoring_probabilities = aggregate_speech_predictions(output_records, probabilities)
        report["participant_records"] = scoring_records
        report["participant_probabilities"] = scoring_probabilities
    report["evaluation_unit"] = "participant" if modality == "audio" else "review"
    if all(labeled):
        labels = [int(row["label"]) for row in scoring_records]
        metrics = classification_metrics(labels, scoring_probabilities, classes)
        if include_intervals and (modality == "audio" or evaluation.get("bootstrap_text", False)):
            metrics["confidence_intervals"] = bootstrap_intervals(
                labels, scoring_probabilities, num_classes=classes,
                n_bootstrap=evaluation.get("bootstrap_samples", 1000),
                confidence=evaluation.get("confidence", .95), seed=config.get("seed", 42),
                stratified=evaluation.get("stratified_bootstrap", False),
            )
    else:
        metrics = {"n": len(scoring_records), "num_classes": classes, "status": "labels_unavailable"}
    metrics["evaluation_unit"] = report["evaluation_unit"]
    report["metrics"] = metrics
    cost["evaluation_wall_seconds"] = time.perf_counter() - started
    return report


def evaluate_domains(model, retrieval_model, collator, records_by_domain: dict,
                     memory, config: dict, *, method: str | None = None,
                     include_intervals: bool = False) -> dict:
    """Evaluate each cohort and select by equal-domain mean accuracy."""
    if not records_by_domain:
        raise ValueError("Domain evaluation requires at least one domain")
    reports = {}
    for domain, records in records_by_domain.items():
        report = predict_records(model, retrieval_model, collator, records, memory, config,
                                 method=method, include_intervals=include_intervals)
        if "accuracy" not in report["metrics"]:
            raise ValueError("Domain evaluation requires scoring labels; use predict_records for unlabeled inference")
        reports[domain] = report
    return {"selection_score": float(np.mean([report["metrics"]["accuracy"] for report in reports.values()])),
            "domains": reports}
