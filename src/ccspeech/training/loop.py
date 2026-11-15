"""Resumable supervised stages with bounded replay and explicit update accounting.

This module performs global updates only. MbPA query-local adaptation belongs
to the validation/prediction callback and never changes the global optimizer.
The live objects remain at the last stage state; callers load ``best_checkpoint``
explicitly when selecting a model for the next stage.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from ccspeech.methods.optim import TimescaleOptimizer
from ccspeech.models import parameter_groups
from ccspeech.training.config import canonical_hash, semantic_config
from ccspeech.training.state import (
    atomic_checkpoint, portable_memory_state, read_checkpoint, rebind_memory,
    restore_rng, rng_state,
)


def _example_id(record: dict) -> str:
    value = record.get("example_id", record.get("segment_id"))
    if value is None:
        raise ValueError("Training records require example_id or segment_id")
    return str(value)


def _permutation(length: int, seed: int, stage: str, epoch: int) -> list[int]:
    # A private RNG leaves dropout/RNG checkpoints independent of data ordering.
    salt = hashlib.sha256(f"{seed}:{stage}:{epoch}".encode()).digest()
    indices = list(range(length))
    random.Random(int.from_bytes(salt, "big")).shuffle(indices)
    return indices


def _training_fingerprint(config: dict, records: list[dict], stage: str,
                          epochs: int, method: str) -> str:
    # Includes ordered stream identity/labels, so resuming against a changed
    # split or reordered records cannot silently skip or repeat examples.
    return canonical_hash({"config": semantic_config(config), "stage": stage,
                           "epochs": epochs, "method": method,
                           "records": [(_example_id(r), int(r["label"])) for r in records]})


def _tensor_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def train_stage(model, retrieval_model, collator, train_records, memory,
                config: dict, output_dir, *, stage: str, epochs: int,
                method: str, validate_fn, resume: bool = False) -> dict:
    """Train a complete stage, atomically checkpointing all resumable state.

    A checkpoint cursor refers to the NEXT current-data batch. Each batch and
    its sparse replay updates form one checkpoint transaction. New examples
    enter reservoir memory once, on epoch zero, with their pre-update logits.
    Sparse replay thresholds count these unique examples, not repeated epochs.
    """
    if method not in {"sequential", "er", "derpp", "sparse_replay", "mbpa", "mbpa_random"}:
        raise ValueError(f"Unknown stage method: {method}")
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    records = list(train_records)
    if not records:
        raise ValueError("A training stage requires records")
    identifiers = [_example_id(record) for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Training stage contains duplicate example identifiers")
    if any(record.get("split") != "train" or record.get("label") is None for record in records):
        raise ValueError("Only labeled training records may enter a training stage")
    training = config["training"]
    memory_config = config.get("memory", {})
    runtime = config.get("runtime", {})
    batch_size = training["batch_size"]
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    checkpoint_every = training.get("checkpoint_every", 500)
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError("checkpoint_every must be a positive integer")
    max_steps = runtime.get("max_steps_per_stage")
    if max_steps is not None:
        if runtime.get("smoke") is not True:
            raise ValueError("max_steps_per_stage requires explicit runtime.smoke=true")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps_per_stage must be positive")
    replay_size = memory_config.get("replay_batch_size", batch_size)
    sparse_frequency = memory_config.get("sparse_frequency", 10000)
    sparse_size = memory_config.get("sparse_batch_size", 100)
    sparse_microbatch = memory_config.get("sparse_microbatch_size", batch_size)
    for name, value in (("replay_batch_size", replay_size), ("sparse_frequency", sparse_frequency),
                        ("sparse_batch_size", sparse_size), ("sparse_microbatch_size", sparse_microbatch)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    alpha, beta = memory_config.get("alpha", .5), memory_config.get("beta", .5)
    if alpha < 0 or beta < 0:
        raise ValueError("DER++ coefficients must be nonnegative")
    if method != "sequential" and memory.capacity < 1:
        raise ValueError("Replay methods require nonzero memory capacity")
    if memory_config.get("capacity", memory.capacity) != memory.capacity:
        raise ValueError("Configured memory capacity differs from the memory object")
    device = torch.device(runtime.get("device", str(next(model.parameters()).device)))
    precision = training.get("precision", "float32")
    if precision not in {"float32", "bfloat16"}:
        raise ValueError("Only float32 and CUDA bfloat16 training are supported")
    use_bfloat16 = device.type == "cuda" and precision == "bfloat16"
    if use_bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("The selected CUDA device does not support bfloat16")
    model.to(device)
    if retrieval_model is model:
        raise ValueError("Retrieval must use a separate fixed encoder")
    if memory.capacity and retrieval_model is None:
        raise ValueError("A fixed retrieval encoder is required for stored keys")
    if retrieval_model is not None:
        retrieval_model.to(device).eval().requires_grad_(False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    groups = parameter_groups(model)
    timescales = config.get("timescales", {})
    optimizer = TimescaleOptimizer(
        groups["head"], groups["encoder"], lr=training["lr"],
        slow_lr=training.get("encoder_lr"),
        slow_interval=timescales.get("encoder_interval", 1),
        head_interval=timescales.get("head_interval", 1),
        weight_decay=training.get("weight_decay", 0.),
        max_grad_norm=training.get("max_grad_norm", 1.),
    )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    best_path, last_path = directory / "best.pt", directory / "last.pt"
    config_hash = canonical_hash(semantic_config(config))
    fingerprint = _training_fingerprint(config, records, stage, epochs, method)
    batch_count = math.ceil(len(records) / batch_size)
    epoch = batch_cursor = 0
    best_score = None
    last_validation = None
    last_validation_signature = None
    admitted: set[str] = set()
    stats = {"current_batches": 0, "current_presentations": 0, "new_examples": 0,
             "replay_presentations": 0, "label_replay_presentations": 0,
             "distill_presentations": 0, "sparse_replay_presentations": 0,
             "sparse_replay_updates": 0, "retrieval_presentations": 0,
             "current_loss_sum": 0., "objective_sum": 0., "sparse_loss_sum": 0.,
             "elapsed_seconds": 0.}
    completed = stopped_early = False
    if resume:
        checkpoint = read_checkpoint(last_path)
        if (checkpoint.get("kind") != "training_stage" or checkpoint.get("config_hash") != config_hash
                or checkpoint.get("training_fingerprint") != fingerprint):
            raise ValueError("Checkpoint stage, data stream, or scientific configuration differs")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        memory.load_state_dict(checkpoint["memory_state"])
        rebind_memory(memory, Path(config["data"]["root"]))
        restore_rng(checkpoint["rng_state"])
        epoch, batch_cursor = checkpoint["epoch"], checkpoint["batch_cursor"]
        if not (0 <= epoch <= epochs and 0 <= batch_cursor <= batch_count):
            raise ValueError("Invalid checkpoint training cursor")
        stats = checkpoint["training_stats"]
        admitted = set(checkpoint["admitted_ids"])
        best_score = checkpoint["best_score"]
        last_validation = checkpoint.get("last_validation")
        last_validation_signature = checkpoint.get("last_validation_signature")
        completed = checkpoint.get("completed", False)
        stopped_early = checkpoint.get("stopped_early", False)
        if best_score is not None and not best_path.is_file():
            raise ValueError("Resume requires the associated best.pt checkpoint")
    elif last_path.exists() or best_path.exists():
        raise FileExistsError("Stage checkpoints already exist; resume explicitly or choose another output directory")
    started = time.perf_counter()
    elapsed_before = stats["elapsed_seconds"]

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16) if use_bfloat16 else nullcontext()

    def batch(record_batch):
        value = collator(record_batch)
        if value["labels"] is None:
            raise ValueError("Training or replay batch has no labels")
        return ({key: tensor.to(device) for key, tensor in value["inputs"].items()},
                value["labels"].to(device))

    def backward(loss):
        if not torch.isfinite(loss.detach()).all():
            raise FloatingPointError(f"Nonfinite training loss in stage {stage}")
        loss.backward()
        return float(loss.detach())

    def signature():
        return [optimizer.fast_steps, optimizer.slow_steps, memory.seen]

    def resource_statistics():
        stats["model_parameter_bytes"] = sum(p.numel() * p.element_size() for p in model.parameters())
        stats["trainable_parameter_bytes"] = sum(p.numel() * p.element_size()
                                                   for p in model.parameters() if p.requires_grad)
        stats["fixed_retrieval_parameter_bytes"] = (
            sum(p.numel() * p.element_size() for p in retrieval_model.parameters())
            if retrieval_model is not None else 0)
        if device.type == "cuda":
            stats["cuda_peak_allocated_bytes"] = max(stats.get("cuda_peak_allocated_bytes") or 0,
                                                      torch.cuda.max_memory_allocated(device))
            stats["cuda_peak_reserved_bytes"] = max(stats.get("cuda_peak_reserved_bytes") or 0,
                                                     torch.cuda.max_memory_reserved(device))
        else:
            stats["cuda_peak_allocated_bytes"] = stats["cuda_peak_reserved_bytes"] = None
        # Tensor payload estimates exclude pickle/record/RNG overhead. No
        # optimizer tensors are copied merely to calculate this estimate.
        selection = _tensor_bytes(model.state_dict()) + sum(
            _tensor_bytes((entry.logits, entry.key)) for entry in memory.entries)
        optimizer_bytes = sum(_tensor_bytes(owner.state) for owner in (optimizer.fast, optimizer.slow)
                              if owner is not None)
        optimizer_bytes += _tensor_bytes((optimizer.pending, optimizer.pending_fast))
        stats["estimated_checkpoint_tensor_bytes"] = {"best": selection, "last": selection + optimizer_bytes}

    def snapshot(*, finished=False, resumable=True):
        stats["elapsed_seconds"] = elapsed_before + time.perf_counter() - started
        stats["optimizer"] = optimizer.statistics()
        resource_statistics()
        current_validation = last_validation if last_validation_signature == signature() else None
        result = {"schema_version": 1, "kind": "training_stage", "config": config,
                "checkpoint_role": "resume" if resumable else "selection",
                "config_hash": config_hash, "training_fingerprint": fingerprint,
                "stage": stage, "method": method, "epochs": epochs,
                "model_metadata": model.metadata(), "model_state": model.state_dict(),
                "memory_state": portable_memory_state(memory), "rng_state": rng_state(),
                "epoch": epoch, "batch_cursor": batch_cursor, "admitted_ids": sorted(admitted),
                "training_stats": stats, "best_score": best_score,
                "selection_score": current_validation["selection_score"] if current_validation else None,
                "validation": current_validation, "last_validation": last_validation,
                "last_validation_signature": last_validation_signature,
                "completed": finished, "stopped_early": stopped_early}
        # Selection checkpoints are also used for inference and source warmup
        # branching; only last.pt needs full Adam moments and pending gradients.
        if resumable:
            result["optimizer_state"] = optimizer.state_dict()
        return result

    def validate():
        nonlocal best_score, last_validation, last_validation_signature
        current_signature = signature()
        if last_validation_signature == current_signature:
            return
        was_training = model.training
        try:
            result = validate_fn(model, memory)
        finally:
            model.train(was_training)
        score = float(result["selection_score"])
        if not math.isfinite(score):
            raise FloatingPointError("Validation selection_score must be finite")
        last_validation = dict(result, selection_score=score)
        last_validation_signature = current_signature
        if best_score is None or score > best_score:
            best_score = score
            atomic_checkpoint(best_path, snapshot(resumable=False))

    def sparse_replay():
        entries = memory.sample(sparse_size)
        if not entries:
            return
        optimizer.zero_grad()
        total_loss = 0.
        # Weight each microbatch by its sample count: this is exactly the mean
        # loss of one sparse replay batch, with a single optimizer transaction.
        for start in range(0, len(entries), sparse_microbatch):
            portion = entries[start:start + sparse_microbatch]
            inputs, labels = batch([entry.record for entry in portion])
            with autocast():
                loss = F.cross_entropy(model(inputs), labels) * (len(portion) / len(entries))
            total_loss += backward(loss)
        optimizer.step()
        stats["sparse_loss_sum"] += total_loss
        stats["sparse_replay_updates"] += 1
        stats["sparse_replay_presentations"] += len(entries)
        stats["label_replay_presentations"] += len(entries)
        stats["replay_presentations"] += len(entries)

    if not completed:
        if training.get("validate_initial", False) and not stats["current_batches"]:
            validate()
        while epoch < epochs:
            order = _permutation(len(records), config["seed"], stage, epoch)
            while batch_cursor < batch_count:
                if max_steps is not None and stats["current_batches"] >= max_steps:
                    stopped_early = True
                    break
                selected = order[batch_cursor * batch_size:(batch_cursor + 1) * batch_size]
                current_records = [records[index] for index in selected]
                new_positions = [i for i, record in enumerate(current_records)
                                 if epoch == 0 and _example_id(record) not in admitted]
                model.train()
                optimizer.zero_grad()
                inputs, labels = batch(current_records)
                with autocast():
                    logits = model(inputs)
                    current_loss = F.cross_entropy(logits, labels)
                historical_logits = logits.detach().float().cpu()
                current_value = backward(current_loss)
                objective_value = current_value
                # Separate backwards preserve the sum objective while avoiding
                # three simultaneous full-encoder graphs on speech workloads.
                if method == "derpp":
                    # Distillation and label replay draw INDEPENDENT minibatches.
                    distill_entries = memory.sample(replay_size)
                    label_entries = memory.sample(replay_size)
                    if distill_entries:
                        if any(entry.logits is None for entry in distill_entries):
                            raise ValueError("DER++ memory lacks historical logits")
                        replay_inputs, _ = batch([entry.record for entry in distill_entries])
                        stored = torch.stack([entry.logits for entry in distill_entries]).to(device)
                        with autocast():
                            predictions = model(replay_inputs)
                            loss = alpha * F.mse_loss(predictions.float(), stored.float())
                        objective_value += backward(loss)
                        stats["distill_presentations"] += len(distill_entries)
                        stats["replay_presentations"] += len(distill_entries)
                else:
                    label_entries = memory.sample(replay_size) if method == "er" else []
                if label_entries:
                    replay_inputs, replay_labels = batch([entry.record for entry in label_entries])
                    with autocast():
                        loss = F.cross_entropy(model(replay_inputs), replay_labels)
                        if method == "derpp":
                            loss = beta * loss
                    objective_value += backward(loss)
                    stats["label_replay_presentations"] += len(label_entries)
                    stats["replay_presentations"] += len(label_entries)
                keys = None
                if new_positions and memory.capacity:
                    # Frozen keys use the original pretrained encoder, never
                    # the updated classifier or query labels.
                    key_inputs = {key: value[new_positions] for key, value in inputs.items()}
                    # Query keys in prediction use float32 too. Training
                    # autocast must not change the fixed retrieval geometry.
                    with torch.no_grad(), torch.autocast(device.type, enabled=False):
                        keys = retrieval_model.encode(key_inputs).detach().float().cpu()
                    stats["retrieval_presentations"] += len(new_positions)
                optimizer.step()
                previous_new = stats["new_examples"]
                for offset, position in enumerate(new_positions):
                    record = current_records[position]
                    memory.add(record, historical_logits[position], None if keys is None else keys[offset])
                    admitted.add(_example_id(record))
                stats["new_examples"] += len(new_positions)
                if method in {"sparse_replay", "mbpa", "mbpa_random"}:
                    crossed = stats["new_examples"] // sparse_frequency - previous_new // sparse_frequency
                    for _ in range(crossed):
                        sparse_replay()
                stats["current_batches"] += 1
                stats["current_presentations"] += len(current_records)
                stats["current_loss_sum"] += current_value
                stats["objective_sum"] += objective_value
                batch_cursor += 1
                if stats["current_batches"] % checkpoint_every == 0:
                    atomic_checkpoint(last_path, snapshot())
            if stopped_early:
                break
            # Flush only at the final stage boundary. Slow updates may span
            # intermediate epoch boundaries, and their pending gradients are
            # saved with the checkpoint for exact continuation.
            if epoch + 1 == epochs:
                optimizer.flush()
            validate()
            epoch += 1
            batch_cursor = 0
            atomic_checkpoint(last_path, snapshot())
        optimizer.flush()
        validate()
        completed = True
        atomic_checkpoint(last_path, snapshot(finished=True))
    if best_score is None:
        raise RuntimeError("Stage finished without a validation selection checkpoint")
    stats["optimizer"] = optimizer.statistics()
    resource_statistics()
    stats["checkpoint_disk_bytes"] = {"best": best_path.stat().st_size, "last": last_path.stat().st_size}
    return {"best_checkpoint": str(best_path), "last_checkpoint": str(last_path),
            "best_score": best_score, "training_stats": stats,
            "stopped_early": stopped_early, "completed": completed,
            "live_state": "last", "validation": last_validation}
