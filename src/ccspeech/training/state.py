"""Portable checkpoints, atomic metadata, and reproducible RNG state."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


@contextmanager
def isolated_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("Refusing symlink output")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_checkpoint(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("Refusing symlink checkpoint output")
    torch.save(cpu_tree(value), temporary)
    temporary.replace(path)


def read_checkpoint(path: Path) -> dict:
    # These checkpoints contain optimizer/RNG state and must be produced by this
    # project. Do not load checkpoints received from untrusted third parties.
    result = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(result, dict) or result.get("schema_version") != 1:
        raise ValueError("Unsupported experiment checkpoint")
    return result


def portable_records(records: list[dict]) -> list[dict]:
    return [{key: value for key, value in record.items() if not key.startswith("_")} for record in records]


def rebind_memory(memory, data_root: Path) -> None:
    root = data_root.resolve()
    for entry in memory.entries:
        if "audio_path" in entry.record:
            path = (root / entry.record["audio_path"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("Checkpoint audio is unavailable under the current data root")
            entry.record["_audio_file"] = str(path)


def portable_memory_state(memory) -> dict:
    state = memory.state_dict()
    for entry in state["entries"]:
        entry["record"] = portable_records([entry["record"]])[0]
    return state


@contextmanager
def exclusive_run(directory: Path):
    """Hold an OS lock; stale metadata alone never proves a run is live."""
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is actively using {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
