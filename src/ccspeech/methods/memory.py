"""Reservoir memory with immutable historical logits and fixed retrieval keys."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import random
from numbers import Integral
from typing import Any

import torch


@dataclass
class MemoryEntry:
    record: dict[str, Any]
    logits: torch.Tensor | None = None
    key: torch.Tensor | None = None

    @property
    def example_id(self) -> str:
        value = self.record.get("example_id", self.record.get("segment_id"))
        if value is None:
            raise ValueError("Memory records require example_id or segment_id")
        return str(value)


class ReservoirMemory:
    """Uniform reservoir over the presented training stream.

    The engine must call ``add`` once per newly encountered training example,
    not again on replay or later optimization epochs. Records reference the
    immutable raw text/audio; only admitted records may be loaded for replay.
    """

    def __init__(self, capacity: int, seed: int = 42):
        if not isinstance(capacity, Integral) or isinstance(capacity, bool) or capacity < 0:
            raise ValueError("Memory capacity must be a nonnegative integer")
        self.capacity = int(capacity)
        self.seen = 0
        self.entries: list[MemoryEntry] = []
        self.rng = random.Random(seed)
        self.replay_rng = random.Random(seed + 1_000_003)
        self.local_rng = random.Random(seed + 2_000_003)

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _vector(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        value = value.detach().to(device="cpu", dtype=torch.float32).flatten().clone()
        if not value.numel() or not torch.isfinite(value).all():
            raise ValueError("Memory vectors must be finite and nonempty")
        return value

    def add(self, record: dict, logits: torch.Tensor | None = None,
            key: torch.Tensor | None = None) -> bool:
        if record.get("split") != "train":
            raise ValueError("Only explicit training records may enter memory")
        if "label" not in record or record["label"] is None:
            raise ValueError("Memory records require training labels")
        entry = MemoryEntry(copy.deepcopy(record), self._vector(logits), self._vector(key))
        entry.example_id  # Validate the identifier before consuming the stream.
        self.seen += 1
        if not self.capacity:
            return False
        if len(self.entries) < self.capacity:
            self.entries.append(entry)
            return True
        candidate = self.rng.randrange(self.seen)
        if candidate < self.capacity:
            self.entries[candidate] = entry
            return True
        return False

    def sample(self, count: int, *, purpose: str = "replay") -> list[MemoryEntry]:
        if not isinstance(count, Integral) or isinstance(count, bool) or count < 0:
            raise ValueError("Sample count must be a nonnegative integer")
        if purpose not in {"replay", "local"}:
            raise ValueError("Sampling purpose must be replay or local")
        # Evaluation frequency must not change reservoir admission or subsequent
        # global replay draws. Use the local stream for random-query controls.
        rng = self.replay_rng if purpose == "replay" else self.local_rng
        return rng.sample(self.entries, min(count, len(self.entries)))

    def nearest(self, query_key: torch.Tensor, count: int) -> list[MemoryEntry]:
        if not isinstance(count, Integral) or isinstance(count, bool) or count < 0:
            raise ValueError("Neighbor count must be a nonnegative integer")
        if not count or not self.entries:
            return []
        if any(entry.key is None for entry in self.entries):
            raise ValueError("Nearest retrieval requires a key for every memory entry")
        query = self._vector(query_key)
        keys = torch.stack([entry.key for entry in self.entries])
        if query.shape != keys.shape[1:]:
            raise ValueError("Query and stored retrieval key dimensions differ")
        distances = (keys - query).square().sum(dim=1)
        # Stable sort provides deterministic tie breaking in reservoir slot order.
        indices = torch.argsort(distances, stable=True)[:min(count, len(self.entries))]
        return [self.entries[int(i)] for i in indices]

    def statistics(self) -> dict:
        vector_bytes = sum(t.numel() * t.element_size() for e in self.entries
                           for t in (e.logits, e.key) if t is not None)
        record_bytes = sum(len(json.dumps(e.record, sort_keys=True, default=str).encode()) for e in self.entries)
        return {"capacity": self.capacity, "entries": len(self), "stream_examples_seen": self.seen,
                "vector_bytes": vector_bytes, "serialized_record_bytes": record_bytes,
                "stored_bytes": vector_bytes + record_bytes,
                "accounting": "Records plus logits/keys; referenced audio bytes are counted separately by the engine."}

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "seen": self.seen, "rng_state": self.rng.getstate(),
                "replay_rng_state": self.replay_rng.getstate(), "local_rng_state": self.local_rng.getstate(),
                "entries": [{"record": copy.deepcopy(e.record),
                             "logits": e.logits.clone() if e.logits is not None else None,
                             "key": e.key.clone() if e.key is not None else None} for e in self.entries]}

    def load_state_dict(self, state: dict) -> None:
        if state["capacity"] != self.capacity or len(state["entries"]) > self.capacity:
            raise ValueError("Checkpoint memory capacity differs")
        entries = []
        for value in state["entries"]:
            record = value["record"]
            if record.get("split") != "train" or record.get("label") is None:
                raise ValueError("Checkpoint contains a nontraining memory record")
            entry = MemoryEntry(copy.deepcopy(record), self._vector(value["logits"]), self._vector(value["key"]))
            entry.example_id
            entries.append(entry)
        if state["seen"] < len(entries):
            raise ValueError("Invalid reservoir stream count")
        self.entries = entries
        self.seen = int(state["seen"])
        self.rng.setstate(state["rng_state"])
        self.replay_rng.setstate(state["replay_rng_state"])
        self.local_rng.setstate(state["local_rng_state"])
