"""Explicit fast-head/slow-encoder optimizer schedules and step accounting."""
from __future__ import annotations

import torch


class TimescaleOptimizer:
    """Average slow-group gradients over k batches; update the fast group each batch.

    This is an update-frequency ablation inspired by multiple timescales. It is
    not an implementation of Nested Learning's full HOPE architecture. A final
    incomplete slow accumulation is flushed using its actual batch count.
    """

    def __init__(self, fast_parameters, slow_parameters, *, lr: float = 3e-5,
                 slow_lr: float | None = None, slow_interval: int = 1,
                 head_interval: int = 1, weight_decay: float = 0.0,
                 max_grad_norm: float | None = 1.0):
        self.fast_parameters = [p for p in fast_parameters if p.requires_grad]
        self.slow_parameters = [p for p in slow_parameters if p.requires_grad]
        if any(value < 1 or int(value) != value for value in (slow_interval, head_interval)):
            raise ValueError("Update intervals must be positive integers")
        if not self.fast_parameters and not self.slow_parameters:
            raise ValueError("No trainable parameters")
        if {id(p) for p in self.fast_parameters} & {id(p) for p in self.slow_parameters}:
            raise ValueError("Fast and slow parameter groups overlap")
        self.slow_interval = int(slow_interval)
        self.head_interval = int(head_interval)
        self.max_grad_norm = max_grad_norm
        self.fast = torch.optim.AdamW(self.fast_parameters, lr=lr, weight_decay=weight_decay) if self.fast_parameters else None
        self.slow = torch.optim.AdamW(self.slow_parameters, lr=slow_lr if slow_lr is not None else lr,
                                     weight_decay=weight_decay) if self.slow_parameters else None
        self.pending = [None for _ in self.slow_parameters]
        self.pending_fast = [None for _ in self.fast_parameters]
        self.pending_batches = 0
        self.pending_fast_batches = 0
        self.fast_steps = self.slow_steps = self.batches = 0

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in (self.fast, self.slow):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.batches += 1
        if self.fast is not None:
            for i, parameter in enumerate(self.fast_parameters):
                if parameter.grad is not None:
                    gradient = parameter.grad.detach()
                    if self.pending_fast[i] is None:
                        self.pending_fast[i] = gradient.clone()
                    else:
                        self.pending_fast[i].add_(gradient)
            self.pending_fast_batches += 1
            if self.pending_fast_batches == self.head_interval:
                self._flush_group("fast")
        if self.slow is not None:
            for i, parameter in enumerate(self.slow_parameters):
                if parameter.grad is not None:
                    gradient = parameter.grad.detach()
                    if self.pending[i] is None:
                        self.pending[i] = gradient.clone()
                    else:
                        self.pending[i].add_(gradient)
            self.pending_batches += 1
            if self.pending_batches == self.slow_interval:
                self._flush_group("slow")

    def flush(self) -> None:
        self._flush_group("fast")
        self._flush_group("slow")

    def _flush_group(self, group: str) -> None:
        fast = group == "fast"
        count = self.pending_fast_batches if fast else self.pending_batches
        if not count:
            return
        parameters = self.fast_parameters if fast else self.slow_parameters
        gradients = self.pending_fast if fast else self.pending
        optimizer = self.fast if fast else self.slow
        for parameter, pending in zip(parameters, gradients):
            parameter.grad = None if pending is None else pending.div(count)
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        optimizer.step()
        if fast:
            self.fast_steps += 1
            self.pending_fast = [None for _ in self.fast_parameters]
            self.pending_fast_batches = 0
        else:
            self.slow_steps += 1
            self.pending = [None for _ in self.slow_parameters]
            self.pending_batches = 0

    def statistics(self) -> dict:
        return {"gradient_batches": self.batches, "fast_optimizer_steps": self.fast_steps,
                "slow_optimizer_steps": self.slow_steps, "slow_interval": self.slow_interval,
                "head_interval": self.head_interval, "pending_slow_batches": self.pending_batches,
                "pending_fast_batches": self.pending_fast_batches}

    def state_dict(self) -> dict:
        return {"fast": self.fast.state_dict() if self.fast is not None else None,
                "slow": self.slow.state_dict() if self.slow is not None else None,
                "pending": [None if g is None else g.detach().cpu().clone() for g in self.pending],
                "pending_fast": [None if g is None else g.detach().cpu().clone() for g in self.pending_fast],
                "pending_batches": self.pending_batches, **self.statistics()}

    def load_state_dict(self, state: dict) -> None:
        if (state["slow_interval"] != self.slow_interval or state["head_interval"] != self.head_interval
                or len(state["pending"]) != len(self.slow_parameters)
                or len(state["pending_fast"]) != len(self.fast_parameters)):
            raise ValueError("Checkpoint optimizer grouping or interval differs")
        for key in ("fast", "slow"):
            optimizer = getattr(self, key)
            if (optimizer is None) != (state[key] is None):
                raise ValueError("Checkpoint optimizer groups differ")
            if optimizer is not None:
                optimizer.load_state_dict(state[key])
        self.pending = [None if g is None else g.to(device=p.device, dtype=p.dtype).clone()
                        for g, p in zip(state["pending"], self.slow_parameters)]
        self.pending_batches = int(state["pending_batches"])
        self.pending_fast = [None if g is None else g.to(device=p.device, dtype=p.dtype).clone()
                             for g, p in zip(state["pending_fast"], self.fast_parameters)]
        self.pending_fast_batches = int(state["pending_fast_batches"])
        self.batches = int(state["gradient_batches"])
        self.fast_steps = int(state["fast_optimizer_steps"])
        self.slow_steps = int(state["slow_optimizer_steps"])
