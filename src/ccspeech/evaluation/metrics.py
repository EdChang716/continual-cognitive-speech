"""Metrics on explicit evaluation units, without sample-level leakage."""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def _inputs(labels, probabilities, num_classes: int | None = None):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 2 or len(labels) != len(probabilities) or not len(labels):
        raise ValueError("Expected nonempty labels[N] and probabilities[N,C]")
    classes = probabilities.shape[1] if num_classes is None else int(num_classes)
    if classes < 2 or probabilities.shape[1] != classes:
        raise ValueError("Invalid probability class count")
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0) or np.any(labels >= classes):
        raise ValueError("Labels must be integer class indices")
    if not np.isfinite(probabilities).all() or np.any(probabilities < -1e-8) or np.any(probabilities > 1 + 1e-8):
        raise ValueError("Probabilities must be finite and between zero and one")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Class probabilities must sum to one")
    return labels.astype(np.int64), probabilities, classes


def classification_metrics(labels, probabilities, num_classes: int | None = None) -> dict:
    """Macro F1 uses all configured classes; undefined AUROC is explicit null."""
    labels, probabilities, classes = _inputs(labels, probabilities, num_classes)
    predictions = probabilities.argmax(axis=1)
    present = np.unique(labels)
    recalls = [float(np.mean(predictions[labels == value] == value)) for value in present]
    result = {
        "n": len(labels), "num_classes": classes,
        "label_counts": {str(c): int(np.sum(labels == c)) for c in range(classes)},
        "accuracy": float(np.mean(predictions == labels)),
        "macro_f1": float(f1_score(labels, predictions, labels=list(range(classes)), average="macro", zero_division=0)),
        "balanced_accuracy": float(np.mean(recalls)),
        "auroc": None,
        "auroc_status": "undefined_missing_classes",
    }
    if len(present) == classes:
        result["auroc"] = float(roc_auc_score(labels, probabilities[:, 1]) if classes == 2
                                else roc_auc_score(labels, probabilities, labels=list(range(classes)),
                                                   average="macro", multi_class="ovr"))
        result["auroc_status"] = "defined"
    return result


def aggregate_speech_predictions(records: Sequence[dict], probabilities) -> tuple[list[dict], np.ndarray]:
    """Average windows per recording, then equally average recordings per person.

    This prevents a long recording with many windows from dominating a person's
    prediction. The caller must evaluate participant-level labels and bootstrap
    these returned people, not the original overlapping windows.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or len(records) != len(probabilities) or not len(records):
        raise ValueError("Expected matching nonempty records and probabilities")
    if not np.isfinite(probabilities).all() or np.any(probabilities < -1e-8) or np.any(probabilities > 1+1e-8):
        raise ValueError("Invalid speech probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Speech probabilities must sum to one")
    grouped = defaultdict(list)
    participant_records = defaultdict(list)
    seen_segments = set()
    recording_owner = {}
    for record, probability in zip(records, probabilities):
        participant = str(record["participant_id"])
        recording = str(record["recording_id"])
        segment = str(record["segment_id"])
        if segment in seen_segments:
            raise ValueError("Duplicate segment prediction")
        seen_segments.add(segment)
        if recording in recording_owner and recording_owner[recording] != participant:
            raise ValueError("A recording is assigned to different participants")
        recording_owner[recording] = participant
        grouped[participant, recording].append(probability)
        participant_records[participant].append(record)
    output_records, output_probabilities = [], []
    for participant in sorted(participant_records):
        source = participant_records[participant]
        row = {"participant_id": participant, "segments": len(source)}
        for field in ("label", "domain", "language", "split"):
            values = [record[field] for record in source if field in record and record[field] not in (None, "")]
            if values:
                if len(values) != len(source) or len({str(value) for value in values}) != 1:
                    raise ValueError(f"Inconsistent participant {field}")
                row[field] = int(values[0]) if field == "label" else values[0]
        recordings = sorted(key for key in grouped if key[0] == participant)
        row["recordings"] = len(recordings)
        output_records.append(row)
        output_probabilities.append(np.mean([np.mean(grouped[key], axis=0) for key in recordings], axis=0))
    return output_records, np.stack(output_probabilities)


def bootstrap_intervals(labels, probabilities, *, num_classes: int | None = None,
                        n_bootstrap: int = 1000, confidence: float = 0.95,
                        seed: int = 42, stratified: bool = False) -> dict:
    """Percentile intervals over evaluation units (participants for speech)."""
    labels, probabilities, classes = _inputs(labels, probabilities, num_classes)
    if n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap configuration")
    rng = np.random.default_rng(seed)
    values = {name: [] for name in ("accuracy", "macro_f1", "balanced_accuracy", "auroc")}
    by_class = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    for _ in range(n_bootstrap):
        if stratified:
            indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in by_class])
        else:
            indices = rng.integers(0, len(labels), size=len(labels))
        metrics = classification_metrics(labels[indices], probabilities[indices], classes)
        for name in values:
            if metrics[name] is not None:
                values[name].append(metrics[name])
    quantiles = [(1-confidence)/2, 1-(1-confidence)/2]
    result = {"confidence": confidence, "n_bootstrap": n_bootstrap, "seed": seed,
              "stratified": stratified, "n_units": len(labels), "intervals": {}}
    for name, samples in values.items():
        lower, upper = np.quantile(samples, quantiles) if samples else (None, None)
        result["intervals"][name] = {"lower": None if lower is None else float(lower),
                                     "upper": None if upper is None else float(upper),
                                     "valid_resamples": len(samples)}
    return result


def continual_summary(score_matrix, domain_order: Sequence[str]) -> dict:
    """Summarize scores after each stage: rows=stages, columns=domains.

    The matrix contains scores in [0,1] for every domain at every stage. For an
    earlier domain, forgetting is its best score since first learning it minus
    its final score. Improvement therefore produces negative forgetting. No
    pre-adaptation or unseen-domain score is used to inflate the maximum.
    """
    matrix = np.asarray(score_matrix, dtype=np.float64)
    n = len(domain_order)
    if matrix.shape != (n, n) or n < 1 or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite square stage-by-domain score matrix")
    if len(set(domain_order)) != n or np.any(matrix < 0) or np.any(matrix > 1):
        raise ValueError("Invalid domains or score range")
    forgetting = {domain_order[i]: float(np.max(matrix[i:-1, i]) - matrix[-1, i]) for i in range(n-1)}
    source_change = {domain_order[i]: float(matrix[-1, i] - matrix[i, i]) for i in range(n-1)}
    return {"domain_order": list(domain_order), "score_matrix": matrix.tolist(),
            "final_domain_averaged_accuracy": float(matrix[-1].mean()),
            "source_score_change": source_change, "forgetting": forgetting,
            "mean_forgetting": float(np.mean(list(forgetting.values()))) if forgetting else None}


def paired_bootstrap_change(labels, before_probabilities, after_probabilities, *,
                            metric: str = "accuracy", n_bootstrap: int = 1000,
                            confidence: float = .95, seed: int = 42) -> dict:
    """Paired participant/sample bootstrap for after-minus-before score change."""
    labels, before, classes = _inputs(labels, before_probabilities)
    _, after, _ = _inputs(labels, after_probabilities, classes)
    if metric not in {"accuracy", "macro_f1", "balanced_accuracy", "auroc"}:
        raise ValueError("Unknown metric for paired bootstrap")
    if n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap configuration")
    first = classification_metrics(labels, before, classes)[metric]
    second = classification_metrics(labels, after, classes)[metric]
    values = []
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(labels), size=len(labels))
        initial = classification_metrics(labels[indices], before[indices], classes)[metric]
        final = classification_metrics(labels[indices], after[indices], classes)[metric]
        if initial is not None and final is not None:
            values.append(final-initial)
    interval = np.quantile(values, [(1-confidence)/2, 1-(1-confidence)/2]) if values else [None, None]
    return {"metric": metric, "change": None if first is None or second is None else second-first,
            "lower": None if interval[0] is None else float(interval[0]),
            "upper": None if interval[1] is None else float(interval[1]),
            "confidence": confidence, "n_bootstrap": n_bootstrap, "valid_resamples": len(values),
            "n_units": len(labels), "seed": seed, "paired": True}
