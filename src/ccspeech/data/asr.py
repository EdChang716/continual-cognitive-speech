"""Explicit, local-only reference-blind ASR for changed participant audio.

This module never downloads a model or starts a paid service. MLX is imported
only after the manifest, local model files, and reusable outputs pass checks.
ASR windows cover the complete recording, independently of training-tail policy.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

from .speech import ASR_REVISION, SAMPLE_RATE, json_bytes, validate_words
from .taukadial import sha256_file

MODEL_REPO = "mlx-community/whisper-large-v3-mlx"
# Hashes of the existing local checkpoint with the pinned download provenance.
MODEL_FILES = {
    "config.json": "34982ce6ae286095000f82ae9583b3431639e8b092bf60c961f203745e6500e3",
    "weights.npz": "05ff791ce3630fae47e7c51004e9666204d786246ec07cac6110af768099b40d",
}
REQUIRED_FIELDS = {"recording_id", "language", "audio_path", "sha256"}
OPTIONAL_FIELDS = {"participant_id", "split"}


def require_local(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"Required local file is missing: {path.name}")
    if getattr(path.stat(), "st_flags", 0) & 0x40000000:
        raise ValueError(f"Cloud placeholder must be materialized first: {path.name}")


def load_manifest(path: Path) -> list[dict]:
    """Accept only explicit identity/audio fields, never transcript or labels."""
    require_local(path)
    root = path.resolve().parent
    rows, identifiers = [], set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (not isinstance(row, dict) or not REQUIRED_FIELDS <= row.keys()
                or row.keys() - REQUIRED_FIELDS - OPTIONAL_FIELDS):
            raise ValueError("ASR manifest accepts only identity, language, path, and hash fields")
        rid = row["recording_id"]
        if (not isinstance(rid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", rid)
                or rid in {"run", "status"} or ".." in rid or rid in identifiers):
            raise ValueError("Invalid, reserved, or duplicate recording_id")
        if row["language"] not in {"en", "zh"}:
            raise ValueError("ASR language must be en or zh")
        if not isinstance(row["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("ASR source sha256 must be a lowercase hexadecimal digest")
        if not isinstance(row["audio_path"], str) or "\\" in row["audio_path"]:
            raise ValueError("Unsafe relative audio_path")
        relative = Path(row["audio_path"])
        audio = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not audio.is_relative_to(root):
            raise ValueError("Unsafe relative audio_path")
        # Deliberately do not carry participant, split, or any reference into ASR.
        rows.append({**{key: row[key] for key in REQUIRED_FIELDS}, "resolved_audio": audio})
        identifiers.add(rid)
    return sorted(rows, key=lambda row: row["recording_id"])


def model_spec(model_path: Path) -> dict:
    model_path = model_path.resolve()
    provenance = model_path / "download_provenance.json"
    require_local(provenance)
    expected = {"repo": MODEL_REPO, "revision": ASR_REVISION}
    if json.loads(provenance.read_text()) != expected:
        raise ValueError("Local model download provenance differs from the pinned revision")
    for name, expected_hash in MODEL_FILES.items():
        path = model_path / name
        require_local(path)
        if sha256_file(path) != expected_hash:
            raise ValueError(f"Pinned local model hash mismatch: {name}")
    versions = {}
    for package in ("mlx", "mlx-metal", "mlx-whisper", "numpy", "soundfile"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "asr_protocol_version": 1, "model": expected,
        "model_files_sha256": dict(MODEL_FILES), "runtime_versions": versions,
        "temperature": 0, "condition_on_previous_text": False, "initial_prompt": None,
        "word_timestamps": True, "task": "transcribe", "random_seed": 0,
        "segment_seconds": 30, "overlap_seconds": 3,
        "overlap_policy": "midpoint ownership with 1.5s margins",
        "reference_blind": True, "local_files_only": True,
    }


def asr_windows(num_samples: int) -> list[tuple[int, int]]:
    """Complete 30s/27s coverage, without a redundant already-covered tail."""
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("ASR requires nonempty audio")
    bounds, start = [], 0
    while True:
        end = min(start + 30 * SAMPLE_RATE, num_samples)
        bounds.append((start, end))
        if end == num_samples:
            return bounds
        start += 27 * SAMPLE_RATE


def transcribe_audio(audio, language: str, decode) -> dict:
    """The backend receives only waveform and language; ownership uses clocks."""
    import numpy as np

    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Expected nonempty finite mono audio")
    duration = len(audio) / SAMPLE_RATE
    bounds = asr_windows(len(audio))
    words, windows = [], []
    for index, (start, end) in enumerate(bounds):
        result = decode(audio[start:end], language)
        if not isinstance(result.get("text"), str) or not isinstance(result.get("segments"), list):
            raise ValueError("ASR backend returned malformed text/segments")
        offset = start / SAMPLE_RATE
        lo = offset + (1.5 if index else 0)
        hi = end / SAMPLE_RATE - (1.5 if index + 1 < len(bounds) else 0)
        count = 0
        segments = []
        for segment in result["segments"]:
            a_seg, b_seg = float(segment["start"]), float(segment["end"])
            if not math.isfinite(a_seg) or not math.isfinite(b_seg) or a_seg > b_seg:
                raise ValueError("ASR segment has invalid timestamps")
            segments.append({"start": a_seg, "end": b_seg, "text": str(segment.get("text", ""))})
            for word in segment.get("words", []):
                a, b = float(word["start"]) + offset, float(word["end"]) + offset
                probability = float(word.get("probability", 0))
                if (not all(math.isfinite(value) for value in (a, b, probability))
                        or a > b or not 0 <= probability <= 1 or not isinstance(word["word"], str)):
                    raise ValueError("ASR word has invalid text, timestamps, or probability")
                midpoint = (a + b) / 2
                if lo <= midpoint < hi or (index + 1 == len(bounds) and midpoint == hi):
                    words.append({"start": max(0., a), "end": min(duration, b),
                                  "word": word["word"], "probability": probability,
                                  "window_index": index})
                    count += 1
        windows.append({"start": offset, "duration": (end - start) / SAMPLE_RATE,
                        "text": result["text"], "retained_words": count, "segments": segments})
    validate_words(words, duration)
    return {"text": "".join(word["word"] for word in words), "words": words, "windows": windows,
            "duration_seconds": duration,
            "quality_flags": [] if any(word["word"].strip() for word in words) else ["empty_asr_transcript"]}


def _load_backend(model_path: Path):
    # Defense in depth: the backend must never fall back to a Hub download.
    os.environ["HF_HUB_OFFLINE"] = "1"
    import mlx.core as mx
    import mlx_whisper

    def decode(audio, language):
        mx.random.seed(0)
        try:
            return mlx_whisper.transcribe(
                audio, path_or_hf_repo=str(model_path.resolve()), language=language,
                task="transcribe", word_timestamps=True, temperature=0,
                condition_on_previous_text=False, initial_prompt=None, verbose=None,
            )
        finally:
            mx.clear_cache()

    return decode


@contextmanager
def _gpu_lock():
    """Serialize this runner's local MLX jobs, including different output dirs."""
    import fcntl

    path = Path(tempfile.gettempdir()) / f"ccspeech-mlx-asr-{os.getuid()}.lock"
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _status(output: Path, value: dict) -> None:
    path = output / "status.json"
    if path.is_symlink():
        raise ValueError("Refusing symlink status output")
    descriptor, temporary = tempfile.mkstemp(prefix=".status-", suffix=".tmp", dir=output)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(json_bytes(value))
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def _write_new(path: Path, value: dict) -> None:
    """Atomic no-overwrite publication; interrupted temporary files aren't caches."""
    content = json_bytes(value)
    if path.is_symlink():
        raise ValueError("Refusing symlink output")
    if path.exists():
        if sha256_file(path) != hashlib.sha256(content).hexdigest():
            raise ValueError("Existing output differs; use a new output directory")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".asr-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and cannot replace an existing result.
        os.link(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def run_asr(manifest: Path, model_path: Path, output: Path, limit: int | None = None,
            *, backend_factory=None) -> dict:
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("limit must be a positive integer")
    rows = load_manifest(manifest)
    selected = rows if limit is None else rows[:limit]
    if output.is_symlink():
        raise ValueError("Refusing symlink output directory")
    output.mkdir(parents=True, exist_ok=True)
    with _gpu_lock():
        spec = model_spec(model_path)
        spec_hash = hashlib.sha256(json_bytes(spec)).hexdigest()
        pending, reused = [], 0
        # Finish all validation before importing or allocating the ASR model.
        for row in selected:
            require_local(row["resolved_audio"])
            if sha256_file(row["resolved_audio"]) != row["sha256"]:
                raise ValueError(f"Source audio hash mismatch: {row['recording_id']}")
            import soundfile as sf

            info = sf.info(row["resolved_audio"])
            if info.samplerate != SAMPLE_RATE or info.channels != 1 or info.frames <= 0:
                raise ValueError("ASR input must already be nonempty 16kHz mono audio")
            dest = output / f"{row['recording_id']}.json"
            if dest.is_symlink():
                raise ValueError("Refusing symlink cached ASR output")
            if dest.exists():
                cached = json.loads(dest.read_text())
                if any(cached.get(key) != expected for key, expected in {
                    "recording_id": row["recording_id"], "language": row["language"],
                    "source_sha256": row["sha256"], "reference_blind": True,
                    "run_settings_sha256": spec_hash,
                }.items()):
                    raise ValueError("Cached ASR source/settings changed; use a new output directory")
                physical_duration = info.frames / info.samplerate
                cached_duration = cached.get("duration_seconds")
                if (isinstance(cached_duration, bool) or not isinstance(cached_duration, (int, float))
                        or not math.isfinite(cached_duration)
                        or abs(cached_duration - physical_duration) > 1 / SAMPLE_RATE):
                    raise ValueError("Cached ASR duration differs from the physical source audio")
                # A cache cannot extend its own valid timestamp range by
                # editing duration_seconds; the verified WAV is authoritative.
                validate_words(cached["words"], physical_duration)
                if cached["text"] != "".join(word["word"] for word in cached["words"]):
                    raise ValueError("Cached ASR text differs from its timed words")
                reused += 1
            else:
                pending.append(row)
        _write_new(output / "run.json", spec)
        done = reused
        status = {"state": "running", "selected_recordings": len(selected),
                  "manifest_recordings": len(rows), "completed_recordings": done,
                  "new_recordings": 0, "reused_recordings": reused,
                  "manifest_sha256": sha256_file(manifest), "run_settings_sha256": spec_hash}
        _status(output, status)
        try:
            if pending:
                decode = (backend_factory or _load_backend)(model_path)
                for row in pending:
                    started = time.monotonic()
                    require_local(row["resolved_audio"])
                    content = row["resolved_audio"].read_bytes()
                    if hashlib.sha256(content).hexdigest() != row["sha256"]:
                        raise ValueError("Source audio changed after preflight")
                    audio, rate = sf.read(io.BytesIO(content), dtype="float32")
                    if rate != SAMPLE_RATE:
                        raise ValueError("ASR input must already be 16kHz mono; no implicit resampling")
                    result = transcribe_audio(audio, row["language"], decode)
                    result.update(recording_id=row["recording_id"], language=row["language"],
                                  source_audio_path=row["audio_path"], source_sha256=row["sha256"],
                                  reference_blind=True, run_settings_sha256=spec_hash,
                                  elapsed_seconds=time.monotonic() - started)
                    _write_new(output / f"{row['recording_id']}.json", result)
                    done += 1
                    status.update(completed_recordings=done, new_recordings=done - reused)
                    _status(output, status)
                    print(f"ASR completed {done}/{len(selected)} selected recordings", flush=True)
        except (Exception, KeyboardInterrupt) as error:
            # Publish only the exception class: backend messages can contain
            # local paths or clinical text. Completed results remain reusable.
            status.update(state="failed", error_type=type(error).__name__,
                          interrupted=isinstance(error, KeyboardInterrupt))
            try:
                _status(output, status)
            except Exception:
                # A second error while reporting (for example a full disk)
                # must not replace the original backend/interruption failure.
                pass
            raise
        status["state"] = "complete" if len(selected) == len(rows) else "selected_limit_complete"
        _status(output, status)
        return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(run_asr(args.manifest, args.model_path, args.output, args.limit), indent=2))


if __name__ == "__main__":
    main()
