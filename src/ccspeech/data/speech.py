"""Versioned TAUKADIAL windows and explicit, provenance-checked text matching.

No ASR model, cloud service or paid compute is started by this module. The
WhisperCARE window policy is 30 seconds, a 27-second stride, and omission of
final tails shorter than 15 seconds. Saved WAVs retain their actual length.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import io

from .taukadial import copy_verified, recording_identity, safe_member_path, sha256_file, write_json

SAMPLE_RATE = 16_000
WINDOW_SECONDS = 30
STRIDE_SECONDS = 27
MIN_TAIL_SECONDS = 15
WHISPERCARE_REVISION = "abc2fe7074aa27e0cf1a6b72b07a98ed7440e1d9"
MATCH_METHODS = {"asr_word_timestamps", "whisper_medium_gpt_reference"}
ASR_REVISION = "49e6aa286ad60c14352c404340ded53710378a11"


def window_bounds(num_samples: int, sample_rate: int = SAMPLE_RATE) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return retained and dropped sample intervals using WhisperCARE's loop."""
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 0:
        raise ValueError("num_samples must be a nonnegative integer")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    retained, dropped = [], []
    start = 0
    while start < num_samples:
        end = min(start + WINDOW_SECONDS * sample_rate, num_samples)
        if end - start < MIN_TAIL_SECONDS * sample_rate:
            dropped.append((start, end))
            break
        retained.append((start, end))
        if end == num_samples:
            break
        start += STRIDE_SECONDS * sample_rate
    return retained, dropped


def validate_words(words: list[dict], duration: float) -> None:
    """Validate word times without treating ASR timestamps as manual truth."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Audio duration must be finite and positive")
    for word in words:
        start, end = float(word["start"]), float(word["end"])
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start <= end <= duration + 1 / SAMPLE_RATE:
            raise ValueError("Word times fall outside the audio clock")
        if not isinstance(word["word"], str):
            raise ValueError("Each ASR word must contain a string")


def match_word_timestamps(words: list[dict], start: float, end: float, duration: float) -> dict:
    """Select existing ASR tokens by midpoint; flag boundary-straddling words.

    Overlap words may appear in both overlapping model windows. This is distinct
    from deduplicating the ASR's own overlapping decoding windows upstream.
    """
    validate_words(words, duration)
    if not 0 <= start < end <= duration + 1 / SAMPLE_RATE:
        raise ValueError("Invalid model window bounds")
    selected = []
    is_last = abs(end - duration) <= 1 / SAMPLE_RATE
    for word in words:
        a, b = float(word["start"]), float(word["end"])
        midpoint = (a + b) / 2
        if start <= midpoint < end or (is_last and midpoint == end):
            selected.append({
                "word": word["word"],
                "start_seconds": max(0.0, a - start),
                "end_seconds": min(end - start, b - start),
                "parent_start_seconds": a,
                "parent_end_seconds": b,
                "boundary_truncated": a < start or b > end,
                "probability": word.get("probability"),
            })
    text = "".join(word["word"] for word in selected).strip()
    flags = []
    if not text:
        flags.append("empty_asr_transcript")
    if any(word["boundary_truncated"] for word in selected):
        flags.append("word_crosses_segment_boundary")
    return {
        "matched_transcript": text,
        "match_method": "asr_word_timestamps",
        "transcript_source": "independent_whisper_large_v3_asr",
        "matched_words": selected,
        "quality_flags": flags,
    }


def match_reference_span(match: dict, segment_id: str, audio_sha256: str, full_reference: str) -> dict:
    """Validate an explicitly supplied Whisper-medium/GPT reference match.

    This adapter imports existing matching outputs; it never requests an API or
    substitutes ASR/full-recording text when an artifact is absent or invalid.
    """
    if match.get("segment_id") != segment_id or match.get("audio_sha256") != audio_sha256:
        raise ValueError("Reference match does not belong to this exact segment audio")
    if match.get("whisper_model") != "medium" or match.get("matching_model") != "gpt-4o":
        raise ValueError("Reference adapter requires recorded Whisper-medium and GPT-4o provenance")
    digest = hashlib.sha256(full_reference.encode("utf-8")).hexdigest()
    if match.get("reference_text_sha256") != digest:
        raise ValueError("Reference match belongs to a different source transcript")
    text = match.get("matched_transcript")
    if not isinstance(text, str) or not text.strip() or text not in full_reference:
        raise ValueError("Matched transcript must be a nonempty verbatim source span")
    flags = list(match.get("quality_flags", []))
    if match.get("fallback", False):
        flags.append("reference_match_fallback_requires_review")
    return {
        "matched_transcript": text,
        "match_method": "whisper_medium_gpt_reference",
        "transcript_source": "provided_reference_transcript",
        "matched_words": [],
        "quality_flags": sorted(set(flags)),
    }


def original_time_pieces(time_map: list[dict], start: float, end: float) -> list[dict]:
    """Map a derived-audio interval back through retained original intervals."""
    pieces = []
    for interval in time_map:
        a = max(start, float(interval["clean_start"]))
        b = min(end, float(interval["clean_end"]))
        if b > a:
            original_start = float(interval["source_start"]) + a - float(interval["clean_start"])
            pieces.append({
                "segment_start_seconds": a - start,
                "segment_end_seconds": b - start,
                "original_start_seconds": original_start,
                "original_end_seconds": original_start + b - a,
            })
    if abs(sum(p["segment_end_seconds"] - p["segment_start_seconds"] for p in pieces) - (end - start)) > 1 / SAMPLE_RATE:
        raise ValueError("Original-time map does not cover the entire model window")
    for previous, current in zip(pieces, pieces[1:]):
        if abs(previous["segment_end_seconds"] - current["segment_start_seconds"]) > 1 / SAMPLE_RATE:
            raise ValueError("Original-time map has a gap or overlap in the derived timeline")
    return pieces


def load_recording_splits(data_root: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """Read the established participant splits without reading locked labels."""
    recordings, hashes, owners = {}, {}, {}
    for split in ("train", "validation", "test"):
        source = data_root / "processed" / "taukadial" / f"{split}.csv"
        hashes[split] = sha256_file(source)
        with source.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if split == "test" and "label" in (reader.fieldnames or []):
                raise ValueError("Default test manifest must not contain labels")
            for row in reader:
                rid, pid = recording_identity(row["recording_id"])
                if row["participant_id"] != pid or row["split"] != split:
                    raise ValueError("Recording identity/split metadata disagree")
                if rid in recordings:
                    raise ValueError("Recording occurs in multiple partitions")
                if pid in owners and owners[pid] != split:
                    raise ValueError("Participant appears in multiple partitions")
                owners[pid] = split
                if split != "test":
                    row["label"] = int(row["label"])
                    if row["label"] not in (0, 1):
                        raise ValueError("Unexpected diagnosis label")
                recordings[rid] = row
    return recordings, hashes


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def stable_write(path: Path, content: bytes) -> None:
    """Idempotent write: never follow links or overwrite conflicting artifacts."""
    if path.is_symlink():
        raise ValueError(f"Refusing symlink output: {path.name}")
    if path.exists():
        if sha256_file(path) != hashlib.sha256(content).hexdigest():
            raise ValueError(f"Existing output differs; choose a new version directory: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    if temporary.is_symlink() or temporary.exists():
        raise ValueError(f"Temporary output already exists: {temporary.name}")
    with temporary.open("xb") as stream:
        stream.write(content)
    temporary.replace(path)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def validate_time_map(mapping: list[dict], candidate_samples: int, source_samples: int) -> None:
    """Validate every retained interval on the original 16 kHz sample clock."""
    cursor, previous_end = 0, 0
    for interval in mapping:
        values = [float(interval[k]) for k in ("source_start", "source_end", "clean_start", "clean_end")]
        if not all(math.isfinite(x) for x in values):
            raise ValueError("Nonfinite original-time map")
        a, b, ca, cb = (round(x * SAMPLE_RATE) for x in values)
        if not (0 <= previous_end <= a < b <= source_samples and ca == cursor and cb - ca == b - a):
            raise ValueError("Invalid original-time map or concatenation gap")
        previous_end, cursor = b, cb
    if cursor != candidate_samples:
        raise ValueError("Time map length does not match candidate audio")


def verify_candidate_pcm(candidate, normalized_source, mapping: list[dict]) -> None:
    """Prove candidate samples are exactly the documented source intervals."""
    import numpy as np

    validate_time_map(mapping, len(candidate), len(normalized_source))
    restored = np.concatenate([
        normalized_source[round(m["source_start"] * SAMPLE_RATE):round(m["source_end"] * SAMPLE_RATE)]
        for m in mapping
    ])
    if not np.array_equal(candidate, restored):
        raise ValueError("Candidate PCM differs from documented original intervals")


def read_pcm(path: Path):
    import soundfile as sf

    info = sf.info(path)
    if info.samplerate != SAMPLE_RATE or info.channels != 1 or info.subtype != "PCM_16":
        raise ValueError(f"Expected 16 kHz mono PCM16: {path.name}")
    samples, rate = sf.read(path, dtype="int16")
    if rate != SAMPLE_RATE or samples.ndim != 1 or not len(samples):
        raise ValueError(f"Invalid or empty WAV: {path.name}")
    return samples


def import_candidate_sources(data_root: Path, source_project: Path) -> Path:
    """Copy approved local candidate inputs into a standalone private package.

    Source annotations may contain diagnosis-bearing original names and human
    transcripts. Only an explicit allowlist is exported; those fields never
    enter this project's model-facing data or sanitized provenance.
    """
    data_root, source_project = data_root.resolve(), source_project.resolve()
    recordings, split_hashes = load_recording_splits(data_root)
    source_base = source_project / "data" / "taukadial_asr"
    parent_path = source_base / "participant_manifest_all_candidates.jsonl"
    original_path = source_base / "reference_manifest.jsonl"
    parents = load_jsonl(parent_path)
    originals = {r["recording_id"]: r for r in load_jsonl(original_path)}
    hyp_dir = source_project / "outputs" / "taukadial_asr" / "participant_whisper_all_candidates_v1"
    if len(parents) != len(recordings) or {r["recording_id"] for r in parents} != set(recordings):
        raise ValueError("Candidate sources must cover every project recording exactly once")
    runs = []
    for name in ("english", "chinese", "english_unresolved"):
        run_path = hyp_dir.parent / f"participant_whisper_{name}_v1" / "run.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if (run.get("model", {}).get("revision") != ASR_REVISION or run.get("reference_blind") is not True
                or run.get("word_timestamps") is not True):
            raise ValueError("Unexpected existing ASR model provenance")
        runs.append({"source_run": f"participant_whisper_{name}_v1", "source_sha256": sha256_file(run_path),
                     "settings": {k: v for k, v in run.items() if k != "model_path"}})
    target = data_root / "sources" / "speech_v2"
    portable = []
    for number, parent in enumerate(sorted(parents, key=lambda row: row["recording_id"]), 1):
        rid = parent["recording_id"]
        known, original = recordings[rid], originals[rid]
        if (parent["participant_id"] != known["participant_id"] or parent["language"] != known["language"]
                or parent["original_wav_sha256"] != known["sha256"]
                or original["original_wav_sha256"] != known["sha256"]):
            raise ValueError(f"Candidate identity or original hash mismatch: {rid}")
        audio = Path(parent["source_audio_path"])
        sidecar_path = audio.with_suffix(".json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        hypothesis_path = hyp_dir / f"{rid}.json"
        hypothesis = json.loads(hypothesis_path.read_text(encoding="utf-8"))
        digest = sha256_file(audio)
        if digest != parent["normalized_wav_sha256"] or digest != hypothesis["source_sha256"]:
            raise ValueError(f"Candidate/ASR audio hash mismatch: {rid}")
        if (hypothesis["recording_id"] != rid or hypothesis["language"] != known["language"]
                or hypothesis.get("reference_blind") is not True or sidecar["recording_id"] != rid):
            raise ValueError(f"ASR/sidecar identity mismatch: {rid}")
        samples = read_pcm(audio)
        duration = len(samples) / SAMPLE_RATE
        if abs(duration - float(parent["duration_seconds"])) > 1 / SAMPLE_RATE:
            raise ValueError(f"Candidate duration mismatch: {rid}")
        validate_words(hypothesis["words"], duration)
        normalized_path = Path(sidecar["source_audio_path"])
        if sha256_file(normalized_path) != original["normalized_wav_sha256"]:
            raise ValueError(f"Normalized original audio hash mismatch: {rid}")
        verify_candidate_pcm(samples, read_pcm(normalized_path), sidecar["time_map"])
        flags = ["speaker_purity_not_manually_verified", "reference_assisted_candidate_audio", "asr_transcript_not_human_gold"]
        if not parent.get("task_verified_by_audio", False):
            flags.append("source_task_scope_unverified")
        if not parent.get("reference_scope_review_complete", False):
            flags.append("reference_scope_review_not_recorded_complete")
        if float(parent.get("unknown_speech_seconds", 0)) > 0:
            flags.append("unresolved_speaker_intervals")
        if sidecar.get("omitted_low_confidence_turns"):
            flags.append("low_confidence_source_turns_omitted")
        if sidecar.get("alignment_issues"):
            flags.append("source_alignment_requires_review")
        relative_audio = f"audio/{rid}.wav"
        copy_verified(audio, safe_member_path(relative_audio, target))
        item = {
            "recording_id": rid, "participant_id": known["participant_id"], "language": known["language"],
            "audio_path": relative_audio, "num_samples": len(samples), "sample_rate": SAMPLE_RATE, "channels": 1,
            "duration_seconds": duration, "sha256": digest, "original_wav_sha256": known["sha256"],
            "normalized_original_sha256": original["normalized_wav_sha256"],
            "normalized_original_num_samples": round(float(original["duration_seconds"]) * SAMPLE_RATE),
            "source_sidecar_sha256": sha256_file(sidecar_path), "source_asr_sha256": sha256_file(hypothesis_path),
            "extraction_condition": sidecar["condition"], "time_map": sidecar["time_map"],
            "words": hypothesis["words"], "quality_flags": sorted(set(flags)),
            "task_scope_verified": bool(parent.get("task_verified_by_audio", False)),
            "manual_speaker_qa_complete": bool(sidecar.get("manual_speaker_QA_complete", False)),
            "unknown_speech_seconds": parent.get("unknown_speech_seconds"),
            "alignment_issues": sidecar.get("alignment_issues", []),
            "omitted_low_confidence_turns": sidecar.get("omitted_low_confidence_turns", []),
            "candidate_pcm_matches_normalized_original_intervals": True,
        }
        stable_write(safe_member_path(f"records/{rid}.json", target), json_bytes(item))
        portable.append({k: item[k] for k in ("recording_id", "participant_id", "language", "audio_path", "sha256", "original_wav_sha256")})
        if number % 50 == 0:
            print(f"Verified and imported {number}/{len(parents)} candidate recordings", flush=True)
    stable_write(target / "manifest.jsonl", ("".join(json.dumps(row, sort_keys=True) + "\n" for row in portable)).encode())
    stable_write(target / "provenance.json", json_bytes({
        "schema_version": 2, "parent_manifest_sha256": sha256_file(parent_path),
        "original_reference_manifest_sha256": sha256_file(original_path), "split_manifest_sha256": split_hashes,
        "asr_runs": runs, "source_recordings": len(portable), "speaker_condition": "participant_candidates",
        "manual_speaker_purity_verified": False, "clinical_external_api_calls": 0,
        "source_normalization": "Existing scipy.signal.resample_poly mono 16 kHz PCM16; candidate samples retained exactly",
        "speaker_preprocessing": "Existing reference-assisted CHAT timing / CTC / acoustic candidate intervals; no new VAD or filtering",
        "candidate_pcm_verified_against_normalized_original": True,
    }))
    return target


def prepare_windows(data_root: Path, source_root: Path | None = None) -> dict:
    """Build versioned model inputs from the portable, validated source package."""
    import numpy as np
    import soundfile as sf

    data_root = data_root.resolve()
    source_root = (source_root or data_root / "sources" / "speech_v2").resolve()
    if not source_root.is_relative_to(data_root):
        raise ValueError("Portable source_root must be inside data_root for relative provenance paths")
    output = data_root / "processed" / "speech_v2"
    report_root = data_root / "reports" / "speech_v2"
    recordings, split_hashes = load_recording_splits(data_root)
    parents = load_jsonl(source_root / "manifest.jsonl")
    if len(parents) != len(recordings) or {r["recording_id"] for r in parents} != set(recordings):
        raise ValueError("Portable source manifest must cover every recording exactly once")
    manifests = {s: [] for s in ("train", "validation", "test")}
    details, exclusions, unresolved, hashes = [], [], [], {}
    source_hours = 0.0
    for parent in sorted(parents, key=lambda row: row["recording_id"]):
        rid = parent["recording_id"]
        known = recordings[rid]
        item_path = safe_member_path(f"records/{rid}.json", source_root)
        item = json.loads(item_path.read_text(encoding="utf-8"))
        if (item["original_wav_sha256"] != known["sha256"] or item["language"] != known["language"]
                or item["participant_id"] != known["participant_id"] or item["recording_id"] != rid):
            raise ValueError(f"Portable source identity mismatch: {rid}")
        source = safe_member_path(item["audio_path"], source_root)
        source_digest = sha256_file(source)
        if source_digest != item["sha256"] or source_digest != parent["sha256"]:
            raise ValueError(f"Portable candidate hash mismatch: {rid}")
        samples = read_pcm(source)
        if len(samples) != item["num_samples"]:
            raise ValueError(f"Portable candidate sample count mismatch: {rid}")
        duration = len(samples) / SAMPLE_RATE
        validate_words(item["words"], duration)
        validate_time_map(item["time_map"], len(samples), item["normalized_original_num_samples"])
        source_hours += duration / 3600
        kept, dropped = window_bounds(len(samples))
        for start, end in dropped:
            exclusions.append({"recording_id": rid, "participant_id": known["participant_id"], "split": known["split"],
                               "start_seconds": start / SAMPLE_RATE, "end_seconds": end / SAMPLE_RATE,
                               "reason": "entire_recording_under_15s" if not kept else "final_tail_under_15s"})
        unresolved.append({"recording_id": rid, "participant_id": known["participant_id"], "split": known["split"],
                           "language": known["language"], "quality_flags": item["quality_flags"],
                           "task_scope_verified": item["task_scope_verified"], "manual_speaker_qa_complete": item["manual_speaker_qa_complete"],
                           "alignment_issues": item["alignment_issues"], "omitted_low_confidence_turns": item["omitted_low_confidence_turns"]})
        for index, (start, end) in enumerate(kept):
            sid = f"{rid}-seg{index:03d}"
            split = known["split"]
            destination = safe_member_path(f"audio/{split}/{sid}.wav", output)
            buffer = io.BytesIO()
            sf.write(buffer, samples[start:end], SAMPLE_RATE, format="WAV", subtype="PCM_16")
            wav_bytes = buffer.getvalue()
            stable_write(destination, wav_bytes)
            restored = read_pcm(destination)
            if not np.array_equal(restored, samples[start:end]):
                raise ValueError(f"Saved segment PCM mismatch: {sid}")
            digest = hashlib.sha256(wav_bytes).hexdigest()
            if digest in hashes and hashes[digest] != split:
                raise ValueError("Byte-identical segment appears across dataset partitions")
            hashes[digest] = split
            match = match_word_timestamps(item["words"], start / SAMPLE_RATE, end / SAMPLE_RATE, duration)
            row = {
                "segment_id": sid, "recording_id": rid, "participant_id": known["participant_id"],
                "audio_path": destination.relative_to(data_root).as_posix(), "language": known["language"], "split": split,
                "start_seconds": start / SAMPLE_RATE, "end_seconds": end / SAMPLE_RATE,
                "duration_seconds": (end - start) / SAMPLE_RATE, "num_samples": end - start,
                "sample_rate": SAMPLE_RATE, "channels": 1, "sha256": digest,
                "source_sha256": source_digest, "original_wav_sha256": known["sha256"],
                "matched_transcript": match["matched_transcript"], "match_method": match["match_method"],
                "transcript_source": match["transcript_source"], "speaker_condition": "participant_candidates",
                "quality_flags": sorted(set(item["quality_flags"] + match["quality_flags"])),
                "original_time_pieces": original_time_pieces(item["time_map"], start / SAMPLE_RATE, end / SAMPLE_RATE),
            }
            if split != "test":
                row["label"] = known["label"]
            manifests[split].append(row)
            details.append({**row, "matched_words": match["matched_words"], "extraction_condition": item["extraction_condition"]})
    source_participants = {r["participant_id"] for r in recordings.values()}
    retained_participants = {r["participant_id"] for rows in manifests.values() for r in rows}
    missing_participants = sorted(source_participants - retained_participants)
    expected_files = {r["audio_path"] for rows in manifests.values() for r in rows}
    existing_files = {p.relative_to(data_root).as_posix() for p in (output / "audio").rglob("*.wav")}
    if expected_files != existing_files:
        raise ValueError("Unexpected stale or missing segment WAVs in output directory")
    for split, rows in manifests.items():
        if not rows:
            raise ValueError(f"No retained data for split: {split}")
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in row.items()})
        stable_write(output / f"{split}.csv", stream.getvalue().encode("utf-8"))
    for name, records in (("segments.jsonl", details), ("excluded_tails.jsonl", exclusions), ("unresolved_cases.jsonl", unresolved)):
        stable_write(output / name, ("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)).encode("utf-8"))
    report = {
        "schema_version": 2, "complete": not missing_participants, "source_recordings": len(parents),
        "source_participants": len(source_participants), "retained_recordings": len({r["recording_id"] for r in details}),
        "retained_participants": len(retained_participants), "participants_without_retained_audio": missing_participants,
        "retained_segments": len(details), "candidate_source_hours": source_hours,
        "segment_hours_including_overlap": sum(r["duration_seconds"] for r in details) / 3600,
        "exclusions": dict(Counter(r["reason"] for r in exclusions)),
        "empty_matched_transcripts": sum(not r["matched_transcript"] for r in details),
        "split_counts": {s: {"segments": len(rows), "recordings": len({r["recording_id"] for r in rows}),
                              "participants": len({r["participant_id"] for r in rows}),
                              "language_segments": dict(Counter(r["language"] for r in rows)),
                              **({"label_segments": dict(Counter(str(r["label"]) for r in rows))} if s != "test" else {})}
                         for s, rows in manifests.items()},
        "quality_flag_recording_counts": dict(Counter(flag for r in unresolved for flag in r["quality_flags"])),
        "quality_flag_segment_counts": dict(Counter(flag for r in details for flag in r["quality_flags"])),
        "window_seconds": WINDOW_SECONDS, "stride_seconds": STRIDE_SECONDS, "min_tail_seconds": MIN_TAIL_SECONDS,
        "tail_padding": False, "matching_method": "asr_word_timestamps", "human_reference_matching": False,
        "speaker_condition": "participant_candidates", "manual_speaker_purity_verified": False,
        "whispercare_revision": WHISPERCARE_REVISION, "asr_revision": ASR_REVISION,
        "split_manifest_sha256": split_hashes,
        "portable_source_manifest_sha256": sha256_file(source_root / "manifest.jsonl"),
        "checks": {"participant_disjointness": True, "audio_readability_and_pcm_integrity": True,
                   "source_audio_hashes": True, "original_time_map_coverage": True,
                   "test_labels_not_read_or_exported": True, "cross_split_byte_duplicates": 0},
    }
    checksums = {p.relative_to(data_root).as_posix(): sha256_file(p)
                 for root in (source_root, output) for p in sorted(root.rglob("*")) if p.is_file()}
    stable_write(report_root / "file_hashes.json", json_bytes(checksums))
    stable_write(report_root / "audit.json", json_bytes(report))
    if missing_participants:
        raise ValueError("Some participants have no retained windows; review audit before training")
    return report


def audit_existing(data_root: Path, parent_manifest: Path, hypotheses_dir: Path) -> dict:
    """Inspect available existing ASR input metadata, without claiming purity."""
    recordings, split_hashes = load_recording_splits(data_root)
    parents = load_jsonl(parent_manifest)
    counts, issues = Counter(), []
    ids = set()
    for parent in parents:
        rid = parent["recording_id"]
        if rid in ids or rid not in recordings:
            raise ValueError("Duplicate or unexpected parent recording")
        ids.add(rid)
        known = recordings[rid]
        if parent["original_wav_sha256"] != known["sha256"] or parent["language"] != known["language"]:
            raise ValueError("Parent provenance does not match this project's raw audio")
        hyp_path = hypotheses_dir / f"{rid}.json"
        if not hyp_path.exists():
            issues.append({"recording_id": rid, "issue": "asr_output_missing"})
            continue
        hypothesis = json.loads(hyp_path.read_text(encoding="utf-8"))
        if hypothesis["recording_id"] != rid or hypothesis["language"] != known["language"]:
            raise ValueError("ASR identity/language disagrees with parent")
        if hypothesis.get("reference_blind") is not True:
            raise ValueError("Existing ASR was not recorded as reference-blind")
        duration = float(parent["duration_seconds"])
        validate_words(hypothesis["words"], duration)
        counts["recordings"] += 1
        counts[f"language_{known['language']}"] += 1
        if not parent.get("task_verified_by_audio", False):
            counts["unverified_source_task_scope"] += 1
        if not parent.get("reference_scope_review_complete", False):
            counts["reference_scope_review_incomplete"] += 1
        if not hypothesis["words"]:
            counts["empty_asr_recordings"] += 1
        kept, dropped = window_bounds(round(duration * SAMPLE_RATE))
        counts["expected_segments"] += len(kept)
        counts["dropped_short_tails"] += len(dropped)
        counts["recordings_without_retained_segment"] += not kept
    if ids != set(recordings):
        raise ValueError("Parent manifest must cover all project recordings")
    return {
        "schema_version": 2,
        "counts": dict(counts),
        "issues": issues,
        "split_manifest_sha256": split_hashes,
        "parent_manifest_sha256": sha256_file(parent_manifest),
        "word_time_validation": "metadata-only; preparation additionally verifies physical audio hashes",
        "speaker_purity_certified": False,
        "reference_matching_complete": False,
        "whispercare_segmentation_revision": WHISPERCARE_REVISION,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="Inspect existing transcript assets before choosing a route")
    audit.add_argument("--data-root", type=Path, default=Path("data"))
    audit.add_argument("--parent-manifest", type=Path, required=True)
    audit.add_argument("--hypotheses-dir", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser("prepare", help="Build candidate-audio windows with local ASR timestamp matching")
    prepare.add_argument("--data-root", type=Path, default=Path("data"))
    prepare.add_argument("--source-project", type=Path, help="Import approved existing candidate assets before preparing; omit after restoring portable sources")
    prepare.add_argument("--source-root", type=Path, help="Portable source package (default: data/sources/speech_v2)")
    args = parser.parse_args()
    if args.command == "audit":
        report = audit_existing(args.data_root.resolve(), args.parent_manifest, args.hypotheses_dir)
        write_json(args.output, report)
    else:
        if args.source_project and args.source_root:
            parser.error("--source-project and --source-root cannot be combined")
        source_root = args.source_root.resolve() if args.source_root else None
        if args.source_project:
            source_root = import_candidate_sources(args.data_root.resolve(), args.source_project.resolve())
        report = prepare_windows(args.data_root.resolve(), source_root)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
