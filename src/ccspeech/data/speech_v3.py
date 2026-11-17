"""Export speech v3 without replacing v2 or promoting uncertain source audio.

The upstream source builder must assign explicit eligibility to every original
TAUKADIAL recording. This exporter preserves every nonempty eligible tail and
retains quarantined sources only for review. It does not run ASR or certify
participant speaker identity; timestamp matching reuses the supplied ASR words.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

from .speech import (
    SAMPLE_RATE, STRIDE_SECONDS, WINDOW_SECONDS, json_bytes, load_jsonl,
    load_recording_splits, match_word_timestamps, original_time_pieces,
    read_pcm, stable_write, validate_time_map, validate_words, window_bounds,
)
from .taukadial import safe_member_path, sha256_file

VERSION = "speech_v3"
ELIGIBILITIES = {"participant_candidate", "quarantined"}


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                         if isinstance(value, (list, dict)) else value for key, value in row.items()})
    stable_write(path, stream.getvalue().encode("utf-8"))


def _write_jsonl(path: Path, records: list[dict]) -> None:
    stable_write(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                               for row in records).encode("utf-8"))


def _eligibility(item: dict, parent: dict) -> str:
    value = item.get("eligibility")
    if value not in ELIGIBILITIES:
        raise ValueError("Every v3 source record requires explicit supported eligibility")
    if "eligibility" in parent and parent["eligibility"] != value:
        raise ValueError("Source manifest and record eligibility disagree")
    reasons = item.get("exclusion_reasons")
    if not isinstance(reasons, list) or any(not isinstance(reason, str) or not reason.strip() for reason in reasons):
        raise ValueError("Source exclusion_reasons must be an explicit list of nonempty strings")
    if value == "quarantined" and not reasons:
        raise ValueError("Quarantined sources require an exclusion reason")
    if value == "participant_candidate" and reasons:
        raise ValueError("An eligible source cannot carry exclusion reasons")
    if not isinstance(item.get("review_intervals", []), list):
        raise ValueError("review_intervals must be a list")
    return value


def prepare_windows_v3(data_root: Path, source_root: Path | None = None) -> dict:
    """Validate complete v3 sources and write only eligible model-facing windows.

    Whole recordings or participants with no eligible input stay excluded; the
    audit names these omissions rather than silently falling back to v2 audio.
    """
    import numpy as np
    import soundfile as sf

    data_root = data_root.resolve()
    source_root = (source_root or data_root / "sources" / VERSION).resolve()
    if not source_root.is_relative_to(data_root):
        raise ValueError("Portable source_root must be inside data_root")
    output, report_root = data_root / "processed" / VERSION, data_root / "reports" / VERSION
    recordings, split_hashes = load_recording_splits(data_root)
    parents = load_jsonl(source_root / "manifest.jsonl")
    if len(parents) != len(recordings) or {row["recording_id"] for row in parents} != set(recordings):
        raise ValueError("The v3 source manifest must cover every recording exactly once")
    source_provenance = json.loads((source_root / "provenance.json").read_text(encoding="utf-8"))
    manifests = {split: [] for split in ("train", "validation", "test")}
    details, quarantined, hashes = [], [], {}
    source_flags, source_hours, eligible_hours = Counter(), 0., 0.
    for parent in sorted(parents, key=lambda row: row["recording_id"]):
        rid = parent["recording_id"]
        known = recordings[rid]
        item_path = safe_member_path(f"records/{rid}.json", source_root)
        item = json.loads(item_path.read_text(encoding="utf-8"))
        for key in ("recording_id", "participant_id", "language"):
            if item.get(key) != known[key] or parent.get(key) != known[key]:
                raise ValueError(f"Portable v3 source identity mismatch: {rid}")
        if item.get("original_wav_sha256") != known["sha256"] or parent.get("original_wav_sha256") != known["sha256"]:
            raise ValueError(f"Portable v3 original hash mismatch: {rid}")
        eligibility = _eligibility(item, parent)
        flags = item.get("quality_flags")
        if not isinstance(flags, list) or any(not isinstance(flag, str) for flag in flags):
            raise ValueError("Source quality_flags must be a list of strings")
        source = safe_member_path(item["audio_path"], source_root)
        if parent.get("audio_path") != item["audio_path"]:
            raise ValueError(f"Portable v3 source audio path mismatch: {rid}")
        digest = sha256_file(source)
        if digest != item["sha256"] or digest != parent["sha256"]:
            raise ValueError(f"Portable v3 source hash mismatch: {rid}")
        samples = read_pcm(source)
        if len(samples) != item["num_samples"]:
            raise ValueError(f"Portable v3 source sample count mismatch: {rid}")
        duration = len(samples) / SAMPLE_RATE
        validate_words(item["words"], duration)
        validate_time_map(item["time_map"], len(samples), item["normalized_original_num_samples"])
        source_flags.update(flags)
        source_hours += duration / 3600
        if eligibility == "quarantined":
            quarantined.append({
                "recording_id": rid, "participant_id": known["participant_id"],
                "language": known["language"], "split": known["split"], "eligibility": eligibility,
                "exclusion_reasons": item["exclusion_reasons"], "quality_flags": flags,
                "review_intervals": item.get("review_intervals", []),
                "audio_path": source.relative_to(data_root).as_posix(), "sha256": digest,
                "num_samples": len(samples), "duration_seconds": duration,
                "source_record_path": item_path.relative_to(data_root).as_posix(),
                "time_map": item["time_map"], "words": item["words"],
            })
            continue
        eligible_hours += duration / 3600
        kept, dropped = window_bounds(len(samples), min_tail_seconds=0)
        if dropped or not kept:
            raise ValueError("V3 must preserve every nonempty eligible input")
        for index, (start, end) in enumerate(kept):
            sid, split = f"{rid}-v3-seg{index:03d}", known["split"]
            destination = safe_member_path(f"audio/{split}/{sid}.wav", output)
            buffer = io.BytesIO()
            sf.write(buffer, samples[start:end], SAMPLE_RATE, format="WAV", subtype="PCM_16")
            wav = buffer.getvalue()
            stable_write(destination, wav)
            if not np.array_equal(read_pcm(destination), samples[start:end]):
                raise ValueError(f"Saved v3 segment PCM mismatch: {sid}")
            segment_digest = hashlib.sha256(wav).hexdigest()
            if segment_digest in hashes and hashes[segment_digest] != split:
                raise ValueError("Byte-identical segment appears across dataset partitions")
            hashes[segment_digest] = split
            match = match_word_timestamps(item["words"], start / SAMPLE_RATE, end / SAMPLE_RATE, duration)
            window_flags = flags + match["quality_flags"]
            if end - start < 15 * SAMPLE_RATE:
                window_flags = window_flags + ["short_window_retained_v3"]
            row = {
                "segment_id": sid, "recording_id": rid, "participant_id": known["participant_id"],
                "audio_path": destination.relative_to(data_root).as_posix(), "language": known["language"],
                "split": split, "dataset_version": VERSION, "eligibility": eligibility,
                "start_seconds": start / SAMPLE_RATE, "end_seconds": end / SAMPLE_RATE,
                "duration_seconds": (end - start) / SAMPLE_RATE, "num_samples": end - start,
                "sample_rate": SAMPLE_RATE, "channels": 1, "sha256": segment_digest,
                "source_sha256": digest, "original_wav_sha256": known["sha256"],
                "matched_transcript": match["matched_transcript"], "match_method": match["match_method"],
                "transcript_source": match["transcript_source"], "speaker_condition": "participant_candidates",
                "quality_flags": sorted(set(window_flags)),
                "original_time_pieces": original_time_pieces(item["time_map"], start / SAMPLE_RATE, end / SAMPLE_RATE),
                "source_review_intervals": item.get("review_intervals", []),
            }
            if split != "test":
                row["label"] = known["label"]
            manifests[split].append(row)
            details.append({**row, "matched_words": match["matched_words"],
                            "source_record_path": item_path.relative_to(data_root).as_posix(),
                            "extraction_condition": item["extraction_condition"]})
    expected = {row["audio_path"] for row in details}
    actual = {path.relative_to(data_root).as_posix() for path in (output / "audio").rglob("*.wav")}
    if expected != actual:
        raise ValueError("Unexpected stale or missing v3 segment WAVs")
    # Derive the shared field order from a populated partition and still write
    # explicit header-only files if an entire split has been quarantined.
    if not details:
        raise ValueError("No eligible participant inputs; no training release can be exported")
    fields = [key for key in next(rows[0] for rows in manifests.values() if rows) if key != "label"]
    for split, rows in manifests.items():
        _write_csv(output / f"{split}.csv", rows, fields + (["label"] if split != "test" else []))
    _write_jsonl(output / "segments.jsonl", details)
    _write_jsonl(output / "quarantined_recordings.jsonl", quarantined)
    participants = {row["participant_id"] for row in recordings.values()}
    retained_participants = {row["participant_id"] for row in details}
    retained_recordings = {row["recording_id"] for row in details}
    report = {
        "schema_version": 3, "dataset_version": VERSION, "complete": True,
        "source_recordings": len(parents), "source_participants": len(participants),
        "retained_recordings": len(retained_recordings), "retained_participants": len(retained_participants),
        "retained_segments": len(details), "quarantined_recordings": len(quarantined),
        "recordings_without_eligible_audio": sorted(set(recordings) - retained_recordings),
        "participants_without_eligible_audio": sorted(participants - retained_participants),
        "participant_coverage_complete": participants == retained_participants,
        "all_splits_have_eligible_input": all(manifests.values()),
        "candidate_source_hours": source_hours, "eligible_source_hours": eligible_hours,
        "segment_hours_including_overlap": sum(row["duration_seconds"] for row in details) / 3600,
        "short_segments_under_15s": sum(row["duration_seconds"] < 15 for row in details),
        "empty_matched_transcripts": sum(not row["matched_transcript"] for row in details),
        "quarantine_reason_counts": dict(Counter(reason for row in quarantined for reason in row["exclusion_reasons"])),
        "split_counts": {split: {
            "segments": len(rows), "recordings": len({row["recording_id"] for row in rows}),
            "participants": len({row["participant_id"] for row in rows}),
            "language_segments": dict(Counter(row["language"] for row in rows)),
            **({"label_segments": dict(Counter(str(row["label"]) for row in rows))} if split != "test" else {}),
        } for split, rows in manifests.items()},
        "quality_flag_recording_counts": dict(source_flags),
        "quality_flag_segment_counts": dict(Counter(flag for row in details for flag in row["quality_flags"])),
        "window_seconds": WINDOW_SECONDS, "stride_seconds": STRIDE_SECONDS, "min_tail_seconds": 0,
        "tail_padding": False, "model_padding": "Whisper feature extractor pads during collation only",
        "matching_method": "asr_word_timestamps", "human_reference_matching": False,
        "speaker_condition": "participant_candidates", "manual_speaker_purity_verified": False,
        "eligibility_policy": "Only explicit participant_candidate sources enter primary windows; quarantined sources remain separate",
        "review_limitations": "Eligibility is an upstream evidence decision, not certification of speaker purity or transcription accuracy",
        "source_provenance": source_provenance,
        "source_provenance_sha256": sha256_file(source_root / "provenance.json"),
        "portable_source_manifest_sha256": sha256_file(source_root / "manifest.jsonl"),
        "split_manifest_sha256": split_hashes,
        "checks": {"participant_disjointness": True, "audio_readability_and_pcm_integrity": True,
                   "source_audio_hashes": True, "original_time_map_coverage": True,
                   "test_labels_not_read_or_exported": True, "cross_split_byte_duplicates": 0,
                   "quarantined_sources_in_primary_windows": 0, "eligible_short_tail_exclusions": 0},
    }
    checksums = {path.relative_to(data_root).as_posix(): sha256_file(path)
                 for folder in (source_root, output) for path in sorted(folder.rglob("*")) if path.is_file()}
    stable_write(report_root / "file_hashes.json", json_bytes(checksums))
    stable_write(report_root / "audit.json", json_bytes(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--data-root", type=Path, default=Path("data"))
    prepare.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare_windows_v3(args.data_root, args.source_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
