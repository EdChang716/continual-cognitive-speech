"""Prepare authorized TAUKADIAL audio with participant-disjoint data manifests.

Only standard-library dependencies are required. No audio transforms or model
inference are performed. Language labels must come from an explicit, hash-bound
map because the official metadata do not contain a language column.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import wave

LABELS = {"NC": 0, "MCI": 1}
RECORDING_PATTERN = re.compile(r"^(taukdial-\d{3})-([123])$")
ARCHIVES = {split: f"TAUKADIAL-24-{split}.tgz" for split in ("train", "test")}
LANGUAGE_SOURCE = (
    "User's existing source-cohort crosswalk: Chou=zh, Delaware=en; "
    "verified against each original WAV SHA-256. Not official language metadata."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_metadata(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"Empty metadata: {path.name}")
    dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    return list(csv.DictReader(io.StringIO(text), dialect=dialect))


def recording_identity(value: str) -> tuple[str, str]:
    # Metadata uses a filename, never a path; reject a path rather than strip it.
    if Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError("Recording metadata must contain a basename")
    recording_id = value[:-4] if value.endswith(".wav") else value
    match = RECORDING_PATTERN.fullmatch(recording_id)
    if match is None:
        raise ValueError(f"Unexpected TAUKADIAL recording name: {value}")
    return recording_id, match.group(1)


def safe_member_path(name: str, root: Path) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Unsafe archive member path: {name}")
    root = root.resolve()
    destination = root.joinpath(*relative.parts)
    if not destination.resolve().is_relative_to(root):
        raise ValueError(f"Archive member escapes destination: {name}")
    # Reject existing symlinks as well as links declared inside the archive.
    current = destination
    while current != root:
        if current.is_symlink():
            raise ValueError(f"Symlink in extraction destination: {name}")
        current = current.parent
    return destination


def copy_verified(source: Path, destination: Path) -> str:
    tmp = destination.with_suffix(destination.suffix + ".part")
    if destination.is_symlink() or tmp.is_symlink():
        raise ValueError(f"Symlink in import destination: {destination.name}")
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = sha256_file(source)
    if destination.exists():
        if sha256_file(destination) == digest:
            return digest
        raise ValueError(f"Existing imported file differs: {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, tmp)
    if sha256_file(tmp) != digest:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch while copying {source.name}")
    tmp.replace(destination)
    return digest


def extract_archive(archive: Path, destination: Path, split: str) -> dict[str, str]:
    """Extract only expected regular files; reject traversal, links and duplicates."""
    hashes: dict[str, str] = {}
    prefix = PurePosixPath("TAUKADIAL-24", split)
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            path = safe_member_path(member.name, destination)
            relative = PurePosixPath(member.name)
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported archive entry: {member.name}")
            if member.isdir():
                continue
            if relative.parent != prefix:
                raise ValueError(f"Unexpected archive location: {member.name}")
            if relative.name == "README.md":
                continue
            if relative.suffix != ".wav" and not (split == "train" and relative.name == "groundtruth.csv"):
                raise ValueError(f"Unexpected archive file: {member.name}")
            if member.name in hashes:
                raise ValueError(f"Duplicate archive entry: {member.name}")
            if relative.suffix == ".wav":
                recording_identity(relative.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = safe_member_path(member.name + ".part", destination)
            digest = hashlib.sha256()
            count = 0
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"Unreadable archive member: {member.name}")
            with source, temporary.open("wb") as output:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
                    output.write(block)
                    count += len(block)
            if count != member.size:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"Truncated archive member: {member.name}")
            if path.exists() and sha256_file(path) != digest.hexdigest():
                temporary.unlink(missing_ok=True)
                raise ValueError(f"Existing extracted file differs: {member.name}")
            temporary.replace(path)
            hashes[member.name] = digest.hexdigest()
    return hashes


def load_language_map(path: Path) -> dict[str, dict[str, str]]:
    """Import only ID, language and audio hash, ignoring any other source fields."""
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    else:
        rows = read_metadata(path)
    result = {}
    for row in rows:
        rid, _ = recording_identity(row["recording_id"])
        language = row["language"].lower()
        digest = row.get("original_wav_sha256", row.get("sha256", ""))
        if language not in {"en", "zh"} or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid language or audio SHA-256 for {rid}")
        if rid in result:
            raise ValueError(f"Duplicate language-map entry: {rid}")
        result[rid] = {
            "recording_id": rid,
            "language": language,
            "original_wav_sha256": digest,
        }
    return result


def check_participants(rows: list[dict]) -> dict[str, dict]:
    people: dict[str, dict] = {}
    recordings: dict[str, set] = defaultdict(set)
    for row in rows:
        pid = row["participant_id"]
        signature = {k: row[k] for k in ("language", "label") if k in row}
        if pid in people and people[pid] != signature:
            raise ValueError(f"Inconsistent participant language or diagnosis: {pid}")
        people[pid] = signature
        rid = row["recording_id"]
        if rid in recordings[pid]:
            raise ValueError(f"Duplicate recording: {rid}")
        recordings[pid].add(rid)
    for pid, ids in recordings.items():
        if ids != {f"{pid}-{i}" for i in (1, 2, 3)}:
            raise ValueError(f"Expected three distinct recordings for {pid}")
    return people


def split_participants(people: dict[str, dict], validation_fraction: float, seed: int) -> dict[str, str]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    strata: dict[tuple, list[str]] = defaultdict(list)
    for pid, attributes in people.items():
        strata[(attributes["language"], attributes["label"])].append(pid)
    assignments = {}
    for key, ids in sorted(strata.items()):
        if len(ids) < 2:
            raise ValueError(f"Cannot split stratum with fewer than two participants: {key}")
        ids.sort(key=lambda pid: (hashlib.sha256(f"{seed}|{pid}".encode()).hexdigest(), pid))
        count = max(1, min(len(ids) - 1, math.floor(len(ids) * validation_fraction + 0.5)))
        for index, pid in enumerate(ids):
            assignments[pid] = "validation" if index < count else "train"
    return assignments


def assert_disjoint(partitions: dict[str, list[dict]]) -> None:
    subject_owner = {}
    hash_owner = {}
    for split, rows in partitions.items():
        for row in rows:
            pid = row["participant_id"]
            if pid in subject_owner and subject_owner[pid] != split:
                raise ValueError(f"Participant leakage between {subject_owner[pid]} and {split}")
            subject_owner[pid] = split
            digest = row["sha256"]
            if digest in hash_owner and hash_owner[digest] != split:
                raise ValueError(f"Identical audio bytes cross {hash_owner[digest]} and {split}")
            hash_owner[digest] = split


def wav_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as audio:
        sample_rate = audio.getframerate()
        channels = audio.getnchannels()
        frames = audio.getnframes()
        width = audio.getsampwidth()
        if sample_rate <= 0 or frames <= 0 or channels <= 0:
            raise ValueError(f"Empty or invalid WAV: {path.name}")
        actual = 0
        for block in iter(lambda: audio.readframes(65536), b""):
            actual += len(block)
        if actual != frames * channels * width:
            raise ValueError(f"Truncated WAV frames: {path.name}")
        return {
            "duration_seconds": round(frames / sample_rate, 6),
            "sample_rate": sample_rate,
            "channels": channels,
            "sample_width_bytes": width,
            "num_frames": frames,
        }


def build_rows(raw_root: Path, data_root: Path, split: str, file_hashes: dict[str, str], languages: dict) -> list[dict]:
    metadata_path = raw_root / "TAUKADIAL-24" / split / "groundtruth.csv" if split == "train" else raw_root / "meta_test.csv"
    metadata = read_metadata(metadata_path)
    rows = []
    expected_files = set()
    for item in metadata:
        rid, pid = recording_identity(item["tkdname"])
        member = f"TAUKADIAL-24/{split}/{rid}.wav"
        if member not in file_hashes or rid not in languages:
            raise ValueError(f"Audio or language metadata missing: {rid}")
        digest = file_hashes[member]
        if digest != languages[rid]["original_wav_sha256"]:
            raise ValueError(f"Language map belongs to different audio: {rid}")
        path = raw_root / member
        row = {
            "recording_id": rid,
            "participant_id": pid,
            "official_split": split,
            "split": split,
            "audio_path": path.relative_to(data_root).as_posix(),
            "language": languages[rid]["language"],
            **wav_info(path),
            "sha256": digest,
            "speaker_scope": "original_recording_roles_not_filtered",
        }
        if split == "train":
            diagnosis = item["dx"].strip().upper()
            if diagnosis not in LABELS:
                raise ValueError(f"Unexpected training diagnosis: {diagnosis}")
            row["label"] = LABELS[diagnosis]
        rows.append(row)
        expected_files.add(member)
    wav_files = {name for name in file_hashes if name.endswith(".wav")}
    if expected_files != wav_files:
        raise ValueError(f"Metadata/audio coverage differs in {split}")
    rows.sort(key=lambda row: row["recording_id"])
    check_participants(rows)
    return rows


def prepare(source_dir: Path, data_root: Path, language_map: Path, seed: int = 42, validation_fraction: float = 0.2) -> dict:
    source_dir = source_dir.resolve()
    data_root = data_root.resolve()
    raw_root = data_root / "raw" / "taukadial"
    output = data_root / "processed" / "taukadial"
    locked = data_root / "locked" / "taukadial"
    report_dir = data_root / "reports" / "taukadial"
    truth_path = source_dir / "testgroundtruth.csv"
    if not truth_path.exists() and any((locked / name).exists() for name in ("testgroundtruth.csv", "test_labels.csv")):
        raise ValueError(
            "Existing locked test labels were found, but the current source directory "
            "does not supply testgroundtruth.csv. Supply the original labels or use a "
            "fresh data root; existing files have not been deleted or changed."
        )
    languages = load_language_map(language_map)
    archive_hashes = {}
    extracted = {}
    for split, filename in ARCHIVES.items():
        archive = raw_root / "archives" / filename
        archive_hashes[filename] = copy_verified(source_dir / filename, archive)
        extracted[split] = extract_archive(archive, raw_root, split)
        print(f"Imported and checked {filename}", flush=True)
    meta_hash = copy_verified(source_dir / "meta_test.csv", raw_root / "meta_test.csv")
    train = build_rows(raw_root, data_root, "train", extracted["train"], languages)
    test = build_rows(raw_root, data_root, "test", extracted["test"], languages)
    if len(train) != 387 or len(test) != 120:
        raise ValueError("This protocol requires the official 387-train/120-test release")
    all_ids = {r["recording_id"] for r in train + test}
    if all_ids != set(languages):
        raise ValueError("Language map must cover exactly the supplied 507 recordings")
    people = check_participants(train)
    if len(people) != 129 or len(check_participants(test)) != 40:
        raise ValueError("Unexpected official participant counts")
    assignments = split_participants(people, validation_fraction, seed)
    partitions = {name: [] for name in ("train", "validation", "test")}
    for row in train:
        row["split"] = assignments[row["participant_id"]]
        partitions[row["split"]].append(row)
    partitions["test"] = test
    assert_disjoint(partitions)
    for name, rows in partitions.items():
        write_csv(output / f"{name}.csv", rows, list(rows[0]))
    map_path = raw_root / "language_map.csv"
    write_csv(map_path, [languages[rid] for rid in sorted(languages)], ["recording_id", "language", "original_wav_sha256"])

    # The official held-out labels are copied separately, only for eventual
    # evaluation. They do not participate in the partitioning algorithm above.
    test_label_status = "not_supplied"
    if truth_path.exists():
        copy_verified(truth_path, locked / "testgroundtruth.csv")
        labels = []
        for item in read_metadata(truth_path):
            rid, pid = recording_identity(item["tkdname"])
            diagnosis = item["dx"].strip().upper()
            if diagnosis not in LABELS:
                raise ValueError("Unexpected official test diagnosis")
            labels.append({"recording_id": rid, "participant_id": pid, "label": LABELS[diagnosis]})
        if len(labels) != len(test) or {r["recording_id"] for r in labels} != {r["recording_id"] for r in test}:
            raise ValueError("Official test labels do not exactly match test recordings")
        consistency = defaultdict(set)
        for row in labels:
            consistency[row["participant_id"]].add(row["label"])
        if any(len(values) != 1 for values in consistency.values()):
            raise ValueError("Official test labels disagree within participant")
        write_csv(locked / "test_labels.csv", sorted(labels, key=lambda r: r["recording_id"]), ["recording_id", "participant_id", "label"])
        test_label_status = "present_in_locked_directory_not_used_for_splitting"
    summary = {
        "schema_version": 1,
        "dataset": "TAUKADIAL-24",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "split_policy": "Official test unchanged; official training participants stratified by language and diagnosis using SHA-256(seed|participant_id) ordering.",
        "label_map": LABELS,
        "language_provenance": LANGUAGE_SOURCE,
        "language_map_sha256": sha256_file(map_path),
        "archive_sha256": archive_hashes,
        "meta_test_sha256": meta_hash,
        "test_label_status": test_label_status,
        "audio_preprocessing": "None; preserve original WAV bytes, duration, sample rate, channels and pauses. Speaker roles have not been filtered.",
        "checks": {
            "participant_overlap": 0,
            "cross_partition_identical_wav_bytes": 0,
            "all_audio_readable": True,
            "language_map_hash_matches": 507,
            "three_recordings_per_participant": True,
            "test_manifest_has_labels": False,
        },
        "partitions": {},
        "manifest_sha256": {},
    }
    for name, rows in partitions.items():
        participants = check_participants(rows)
        stats = {
            "recordings": len(rows),
            "participants": len(participants),
            "participant_languages": dict(sorted(Counter(p["language"] for p in participants.values()).items())),
            "hours": round(sum(r["duration_seconds"] for r in rows) / 3600, 6),
            "sample_rates": dict(sorted(Counter(r["sample_rate"] for r in rows).items())),
            "channels": dict(sorted(Counter(r["channels"] for r in rows).items())),
            "unique_wav_hashes": len({r["sha256"] for r in rows}),
        }
        if name != "test":
            stats["participant_language_label_counts"] = dict(sorted(Counter(f"{p['language']}:{p['label']}" for p in participants.values()).items()))
        summary["partitions"][name] = stats
        summary["manifest_sha256"][f"{name}.csv"] = sha256_file(output / f"{name}.csv")
    write_json(report_dir / "audit.json", summary)
    write_json(report_dir / "file_hashes.json", {name: digest for split in extracted.values() for name, digest in split.items()})
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Authorized original archives and metadata")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--language-map", type=Path, help="CSV or JSONL containing recording_id, language, original_wav_sha256")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    mapping = args.language_map or args.source_dir / "language_map.csv"
    if not mapping.is_file():
        parser.error("A hash-bound --language-map is required; official metadata do not identify language")
    prepare(args.source_dir, args.data_root, mapping, args.seed, args.validation_fraction)


if __name__ == "__main__":
    main()
