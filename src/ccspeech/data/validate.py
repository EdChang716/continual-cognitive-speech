"""Validate installed dataset contents, split integrity, and optional raw bytes."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import pyarrow.parquet as pq

from .download import safe_path, sha256_file, verify
from .taukadial import assert_disjoint, check_participants, recording_identity, wav_info
from .text import DOMAINS, SPLITS, PROTOCOL_VERSION, normalized_text_hash


def _checksum(path: Path, digest: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != digest:
        raise ValueError(f"SHA-256 mismatch: {path}")


def validate_text(root: Path, *, full: bool = False, sources: Path = Path("configs/data/sources.json")) -> dict:
    root = root.resolve()
    report = json.loads((root / "reports/text.json").read_text())
    if report.get("protocol") != PROTOCOL_VERSION:
        raise ValueError("Unsupported text protocol")
    seen = set()
    result = {}
    for domain in DOMAINS:
        result[domain] = {}
        for split in SPLITS:
            entry = report["prepared"]["files"][domain][split]
            relative = f"processed/text/{domain}/{split}.parquet"
            if entry["path"] != relative:
                raise ValueError("Unexpected prepared text path")
            path = safe_path(root, relative)
            _checksum(path, entry["sha256"])
            if path.stat().st_size != entry["size_bytes"]:
                raise ValueError(f"Prepared file size mismatch: {relative}")
            counts = Counter()
            rows = 0
            source_split = "test" if split == "test" else "train"
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
                for row in batch.to_pylist():
                    label = row["label"]
                    if type(label) is not int or label not in range(5):
                        raise ValueError("Invalid text label")
                    if row["domain"] != domain or row["source_split"] != source_split:
                        raise ValueError("Text domain or source split mismatch")
                    source_row = row["source_row"]
                    if type(source_row) is not int or source_row < 0:
                        raise ValueError("Invalid text source row")
                    if row["example_id"] != f"{domain}:{source_split}:{source_row:010d}":
                        raise ValueError("Text example ID does not match source row")
                    digest = normalized_text_hash(row["text"])
                    if digest.hex() != row["text_sha256"]:
                        raise ValueError("Stored text digest does not match actual normalized text")
                    if digest in seen:
                        raise ValueError("Normalized text overlaps within or across prepared splits")
                    seen.add(digest)
                    counts[str(label)] += 1
                    rows += 1
            requested = report["requested_rows_per_domain"][split]
            if rows != entry["rows"] or rows != requested:
                raise ValueError(f"Text row count mismatch: {domain}/{split}")
            if dict(counts) != entry["label_counts"] or any(counts[str(i)] != rows // 5 for i in range(5)):
                raise ValueError(f"Text class balance mismatch: {domain}/{split}")
            result[domain][split] = rows
    if len(seen) != report["prepared"]["total_rows"]:
        raise ValueError("Total prepared text count mismatch")
    index = report["prepared"]["index"]
    if index["path"] != "processed/text/index.csv":
        raise ValueError("Unexpected source index path")
    _checksum(safe_path(root, index["path"]), index["sha256"])
    if full:
        _checksum(sources, report["source_manifest_sha256"])
        source_manifest = json.loads(sources.read_text())
        if source_manifest["datasets"] != report["sources"]:
            raise ValueError("Raw source definitions differ from preparation report")
        for domain in DOMAINS:
            for item in source_manifest["datasets"][domain]["files"]:
                verify(safe_path(root, item["local_path"]), item)
    return {"dataset": "text", "status": "passed", "full": full, "rows": result,
            "normalized_text_overlap": 0}


def validate_taukadial(root: Path, *, full: bool = False) -> dict:
    root = root.resolve()
    report = json.loads((root / "reports/taukadial/audit.json").read_text())
    if report.get("schema_version") != 1 or report.get("dataset") != "TAUKADIAL-24":
        raise ValueError("Unsupported TAUKADIAL audit format")
    partitions = {}
    result = {}
    total = 0
    for split in SPLITS:
        path = safe_path(root, f"processed/taukadial/{split}.csv")
        _checksum(path, report["manifest_sha256"][f"{split}.csv"])
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or [])
            if split == "test" and fields.intersection({"label", "dx", "diagnosis", "mmse"}):
                raise ValueError("Test manifest must not contain diagnosis or cognitive-score labels")
            if split != "test" and "label" not in fields:
                raise ValueError("Training/validation diagnosis labels are missing")
            rows = list(reader)
        if not rows:
            raise ValueError(f"Empty TAUKADIAL manifest: {split}")
        for row in rows:
            rid, pid = recording_identity(row["recording_id"])
            if pid != row["participant_id"] or row["split"] != split:
                raise ValueError("TAUKADIAL identity or split mismatch")
            official_split = "test" if split == "test" else "train"
            if row["official_split"] != official_split:
                raise ValueError("Official TAUKADIAL test assignment changed")
            if row["language"] not in {"en", "zh"}:
                raise ValueError("Invalid TAUKADIAL language")
            if split != "test":
                if row["label"] not in {"0", "1"}:
                    raise ValueError("Invalid TAUKADIAL diagnosis label")
                row["label"] = int(row["label"])
            audio = safe_path(root, row["audio_path"])
            expected = f"raw/taukadial/TAUKADIAL-24/{official_split}/{rid}.wav"
            if row["audio_path"] != expected:
                raise ValueError("TAUKADIAL audio path does not match recording identity")
            if not audio.is_file():
                raise FileNotFoundError(audio)
            if full:
                _checksum(audio, row["sha256"])
                actual = wav_info(audio)
                for field, value in actual.items():
                    if field == "duration_seconds":
                        if abs(float(row[field]) - value) > 1e-6:
                            raise ValueError("WAV duration differs from manifest")
                    elif int(row[field]) != value:
                        raise ValueError(f"WAV {field} differs from manifest")
        people = check_participants(rows)
        expected_stats = report["partitions"][split]
        if len(rows) != expected_stats["recordings"] or len(people) != expected_stats["participants"]:
            raise ValueError(f"TAUKADIAL counts differ from audit: {split}")
        if dict(Counter(p["language"] for p in people.values())) != expected_stats["participant_languages"]:
            raise ValueError(f"TAUKADIAL language counts differ from audit: {split}")
        if split != "test":
            counts = Counter(f"{p['language']}:{p['label']}" for p in people.values())
            if dict(counts) != expected_stats["participant_language_label_counts"]:
                raise ValueError(f"TAUKADIAL class counts differ from audit: {split}")
        partitions[split] = rows
        total += len(rows)
        result[split] = {"participants": len(people), "recordings": len(rows)}
    assert_disjoint(partitions)
    if total != 507 or result["test"] != {"participants": 40, "recordings": 120}:
        raise ValueError("TAUKADIAL v1 requires 507 recordings and the unchanged 40-participant official test")
    # Held-out label values are deliberately never loaded by this validator.
    return {"dataset": "taukadial", "status": "passed", "full": full, "splits": result,
            "participant_overlap": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "text", "taukadial"], default="all")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--sources", type=Path, default=Path("configs/data/sources.json"))
    parser.add_argument("--full", action="store_true", help="Also verify raw source checksums and every WAV frame")
    args = parser.parse_args()
    if args.dataset in {"all", "text"}:
        print(json.dumps(validate_text(args.data_root, full=args.full, sources=args.sources), sort_keys=True))
    if args.dataset in {"all", "taukadial"}:
        print(json.dumps(validate_taukadial(args.data_root, full=args.full), sort_keys=True))


if __name__ == "__main__":
    main()
