"""Detect corrupted contents and structural leakage, even with rehashed files."""
import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ccspeech.data.download import sha256_file
from ccspeech.data.text import prepare_text
from ccspeech.data.validate import validate_taukadial, validate_text


def text_fixture(root):
    manifest = {"datasets": {}}
    for domain in ("amazon", "yelp"):
        spec = {"files": [], "expected_rows": {}}
        for split, count in (("train", 3), ("test", 1)):
            rows = [{"label": label, "text": f"{domain} {split} {label} {i}"}
                    for label in range(5) for i in range(count)]
            path = root / f"raw/text/{domain}/{split}{'.csv' if domain == 'amazon' else '.parquet'}"
            path.parent.mkdir(parents=True, exist_ok=True)
            if domain == "amazon":
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["title", "content", "label"])
                    writer.writeheader()
                    writer.writerows({"title": "", "content": r["text"], "label": r["label"]} for r in rows)
            else:
                pq.write_table(pa.Table.from_pylist(rows), path)
            spec["files"].append({"local_path": str(path.relative_to(root)), "size_bytes": path.stat().st_size,
                                  "sha256": sha256_file(path)})
            spec["expected_rows"][split] = len(rows)
        manifest["datasets"][domain] = spec
    sources = root / "sources.json"
    sources.write_text(json.dumps(manifest))
    prepare_text(root, sources, train_size=10, validation_size=5, test_size=5)
    return sources


def test_validate_text_full_and_detect_raw_corruption(tmp_path):
    sources = text_fixture(tmp_path)
    assert validate_text(tmp_path, full=True, sources=sources)["status"] == "passed"
    with (tmp_path / "raw/text/amazon/train.csv").open("a") as stream:
        stream.write("tampered")
    with pytest.raises(ValueError, match="Unexpected size"):
        validate_text(tmp_path, full=True, sources=sources)


def test_recomputes_text_hash_instead_of_trusting_column(tmp_path):
    text_fixture(tmp_path)
    path = tmp_path / "processed/text/amazon/train.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["text"] = "changed content but stale digest"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    report_path = tmp_path / "reports/text.json"
    report = json.loads(report_path.read_text())
    entry = report["prepared"]["files"]["amazon"]["train"]
    entry.update(sha256=sha256_file(path), size_bytes=path.stat().st_size)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Stored text digest"):
        validate_text(tmp_path)


def minimal_audio_fixture(root, *, test_labels=False, path_escape=False):
    """Three subjects suffice to reach rejection before the final v1 size check."""
    report = {"schema_version": 1, "dataset": "TAUKADIAL-24", "manifest_sha256": {}, "partitions": {}}
    for person, split in enumerate(("train", "validation", "test"), 1):
        rows = []
        for number in (1, 2, 3):
            rid = f"taukdial-{person:03d}-{number}"
            official = "test" if split == "test" else "train"
            audio_path = f"raw/taukadial/TAUKADIAL-24/{official}/{rid}.wav"
            audio = root / audio_path
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(rid.encode())
            row = {"recording_id": rid, "participant_id": f"taukdial-{person:03d}", "split": split,
                   "official_split": official, "audio_path": audio_path, "language": "en", "sha256": sha256_file(audio)}
            if split != "test" or test_labels:
                row["label"] = "0"
            if path_escape and split == "train":
                row["audio_path"] = "../escape.wav"
            rows.append(row)
        path = root / f"processed/taukadial/{split}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report["manifest_sha256"][f"{split}.csv"] = sha256_file(path)
        report["partitions"][split] = {"recordings": 3, "participants": 1, "participant_languages": {"en": 1},
                                        "participant_language_label_counts": {"en:0": 1}}
    path = root / "reports/taukadial/audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report))


def test_rejects_test_labels_even_when_manifest_checksum_matches(tmp_path):
    minimal_audio_fixture(tmp_path, test_labels=True)
    with pytest.raises(ValueError, match="Test manifest must not contain"):
        validate_taukadial(tmp_path)


def test_rejects_audio_path_outside_data_root(tmp_path):
    minimal_audio_fixture(tmp_path, path_escape=True)
    with pytest.raises(ValueError, match="Path escapes data root"):
        validate_taukadial(tmp_path)


def test_rejects_manifest_checksum_mismatch(tmp_path):
    minimal_audio_fixture(tmp_path)
    with (tmp_path / "processed/taukadial/train.csv").open("a") as stream:
        stream.write("tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_taukadial(tmp_path)
