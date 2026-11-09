"""Data-contract tests for fixed labels, sampling, and leakage prevention."""
import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ccspeech.data.download import sha256_file
from ccspeech.data.text import normalized_text_hash, prepare_text


def _sources(tmp_path: Path, records: dict | None = None) -> tuple[Path, Path]:
    data = tmp_path / "data"
    manifest = {"version": 1, "datasets": {}}
    for domain in ("amazon", "yelp"):
        dataset = {"repository": f"fixture/{domain}", "revision": "fixture-v1", "files": [], "expected_rows": {}}
        for split, per_class in (("train", 10), ("test", 4)):
            rows = []
            for label in range(5):
                for i in range(per_class):
                    text = f"{domain} {split} class {label} example {i}"
                    rows.append({"label": label, "text": text})
            rows.extend((records or {}).get((domain, split), []))
            path = data / "raw/text" / domain / (split + (".csv" if domain == "amazon" else ".parquet"))
            path.parent.mkdir(parents=True, exist_ok=True)
            if domain == "amazon":
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["label", "title", "content"])
                    writer.writeheader()
                    for row in rows:
                        writer.writerow({"label": row["label"], "title": "", "content": row["text"]})
            else:
                pq.write_table(pa.Table.from_pylist(rows), path)
            dataset["files"].append({"local_path": str(path.relative_to(data)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
            dataset["expected_rows"][split] = len(rows)
        manifest["datasets"][domain] = dataset
    manifest_path = tmp_path / "sources.json"
    manifest_path.write_text(json.dumps(manifest))
    return data, manifest_path


def _prepare(data: Path, manifest: Path, **kwargs):
    return prepare_text(data, manifest, train_size=10, validation_size=5, test_size=5, **kwargs)


def _records(data: Path):
    for domain in ("amazon", "yelp"):
        for split in ("train", "validation", "test"):
            for record in pq.read_table(data / "processed/text" / domain / f"{split}.parquet").to_pylist():
                yield split, record


def test_preserves_labels_source_text_and_balanced_disjoint_splits(tmp_path):
    data, manifest = _sources(tmp_path)
    report = _prepare(data, manifest)
    records = list(_records(data))
    assert report["prepared"]["total_rows"] == 40
    assert len({r["text_sha256"] for _, r in records}) == 40
    assert {r["label"] for _, r in records} == {0, 1, 2, 3, 4}
    for split, row in records:
        assert row["source_split"] == ("test" if split == "test" else "train")
        assert row["text_sha256"] == normalized_text_hash(row["text"]).hex()
        if row["domain"] == "amazon":
            assert row["text"].startswith("\n")  # Original empty title + required newline.
        else:
            assert not row["text"].startswith("\n")
    assert report["prepared"]["normalized_text_overlap_count"] == 0
    assert (data / "processed/text/index.csv").exists()


def test_excludes_all_official_test_texts_even_unselected_and_cross_domain(tmp_path):
    extras = {
        ("amazon", "train"): [{"label": 4, "text": "  YELP  TEST CLASS 0 EXAMPLE 0 "}],
        ("yelp", "train"): [{"label": 0, "text": "ＡＭＡＺＯＮ train class 0 example 0"}],
        ("yelp", "test"): [{"label": 1, "text": " AMAZON TEST CLASS 0 EXAMPLE 0 "}],
    }
    data, manifest = _sources(tmp_path, extras)
    report = _prepare(data, manifest)
    assert report["audit"]["amazon"]["train"]["excluded_due_to_official_test"] == 1
    assert report["audit"]["amazon"]["train"]["conflicting_label_duplicates"] == 1
    assert report["audit"]["yelp"]["train"]["duplicate_excluded"] == 1
    assert report["audit"]["yelp"]["test"]["duplicate_excluded"] == 1
    for split, record in _records(data):
        if split != "test":
            assert "test" not in record["text"].casefold()


def test_deterministic_for_same_seed_and_changes_with_seed(tmp_path):
    data, manifest = _sources(tmp_path)
    one = _prepare(data, manifest, seed=42)
    hashes_one = {p.relative_to(data): sha256_file(p) for p in (data / "processed/text").rglob("*") if p.is_file()}
    two = _prepare(data, manifest, seed=42, overwrite=True)
    hashes_two = {p.relative_to(data): sha256_file(p) for p in (data / "processed/text").rglob("*") if p.is_file()}
    assert one == two
    assert hashes_one == hashes_two
    three = _prepare(data, manifest, seed=43, overwrite=True)
    assert one["prepared"]["index"]["sha256"] != three["prepared"]["index"]["sha256"]


def test_rejects_invalid_label_without_publishing_data(tmp_path):
    data, manifest = _sources(tmp_path, {("yelp", "train"): [{"label": 5, "text": "invalid label"}]})
    with pytest.raises(ValueError, match="Label must be"):
        _prepare(data, manifest)
    assert not (data / "processed/text").exists()
    assert not (data / "reports/text.json").exists()


def test_excludes_empty_text_and_reports_it(tmp_path):
    data, manifest = _sources(tmp_path, {("yelp", "train"): [{"label": 0, "text": " \t \n "}]})
    report = _prepare(data, manifest)
    assert report["audit"]["yelp"]["train"]["empty_text_excluded"] == 1
    assert all(row["text"].strip() for _, row in _records(data))


def test_rejects_insufficient_eligible_source(tmp_path):
    data, manifest = _sources(tmp_path)
    with pytest.raises(ValueError, match="Insufficient eligible"):
        prepare_text(data, manifest, train_size=50, validation_size=5, test_size=5)
    assert not (data / "processed/text").exists()


def test_rejects_wrong_source_checksum_before_preparing(tmp_path):
    data, manifest = _sources(tmp_path)
    parsed = json.loads(manifest.read_text())
    parsed["datasets"]["amazon"]["files"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(parsed))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _prepare(data, manifest)
    assert not (data / "processed/text").exists()


def test_rejects_wrong_source_row_count(tmp_path):
    data, manifest = _sources(tmp_path)
    parsed = json.loads(manifest.read_text())
    parsed["datasets"]["amazon"]["expected_rows"]["test"] += 1
    manifest.write_text(json.dumps(parsed))
    with pytest.raises(ValueError, match="Unexpected row count"):
        _prepare(data, manifest)


@pytest.mark.parametrize("count", [0, 6, -5])
def test_rejects_invalid_requested_counts(tmp_path, count):
    with pytest.raises(ValueError, match="positive multiples of five"):
        prepare_text(tmp_path, tmp_path / "unused.json", train_size=count)
