"""Prepare deterministic, duplicate-disjoint Amazon/Yelp five-star DIL datasets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import re
import sqlite3
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from .download import safe_path, sha256_file, verify

DOMAINS = ("amazon", "yelp")
SPLITS = ("train", "validation", "test")
PROTOCOL_VERSION = "amazon-yelp-dil-v1"
SCHEMA = pa.schema([
    ("example_id", pa.string()),
    ("domain", pa.string()),
    ("source_split", pa.string()),
    ("source_row", pa.int64()),
    ("text", pa.string()),
    ("label", pa.int64()),
    ("text_sha256", pa.string()),
])
INDEX_FIELDS = [
    "example_id", "domain", "split", "source_split", "source_row", "label", "text_sha256",
]


def normalized_text_hash(text: str) -> bytes:
    """Hash NFKC/casefold/whitespace-normalized text; never alter model input."""
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    if not normalized:
        raise ValueError("Empty normalized text")
    return hashlib.sha256(normalized.encode("utf-8")).digest()


def _label(value: object, location: str) -> int:
    # Both pinned mirrors are zero-based. In particular, do not subtract one.
    if isinstance(value, bool) or not re.fullmatch(r"[0-4]", str(value)):
        raise ValueError(f"Label must be an integer in 0..4 at {location}")
    return int(str(value))


def _raw_rows(path: Path, domain: str) -> Iterator[tuple[int, str, int]]:
    """Yield zero-based source row, unchanged input text, and zero-based label."""
    if domain == "amazon":
        csv.field_size_limit(2**31 - 1)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"label", "title", "content"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing Amazon CSV columns: {path}")
            for row_number, row in enumerate(reader):
                title, content = row["title"], row["content"]
                if not isinstance(title, str) or not isinstance(content, str):
                    raise ValueError(f"Missing text field at amazon/{path.stem}:{row_number}")
                yield row_number, title + "\n" + content, _label(
                    row["label"], f"amazon/{path.stem}:{row_number}",
                )
    elif domain == "yelp":
        parquet = pq.ParquetFile(path)
        if not {"label", "text"}.issubset(parquet.schema_arrow.names):
            raise ValueError(f"Missing Yelp parquet columns: {path}")
        row_number = 0
        for batch in parquet.iter_batches(batch_size=8192, columns=["label", "text"]):
            columns = batch.to_pydict()
            for label, text in zip(columns["label"], columns["text"]):
                if not isinstance(text, str):
                    raise ValueError(f"Missing text field at yelp/{path.stem}:{row_number}")
                yield row_number, text, _label(label, f"yelp/{path.stem}:{row_number}")
                row_number += 1
    else:
        raise ValueError(f"Unknown domain: {domain}")


def _priority(seed: int, purpose: str, domain: str, split: str, row: int) -> int:
    key = f"{PROTOCOL_VERSION}|{seed}|{purpose}|{domain}|{split}|{row}"
    return int.from_bytes(hashlib.sha256(key.encode("ascii")).digest(), "big")


class _BalancedSample:
    """Keep the smallest deterministic pseudorandom priorities per label."""

    def __init__(self, per_class: int, seed: int, domain: str, source_split: str):
        self.per_class = per_class
        self.seed = seed
        self.domain = domain
        self.source_split = source_split
        self.heaps: dict[int, list[tuple[int, int, bytes]]] = {label: [] for label in range(5)}

    def add(self, row: int, label: int, digest: bytes) -> None:
        entry = (-_priority(self.seed, "sample", self.domain, self.source_split, row), -row, digest)
        heap = self.heaps[label]
        if len(heap) < self.per_class:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)

    def selected(self, validation_per_class: int) -> dict[int, tuple[str, int, bytes]]:
        result: dict[int, tuple[str, int, bytes]] = {}
        for label, heap in self.heaps.items():
            if len(heap) != self.per_class:
                raise ValueError(
                    f"Insufficient eligible {self.domain}/{self.source_split} label {label}: "
                    f"need {self.per_class}, found {len(heap)}",
                )
            candidates = sorted(
                ((-negative_row, digest) for _, negative_row, digest in heap),
                key=lambda item: (
                    _priority(self.seed, "validation", self.domain, self.source_split, item[0]),
                    item[0],
                ),
            )
            for position, (row, digest) in enumerate(candidates):
                if self.source_split == "test":
                    prepared_split = "test"
                else:
                    prepared_split = "validation" if position < validation_per_class else "train"
                result[row] = (prepared_split, label, digest)
        return result


def _sources(data_root: Path, sources_path: Path) -> tuple[dict, dict]:
    sources = json.loads(sources_path.read_text())
    paths: dict[tuple[str, str], tuple[Path, dict]] = {}
    for domain in DOMAINS:
        dataset = sources["datasets"][domain]
        for item in dataset["files"]:
            path = safe_path(data_root, item["local_path"])
            split = path.stem
            if split not in ("train", "test") or (domain, split) in paths:
                raise ValueError(f"Invalid or duplicate source file for {domain}: {path.name}")
            verify(path, item)
            paths[domain, split] = path, item
        if set(dataset["expected_rows"]) != {"train", "test"}:
            raise ValueError(f"Expected row counts must specify train and test for {domain}")
    if len(paths) != 4:
        raise ValueError("Expected exactly one training file and one test file per domain")
    return sources, paths


def _scan(
    connection: sqlite3.Connection,
    path: Path,
    domain: str,
    source_split: str,
    expected_rows: int,
    sampler: _BalancedSample,
) -> dict:
    counts: Counter = Counter()
    labels: Counter = Counter()
    eligible_labels: Counter = Counter()
    duplicate_origins: Counter = Counter()
    cursor = connection.cursor()
    for row, text, label in _raw_rows(path, domain):
        counts["rows"] += 1
        labels[str(label)] += 1
        try:
            digest = normalized_text_hash(text)
        except ValueError:
            counts["empty_text_excluded"] += 1
            continue
        cursor.execute(
            "INSERT OR IGNORE INTO seen VALUES (?, ?, ?, ?)",
            (digest, label, domain, source_split == "test"),
        )
        if cursor.rowcount == 0:
            original_label, original_domain, original_test = cursor.execute(
                "SELECT label, domain, official_test FROM seen WHERE digest = ?", (digest,),
            ).fetchone()
            counts["duplicate_excluded"] += 1
            duplicate_origins[f"{original_domain}/{'test' if original_test else 'train'}"] += 1
            if original_label != label:
                counts["conflicting_label_duplicates"] += 1
            if source_split == "train" and original_test:
                counts["excluded_due_to_official_test"] += 1
        else:
            counts["eligible_unique_rows"] += 1
            eligible_labels[str(label)] += 1
            sampler.add(row, label, digest)
        if counts["rows"] % 100_000 == 0:
            connection.commit()
        if counts["rows"] % 500_000 == 0:
            print(f"Scanned {domain}/{source_split}: {counts['rows']:,} rows", flush=True)
    connection.commit()
    if counts["rows"] != expected_rows:
        raise ValueError(
            f"Unexpected row count for {domain}/{source_split}: "
            f"expected {expected_rows}, found {counts['rows']}",
        )
    print(f"Audited {domain}/{source_split}: {counts['rows']:,} rows", flush=True)
    return {
        **{key: counts[key] for key in (
            "rows", "empty_text_excluded", "eligible_unique_rows", "duplicate_excluded",
            "conflicting_label_duplicates", "excluded_due_to_official_test",
        )},
        "label_counts": {str(label): labels[str(label)] for label in range(5)},
        "eligible_label_counts": {str(label): eligible_labels[str(label)] for label in range(5)},
        "duplicate_origins": dict(sorted(duplicate_origins.items())),
    }


def _materialize(
    connection: sqlite3.Connection,
    stage: Path,
    paths: dict,
    selections: dict,
    requested_counts: dict[str, int],
) -> dict:
    """Read original sources again and write only selected records, in source order."""
    connection.execute("CREATE TABLE prepared_seen (digest BLOB PRIMARY KEY) WITHOUT ROWID")
    counts: Counter = Counter()
    labels: dict[tuple[str, str], Counter] = {}
    writers = {}
    buffers: dict[tuple[str, str], list] = {}
    for domain in DOMAINS:
        (stage / domain).mkdir()
        for split in SPLITS:
            key = domain, split
            writers[key] = pq.ParquetWriter(stage / domain / f"{split}.parquet", SCHEMA, compression="zstd")
            buffers[key] = []
            labels[key] = Counter()

    def flush(key: tuple[str, str]) -> None:
        if buffers[key]:
            writers[key].write_table(pa.Table.from_pylist(buffers[key], schema=SCHEMA))
            buffers[key].clear()

    try:
        with (stage / "index.csv").open("w", encoding="utf-8", newline="") as stream:
            index_writer = csv.DictWriter(stream, fieldnames=INDEX_FIELDS)
            index_writer.writeheader()
            for domain in DOMAINS:
                for source_split in ("train", "test"):
                    selected = selections[domain, source_split]
                    path, _ = paths[domain, source_split]
                    written = 0
                    for row, text, label in _raw_rows(path, domain):
                        if row not in selected:
                            continue
                        split, expected_label, digest = selected[row]
                        if label != expected_label or normalized_text_hash(text) != digest:
                            raise ValueError(f"Source changed during preparation: {domain}/{source_split}:{row}")
                        try:
                            connection.execute("INSERT INTO prepared_seen VALUES (?)", (digest,))
                        except sqlite3.IntegrityError as error:
                            raise ValueError("Normalized text overlaps across prepared datasets") from error
                        record = {
                            "example_id": f"{domain}:{source_split}:{row:010d}",
                            "domain": domain,
                            "source_split": source_split,
                            "source_row": row,
                            "text": text,
                            "label": label,
                            "text_sha256": digest.hex(),
                        }
                        key = domain, split
                        buffers[key].append(record)
                        counts[key] += 1
                        labels[key][str(label)] += 1
                        written += 1
                        index_writer.writerow({
                            **{field: record[field] for field in INDEX_FIELDS if field != "split"},
                            "split": split,
                        })
                        if len(buffers[key]) >= 4096:
                            flush(key)
                    if written != len(selected):
                        raise ValueError(f"Missing selected rows in {domain}/{source_split}")
                print(f"Wrote prepared {domain} train/validation/test", flush=True)
        for key in buffers:
            flush(key)
    finally:
        for writer in writers.values():
            writer.close()

    total_rows = sum(counts.values())
    unique_rows = connection.execute("SELECT COUNT(*) FROM prepared_seen").fetchone()[0]
    if unique_rows != total_rows:
        raise ValueError("Prepared normalized-text uniqueness check failed")
    files = {}
    for domain in DOMAINS:
        files[domain] = {}
        for split in SPLITS:
            key = domain, split
            expected = requested_counts[split]
            if counts[key] != expected or any(labels[key][str(label)] != expected // 5 for label in range(5)):
                raise ValueError(f"Unbalanced or unexpected prepared counts: {domain}/{split}")
            path = stage / domain / f"{split}.parquet"
            files[domain][split] = {
                "path": f"processed/text/{domain}/{split}.parquet",
                "rows": counts[key],
                "label_counts": dict(sorted(labels[key].items())),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    return {
        "files": files,
        "total_rows": total_rows,
        "normalized_text_overlap_count": total_rows - unique_rows,
        "index": {"path": "processed/text/index.csv", "sha256": sha256_file(stage / "index.csv")},
    }


def prepare_text(
    data_root: Path,
    sources_path: Path,
    *,
    seed: int = 42,
    train_size: int = 115_000,
    validation_size: int = 10_000,
    test_size: int = 7_600,
    overwrite: bool = False,
) -> dict:
    """Verify all raw sources and publish a complete prepared text dataset."""
    requested = {"train": train_size, "validation": validation_size, "test": test_size}
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 or n % 5 for n in requested.values()):
        raise ValueError("Train, validation, and test sizes must be positive multiples of five")
    data_root = data_root.resolve()
    target = data_root / "processed/text"
    report_path = data_root / "reports/text.json"
    if target.exists() and not overwrite:
        raise FileExistsError(f"Prepared text data already exists: {target}; use --overwrite to rebuild")
    sources, paths = _sources(data_root, sources_path)
    print("Verified checksums for all four raw source files", flush=True)
    work_root = data_root / "tmp"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="text-", dir=work_root) as work:
        work = Path(work)
        stage = work / "prepared"
        stage.mkdir()
        connection = sqlite3.connect(work / "seen.sqlite")
        try:
            # This is a rebuildable scratch database, never a source-of-record.
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA cache_size=-65536")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute(
                "CREATE TABLE seen (digest BLOB PRIMARY KEY, label INTEGER NOT NULL, "
                "domain TEXT NOT NULL, official_test INTEGER NOT NULL) WITHOUT ROWID",
            )
            audit = {domain: {} for domain in DOMAINS}
            selections = {}
            # Protect every official test text, including test rows not selected below.
            for source_split in ("test", "train"):
                for domain in DOMAINS:
                    per_class = (test_size if source_split == "test" else train_size + validation_size) // 5
                    sampler = _BalancedSample(per_class, seed, domain, source_split)
                    path, _ = paths[domain, source_split]
                    audit[domain][source_split] = _scan(
                        connection, path, domain, source_split,
                        sources["datasets"][domain]["expected_rows"][source_split], sampler,
                    )
                    selections[domain, source_split] = sampler.selected(validation_size // 5)
            prepared = _materialize(connection, stage, paths, selections, requested)
        finally:
            connection.close()
        report = {
            "protocol": PROTOCOL_VERSION,
            "seed": seed,
            "requested_rows_per_domain": requested,
            "source_manifest_sha256": sha256_file(sources_path),
            "sources": sources["datasets"],
            "rules": {
                "domain_order_for_duplicate_ownership": list(DOMAINS),
                "scan_order": ["amazon/test", "yelp/test", "amazon/train", "yelp/train"],
                "label_mapping": "0,1,2,3,4 represent 1,2,3,4,5 stars; no relabeling",
                "amazon_input": "title + newline + content",
                "yelp_input": "text unchanged",
                "duplicate_key": "SHA-256 of UTF-8(NFKC(text).casefold(), whitespace collapsed)",
                "duplicate_policy": "Keep first globally; exclude all official-test hashes from training eligibility",
                "empty_text_policy": "Exclude whitespace-only normalized inputs; reject missing fields and invalid labels",
                "sampling": "Smallest SHA-256 pseudorandom priorities per class over all unique eligible rows",
                "validation": "Independent SHA-256 ordering partitions selected official-training pool per class",
                "prepared_row_order": "Original source row order; shuffle only in the training loader",
                "scope": "Adapted two-domain DIL comparison; not an exact MbPA++ benchmark reproduction",
            },
            "audit": audit,
            "prepared": prepared,
        }
        # Publish only after counts, labels, source hashes, and split disjointness pass.
        target.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        staged_report = work / "text.json"
        staged_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        previous = work / "previous"
        if target.exists():
            target.rename(previous)
        try:
            stage.rename(target)
        except BaseException:
            if previous.exists():
                previous.rename(target)
            raise
        staged_report.replace(report_path)
    print(f"Prepared {prepared['total_rows']:,} text records; report: {report_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--sources", type=Path, default=Path("configs/data/sources.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=115_000)
    parser.add_argument("--validation-size", type=int, default=10_000)
    parser.add_argument("--test-size", type=int, default=7_600)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare_text(
        args.data_root, args.sources, seed=args.seed, train_size=args.train_size,
        validation_size=args.validation_size, test_size=args.test_size, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
