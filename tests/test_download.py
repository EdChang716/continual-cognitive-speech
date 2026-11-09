"""Interrupted/corrupted transfer recovery without network access."""
import hashlib
import json
from pathlib import Path

import pytest

from ccspeech.data.download import acquire, download_verified


PAYLOAD = b"verified dataset bytes"
ITEM = {"size_bytes": len(PAYLOAD), "sha256": hashlib.sha256(PAYLOAD).hexdigest()}


def fake_curl(monkeypatch, payloads):
    calls = []

    def run(command, *, check):
        assert check
        target = Path(command[command.index("--output") + 1])
        calls.append(target.exists())
        target.write_bytes(next(payloads))

    monkeypatch.setattr("ccspeech.data.download.subprocess.run", run)
    return calls


def test_complete_valid_partial_is_published_without_network(tmp_path, monkeypatch):
    target = tmp_path / "train.csv"
    target.with_suffix(".csv.part").write_bytes(PAYLOAD)
    calls = fake_curl(monkeypatch, iter(()))
    download_verified(target, ITEM, "https://example.invalid/data")
    assert calls == []
    assert target.read_bytes() == PAYLOAD


def test_complete_corrupt_partial_is_quarantined_then_restarted(tmp_path, monkeypatch):
    target = tmp_path / "train.csv"
    corrupt = b"x" * len(PAYLOAD)
    target.with_suffix(".csv.part").write_bytes(corrupt)
    calls = fake_curl(monkeypatch, iter([PAYLOAD]))
    download_verified(target, ITEM, "https://example.invalid/data")
    assert calls == [False]
    assert target.read_bytes() == PAYLOAD
    assert next(tmp_path.glob("*.invalid-*")).read_bytes() == corrupt


def test_corrupt_resumed_transfer_gets_only_one_fresh_retry(tmp_path, monkeypatch):
    target = tmp_path / "train.csv"
    target.with_suffix(".csv.part").write_bytes(b"corrupt prefix")
    calls = fake_curl(monkeypatch, iter([b"wrong", PAYLOAD]))
    download_verified(target, ITEM, "https://example.invalid/data")
    assert calls == [True, False]
    assert target.read_bytes() == PAYLOAD


def test_repeated_corruption_fails_without_publishing(tmp_path, monkeypatch):
    target = tmp_path / "train.csv"
    calls = fake_curl(monkeypatch, iter([b"wrong", b"still wrong"]))
    with pytest.raises(ValueError, match="Unexpected size"):
        download_verified(target, ITEM, "https://example.invalid/data")
    assert len(calls) == 2
    assert not target.exists()


def test_existing_final_corruption_is_preserved_and_rejected(tmp_path, monkeypatch):
    target = tmp_path / "train.csv"
    target.write_bytes(b"user's existing file")
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps({"datasets": {"toy": {
        "repository": "example/toy", "revision": "pinned",
        "files": [{**ITEM, "local_path": "train.csv", "remote_path": "train.csv"}],
    }}}))
    calls = fake_curl(monkeypatch, iter(()))
    with pytest.raises(ValueError):
        acquire(tmp_path, sources)
    assert target.read_bytes() == b"user's existing file"
    assert calls == []


def test_partial_symlink_cannot_redirect_output(tmp_path, monkeypatch):
    target = tmp_path / "train.csv"
    elsewhere = tmp_path / "keep.txt"
    elsewhere.write_bytes(b"keep")
    target.with_suffix(".csv.part").symlink_to(elsewhere)
    calls = fake_curl(monkeypatch, iter(()))
    with pytest.raises(ValueError, match="symlink"):
        download_verified(target, ITEM, "https://example.invalid/data")
    assert elsewhere.read_bytes() == b"keep"
    assert calls == []
