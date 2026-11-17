import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from ccspeech.data.bundle import build, restore, safe_destination, verify_members


def fixture_data(root):
    paths = ["raw/text/amazon/train.csv", "raw/text/amazon/test.csv",
             "raw/text/yelp/train.parquet", "raw/text/yelp/test.parquet",
             "reports/text.json", "processed/text/index.csv"]
    paths += [f"processed/text/{d}/{s}.parquet" for d in ("amazon", "yelp")
              for s in ("train", "validation", "test")]
    for path in paths:
        dest = root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"toy data\n")


def test_bundle_reproducible_restore_and_conflict(tmp_path):
    source = tmp_path / "source"
    fixture_data(source)
    a = build(source, tmp_path / "a.zip", "text")
    b = build(source, tmp_path / "b.zip", "text")
    assert a["sha256"] == b["sha256"]
    dest = tmp_path / "restored"
    restore(tmp_path / "a.zip", dest, a["sha256"])
    assert (dest / "raw/text/amazon/train.csv").read_bytes() == b"toy data\n"
    (dest / "raw/text/amazon/train.csv").write_text("existing user data")
    with pytest.raises(ValueError, match="Existing data differ"):
        restore(tmp_path / "a.zip", dest, a["sha256"])
    assert (dest / "raw/text/amazon/train.csv").read_text() == "existing user data"


@pytest.mark.parametrize("name", ["../escape", "/raw/escape", "raw/../../escape", ".git/config", "raw\\escape"])
def test_reject_escape(tmp_path, name):
    with pytest.raises(ValueError):
        safe_destination(tmp_path, name)


def test_reject_corrupt_payload(tmp_path):
    path = tmp_path / "bad.zip"
    manifest = {"format_version": 1, "dataset": "text", "files": [
        {"path": "raw/text/x", "size_bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    ]}
    with ZipFile(path, "w") as z:
        z.writestr("raw/text/x", "bad")
        z.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest))
    with ZipFile(path) as z, pytest.raises(ValueError, match="checksum"):
        verify_members(z)


def test_ignore_download_quarantine_and_reject_temporary_symlink(tmp_path):
    source = tmp_path / "source"
    fixture_data(source)
    (source / "raw/text/amazon/train.csv.part.invalid-1").write_text("quarantine")
    result = build(source, tmp_path / "a.zip", "text")
    with ZipFile(tmp_path / "a.zip") as z:
        assert not any("invalid" in name for name in z.namelist())
    dest = tmp_path / "restored"
    partial = dest / "raw/text/amazon/train.csv.part"
    partial.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_text("keep")
    partial.symlink_to(outside)
    with pytest.raises(ValueError):
        restore(tmp_path / "a.zip", dest, result["sha256"])
    assert outside.read_text() == "keep"


def _restore_script(tmp_path, monkeypatch, release, datasets):
    import importlib.util

    spec = importlib.util.spec_from_file_location("restore_data_test", Path(__file__).resolve().parents[1] / "scripts/restore_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = tmp_path / f"configs/data/release_v{release}.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"bundles": [{"dataset": dataset, "file": dataset + ".zip", "sha256": "test"}
                                               for dataset in datasets]}))
    monkeypatch.setattr(module, "REPO", tmp_path)
    calls = []
    monkeypatch.setattr(module, "restore", lambda bundle, root, sha: calls.append((bundle, root, sha)))
    return module, calls


def test_restore_default_v3_selects_latest_training_bundle(tmp_path, monkeypatch):
    import sys

    module, calls = _restore_script(tmp_path, monkeypatch, 3, ["text", "taukadial", "speech_v2", "speech_v3"])
    monkeypatch.setattr(sys, "argv", ["restore_data.py"])
    module.main()
    assert [bundle for bundle, _, _ in calls] == [tmp_path / "artifacts/data_v3/text.zip",
                                                tmp_path / "artifacts/data_v3/speech_v3.zip"]


def test_restore_explicit_v2_still_resolves_its_own_bundle_directory(tmp_path, monkeypatch):
    import sys

    module, calls = _restore_script(tmp_path, monkeypatch, 2, ["text", "speech_v2"])
    monkeypatch.setattr(sys, "argv", ["restore_data.py", "--release", "2"])
    module.main()
    assert [bundle.parent for bundle, _, _ in calls] == [tmp_path / "artifacts/data_v2"] * 2
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["restore_data.py", "--release", "2", "--dataset", "speech_v3"])
    with pytest.raises(ValueError, match="unavailable"):
        module.main()
    assert not calls
