"""Versioned all-tail export, explicit quarantine, and portable model inputs."""
import csv
import hashlib
import json

import numpy as np
import pytest
import soundfile as sf

from ccspeech.data.bundle import build, restore
from ccspeech.data.speech import window_bounds
from ccspeech.data.speech_v3 import prepare_windows_v3


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def sources(tmp_path):
    root = tmp_path / "data"
    source = root / "sources/speech_v3"
    (source / "audio").mkdir(parents=True)
    (source / "records").mkdir()
    manifests = {split: [] for split in ("train", "validation", "test")}
    portable = []
    for i, (split, seconds, eligibility) in enumerate([
        ("train", 14, "participant_candidate"), ("validation", 31, "participant_candidate"),
        ("test", 60, "participant_candidate"), ("train", 42, "quarantined"),
    ], 1):
        rid, pid = f"taukdial-{i:03d}-1", f"taukdial-{i:03d}"
        samples = np.random.default_rng(i).integers(-2000, 2000, seconds * 16000, dtype=np.int16)
        audio = source / f"audio/{rid}.wav"
        sf.write(audio, samples, 16000, subtype="PCM_16")
        digest = hashlib.sha256(audio.read_bytes()).hexdigest()
        known = {"recording_id": rid, "participant_id": pid, "language": "en", "split": split,
                 "sha256": f"raw{i}", "audio_path": f"raw/{rid}.wav"}
        if split != "test":
            known["label"] = i % 2
        manifests[split].append(known)
        item = {
            "recording_id": rid, "participant_id": pid, "language": "en", "audio_path": f"audio/{rid}.wav",
            "sha256": digest, "original_wav_sha256": f"raw{i}", "num_samples": len(samples),
            "normalized_original_num_samples": len(samples) + 16000,
            "time_map": [{"source_start": 1., "source_end": seconds + 1., "clean_start": 0., "clean_end": float(seconds)}],
            "words": [{"word": " first", "start": .1, "end": .4},
                      {"word": " last", "start": seconds - .4, "end": seconds - .1}],
            "quality_flags": ["speaker_purity_not_manually_verified"],
            "extraction_condition": "test_participant_candidate", "eligibility": eligibility,
            "exclusion_reasons": ["unresolved_speaker_identity"] if eligibility == "quarantined" else [],
            "review_intervals": [{"start_seconds": 1., "end_seconds": 2., "reason": "speaker_review"}]
                                if eligibility == "quarantined" else [],
        }
        (source / f"records/{rid}.json").write_text(json.dumps(item))
        portable.append({key: item[key] for key in (
            "recording_id", "participant_id", "language", "audio_path", "sha256", "original_wav_sha256", "eligibility")})
    for split, rows in manifests.items():
        write_csv(root / f"processed/taukadial/{split}.csv", rows)
    (source / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in portable))
    (source / "provenance.json").write_text(json.dumps({"schema_version": 3, "manual_speaker_purity_verified": False}))
    # A pre-existing version must remain byte-identical and outside the bundle.
    (root / "processed/speech_v2").mkdir()
    (root / "processed/speech_v2/old.txt").write_text("preserve v2\n")
    return root, source


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("samples,expected", [
    (1, [(0, 1)]), (14 * 16000, [(0, 14 * 16000)]),
    (30 * 16000, [(0, 30 * 16000)]),
    (30 * 16000 + 1, [(0, 30 * 16000), (27 * 16000, 30 * 16000 + 1)]),
    (57 * 16000, [(0, 30 * 16000), (27 * 16000, 57 * 16000)]),
])
def test_v3_retains_every_nonempty_sample_without_redundant_exact_tail(samples, expected):
    assert window_bounds(samples, min_tail_seconds=0) == (expected, [])
    if samples < 15 * 16000:
        assert window_bounds(samples)[0] == []  # The v2 default remains unchanged.


def test_v3_export_is_reproducible_preserves_shorts_and_separates_uncertainty(sources):
    root, _ = sources
    first = prepare_windows_v3(root)
    second = prepare_windows_v3(root)
    assert first == second
    assert first["retained_segments"] == 6
    assert first["short_segments_under_15s"] == 3
    assert first["quarantined_recordings"] == 1
    assert first["participants_without_eligible_audio"] == ["taukdial-004"]
    assert not first["participant_coverage_complete"]
    assert first["checks"]["quarantined_sources_in_primary_windows"] == 0
    output = root / "processed/speech_v3"
    expected_seconds = {"train": [14.], "validation": [30., 4.], "test": [30., 30., 6.]}
    for split, durations in expected_seconds.items():
        rows = read_csv(output / f"{split}.csv")
        assert [float(row["duration_seconds"]) for row in rows] == durations
        for row in rows:
            assert row["eligibility"] == "participant_candidate"
            assert "taukdial-004" not in row["segment_id"]
            assert row["dataset_version"] == "speech_v3"
            assert sf.info(root / row["audio_path"]).frames == int(row["num_samples"])
            assert sf.info(root / row["audio_path"]).frames <= 480000
            pieces = json.loads(row["original_time_pieces"])
            assert pieces[0]["original_start_seconds"] == float(row["start_seconds"]) + 1.
            assert bool("label" in row) == (split != "test")
    quarantine = [json.loads(line) for line in (output / "quarantined_recordings.jsonl").read_text().splitlines()]
    assert len(quarantine) == 1 and quarantine[0]["exclusion_reasons"]
    assert "label" not in quarantine[0]
    assert (root / quarantine[0]["audio_path"]).is_file()
    assert quarantine[0]["words"] and quarantine[0]["time_map"] and quarantine[0]["review_intervals"]
    assert (root / "processed/speech_v2/old.txt").read_text() == "preserve v2\n"
    assert str(root) not in (output / "segments.jsonl").read_text()


@pytest.mark.parametrize("change", [
    {"eligibility": None}, {"eligibility": "participant"},
    {"exclusion_reasons": ["unresolved_but_not_quarantined"]},
    {"participant_id": "taukdial-099"},
])
def test_v3_rejects_implicit_eligibility_or_identity_conflict(sources, change):
    root, source = sources
    path = source / "records/taukdial-001-1.json"
    item = json.loads(path.read_text())
    item.update(change)
    path.write_text(json.dumps(item))
    with pytest.raises(ValueError):
        prepare_windows_v3(root)


def test_v3_checks_quarantined_source_corruption_too(sources):
    root, source = sources
    audio = source / "audio/taukdial-004-1.wav"
    value = bytearray(audio.read_bytes())
    value[-1] ^= 1
    audio.write_bytes(value)
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_windows_v3(root)


def test_v3_empty_asr_tail_is_flagged_and_kept_for_audio(sources):
    root, source = sources
    path = source / "records/taukdial-002-1.json"
    item = json.loads(path.read_text())
    item["words"] = []
    path.write_text(json.dumps(item))
    audit = prepare_windows_v3(root)
    rows = read_csv(root / "processed/speech_v3/validation.csv")
    assert audit["empty_matched_transcripts"] == 3  # Two validation windows plus the already-empty middle test window.
    assert all(not row["matched_transcript"] and "empty_asr_transcript" in json.loads(row["quality_flags"]) for row in rows)


def test_v3_test_label_gate_participant_split_and_input_maps_are_enforced(sources):
    root, source = sources
    test_path = root / "processed/taukadial/test.csv"
    rows = read_csv(test_path)
    rows[0]["label"] = 1
    write_csv(test_path, rows)
    with pytest.raises(ValueError, match="must not contain labels"):
        prepare_windows_v3(root)
    rows[0].pop("label")
    write_csv(test_path, rows)
    path = source / "records/taukdial-001-1.json"
    item = json.loads(path.read_text())
    item["time_map"][0]["clean_end"] -= .1
    path.write_text(json.dumps(item))
    with pytest.raises(ValueError, match="time map"):
        prepare_windows_v3(root)


def test_v3_rejects_participant_crossing_original_splits(sources):
    root, _ = sources
    path = root / "processed/taukadial/validation.csv"
    rows = read_csv(path)
    rows[0]["recording_id"] = "taukdial-001-2"
    rows[0]["participant_id"] = "taukdial-001"
    write_csv(path, rows)
    with pytest.raises(ValueError, match="Participant appears in multiple partitions"):
        prepare_windows_v3(root)


def test_v3_bundle_restore_and_training_short_audio_collation(sources, tmp_path):
    from ccspeech.training.data import load_records, make_collator

    root, _ = sources
    prepare_windows_v3(root)
    bundle = tmp_path / "speech_data_v3.zip"
    result = build(root, bundle, "speech_v3")
    destination = tmp_path / "restored"
    restore(bundle, destination, result["sha256"])
    assert not (destination / "processed/speech_v2").exists()
    assert (destination / "processed/speech_v3/quarantined_recordings.jsonl").exists()
    records = load_records({"modality": "audio", "root": str(destination),
                            "speech_dir": "processed/speech_v3"}, "validation", "en")
    assert [int(row["num_samples"]) for row in records] == [480000, 64000]
    collator = make_collator({"modality": "audio", "name": "offline-test", "pretrained": False,
                             "random_config": {"num_mel_bins": 80, "max_source_positions": 1500}})
    batch = collator(records)
    assert batch["inputs"]["input_features"].shape == (2, 80, 3000)
    assert batch["inputs"]["attention_mask"].sum(dim=1).tolist() == [3000, 400]
    assert [row["eligibility"] for row in batch["records"]] == ["participant_candidate"] * 2
