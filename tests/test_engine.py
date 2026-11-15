"""Offline end-to-end experiments with real tiny BERT and Whisper encoders."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
import torch

from ccspeech.training.config import base_config
from ccspeech.training.engine import final_test, run_experiment
from ccspeech.training.state import read_checkpoint
from ccspeech.training.suites import run_name


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json(path):
    return json.loads(path.read_text())


def _assert_tree(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_tree(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            _assert_tree(a, b)
    else:
        assert first == second


@pytest.fixture
def prepared(tmp_path):
    """Create only synthetic inputs; no downloaded model or human data needed."""
    original_threads = torch.get_num_threads()
    root = tmp_path / "data"
    vocabulary = tmp_path / "vocab.txt"
    vocabulary.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nbad\npoor\nokay\ngood\ngreat\nreview\nfood\nbook\n")
    for domain in ("amazon", "yelp"):
        for split, count in (("train", 6), ("validation", 5), ("test", 5)):
            rows = [{"example_id": f"{domain}-{split}-{i}", "domain": domain,
                     "source_split": "test" if split == "test" else "train",
                     "text": ["bad", "poor", "okay", "good", "great"][i % 5] +
                             (" book review" if domain == "amazon" else " food review"),
                     "label": i % 5} for i in range(count)]
            path = root / "processed/text" / domain / f"{split}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows), path)
    truth = []
    for split, count in (("train", 6), ("validation", 2), ("test", 2)):
        rows = []
        for domain in ("en", "zh"):
            for i in range(count):
                identity = f"{domain}-{split}-{i}"
                relative = f"processed/speech_v2/audio/{identity}.wav"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                waveform = .15 * np.sin(np.arange(4000) * (0.03 + i * .003)).astype(np.float32)
                sf.write(path, waveform, 16000, subtype="PCM_16")
                row = {"segment_id": identity + "-s0", "recording_id": identity + "-r0",
                       "participant_id": identity, "language": domain, "split": split,
                       "audio_path": relative, "matched_transcript": "synthetic example",
                       "num_samples": 4000, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                if split == "test":
                    truth.append({"recording_id": row["recording_id"], "participant_id": identity, "label": i % 2})
                else:
                    row["label"] = i % 2
                rows.append(row)
        _write_csv(root / "processed/speech_v2" / f"{split}.csv", rows)
    _write_csv(root / "locked/taukadial/test_labels.csv", truth)

    def config(phase="text", method="derpp"):
        value = base_config(phase)
        value["method"] = method
        value["seed"] = 19
        value["data"]["root"] = str(root)
        value["runtime"] = {"device": "cpu", "num_threads": 1, "smoke": True}
        value["training"].update(source_epochs=1, target_epochs=1, batch_size=2, lr=.003,
                                  weight_decay=.0, checkpoint_every=1, max_length=8)
        value["memory"].update(capacity=2, replay_batch_size=2, sparse_frequency=2,
                               sparse_batch_size=2, sparse_microbatch_size=1)
        value["local_adaptation"].update(neighbors=2, steps=1, lr=.001, microbatch_size=1)
        value["evaluation"].update(batch_size=3, bootstrap_samples=12)
        if phase == "text":
            value["model"] = {"modality": "text", "name": "offline-tiny-bert", "pretrained": False,
                              "vocab_file": str(vocabulary), "random_config": {
                                  "vocab_size": 13, "hidden_size": 8, "num_hidden_layers": 2,
                                  "num_attention_heads": 2, "intermediate_size": 12,
                                  "max_position_embeddings": 24, "hidden_dropout_prob": .1,
                                  "attention_probs_dropout_prob": .1}}
        else:
            value["model"] = {"modality": "audio", "name": "offline-tiny-whisper", "pretrained": False,
                              "random_config": {"num_mel_bins": 8, "d_model": 8, "encoder_layers": 2,
                                  "encoder_attention_heads": 2, "encoder_ffn_dim": 12,
                                  "decoder_layers": 1, "decoder_attention_heads": 2, "decoder_ffn_dim": 12,
                                  "max_source_positions": 32, "vocab_size": 13, "pad_token_id": 0,
                                  "bos_token_id": 1, "eos_token_id": 2, "decoder_start_token_id": 1,
                                  "dropout": .1, "attention_dropout": .1}}
        return value

    yield root, config
    torch.set_num_threads(original_threads)


def test_all_text_methods_source_cache_test_gate_and_final_reports(prepared, tmp_path, monkeypatch):
    _, configure = prepared
    runs = tmp_path / "runs"
    from ccspeech.training import engine
    original_load = engine.load_split
    training = True

    def guarded(config, split, **kwargs):
        assert not (training and split == "test"), "Training attempted to open official test records"
        if split != "test":
            assert not kwargs.get("allow_test_labels", False)
        return original_load(config, split, **kwargs)

    monkeypatch.setattr(engine, "load_split", guarded)
    caches = []
    for method in ("sequential", "er", "derpp", "sparse_replay", "mbpa", "mbpa_random", "joint", "separate"):
        training = True
        config = configure(method=method)
        output = run_experiment(config, output_root=runs)
        result = _json(output / "result.json")
        assert result["smoke"] is True
        assert set(result["validation"]["final"]["domains"]) == {"amazon", "yelp"}
        assert not (output / "test.json").exists()
        if method not in {"joint", "separate"}:
            caches.append(result["source_cache"])
            checkpoint = read_checkpoint(output / "target/best.pt")
            assert checkpoint["training_stats"]["current_batches"] == 3
            assert len(checkpoint["memory_state"]["entries"]) == (0 if method == "sequential" else 2)
            assert checkpoint["memory_state"]["seen"] == (6 if method == "sequential" else 12)
            if method == "er":
                assert checkpoint["training_stats"]["label_replay_presentations"] == 6
            elif method == "derpp":
                assert checkpoint["training_stats"]["distill_presentations"] == 6
                assert checkpoint["training_stats"]["label_replay_presentations"] == 6
            elif method in {"mbpa", "mbpa_random"}:
                assert checkpoint["training_stats"]["sparse_replay_updates"] == 3
                assert result["validation"]["final"]["domains"]["amazon"]["cost"]["adapted_queries"] == 5
        if method == "sequential":
            with pytest.raises(ValueError, match="Smoke"):
                final_test(output)
        training = False
        test_path = final_test(output, allow_smoke=True)
        tested = _json(test_path)
        assert set(tested["final"]["domains"]) == {"amazon", "yelp"}
        assert all(report["metrics"]["n"] == 5 for report in tested["final"]["domains"].values())
        assert "source" in tested if method not in {"joint", "separate"} else "source" not in tested
        original_report = test_path.read_bytes()
        with pytest.raises(FileExistsError, match="immutable"):
            final_test(output, allow_smoke=True)
        assert test_path.read_bytes() == original_report
    assert len(set(caches)) == 1
    assert len(list((runs / "_source").glob("*/complete.json"))) == 1


@pytest.mark.parametrize("method", ["sequential", "derpp", "mbpa"])
def test_whisper_pipeline_keeps_test_labels_locked_and_scores_participants(prepared, tmp_path, monkeypatch, method):
    root, configure = prepared
    config = configure("speech", method)
    original_open = Path.open
    blocked = {root / "locked/taukadial/test_labels.csv", root / "processed/speech_v2/test.csv"}

    def guarded(path, *args, **kwargs):
        if path in blocked:
            raise AssertionError("Training accessed a locked test file")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guarded)
        output = run_experiment(config, output_root=tmp_path / "runs")
    test_path = final_test(output, allow_smoke=True)
    report = _json(test_path)
    assert "locked/taukadial/test_labels.csv" in report["test_data_fingerprint"]
    assert report["paired_source_change"]["n_units"] == 2
    for domain in config["domains"]:
        metrics = report["final"]["domains"][domain]["metrics"]
        assert metrics["n"] == 2 and metrics["evaluation_unit"] == "participant"
        assert metrics["confidence_intervals"]["n_units"] == 2
        assert (output / "test/final" / f"{domain}_participants.csv").is_file()
    checkpoint = read_checkpoint(output / "target/best.pt")
    for entry in checkpoint["memory_state"]["entries"]:
        assert entry["record"]["split"] == "train"
        assert "_audio_file" not in entry["record"]
        assert entry["logits"].shape == (2,)


@pytest.mark.parametrize("policy", ["classifier_only", "encoder_lora_only", "lora_classifier"])
def test_module_updates_and_timescale_budgets_share_source(prepared, tmp_path, policy):
    _, configure = prepared
    config = configure(method="derpp")
    config["updates"] = {"policy": policy, "rank": 2, "alpha": 4, "dropout": 0.,
                         "placement": "late", "num_layers": 1}
    config["timescales"] = {"head_interval": 1, "encoder_interval": 2}
    output = run_experiment(config, output_root=tmp_path / "runs")
    result = _json(output / "result.json")
    counters = result["training_stats"]["optimizer"]
    assert counters["gradient_batches"] == 3
    assert counters["fast_optimizer_steps"] == (0 if policy == "encoder_lora_only" else 3)
    assert counters["slow_optimizer_steps"] == (0 if policy == "classifier_only" else 2)
    checkpoint = read_checkpoint(output / "target/best.pt")
    source = read_checkpoint(tmp_path / "runs" / result["source_cache"] / "best.pt")
    if policy == "classifier_only":
        for name, parameter in source["model_state"].items():
            if name.startswith("encoder."):
                assert torch.equal(parameter, checkpoint["model_state"][name])
    if policy == "encoder_lora_only":
        for name in ("classifier.weight", "classifier.bias"):
            assert torch.equal(source["model_state"][name], checkpoint["model_state"][name])
    # A second method and uniform optimizer reuse the exact same source run.
    alternate = deepcopy(config)
    alternate["method"] = "sequential"
    alternate["timescales"] = {"head_interval": 1, "encoder_interval": 1}
    another = run_experiment(alternate, output_root=tmp_path / "runs")
    assert _json(another / "result.json")["source_cache"] == result["source_cache"]
    final_test(output, allow_smoke=True)


def test_speech_resume_and_final_test_after_data_root_move_are_exact(prepared, tmp_path, monkeypatch):
    root, configure = prepared
    config = configure("speech", "derpp")
    config["updates"] = {"policy": "lora_classifier", "rank": 2, "alpha": 4, "dropout": .1,
                         "placement": "late", "num_layers": 1}
    config["timescales"] = {"head_interval": 1, "encoder_interval": 2}
    uninterrupted = run_experiment(config, output_root=tmp_path / "complete-runs")
    from ccspeech.training import loop
    checkpoint_writer = loop.atomic_checkpoint
    interrupted = False

    def stop_after_saved_batch(path, state):
        nonlocal interrupted
        checkpoint_writer(path, state)
        if path.parent.name == "target" and path.name == "last.pt" and not interrupted:
            interrupted = True
            raise RuntimeError("simulated interruption after durable checkpoint")

    resumed_root = tmp_path / "resume-runs"
    with monkeypatch.context() as patch:
        patch.setattr(loop, "atomic_checkpoint", stop_after_saved_batch)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            run_experiment(config, output_root=resumed_root)
    partial = read_checkpoint(resumed_root / run_name(config) / "target/last.pt")
    assert partial["training_stats"]["current_batches"] == 1
    assert partial["optimizer_state"]["pending_batches"] == 1
    assert not partial["completed"]
    moved = tmp_path / "relocated-data"
    shutil.move(str(root), moved)
    config["data"]["root"] = str(moved)
    resumed = run_experiment(config, output_root=resumed_root, resume=True)
    first = read_checkpoint(uninterrupted / "target/last.pt")
    second = read_checkpoint(resumed / "target/last.pt")
    _assert_tree(first["model_state"], second["model_state"])
    _assert_tree(first["optimizer_state"], second["optimizer_state"])
    _assert_tree(first["memory_state"], second["memory_state"])
    assert first["training_stats"]["current_batches"] == second["training_stats"]["current_batches"] == 3
    report = final_test(resumed, data_root=str(moved), allow_smoke=True)
    assert _json(report)["final"]["domains"]["en"]["metrics"]["n"] == 2
    # A completed run has a stable identity despite relocating its data root.
    assert run_experiment(config, output_root=resumed_root, resume=True) == resumed


def test_speech_integrity_and_participant_leakage_fail_before_training(prepared, tmp_path):
    root, configure = prepared
    config = configure("speech")
    manifest = root / "processed/speech_v2/validation.csv"
    with manifest.open(newline="") as source:
        rows = list(csv.DictReader(source))
    rows[0]["participant_id"] = "en-train-0"
    _write_csv(manifest, rows)
    with pytest.raises(ValueError, match="participant leakage"):
        run_experiment(config, output_root=tmp_path / "leaking-runs")
    audio = root / rows[0]["audio_path"]
    original = audio.read_bytes()
    audio.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="integrity failure"):
        run_experiment(config, output_root=tmp_path / "corrupt-runs")
