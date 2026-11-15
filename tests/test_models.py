"""Exercise real tiny BERT/Whisper modules and actual prepared data contracts."""
from copy import deepcopy
import csv

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
import torch

from ccspeech.models import LoRALinear, build_model, configure_updates, parameter_groups
from ccspeech.training.data import load_records, make_collator


def text_config(**extra):
    return {"modality": "text", "name": "tiny-test-bert", "pretrained": False,
            "random_config": {"vocab_size": 16, "hidden_size": 16, "num_hidden_layers": 3,
                              "num_attention_heads": 2, "intermediate_size": 24,
                              "max_position_embeddings": 64, "hidden_dropout_prob": 0.0,
                              "attention_probs_dropout_prob": 0.0}, **extra}


def audio_config(**extra):
    return {"modality": "audio", "name": "tiny-test-whisper", "pretrained": False,
            "random_config": {"num_mel_bins": 8, "d_model": 16, "encoder_layers": 3,
                              "encoder_attention_heads": 2, "encoder_ffn_dim": 24,
                              "decoder_layers": 1, "decoder_attention_heads": 2,
                              "decoder_ffn_dim": 24, "max_source_positions": 16,
                              "vocab_size": 16, "pad_token_id": 0, "bos_token_id": 1,
                              "eos_token_id": 2, "decoder_start_token_id": 1,
                              "dropout": 0.0, "attention_dropout": 0.0}, **extra}


def inputs(modality):
    if modality == "text":
        return {"input_ids": torch.tensor([[2, 4, 3, 0], [2, 5, 6, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])}
    return {"input_features": torch.randn(2, 8, 32),
            "attention_mask": torch.tensor([[1] * 20 + [0] * 12, [1] * 32])}


@pytest.mark.parametrize("modality", ["text", "audio"])
@pytest.mark.parametrize("policy", ["full", "classifier_only", "encoder_lora_only", "lora_classifier"])
def test_real_models_forward_and_exact_trainable_updates(modality, policy):
    torch.manual_seed(10)
    model = build_model(text_config() if modality == "text" else audio_config(), 5 if modality == "text" else 2)
    config = {"policy": policy, "rank": 2, "alpha": 4, "dropout": 0.0, "placement": "late", "num_layers": 1}
    stats = configure_updates(model, config)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    model.train()
    batch = inputs(modality)
    logits = model(batch)
    assert logits.shape == (2, model.num_classes)
    assert torch.isfinite(logits).all()
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.3)
    optimizer.step()
    changed = {name for name, value in model.named_parameters() if not torch.equal(before[name], value)}
    assert changed
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert torch.equal(before[name], parameter), name
            assert parameter.grad is None
    if policy == "classifier_only":
        assert all(name.startswith("classifier.") for name in changed)
        assert not model.encoder.training
    elif policy == "encoder_lora_only":
        assert all("lora_" in name for name in changed)
        assert stats["head_trainable_parameters"] == 0
    elif policy == "lora_classifier":
        assert any("lora_B" in name for name in changed)
        assert any(name.startswith("classifier.") for name in changed)
    groups = parameter_groups(model)
    assert not ({id(p) for p in groups["head"]} & {id(p) for p in groups["encoder"]})
    assert stats["trainable_parameters"] == sum(p.numel() for g in groups.values() for p in g)


@pytest.mark.parametrize("modality", ["text", "audio"])
def test_lora_initial_function_and_equal_early_late_budget(modality):
    base = build_model(text_config() if modality == "text" else audio_config(), 2).eval()
    early, late = deepcopy(base), deepcopy(base)
    first = configure_updates(early, {"policy": "lora_classifier", "placement": "early", "num_layers": 1,
                                      "rank": 2, "alpha": 4, "dropout": 0.0})
    last = configure_updates(late, {"policy": "lora_classifier", "placement": "late", "num_layers": 1,
                                    "rank": 2, "alpha": 4, "dropout": 0.0})
    assert first["trainable_parameters"] == last["trainable_parameters"]
    assert first["selected_layers"] == [0] and last["selected_layers"] == [2]
    batch = inputs(modality)
    with torch.no_grad():
        assert torch.allclose(base(batch), early(batch), atol=1e-6)
        assert torch.allclose(base(batch), late(batch), atol=1e-6)
    assert any(isinstance(module, LoRALinear) for module in late.modules())


def test_lora_idempotent_reapplication_and_invalid_layout():
    model = build_model(text_config(), 2)
    config = {"policy": "encoder_lora_only", "placement": "late", "num_layers": 1, "rank": 2}
    first = configure_updates(model, config)
    second = configure_updates(model, config)
    assert first == second
    with pytest.raises(ValueError, match="different layer"):
        configure_updates(model, {**config, "placement": "early"})


@pytest.mark.parametrize("modality", ["text", "audio"])
def test_frozen_input_gradient_checkpointing_retains_lora_gradients(modality):
    config = text_config(gradient_checkpointing=True) if modality == "text" else audio_config(gradient_checkpointing=True)
    model = build_model(config, 2)
    configure_updates(model, {"policy": "encoder_lora_only", "rank": 2})
    model.train()
    model(inputs(modality)).square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for name, p in model.named_parameters() if "lora_B" in name)


@pytest.mark.parametrize("modality", ["text", "audio"])
def test_native_adapter_checkpoint_roundtrip_preserves_predictions(modality):
    config = text_config() if modality == "text" else audio_config()
    model = build_model(config, 2)
    updates = {"policy": "lora_classifier", "rank": 2, "dropout": 0.0, "placement": "early", "num_layers": 1}
    configure_updates(model, updates)
    batch = inputs(modality)
    torch.nn.functional.cross_entropy(model(batch), torch.tensor([0, 1])).backward()
    torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.2).step()
    restored = build_model(config, 2)
    configure_updates(restored, updates)
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    restored.eval()
    with torch.no_grad():
        assert torch.equal(model(batch), restored(batch))


def test_text_collator_and_test_label_gate(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ngood\nbad\nreview\n")
    config = text_config(vocab_file=str(vocab))
    collate = make_collator(config, max_length=8)
    batch = collate([{"text": "good review", "label": 4}, {"text": "bad", "label": 0}])
    assert batch["inputs"]["input_ids"].shape[0] == 2
    assert batch["labels"].tolist() == [4, 0]
    path = tmp_path / "processed/text/amazon/test.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"example_id": "a", "domain": "amazon", "source_split": "test",
                                        "label": 4, "text": "good review"}]), path)
    data = {"root": str(tmp_path), "modality": "text"}
    assert "label" not in load_records(data, "test", "amazon")[0]
    assert load_records(data, "test", "amazon", allow_test_labels=True)[0]["label"] == 4


def _speech_fixture(root, split="test", malicious_path=False):
    audio = root / "processed/speech_v2/audio/example.wav"
    audio.parent.mkdir(parents=True)
    sf.write(audio, np.sin(np.arange(2400) / 20).astype(np.float32), 16000, subtype="PCM_16")
    row = {"segment_id": "segment-1", "participant_id": "participant-1", "recording_id": "recording-1",
           "split": split, "language": "en", "audio_path": "../escape.wav" if malicious_path else str(audio.relative_to(root)),
           "matched_transcript": "hello", "num_samples": "2400", "quality_flags": "source_verified"}
    if split != "test":
        row["label"] = "1"
    with (root / f"processed/speech_v2/{split}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    locked = root / "locked/taukadial/test_labels.csv"
    locked.parent.mkdir(parents=True)
    locked.write_text("recording_id,participant_id,label\nrecording-1,participant-1,1\n")


def test_audio_collator_masks_padding_and_gate_joins_by_identity(tmp_path):
    _speech_fixture(tmp_path)
    data = {"root": str(tmp_path), "modality": "audio"}
    records = load_records(data, "test", "en")
    assert "label" not in records[0]
    assert records[0]["quality_flags"] == "source_verified"
    batch = make_collator(audio_config())(records)
    assert batch["labels"] is None
    assert batch["inputs"]["input_features"].shape == (1, 8, 32)
    assert 0 < batch["inputs"]["attention_mask"].sum() < 32
    prediction = build_model(audio_config(), 2)(batch["inputs"])
    assert prediction.shape == (1, 2)
    assert load_records(data, "test", "en", allow_test_labels=True)[0]["label"] == 1
    (tmp_path / "locked/taukadial/test_labels.csv").write_text(
        "recording_id,participant_id,label\nrecording-1,wrong-participant,1\n")
    with pytest.raises(ValueError, match="identities"):
        load_records(data, "test", "en", allow_test_labels=True)


def test_audio_loader_rejects_path_escape(tmp_path):
    _speech_fixture(tmp_path, malicious_path=True)
    with pytest.raises(ValueError, match="escapes"):
        load_records({"root": str(tmp_path), "modality": "audio"}, "test", "en")


def test_no_implicit_random_model_fallback():
    with pytest.raises(ValueError, match="random_config"):
        build_model({"modality": "text", "pretrained": False}, 2)


@pytest.mark.parametrize("modality", ["text", "audio"])
def test_pretrained_loading_branch_preserves_saved_encoder_weights(tmp_path, modality):
    from transformers import BertConfig, BertModel, WhisperConfig, WhisperModel

    if modality == "text":
        original = BertModel(BertConfig(**text_config()["random_config"]), add_pooling_layer=False)
        expected = original.state_dict()
    else:
        original = WhisperModel(WhisperConfig(**audio_config()["random_config"]))
        expected = original.encoder.state_dict()
    original.save_pretrained(tmp_path)
    loaded = build_model({"modality": modality, "name": str(tmp_path),
                          "pretrained": True, "local_files_only": True}, 2)
    assert set(expected) == set(loaded.encoder.state_dict())
    for key, value in expected.items():
        assert torch.equal(value, loaded.encoder.state_dict()[key]), key
