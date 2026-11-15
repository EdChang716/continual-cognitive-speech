"""Evaluation must preserve training state and use the intended scoring units."""
from copy import deepcopy
import random

import numpy as np
import pytest
import torch
from torch import nn

from ccspeech.methods.memory import ReservoirMemory
from ccspeech.models import build_model
from ccspeech.training.data import make_collator
from ccspeech.training.prediction import evaluate_domains, predict_records
from ccspeech.training.state import rng_state


def assert_tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, np.ndarray):
        assert np.array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_tree_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_tree_equal(a, b)
    else:
        assert first == second


@pytest.fixture
def tiny_text(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ngood\nbad\nreview\nbook\nfood\n")
    model_config = {
        "modality": "text", "name": "tiny-evaluation-test", "pretrained": False,
        "vocab_file": str(vocab),
        "random_config": {"vocab_size": 10, "hidden_size": 8, "num_hidden_layers": 1,
                          "num_attention_heads": 2, "intermediate_size": 12,
                          "max_position_embeddings": 32, "hidden_dropout_prob": .2,
                          "attention_probs_dropout_prob": .2},
    }
    model = build_model(model_config, 2)
    retrieval = deepcopy(model).requires_grad_(False).eval()
    collator = make_collator(model_config, max_length=12)
    train = [{"example_id": f"train-{i}", "text": text, "label": i % 2, "split": "train"}
             for i, text in enumerate(["bad review", "good review", "bad food", "good book"])]
    memory = ReservoirMemory(4, 18)
    with torch.no_grad():
        keys = retrieval.encode(collator(train)["inputs"])
    for row, key in zip(train, keys):
        memory.add(row, key=key)
    records = [{"example_id": "query-0", "text": "good food", "label": 1, "split": "validation"},
               {"example_id": "query-1", "text": "bad book", "label": 0, "split": "validation"}]
    config = {"model": model_config, "method": "mbpa", "seed": 27,
              "training": {"precision": "float32"}, "evaluation": {"batch_size": 2},
              "local_adaptation": {"neighbors": 3, "steps": 2, "lr": .01,
                                   "anchor_strength": .001, "microbatch_size": 2}}
    yield model, retrieval, collator, memory, records, config
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("method", ["mbpa", "mbpa_random"])
def test_tiny_bert_query_label_blindness_and_complete_state_restoration(tiny_text, method):
    model, retrieval, collator, memory, records, config = tiny_text
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    batch = collator([entry.record for entry in memory.entries])
    torch.nn.functional.cross_entropy(model(batch["inputs"]), batch["labels"]).backward()
    optimizer.step()
    model.encoder.encoder.layer[0].eval()  # Preserve mixed modes, not just root.
    retrieval.train()
    model_state = deepcopy(model.state_dict())
    retrieval_state = deepcopy(retrieval.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    modes = [module.training for module in model.modules()]
    retrieval_modes = [module.training for module in retrieval.modules()]
    memory_state = memory.state_dict()
    initial_rng = rng_state()

    def label_guard(rows):
        for row in rows:
            if row["split"] == "validation":
                assert "label" not in row
        # Exercise restoration of all RNG sources around evaluation.
        random.random()
        np.random.random()
        torch.rand(1)
        return collator(rows)

    first = predict_records(model, retrieval, label_guard, records, memory, config, method=method)
    flipped = [{**row, "label": 1 - row["label"]} for row in records]
    second = predict_records(model, retrieval, label_guard, flipped, memory, config, method=method)
    np.testing.assert_array_equal(first["probabilities"], second["probabilities"])
    assert first["metrics"]["accuracy"] == 1 - second["metrics"]["accuracy"]
    assert first["cost"]["local_optimizer_steps"] == 4
    assert first["cost"]["local_neighbor_forward_examples"] == 12
    assert first["cost"]["neighbors_per_query"]["total"] == 6
    assert first["cost"]["retrieval_encoder_examples"] == (2 if method == "mbpa" else 0)
    assert first["cost"]["local_wall_seconds"] > 0
    assert_tree_equal(model_state, model.state_dict())
    assert_tree_equal(retrieval_state, retrieval.state_dict())
    assert_tree_equal(optimizer_state, optimizer.state_dict())
    assert_tree_equal(memory_state, memory.state_dict())
    assert_tree_equal(initial_rng, rng_state())
    assert_tree_equal(gradients, [p.grad for p in model.parameters()])
    assert modes == [module.training for module in model.modules()]
    assert retrieval_modes == [module.training for module in retrieval.modules()]


def test_failed_query_restores_memory_rng_model_and_all_rngs(tiny_text):
    model, retrieval, collator, memory, records, config = tiny_text
    model.train()
    initial = deepcopy(model.state_dict())
    initial_rng = rng_state()
    memory_state = memory.state_dict()

    def broken(rows):
        random.random()
        np.random.random()
        torch.rand(1)
        if rows[0]["example_id"] == "query-1":
            raise RuntimeError("deliberate query collation failure")
        return collator(rows)

    with pytest.raises(RuntimeError, match="deliberate"):
        predict_records(model, retrieval, broken, records, memory, config, method="mbpa_random")
    assert_tree_equal(initial, model.state_dict())
    assert_tree_equal(initial_rng, rng_state())
    assert_tree_equal(memory_state, memory.state_dict())
    assert model.training


def test_source_override_and_unlabeled_direct_prediction(tiny_text):
    model, _, collator, _, records, config = tiny_text
    unlabeled = [{key: value for key, value in row.items() if key != "label"} for row in records]
    result = predict_records(model, None, collator, unlabeled, None, config, method="sequential")
    assert result["probabilities"].shape == (2, 2)
    assert result["metrics"]["status"] == "labels_unavailable"
    assert result["cost"]["direct_queries"] == 2
    assert result["cost"]["local_queries"] == 0
    with pytest.raises(ValueError, match="scoring labels"):
        evaluate_domains(model, None, collator, {"amazon": unlabeled}, None, config, method="sequential")


class KnownOutputModel(nn.Module):
    def __init__(self, modality="audio"):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.modality = modality
        self.num_classes = 2

    def forward(self, batch):
        return batch["logits"] + self.anchor * 0


def known_collator(rows):
    return {"inputs": {"logits": torch.tensor([row["scores"] for row in rows]).log()},
            "labels": torch.tensor([row["label"] for row in rows]) if all("label" in row for row in rows) else None,
            "records": rows}


def test_audio_uses_equal_recording_participant_means_and_bootstrap():
    records = []
    for i in range(3):
        records.append({"participant_id": "p1", "recording_id": "r1", "segment_id": f"r1-{i}",
                        "scores": [.9, .1], "label": 0, "_audio_file": "/private/example.wav"})
    records.extend([
        {"participant_id": "p1", "recording_id": "r2", "segment_id": "r2-0", "scores": [.1, .9], "label": 0},
        {"participant_id": "p2", "recording_id": "r3", "segment_id": "r3-0", "scores": [.2, .8], "label": 1},
    ])
    model = KnownOutputModel()
    config = {"model": {"modality": "audio"}, "evaluation": {"batch_size": 2, "bootstrap_samples": 20}, "seed": 3}
    report = predict_records(model, None, known_collator, records, None, config, include_intervals=True)
    np.testing.assert_allclose(report["participant_probabilities"], [[.5, .5], [.2, .8]], atol=1e-7)
    assert report["probabilities"].shape == (5, 2)
    assert len(report["participant_records"]) == 2
    assert report["participant_records"][0]["recordings"] == 2
    assert report["metrics"]["n"] == 2
    assert report["metrics"]["evaluation_unit"] == "participant"
    assert report["metrics"]["accuracy"] == 1
    assert report["metrics"]["confidence_intervals"]["n_units"] == 2
    assert report["metrics"]["confidence_intervals"]["n_bootstrap"] == 20
    assert all("_audio_file" not in row for row in report["records"])
    assert "_audio_file" in records[0]


def test_domain_selection_is_equal_domain_mean_and_text_bootstrap_explicit():
    model = KnownOutputModel("text")
    config = {"model": {"modality": "text"}, "evaluation": {"batch_size": 3, "bootstrap_samples": 10}}
    first = [{"scores": [.9, .1], "label": 0} for _ in range(4)]
    second = [{"scores": [.9, .1], "label": 1}]
    result = evaluate_domains(model, None, known_collator, {"amazon": first, "yelp": second}, None,
                              config, include_intervals=True)
    assert result["selection_score"] == .5  # Pooled accuracy would be .8.
    assert "confidence_intervals" not in result["domains"]["amazon"]["metrics"]
    config["evaluation"]["bootstrap_text"] = True
    report = predict_records(model, None, known_collator, first, None, config, include_intervals=True)
    assert report["metrics"]["confidence_intervals"]["n_bootstrap"] == 10
