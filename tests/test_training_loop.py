"""Real tiny-BERT stage updates, memory semantics, and exact interrupted resume."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from ccspeech.methods.memory import ReservoirMemory
from ccspeech.models import build_model, configure_updates
from ccspeech.training.config import base_config
from ccspeech.training.data import make_collator
from ccspeech.training import loop
from ccspeech.training.state import read_checkpoint, seed_everything


@pytest.fixture
def setup(tmp_path):
    torch.set_num_threads(1)
    seed_everything(17)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ngood\nbad\nreview\nnew\nold\n")
    config = base_config("text")
    config["model"] = {
        "modality": "text", "name": "offline-tiny-bert", "pretrained": False,
        "vocab_file": str(vocab),
        "random_config": {"vocab_size": 16, "hidden_size": 12, "num_hidden_layers": 2,
                          "num_attention_heads": 2, "intermediate_size": 20,
                          "max_position_embeddings": 32, "hidden_dropout_prob": 0.,
                          "attention_probs_dropout_prob": 0.},
    }
    config["training"].update(batch_size=2, lr=.005, checkpoint_every=1,
                               source_epochs=2, target_epochs=2, max_length=12,
                               weight_decay=0., max_grad_norm=1.)
    config["memory"].update(capacity=12, replay_batch_size=2, sparse_frequency=2,
                             sparse_batch_size=3, sparse_microbatch_size=2)
    config["runtime"] = {"device": "cpu", "smoke": True}
    config["data"]["root"] = str(tmp_path / "data")
    model = build_model(config["model"], 5)
    configure_updates(model, {"policy": "full"})
    retrieval = deepcopy(model).eval().requires_grad_(False)
    collator = make_collator(config["model"], max_length=12)
    current = [{"example_id": f"new-{i}", "split": "train", "domain": "yelp",
                "text": "new good review" if i % 2 else "new bad review", "label": i % 5}
               for i in range(4)]
    old = [{"example_id": f"old-{i}", "split": "train", "domain": "amazon",
            "text": "old good review" if i % 2 else "old bad review", "label": (i + 1) % 5}
           for i in range(4)]
    return config, model, retrieval, collator, current, old


def seeded_memory(model, retrieval, collator, old, capacity=12):
    memory = ReservoirMemory(capacity, seed=23)
    values = collator(old)
    model.eval()
    with torch.no_grad():
        logits = model(values["inputs"])
        keys = retrieval.encode(values["inputs"])
    for record, logit, key in zip(old, logits, keys):
        memory.add(record, logit, key)
    return memory


def evaluator(collator, records, calls):
    def evaluate(model, memory):
        model.eval()
        values = collator(records)
        with torch.no_grad():
            logits = model(values["inputs"])
            score = -float(F.cross_entropy(logits, values["labels"]))
        calls.append((memory.seen, score))
        return {"selection_score": score, "examples": len(records)}
    return evaluate


@pytest.mark.parametrize("method,labels,distill,sparse", [
    ("er", 8, 0, 0), ("derpp", 8, 8, 0),
    ("sparse_replay", 6, 0, 2), ("mbpa", 6, 0, 2), ("mbpa_random", 6, 0, 2),
])
def test_real_bert_replay_stages_and_exact_presentation_accounting(setup, tmp_path, method, labels, distill, sparse):
    config, model, retrieval, collator, current, old = setup
    config["method"] = method
    memory = seeded_memory(model, retrieval, collator, old)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    frozen = {name: value.clone() for name, value in retrieval.state_dict().items()}
    calls = []
    result = loop.train_stage(model, retrieval, collator, current, memory, config,
                              tmp_path / method, stage="target", epochs=2, method=method,
                              validate_fn=evaluator(collator, current, calls))
    stats = result["training_stats"]
    assert stats["current_batches"] == 4
    assert stats["current_presentations"] == 8
    assert stats["new_examples"] == stats["retrieval_presentations"] == 4
    assert stats["label_replay_presentations"] == labels
    assert stats["distill_presentations"] == distill
    assert stats["replay_presentations"] == labels + distill
    assert stats["sparse_replay_updates"] == sparse
    assert stats["optimizer"]["gradient_batches"] == 4 + sparse
    assert memory.seen == 8 and len(memory) == 8
    assert len(calls) == 2  # No duplicate final evaluation after the last epoch.
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert all(torch.equal(frozen[name], value) for name, value in retrieval.state_dict().items())
    checkpoint = read_checkpoint(Path(result["last_checkpoint"]))
    assert checkpoint["completed"] and checkpoint["kind"] == "training_stage"
    assert checkpoint["epoch"] == 2 and checkpoint["batch_cursor"] == 0
    assert result["live_state"] == "last"
    assert Path(result["best_checkpoint"]).is_file()
    selection = read_checkpoint(Path(result["best_checkpoint"]))
    assert selection["checkpoint_role"] == "selection" and "optimizer_state" not in selection
    assert checkpoint["checkpoint_role"] == "resume" and "optimizer_state" in checkpoint
    assert stats["fixed_retrieval_parameter_bytes"] == sum(p.numel() * p.element_size() for p in retrieval.parameters())
    assert stats["cuda_peak_allocated_bytes"] is None
    assert stats["estimated_checkpoint_tensor_bytes"]["last"] > stats["estimated_checkpoint_tensor_bytes"]["best"]
    assert stats["checkpoint_disk_bytes"]["last"] > 0


def test_source_memory_stores_prestep_predictions_once_across_epochs(setup, tmp_path):
    config, model, retrieval, collator, current, _ = setup
    memory = ReservoirMemory(12)
    order = loop._permutation(len(current), config["seed"], "source", 0)
    first = [current[index] for index in order[:2]]
    model.train()
    with torch.no_grad():
        first_logits = model(collator(first)["inputs"]).clone()
    result = loop.train_stage(model, retrieval, collator, current, memory, config,
                              tmp_path / "source", stage="source", epochs=2, method="sequential",
                              validate_fn=evaluator(collator, current, []))
    entries = {entry.example_id: entry for entry in memory.entries}
    for record, logits in zip(first, first_logits):
        assert torch.equal(entries[record["example_id"]].logits, logits)
    assert memory.seen == len(current)
    assert result["training_stats"]["replay_presentations"] == 0
    assert result["training_stats"]["retrieval_presentations"] == len(current)
    with torch.no_grad():
        assert not torch.equal(model(collator(first)["inputs"]), first_logits)


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert_tree_equal(first, second)
    else:
        assert left == right


def test_sparse_replay_control_has_identical_global_updates_to_mbpa(setup, tmp_path):
    config, base, retrieval, collator, current, old = setup
    memory_state = seeded_memory(base, retrieval, collator, old).state_dict()
    results = []
    for method in ("sparse_replay", "mbpa"):
        model = deepcopy(base)
        memory = ReservoirMemory(12)
        memory.load_state_dict(memory_state)
        settings = deepcopy(config)
        settings["method"] = method
        seed_everything(67)
        result = loop.train_stage(model, retrieval, collator, current, memory, settings,
                                  tmp_path / method, stage="target", epochs=2, method=method,
                                  validate_fn=lambda *_: {"selection_score": 0.})
        results.append((model.state_dict(), memory.state_dict(),
                        read_checkpoint(Path(result["last_checkpoint"]))["optimizer_state"]))
    assert_tree_equal(results[0], results[1])


def test_interrupted_resume_matches_uninterrupted_with_pending_timescale_gradients(setup, tmp_path, monkeypatch):
    config, base, retrieval, collator, current, old = setup
    config["model"]["random_config"].update(hidden_dropout_prob=.15, attention_probs_dropout_prob=.1)
    base = build_model(config["model"], 5)
    configure_updates(base, {"policy": "full"})
    retrieval = deepcopy(base).eval().requires_grad_(False)
    config["timescales"] = {"head_interval": 2, "encoder_interval": 3}
    memory_state = seeded_memory(base, retrieval, collator, old).state_dict()

    def fresh():
        memory = ReservoirMemory(12)
        memory.load_state_dict(memory_state)
        return deepcopy(base), memory

    uninterrupted, full_memory = fresh()
    seed_everything(81)
    full = loop.train_stage(uninterrupted, retrieval, collator, current, full_memory,
                            config, tmp_path / "full", stage="target", epochs=2, method="derpp",
                            validate_fn=evaluator(collator, current, []))
    interrupted, partial_memory = fresh()
    save = loop.atomic_checkpoint

    def crash_after_first_transaction(path, checkpoint):
        save(path, checkpoint)
        if path.name == "last.pt" and checkpoint["training_stats"]["current_batches"] == 1:
            raise RuntimeError("simulated process interruption")

    monkeypatch.setattr(loop, "atomic_checkpoint", crash_after_first_transaction)
    seed_everything(81)
    with pytest.raises(RuntimeError, match="simulated"):
        loop.train_stage(interrupted, retrieval, collator, current, partial_memory,
                         config, tmp_path / "resumed", stage="target", epochs=2, method="derpp",
                         validate_fn=evaluator(collator, current, []))
    partial = read_checkpoint(tmp_path / "resumed/last.pt")
    assert partial["optimizer_state"]["pending_batches"] == 1
    assert partial["optimizer_state"]["pending_fast_batches"] == 1
    assert partial["epoch"] == 0 and partial["batch_cursor"] == 1
    monkeypatch.setattr(loop, "atomic_checkpoint", save)
    restored, resumed_memory = fresh()
    relocated = deepcopy(config)
    relocated["data"]["root"] = str(tmp_path / "relocated-data")
    relocated["model"]["cache_dir"] = str(tmp_path / "other-cache")
    seed_everything(999)  # Checkpoint RNG, not this fresh process seed, must win.
    resumed = loop.train_stage(restored, retrieval, collator, current, resumed_memory,
                               relocated, tmp_path / "resumed", stage="target", epochs=2,
                               method="derpp", validate_fn=evaluator(collator, current, []), resume=True)
    assert_tree_equal(uninterrupted.state_dict(), restored.state_dict())
    assert_tree_equal(full_memory.state_dict(), resumed_memory.state_dict())
    full_checkpoint = read_checkpoint(Path(full["last_checkpoint"]))
    resumed_checkpoint = read_checkpoint(Path(resumed["last_checkpoint"]))
    assert_tree_equal(full_checkpoint["optimizer_state"], resumed_checkpoint["optimizer_state"])
    assert torch.equal(full_checkpoint["rng_state"]["torch"], resumed_checkpoint["rng_state"]["torch"])
    assert resumed["best_score"] == full["best_score"]
    assert resumed["training_stats"]["optimizer"]["pending_slow_batches"] == 0
    assert resumed["training_stats"]["optimizer"]["pending_fast_batches"] == 0
    # Resume on a completed stage must neither update nor re-evaluate.
    resumed_again = loop.train_stage(restored, retrieval, collator, current, resumed_memory,
                                     relocated, tmp_path / "resumed", stage="target", epochs=2,
                                     method="derpp", validate_fn=lambda *_: pytest.fail("duplicate validation"),
                                     resume=True)
    assert resumed_again["training_stats"]["current_batches"] == 4


def test_resume_rejects_scientific_changes_and_reordered_data(setup, tmp_path):
    config, model, retrieval, collator, current, _ = setup
    output = tmp_path / "source"
    loop.train_stage(model, retrieval, collator, current, ReservoirMemory(12), config,
                     output, stage="source", epochs=1, method="sequential",
                     validate_fn=evaluator(collator, current, []))
    changed = deepcopy(config)
    changed["training"]["lr"] *= 2
    for candidate, records in ((changed, current), (config, list(reversed(current)))):
        with pytest.raises(ValueError, match="scientific configuration"):
            loop.train_stage(model, retrieval, collator, records, ReservoirMemory(12), candidate,
                             output, stage="source", epochs=1, method="sequential",
                             validate_fn=evaluator(collator, current, []), resume=True)


def test_smoke_limit_flushes_before_selection_and_production_rejects_limit(setup, tmp_path):
    config, model, retrieval, collator, current, _ = setup
    config["runtime"]["max_steps_per_stage"] = 1
    config["timescales"] = {"head_interval": 3, "encoder_interval": 3}
    calls = []
    result = loop.train_stage(model, retrieval, collator, current, ReservoirMemory(12), config,
                              tmp_path / "smoke", stage="source", epochs=2, method="sequential",
                              validate_fn=evaluator(collator, current, calls))
    assert result["stopped_early"]
    assert result["training_stats"]["current_batches"] == 1
    stats = result["training_stats"]["optimizer"]
    assert stats["fast_optimizer_steps"] == stats["slow_optimizer_steps"] == 1
    assert stats["pending_slow_batches"] == stats["pending_fast_batches"] == 0
    assert len(calls) == 1
    best = read_checkpoint(Path(result["best_checkpoint"]))
    assert_tree_equal(best["model_state"], model.state_dict())
    config["runtime"]["smoke"] = False
    with pytest.raises(ValueError, match="smoke=true"):
        loop.train_stage(model, retrieval, collator, current, ReservoirMemory(12), config,
                         tmp_path / "invalid-production", stage="source", epochs=1,
                         method="sequential", validate_fn=evaluator(collator, current, []))


def test_nonfinite_loss_is_rejected_without_checkpoint(setup, tmp_path):
    config, model, retrieval, collator, current, _ = setup
    with torch.no_grad():
        model.classifier.weight.fill_(float("nan"))
    output = tmp_path / "nan"
    with pytest.raises(FloatingPointError, match="Nonfinite training loss"):
        loop.train_stage(model, retrieval, collator, current, ReservoirMemory(12), config,
                         output, stage="source", epochs=1, method="sequential",
                         validate_fn=evaluator(collator, current, []))
    assert not (output / "last.pt").exists()
