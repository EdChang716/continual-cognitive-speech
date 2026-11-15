import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ccspeech.methods.memory import ReservoirMemory
from ccspeech.methods.losses import er_loss, derpp_loss
from ccspeech.methods.local_adaptation import local_adapt_predict
from ccspeech.methods.optim import TimescaleOptimizer


def record(i, split="train"):
    return {"example_id": str(i), "label": i % 2, "split": split, "text": f"example {i}"}


def test_reservoir_bounded_immutable_and_resumable():
    memory = ReservoirMemory(5, seed=17)
    original = torch.tensor([1.0, 2.0], requires_grad=True)
    memory.add(record(0), logits=original, key=original)
    with torch.no_grad():
        original.add_(100)
    assert memory.entries[0].logits.tolist() == [1, 2]
    assert not memory.entries[0].logits.requires_grad
    for i in range(1, 40):
        memory.add(record(i))
    assert len(memory) == 5 and memory.seen == 40
    assert any(int(item.example_id) < 30 for item in memory.entries)
    other = ReservoirMemory(5)
    other.load_state_dict(memory.state_dict())
    assert [v.example_id for v in memory.sample(3)] == [v.example_id for v in other.sample(3)]
    for i in range(40, 70):
        memory.add(record(i)); other.add(record(i))
    assert [v.example_id for v in memory.entries] == [v.example_id for v in other.entries]
    with pytest.raises(ValueError, match="training"):
        memory.add(record(70, "validation"))


def test_reservoir_inclusion_is_not_biased_to_recent_examples():
    counts = [0] * 10
    for seed in range(400):
        memory = ReservoirMemory(3, seed)
        for i in range(10):
            memory.add(record(i))
        for value in memory.entries:
            counts[int(value.example_id)] += 1
    assert all(80 < count < 160 for count in counts)


def test_evaluation_sampling_does_not_change_admission_or_training_replay():
    first, second = ReservoirMemory(5, 2), ReservoirMemory(5, 2)
    for i in range(30):
        first.add(record(i)); second.add(record(i))
        first.sample(3, purpose="local")
    assert [entry.example_id for entry in first.entries] == [entry.example_id for entry in second.entries]
    assert [entry.example_id for entry in first.sample(4)] == [entry.example_id for entry in second.sample(4)]


def test_nearest_is_euclidean_and_random_sample_is_without_replacement():
    memory = ReservoirMemory(3)
    for i, key in enumerate(([100.0, 0.0], [1.0, 1.0], [0.0, 4.0])):
        memory.add(record(i), key=torch.tensor(key))
    assert memory.nearest(torch.tensor([1.0, 0.0]), 1)[0].example_id == "1"
    assert len({entry.example_id for entry in memory.sample(100)}) == 3
    assert ReservoirMemory(0).add(record(0)) is False


def test_er_and_derpp_objectives_and_detached_targets():
    current = torch.tensor([[0.1, 0.9], [1.0, 0.0]], requires_grad=True)
    labels = torch.tensor([1, 0])
    replay = torch.tensor([[0.4, 0.6]], requires_grad=True)
    replay_labels = torch.tensor([0])
    predicted = torch.tensor([[2.0, 0.0]], requires_grad=True)
    stored = torch.tensor([[0.0, 1.0]], requires_grad=True)
    assert torch.allclose(er_loss(current, labels, replay, replay_labels),
                          F.cross_entropy(current, labels) + F.cross_entropy(replay, replay_labels))
    loss = derpp_loss(current, labels, predicted, stored, replay, replay_labels, alpha=.2, beta=.3)
    expected = F.cross_entropy(current, labels) + .2 * F.mse_loss(predicted, stored) + .3 * F.cross_entropy(replay, replay_labels)
    assert torch.allclose(loss, expected)
    loss.backward()
    assert stored.grad is None and predicted.grad is not None


class ToyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2, bias=False)
        self.classifier = nn.Linear(2, 2)
        self.register_buffer("calls", torch.zeros((), dtype=torch.long))
        with torch.no_grad():
            self.encoder.weight.copy_(torch.eye(2))
            self.classifier.weight.zero_()
            self.classifier.bias.zero_()
        self.encoder.requires_grad_(False)
        self.fail_after = None

    def forward(self, batch):
        self.calls += 1
        if self.fail_after is not None and self.calls.item() >= self.fail_after:
            raise RuntimeError("deliberate forward failure")
        return self.classifier(self.encoder(batch["x"]))


def test_local_adaptation_changes_prediction_but_restores_all_state():
    model = ToyClassifier().eval()
    model.classifier.train()  # Mixed module mode must be restored exactly.
    for parameter in model.classifier.parameters():
        parameter.grad = torch.ones_like(parameter) * 9
    snapshot = copy.deepcopy(model.state_dict())
    gradients = [p.grad.clone() for p in model.classifier.parameters()]
    with torch.no_grad():
        output = local_adapt_predict(model, {"x": torch.ones(1, 2)}, {"x": torch.ones(5, 2)},
                                     torch.ones(5, dtype=torch.long), steps=3, lr=.1,
                                     microbatch_size=2, optimizer_name="sgd")
    assert output[0, 1] > output[0, 0]
    assert all(torch.equal(value, snapshot[name]) for name, value in model.state_dict().items())
    assert all(torch.equal(p.grad, grad) for p, grad in zip(model.classifier.parameters(), gradients))
    assert not model.training and model.classifier.training


def test_local_exception_restores_parameters_buffers_and_modes():
    model = ToyClassifier().eval()
    model.fail_after = 2
    snapshot = copy.deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match="deliberate"):
        local_adapt_predict(model, {"x": torch.ones(1, 2)}, {"x": torch.ones(4, 2)},
                            torch.ones(4, dtype=torch.long), steps=2, lr=.1, microbatch_size=4)
    assert all(torch.equal(value, snapshot[name]) for name, value in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    assert not model.training


def test_local_microbatch_has_same_mean_objective():
    model = ToyClassifier()
    neighbor = {"x": torch.tensor([[1., 1.], [2., 1.], [0., 1.], [-1., 2.], [3., -1.]])}
    labels = torch.tensor([1, 1, 0, 1, 0])
    a = local_adapt_predict(model, {"x": torch.ones(1, 2)}, neighbor, labels, steps=3,
                           lr=.01, anchor_strength=.1, optimizer_name="sgd", microbatch_size=5)
    b = local_adapt_predict(model, {"x": torch.ones(1, 2)}, neighbor, labels, steps=3,
                           lr=.01, anchor_strength=.1, optimizer_name="sgd", microbatch_size=2)
    assert torch.allclose(a, b, atol=1e-7)
    with pytest.raises(ValueError, match="exactly one"):
        local_adapt_predict(model, {"x": torch.ones(2, 2)}, neighbor, labels)


def test_timescales_average_gradients_flush_and_resume():
    head = nn.Parameter(torch.tensor(1.0))
    encoder = nn.Parameter(torch.tensor(1.0))
    optimizer = TimescaleOptimizer([head], [encoder], lr=.01, slow_interval=2, max_grad_norm=None)
    for gradient in (2., 6.):
        optimizer.zero_grad(); head.grad = torch.tensor(1.); encoder.grad = torch.tensor(gradient)
        optimizer.step()
    assert optimizer.statistics()["fast_optimizer_steps"] == 2
    assert optimizer.statistics()["slow_optimizer_steps"] == 1
    assert torch.allclose(optimizer.slow.state[encoder]["exp_avg"], torch.tensor(.4))
    optimizer.zero_grad(); head.grad = torch.tensor(1.); encoder.grad = torch.tensor(10.)
    optimizer.step()
    restored_head = nn.Parameter(head.detach().clone())
    restored_encoder = nn.Parameter(encoder.detach().clone())
    resumed = TimescaleOptimizer([restored_head], [restored_encoder], lr=.01, slow_interval=2, max_grad_norm=None)
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    optimizer.flush(); resumed.flush()
    assert optimizer.statistics() == resumed.statistics()
    assert torch.equal(encoder, restored_encoder)
    assert optimizer.slow_steps == 2 and optimizer.fast_steps == 3


def test_uniform_slow_control_delays_both_groups():
    head, encoder = nn.Parameter(torch.tensor(1.)), nn.Parameter(torch.tensor(1.))
    optimizer = TimescaleOptimizer([head], [encoder], lr=.01, head_interval=4, slow_interval=4)
    for _ in range(3):
        optimizer.zero_grad(); head.grad = torch.tensor(1.); encoder.grad = torch.tensor(1.)
        optimizer.step()
    assert head.item() == encoder.item() == 1
    assert optimizer.fast_steps == optimizer.slow_steps == 0
    optimizer.flush()
    assert optimizer.fast_steps == optimizer.slow_steps == 1


def test_timescale_checkpoint_resumes_both_pending_groups_with_zero_grad():
    first = [nn.Parameter(torch.tensor(1.)), nn.Parameter(torch.tensor(2.))]
    optimizer = TimescaleOptimizer(first[:1], first[1:], lr=.03,
                                   head_interval=3, slow_interval=4, max_grad_norm=None)

    def batch(parameters, schedule, coefficient):
        schedule.zero_grad()
        loss = (parameters[0] * parameters[1] - coefficient).square()
        loss.backward()
        schedule.step()

    for coefficient in (0., .5):
        batch(first, optimizer, coefficient)
    second = [nn.Parameter(p.detach().clone()) for p in first]
    resumed = TimescaleOptimizer(second[:1], second[1:], lr=.03,
                                 head_interval=3, slow_interval=4, max_grad_norm=None)
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for coefficient in (.25, -1., .75, -.5, .1):
        batch(first, optimizer, coefficient)
        batch(second, resumed, coefficient)
    optimizer.flush()
    resumed.flush()
    assert optimizer.statistics() == resumed.statistics()
    assert optimizer.fast_steps == 3 and optimizer.slow_steps == 2
    for original, restored in zip(first, second):
        assert torch.equal(original, restored)
