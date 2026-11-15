import numpy as np
import pytest

from ccspeech.evaluation.metrics import (
    aggregate_speech_predictions, bootstrap_intervals, classification_metrics, continual_summary,
    paired_bootstrap_change,
)


def test_classification_metrics_and_missing_class_auc():
    probabilities = np.array([[.9, .1], [.2, .8], [.4, .6], [.3, .7]])
    metrics = classification_metrics([0, 0, 1, 1], probabilities)
    assert metrics["accuracy"] == .75
    assert metrics["balanced_accuracy"] == .75
    assert metrics["auroc"] == .5
    assert metrics["macro_f1"] == pytest.approx((2/3 + .8)/2)
    missing = classification_metrics([1, 1], probabilities[2:])
    assert missing["auroc"] is None
    assert missing["auroc_status"] == "undefined_missing_classes"
    five = classification_metrics([0, 1, 2, 3, 4], np.eye(5))
    assert five["auroc"] == 1 and five["macro_f1"] == 1
    with pytest.raises(ValueError, match="sum"):
        classification_metrics([0], [[.1, .1]])


def test_speech_hierarchical_aggregation_equal_recording_weight():
    records = [dict(participant_id="p", recording_id="a", segment_id=f"a{i}", label=1, language="en") for i in range(9)]
    records += [dict(participant_id="p", recording_id="b", segment_id="b0", label=1, language="en")]
    probabilities = np.array([[1., 0.]] * 9 + [[0., 1.]])
    participants, scores = aggregate_speech_predictions(records, probabilities)
    assert np.allclose(scores, [[.5, .5]])  # Flat window average would be .9/.1.
    assert participants[0]["recordings"] == 2 and participants[0]["segments"] == 10
    assert participants[0]["label"] == 1
    with pytest.raises(ValueError, match="Duplicate segment"):
        aggregate_speech_predictions(records + [records[0]], np.vstack([probabilities, probabilities[0]]))
    records[-1]["label"] = 0
    with pytest.raises(ValueError, match="Inconsistent participant label"):
        aggregate_speech_predictions(records, probabilities)


def test_bootstrap_uses_units_reproducibly_and_handles_undefined_auc():
    labels = [0, 0, 1, 1]
    probabilities = np.array([[.9, .1], [.7, .3], [.2, .8], [.6, .4]])
    a = bootstrap_intervals(labels, probabilities, n_bootstrap=30, seed=9)
    b = bootstrap_intervals(labels, probabilities, n_bootstrap=30, seed=9)
    assert a == b and a["n_units"] == 4
    assert 0 < a["intervals"]["auroc"]["valid_resamples"] < 30
    undefined = bootstrap_intervals([1, 1], probabilities[2:], n_bootstrap=5)
    assert undefined["intervals"]["auroc"] == {"lower": None, "upper": None, "valid_resamples": 0}


def test_continual_summary_excludes_unseen_scores_from_forgetting():
    scores = [[.8, .99], [.6, .7]]
    result = continual_summary(scores, ["amazon", "yelp"])
    assert result["final_domain_averaged_accuracy"] == pytest.approx(.65)
    assert result["forgetting"]["amazon"] == pytest.approx(.2)
    assert result["source_score_change"]["amazon"] == pytest.approx(-.2)
    improved = continual_summary([[.5, .1], [.7, .8]], ["en", "zh"])
    assert improved["mean_forgetting"] == pytest.approx(-.2)


def test_paired_change_identical_predictions_have_zero_interval():
    probabilities = np.array([[.9, .1], [.7, .3], [.2, .8], [.6, .4]])
    result = paired_bootstrap_change([0, 0, 1, 1], probabilities, probabilities, n_bootstrap=20)
    assert result["change"] == result["lower"] == result["upper"] == 0
    assert result["paired"] and result["valid_resamples"] == 20
