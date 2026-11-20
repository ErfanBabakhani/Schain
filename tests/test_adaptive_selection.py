import numpy as np
import pytest

from schain.expansion.adaptive import (
    ACTIONS, LinUCB, choose_action, context_vector, normalize_reward,
    similarity_features,
)


def test_context_is_exactly_eight_bounded_ordered_features():
    value = context_vector(
        supplier_relevance=.7, irrelevant_content=.2, target_match=True,
        official_source=False, search_rank=4, document_novelty=.8,
        entity_novelty=.5,
    )
    assert value.tolist() == [1, .7, .2, 1, 0, .25, .8, .5]
    with pytest.raises(ValueError):
        context_vector(supplier_relevance=1.1, irrelevant_content=0,
                       target_match=False, official_source=False, search_rank=1,
                       document_novelty=1, entity_novelty=0)


def test_linucb_updates_only_selected_row_and_round_trips():
    model = LinUCB()
    x = np.array([1., .5, 0, 1, 0, .5, 1, .2])
    model.update("snippet", x, 0)
    assert model.A.shape == (3, 8, 8)
    assert model.b.shape == model.W.shape == (3, 8)
    assert model.counts == {"snippet": 1, "full_text": 0, "both": 0}
    assert np.array_equal(model.A[1], np.eye(8))
    restored = LinUCB.from_dict(model.to_dict())
    assert np.allclose(restored.A, model.A)
    assert np.allclose(restored.W, model.W)


def test_greedy_selection_respects_cost_and_warmup_probabilities():
    model = LinUCB(alpha=0)
    x = np.array([1., 0, 0, 0, 0, 0, 0, 0])
    model.update("snippet", x, 1)
    action, details = choose_action(
        model, x, estimated_costs={"snippet": 100, "full_text": 1000},
        virtual_price=0, warmup=False,
    )
    assert action == "snippet"
    action, details = choose_action(
        LinUCB(alpha=0), x,
        estimated_costs={"snippet": 100, "full_text": 1000, "both": 1000},
        virtual_price=0, warmup=True,
    )
    assert details["probabilities"] == pytest.approx(
        {"snippet": .6, "full_text": .05, "both": .05, "skip": .3}
    )
    assert action in (*ACTIONS, "skip")


def test_similarity_features_clip_cosine_values():
    candidates = np.array([[1., 0], [0., 1.]])
    positive = np.array([[1., 0]])
    negative = np.array([[0., 1.]])
    pos, neg = similarity_features(candidates, positive, negative)
    assert pos.tolist() == [1, 0]
    assert neg.tolist() == [0, 1]


def test_reward_is_clipped_then_normalized_to_unit_interval():
    assert normalize_reward(2.5, 5) == .5
    assert normalize_reward(8, 5) == 1


def test_greedy_returns_skip_when_no_processing_pair_has_positive_score():
    action, details = choose_action(
        LinUCB(alpha=0), np.array([1., 0, 0, 0, 0, 0, 0, 0]),
        estimated_costs={"snippet": 100}, virtual_price=2, warmup=False,
    )
    assert action == "skip"
    assert details["scores"]["skip"] == 0


def test_checkpoint_rejects_feature_or_action_version_mismatch():
    state = LinUCB().to_dict()
    state["feature_version"] = "future"
    with pytest.raises(ValueError, match="incompatible"):
        LinUCB.from_dict(state)
