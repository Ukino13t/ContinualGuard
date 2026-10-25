import copy
import random
from pathlib import Path

import pytest
import torch

from continualguard.replay_candidate_b import (
    DERPP_ALPHA, DERPP_BETA, MEMORY_CAPACITY, ReplayBuffer, entry_from_row,
    approved_config, normalized_replay_loss, security_risk, validate_replay_candidate,
)
from continualguard.stronger_pilot_no_replay import approved_config as baseline_config


def row(index, stage="AF2", label=0, split="train"):
    return {
        "example_id": f"x-{stage}-{label}-{index}", "label": label, "task_id": stage,
        "split": split, "text_sha256": f"h{index}", "attack_family": stage,
        "benign_group": "B0", "injection_position": "none", "language": "en",
        "domain": "d", "source": {"dataset": "fixture"},
        "project_template_group": f"g{index % 31}",
    }


def test_frozen_loss_and_security_score():
    assert (DERPP_ALPHA, DERPP_BETA, MEMORY_CAPACITY) == (0.5, 0.5, 200)
    assert security_risk(1, .8) == pytest.approx(.2)
    assert security_risk(0, .8) == pytest.approx(.8)
    current, replay, mse = torch.tensor(2.0), torch.tensor(4.0), torch.tensor(6.0)
    assert normalized_replay_loss("reservoir", current, replay).item() == 3
    assert normalized_replay_loss("security_aware", current, replay).item() == 3
    assert normalized_replay_loss("derpp", current, replay, mse).item() == pytest.approx(3.5)


def test_reject_nontrain_current_future():
    with pytest.raises(ValueError):
        validate_replay_candidate(row(1, split="test"))
    with pytest.raises(ValueError):
        validate_replay_candidate(row(1, stage="AF3"), current_stage_index=2)


def test_reservoir_determinism_capacity_and_state_roundtrip():
    entries = [entry_from_row(row(i, stage=("AF2", "AF3", "AF4")[i % 3], label=i % 2)) for i in range(1200)]
    a, b = ReplayBuffer("reservoir", 42), ReplayBuffer("reservoir", 42)
    a.reservoir_update(entries)
    b.reservoir_update(entries)
    assert len(a.entries) == 200
    assert [x["example_id"] for x in a.entries] == [x["example_id"] for x in b.entries]
    restored = ReplayBuffer.from_state_dict(copy.deepcopy(a.state_dict()))
    assert restored.sample(4, 4) == a.sample(4, 4)


def test_reservoir_matches_independent_standard_algorithm():
    candidates = [entry_from_row(row(i)) for i in range(500)]
    observed = ReplayBuffer("reservoir", 17)
    observed.reservoir_update(candidates)
    rng = random.Random(17 * 1009 + 11)
    expected = list(candidates[:200])
    for seen_count, candidate in enumerate(candidates[200:], 201):
        index = rng.randrange(seen_count)
        if index < 200:
            expected[index] = candidate
    assert [x["example_id"] for x in observed.entries] == [x["example_id"] for x in expected]
    assert len({x["example_id"] for x in observed.sample(4, 2)}) == 4


def test_derpp_requires_and_roundtrips_logits():
    b = ReplayBuffer("derpp", 17)
    b.reservoir_update([entry_from_row(row(i), logits=[i / 10, -i / 10]) for i in range(210)])
    restored = ReplayBuffer.from_state_dict(b.state_dict())
    assert all(len(x["historical_logits"]) == 2 for x in restored.entries)


def test_security_quota_balance_stage_balance_and_determinism():
    candidates = []
    stages = ("AF2", "AF3", "AF4", "AF5")
    for sidx, stage in enumerate(stages):
        for label in (0, 1):
            for i in range(80):
                r = row(i + sidx * 1000, stage, label)
                candidates.append(entry_from_row(r, p_attack=.01 * (i % 80), risk=.99 - .01 * (i % 80), selection_seed=42))
    a, b = ReplayBuffer("security_aware", 42), ReplayBuffer("security_aware", 42)
    qa = a.security_update(candidates, stages)
    qb = b.security_update(candidates, stages)
    assert len(a.entries) == 200
    assert sum(x["label"] == 0 for x in a.entries) == 100
    assert sum(x["label"] == 1 for x in a.entries) == 100
    assert [x["example_id"] for x in a.entries] == [x["example_id"] for x in b.entries]
    assert qa == qb
    assert all(v["selected"] == 25 for v in qa.values())
    assert not any("text" in x for x in a.entries)


def test_common_protocol_matches_frozen_baseline():
    root = Path(r"E:\ContinualGuard")
    replay = approved_config(root, "reservoir", 42)
    baseline = baseline_config(root, 42)
    assert replay["model"] == baseline["model"]
    assert replay["training"] == baseline["training"]
    for key in ("data_path", "stage_order", "device", "precision", "deterministic", "threads"):
        assert replay[key] == baseline[key]
    assert replay["replay"]["memory_capacity_examples"] == 200
    assert replay["replay"]["replay_batch_size"] == 4
