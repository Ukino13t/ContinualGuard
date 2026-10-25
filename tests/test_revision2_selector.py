import copy
import random

import pytest

from scripts.revision2_selector import MatchedBuffer, RANK_NAMESPACE, digest, ranked_entry


def row(stage, label, i, group=None):
    return {"example_id": f"{stage}-{label}-{i}", "task_id": stage, "label": label,
            "split": "train", "text_sha256": f"hash-{stage}-{label}-{i}",
            "attack_family": stage, "benign_group": "B0", "injection_position": "head",
            "source": {"dataset": "synthetic"}, "project_template_group": group or f"g{i}"}


def candidates(stage, seed=42, n=200, groups=200):
    return [ranked_entry(row(stage, label, i, f"g{i % groups}"), seed)
            for label in (0, 1) for i in range(n)]


def ids(buffer):
    return [x["example_id"] for x in buffer.entries]


def test_shared_rank_and_separate_sampling_rng():
    entry = ranked_entry(row("AF2", 0, 5), 42)
    assert entry["random_rank_sha256"] == digest(42, RANK_NAMESPACE, entry["example_id"])
    a, b = MatchedBuffer("a1_matched", 42), MatchedBuffer("a2_matched", 42)
    a.update(candidates("AF2"), ("AF2",))
    b.update(candidates("AF2"), ("AF2",))
    assert ids(a) == ids(b)
    assert a.sample(4, 2) == b.sample(4, 2)
    random.seed(983)
    before = random.getstate()
    a.update(candidates("AF3"), ("AF2", "AF3"))
    assert random.getstate() == before


@pytest.mark.parametrize("capacity", [50, 200, 400])
def test_quota_capacity_online_pool_and_nonbinding_parity(capacity):
    a2 = MatchedBuffer("a2_matched", 17, capacity)
    a3 = MatchedBuffer("a3_matched", 17, capacity)
    for index, stage in enumerate(("AF2", "AF3", "AF4", "AF5"), 1):
        new = candidates(stage, 17)
        for buffer in (a2, a3):
            result = buffer.update(new, ("AF2", "AF3", "AF4", "AF5")[:index])
            assert len(buffer.entries) == capacity
            assert result["duplicate_coverage_key_count"] == 0
            assert sum(x["label"] == 0 for x in buffer.entries) == capacity // 2
            assert sum(x["label"] == 1 for x in buffer.entries) == capacity // 2
            for label in (0, 1):
                quotas = [sum(x["label"] == label and x["task_id"] == task for x in buffer.entries)
                          for task in ("AF2", "AF3", "AF4", "AF5")[:index]]
                assert max(quotas) - min(quotas) <= 1
        assert ids(a2) == ids(a3)
        assert a2.state_dict()["sample_rng_state"] == a3.state_dict()["sample_rng_state"]
    assert MatchedBuffer.from_state_dict(copy.deepcopy(a2.state_dict())).state_dict() == a2.state_dict()


def test_binding_coverage_changes_selection():
    a2, a3 = MatchedBuffer("a2_matched", 42, 50), MatchedBuffer("a3_matched", 42, 50)
    # Force one repeated key at the top of a sorted bucket and a fresh key just
    # outside quota. The coverage preference must be observable in this fixture.
    new = candidates("AF2", 42, n=30)
    bucket = sorted([x for x in new if x["label"] == 0], key=lambda x: x["random_rank_sha256"])
    bucket[1]["coverage_key"] = bucket[0]["coverage_key"]
    for buffer in (a2, a3):
        buffer.update(new, ("AF2",))
    assert set(ids(a2)) != set(ids(a3))


def test_reject_leakage_resurrection_and_future_replay():
    with pytest.raises(ValueError):
        ranked_entry({**row("AF2", 0, 1), "split": "test"}, 42)
    buffer = MatchedBuffer("a1_matched", 42)
    buffer.update(candidates("AF2"), ("AF2",))
    with pytest.raises(ValueError):
        buffer.sample(4, 1)
    with pytest.raises(ValueError):
        buffer.update([ranked_entry(row("AF5", 0, 1), 42)], ("AF2", "AF3"))
    with pytest.raises(ValueError):
        buffer.update([ranked_entry(row("AF2", 0, 1), 42)], ("AF2", "AF3"))
