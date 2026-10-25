"""Focused checks for the unstratified matched-priority intervention."""
import copy
import random

import pytest

from scripts.revision2_selector import MatchedBuffer, ranked_entry
from scripts.revision3_selector import A0MatchedBuffer


def candidate(stage, label, index, seed=42):
    row = {"example_id": f"{stage}-{label}-{index}", "task_id": stage,
           "label": label, "split": "train", "text_sha256": f"hash-{stage}-{label}-{index}",
           "attack_family": stage, "benign_group": "B0", "injection_position": "head",
           "source": {"dataset": "synthetic"}, "project_template_group": f"g{index}"}
    return ranked_entry(row, seed)


@pytest.mark.parametrize("capacity", [50, 200, 400])
def test_only_quota_intervention_and_capacity(capacity):
    a0 = A0MatchedBuffer("a0_matched", 42, capacity)
    a1 = MatchedBuffer("a1_matched", 42, capacity)
    for stage_index, stage in enumerate(("AF2", "AF3", "AF4", "AF5"), 1):
        current = [candidate(stage, label, i) for label in (0, 1) for i in range(200)]
        before_rng = random.getstate()
        a0.update(current, ("AF2", "AF3", "AF4", "AF5")[:stage_index])
        a1.update(current, ("AF2", "AF3", "AF4", "AF5")[:stage_index])
        assert random.getstate() == before_rng
        assert len(a0.entries) == len(a1.entries) == capacity
        assert sum(x["label"] == 0 for x in a1.entries) == capacity // 2
        assert a0.sample_rng.getstate() == a1.sample_rng.getstate()
        assert all(x["random_rank_sha256"] == candidate(x["task_id"], x["label"],
                       int(x["example_id"].rsplit("-", 1)[1]))["random_rank_sha256"]
                   for x in [*a0.entries, *a1.entries])
        assert [x["example_id"] for x in A0MatchedBuffer.from_state_dict(
            copy.deepcopy(a0.state_dict())).entries] == [x["example_id"] for x in a0.entries]


def test_unstratified_top_priority_and_no_discarded_resurrection():
    a0 = A0MatchedBuffer("a0_matched", 17, 50)
    first = [candidate("AF2", label, i, 17) for label in (0, 1) for i in range(200)]
    a0.update(first, ("AF2",))
    assert [x["example_id"] for x in a0.entries] == [x["example_id"] for x in
        sorted(first, key=lambda x: (x["random_rank_sha256"], x["example_id"]))[:50]]
    discarded = {x["example_id"] for x in first} - {x["example_id"] for x in a0.entries}
    second = [candidate("AF3", label, i, 17) for label in (0, 1) for i in range(200)]
    a0.update(second, ("AF2", "AF3"))
    assert not ({x["example_id"] for x in a0.entries} & discarded)
    with pytest.raises(ValueError):
        a0.update([candidate("AF5", 0, 0, 17)], ("AF2", "AF3", "AF4"))
    with pytest.raises(ValueError):
        a0.sample(4, 1)
