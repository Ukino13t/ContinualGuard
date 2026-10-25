import copy
from pathlib import Path

import pytest
import torch

from continualguard.loss_aware_control import (
    LossAwareBuffer, approved_config, generic_difficulty_from_logits, loss_entry,
)
from continualguard.replay_candidate_b import ReplayBuffer, approved_config as replay_config, normalized_replay_loss


def row(index, label=0, stage="AF2", group=None, split="train"):
    return {
        "example_id": f"loss-{stage}-{label}-{index}", "label": label, "task_id": stage,
        "split": split, "text_sha256": f"h-{stage}-{label}-{index}", "attack_family": stage,
        "benign_group": "B0", "injection_position": "none" if label == 0 else "head",
        "language": "en", "domain": "fixture", "source": {"dataset": "fixture"},
        "project_template_group": group if group is not None else f"g{index}",
    }


def entry(index, label=0, stage="AF2", loss=0.0, probability=0.5, group=None, seed=42):
    return loss_entry(row(index, label, stage, group), generic_loss=loss, p_attack=probability, seed=seed)


def test_generic_difficulty_is_float32_true_label_ce_and_symmetric():
    logits = torch.tensor([[3.0, 1.0], [3.0, 1.0]], dtype=torch.float16)
    observed = generic_difficulty_from_logits(logits, torch.tensor([0, 1]))
    expected = torch.nn.functional.cross_entropy(logits.float(), torch.tensor([0, 1]), reduction="none")
    assert observed.dtype == torch.float32
    assert torch.equal(observed, expected)
    assert observed[1] > observed[0]
    a = loss_entry(row(1, 0), generic_loss=1.25, p_attack=.8, seed=42)
    b = loss_entry(row(2, 1), generic_loss=1.25, p_attack=.8, seed=42)
    assert a["pre_update_generic_loss"] == b["pre_update_generic_loss"] == 1.25
    assert a["pre_update_security_risk"] != b["pre_update_security_risk"]


def test_true_label_ce_and_security_risk_are_rank_equivalent_in_exact_arithmetic():
    logits=torch.tensor([[2.0,-1.0],[.3,.2],[-.7,1.4],[1.1,2.2]],dtype=torch.float32)
    for label in (0,1):
        labels=torch.full((len(logits),),label,dtype=torch.long)
        losses=generic_difficulty_from_logits(logits,labels)
        p_attack=torch.softmax(logits,dim=-1)[:,1]
        risks=p_attack if label==0 else 1-p_attack
        assert torch.equal(torch.argsort(losses),torch.argsort(risks))
        assert torch.allclose(losses,-torch.log1p(-risks),atol=1e-6)


def test_loss_normalization_matches_security_aware_exactly():
    current, replay = torch.tensor(2.0), torch.tensor(4.0)
    assert normalized_replay_loss("loss_aware", current, replay).item() == 3.0
    assert normalized_replay_loss("loss_aware", current, replay).item() == normalized_replay_loss("security_aware", current, replay).item()


def test_quota_capacity_coverage_tie_determinism_and_roundtrip():
    candidates = []
    for sidx, stage in enumerate(("AF2", "AF3", "AF4", "AF5")):
        for label in (0, 1):
            for i in range(80):
                candidates.append(entry(i + sidx * 1000, label, stage, loss=i / 80, probability=.2, group=f"g{i % 40}"))
    a, b = LossAwareBuffer(42), LossAwareBuffer(42)
    qa = a.update(candidates, ("AF2", "AF3", "AF4", "AF5"))
    qb = b.update(candidates, ("AF2", "AF3", "AF4", "AF5"))
    assert len(a.entries) == 200
    assert sum(x["label"] == 0 for x in a.entries) == 100
    assert sum(x["label"] == 1 for x in a.entries) == 100
    assert all(v["selected"] == 25 for v in qa.values())
    assert qa == qb
    assert [x["example_id"] for x in a.entries] == [x["example_id"] for x in b.entries]
    assert not any("text" in x for x in a.entries)
    restored = LossAwareBuffer.from_state_dict(copy.deepcopy(a.state_dict()))
    assert restored.state_dict() == a.state_dict()
    assert restored.sample(4, 5) == a.sample(4, 5)


def test_same_pool_security_and_loss_selection_differs_only_by_score():
    # Synthetic independently assigned scores isolate the buffer routine. Actual
    # model-derived CE and risk rankings are monotone-equivalent absent numeric ties.
    loss_candidates, security_candidates = [], []
    for label in (0, 1):
        for i in range(120):
            probability = i / 121 if label == 0 else 1 - i / 121
            item = entry(i, label, "AF2", loss=float(119 - i), probability=probability, group=f"unique-{i}")
            loss_candidates.append(item)
            security_candidates.append(copy.deepcopy(item))
    loss_buffer = LossAwareBuffer(42)
    security_buffer = ReplayBuffer("security_aware", 42)
    loss_quota = loss_buffer.update(loss_candidates, ("AF2",))
    security_quota = security_buffer.security_update(security_candidates, ("AF2",))
    assert loss_quota == security_quota
    assert {x["example_id"] for x in loss_buffer.entries} != {x["example_id"] for x in security_buffer.entries}
    # The candidate metadata, structural procedure, and tie hashes are identical.
    for left, right in zip(loss_candidates, security_candidates):
        assert left == right
    assert loss_buffer.sample_rng.getstate() == security_buffer.sample_rng.getstate()


def test_reject_nonprefix_and_current_or_future_replay():
    buffer = LossAwareBuffer(17)
    with pytest.raises(ValueError):
        buffer.update([entry(1, stage="AF3")], ("AF3",))
    buffer.update([entry(i, label=i % 2) for i in range(10)], ("AF2",))
    with pytest.raises(ValueError):
        buffer.sample(4, 1)


def test_control_config_matches_frozen_security_factors():
    root = Path(r"E:\ContinualGuard")
    loss = approved_config(root, 42)
    security = replay_config(root, "security_aware", 42)
    assert loss["model"] == security["model"]
    assert loss["training"] == security["training"]
    for key in ("data_path", "stage_order", "device", "precision", "deterministic", "threads"):
        assert loss[key] == security[key]
    assert loss["replay"]["memory_capacity_examples"] == security["replay"]["memory_capacity_examples"] == 200
    assert loss["replay"]["replay_batch_size"] == security["replay"]["replay_batch_size"] == 4
    assert loss["replay"]["tie_break"] == "sha256(seed|security-aware|example_id)"
