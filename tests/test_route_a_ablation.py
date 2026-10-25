import copy
import math

import pytest
import torch

from scripts.structured_ablation import AblationBuffer, random_entry
from continualguard.loss_aware_control import generic_difficulty_from_logits
from continualguard.replay_candidate_b import security_risk


def row(stage, label, i, group=None):
    return {"example_id": f"{stage}-{label}-{i}", "task_id": stage, "label": label,
            "split": "train", "text_sha256": f"h-{stage}-{label}-{i}",
            "attack_family": stage, "benign_group": "B0",
            "injection_position": "head" if label else "none",
            "source": {"dataset": "synthetic"},
            "project_template_group": group or f"g{i}"}


def entries(stage, variant, seed=42, n=200, groups=200):
    return [random_entry(row(stage, label, i, f"g{i % groups}"), variant, seed)
            for label in (0, 1) for i in range(n)]


@pytest.mark.parametrize("variant", ("a1_class", "a2_class_stage", "a3_coverage"))
def test_capacity_class_balance_repeatability_no_future(variant):
    a, b = AblationBuffer(variant, 42), AblationBuffer(variant, 42)
    for stage in ("AF2", "AF3", "AF4", "AF5"):
        candidates = entries(stage, variant)
        prefix = tuple(("AF2", "AF3", "AF4", "AF5")[:("AF2", "AF3", "AF4", "AF5").index(stage) + 1])
        a.update(candidates, prefix)
        b.update(candidates, prefix)
        assert len(a.entries) == 200
        assert sum(x["label"] == 0 for x in a.entries) == 100
        assert sum(x["label"] == 1 for x in a.entries) == 100
        assert [x["example_id"] for x in a.entries] == [x["example_id"] for x in b.entries]
        assert all(x["task_id"] in prefix for x in a.entries)
    assert AblationBuffer.from_state_dict(copy.deepcopy(a.state_dict())).state_dict() == a.state_dict()
    assert len(a.sample(4, 5)) == 4
    with pytest.raises(ValueError):
        a.sample(4, 4)


def test_stage_quotas_and_coverage():
    buffers = {v: AblationBuffer(v, 17) for v in ("a1_class", "a2_class_stage", "a3_coverage")}
    for stage, prefix in (("AF2", ("AF2",)), ("AF3", ("AF2", "AF3"))):
        for variant, buffer in buffers.items():
            buffer.update(entries(stage, variant, n=200, groups=120), prefix)
    for variant in ("a2_class_stage", "a3_coverage"):
        buffer = buffers[variant]
        for label in (0, 1):
            for stage in ("AF2", "AF3"):
                assert sum(x["label"] == label and x["task_id"] == stage for x in buffer.entries) == 50
    # Coverage priority should use at least as many distinct groups per cell.
    a2, a3 = buffers["a2_class_stage"], buffers["a3_coverage"]
    for label in (0, 1):
        for stage in ("AF2", "AF3"):
            groups2 = {x["coverage_key"] for x in a2.entries if x["label"] == label and x["task_id"] == stage}
            groups3 = {x["coverage_key"] for x in a3.entries if x["label"] == label and x["task_id"] == stage}
            assert len(groups3) >= len(groups2)


def test_split_and_stage_leakage_rejected():
    buffer = AblationBuffer("a1_class", 42)
    bad = row("AF2", 0, 1)
    bad["split"] = "test"
    with pytest.raises(ValueError):
        random_entry(bad, "a1_class", 42)
    with pytest.raises(ValueError):
        buffer.update(entries("AF3", "a1_class"), ("AF2",))


def test_risk_ce_equivalence_and_fp16_ties():
    logits = torch.tensor([[4., 1.], [1., 4.], [.2, .1], [0., 0.]], dtype=torch.float32)
    for label in (0, 1):
        labels = torch.full((len(logits),), label, dtype=torch.long)
        ce = generic_difficulty_from_logits(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]
        risks = torch.tensor([security_risk(label, float(p)) for p in probs])
        assert torch.equal(torch.argsort(ce), torch.argsort(risks))
        assert torch.allclose(ce, -torch.log1p(-risks), atol=1e-6)
    # FP16 softmax can tie two scores that FP32 CE still separates when both
    # true-label probabilities round to one in the half-precision forward.
    saturated = torch.tensor([[0., 9.], [0., 10.]], dtype=torch.float16)
    half_risk = 1 - torch.softmax(saturated, dim=1)[:, 1]
    full_ce = generic_difficulty_from_logits(saturated, torch.ones(2, dtype=torch.long))
    assert half_risk[0] == half_risk[1]
    assert full_ce[0] > full_ce[1]
