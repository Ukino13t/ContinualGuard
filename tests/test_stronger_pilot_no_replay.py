import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from continualguard.stronger_pilot_no_replay import (
    ARMS,
    MATERIALIZATION_MANIFEST_SHA256,
    MATERIALIZATION_VALIDATION_SHA256,
    SEEDS,
    STAGES,
    CandidateBData,
    artifact_integrity_preflight,
    metric_block,
    validate_config,
)

ROOT = Path(r"E:\ContinualGuard")


def test_frozen_execution_config():
    config = yaml.safe_load((ROOT / "configs/stronger_pilot_no_replay.yaml").read_text(encoding="utf-8"))
    for seed in SEEDS:
        configured = dict(config)
        configured["seed"] = seed
        validate_config(configured)
    assert STAGES == ("AF2", "AF3", "AF4", "AF5")
    assert ARMS == {"doc": "B-DOC", "short": "B-SHORT", "hard": "B-HARD"}


def test_integrity_preflight_and_frozen_hashes():
    result = artifact_integrity_preflight(ROOT)
    assert result["status"] == "PASS"
    assert result["manifest_sha256"] == MATERIALIZATION_MANIFEST_SHA256
    assert result["validation_sha256"] == MATERIALIZATION_VALIDATION_SHA256
    assert result["arm_memberships"] == 6960
    assert result["unique_selected_records"] == 3770
    assert all(result["checks"].values())


@pytest.mark.parametrize("arm", tuple(ARMS))
def test_dataset_contract_and_no_cross_stage_identity(arm):
    data = CandidateBData(ROOT, arm)
    seen = set()
    for stage in STAGES:
        for split, expected in (("train", (400, 200, 200)), ("val", (60, 30, 30)), ("test", (120, 60, 60))):
            rows = data.records(stage, split)
            assert len(rows) == expected[0]
            assert Counter(row["label"] for row in rows) == Counter({0: expected[1], 1: expected[2]})
            ids = {row["example_id"] for row in rows}
            assert not (seen & ids)
            seen |= ids
    assert len(seen) == 2320
    assert len(data.notinject) == 339
    assert len(data.bshort) == 120


def test_matched_malicious_and_benign_anchors():
    data = {arm: CandidateBData(ROOT, arm) for arm in ARMS}
    for stage in STAGES:
        for split in ("train", "val", "test"):
            values = [[row["example_id"] for row in data[arm].records(stage, split) if row["label"] == 1] for arm in ARMS]
            assert values[0] == values[1] == values[2]
    for split in ("train", "val", "test"):
        stage1 = [[row["example_id"] for row in data[arm].records("AF2", split) if row["label"] == 0] for arm in ARMS]
        assert stage1[0] == stage1[1] == stage1[2]
        short = [row["example_id"] for row in data["short"].records("AF3", split) if row["label"] == 0]
        hard = [row["example_id"] for row in data["hard"].records("AF3", split) if row["label"] == 0]
        assert short == hard


def test_score_margin_metrics_are_predeclared_and_complete():
    predictions = [
        {"true_label": 1, "predicted_label": 1, "p_attack": 0.999},
        {"true_label": 1, "predicted_label": 0, "p_attack": 0.4},
        {"true_label": 0, "predicted_label": 0, "p_attack": 0.001},
        {"true_label": 0, "predicted_label": 1, "p_attack": 0.6},
    ]
    result = metric_block(predictions)
    assert set(result["p_attack_summary"]) == {"mean", "median", "p05", "p10", "p90", "p95", "p99"}
    assert result["fraction_p_attack_ge_0_99"] == 0.25
    assert result["fraction_p_attack_le_0_01"] == 0.25
    assert result["attack_tpr"] == 0.5
    assert result["benign_fpr"] == 0.5


def test_protocol_explicitly_excludes_replay_and_memory_budget():
    source = (ROOT / "src/continualguard/stronger_pilot_no_replay.py").read_text(encoding="utf-8")
    assert '"historical_training_records": 0' in source
    assert '"replay": False' in source
    assert '"selective_adaptation": False' in source
    assert '"numerical_memory_budget": None' in source
    assert '"optimizer_state_saved": False' in source
