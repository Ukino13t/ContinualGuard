import json
from pathlib import Path

import pytest

ROOT = Path(r"E:\ContinualGuard")


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


@pytest.mark.skipif(not (ROOT / "results/replay_artifact_validation.json").exists(), reason="comparison not summarized")
def test_replay_artifact_validation_passes():
    value = load("results/replay_artifact_validation.json")
    assert value["status"] == "PASS"
    assert value["new_runs"] == 27
    assert value["new_checkpoints"] == 108
    assert value["failed_runs"] == 0
    assert value["initialization_matches_no_replay"] is True
    assert value["candidate_b_hashes_verified"] is True


@pytest.mark.skipif(not (ROOT / "results/replay_fairness_validation.json").exists(), reason="comparison not summarized")
def test_replay_fairness_validation_passes():
    value = load("results/replay_fairness_validation.json")
    assert value["status"] == "PASS"
    assert value["capacity"] == 200
    assert value["replay_batch_size"] == 4
    assert value["example_capacity_equality"] == "PASS"
    assert value["replay_exposure_equality"] == "PASS"
    assert value["dataset_seed_config_equality"] == "PASS"
    assert value["no_test_validation_external_selection"] == "PASS"


@pytest.mark.skipif(not (ROOT / "results/replay_comparison_aggregate.json").exists(), reason="comparison not summarized")
def test_complete_four_method_aggregate():
    value = load("results/replay_comparison_aggregate.json")
    assert value["methods"] == ["no_replay", "reservoir", "derpp", "security_aware"]
    assert len(value["run_summary_per_seed"]) == 36
    assert len(value["acquisition_per_seed"]) == 144
    assert len(value["acquisition_gates"]) == 48
    assert len(value["key_b_short_af3"]) == 12
    assert len(value["table_ready"]) == 11


@pytest.mark.skipif(not (ROOT / "results/replay_memory_composition.json").exists(), reason="comparison not summarized")
def test_memory_accounting_and_security_balance():
    rows = load("results/replay_memory_composition.json")["rows"]
    assert len(rows) == 108
    assert all(row["buffer_size"] <= 200 for row in rows)
    security = [row for row in rows if row["method"] == "security_aware"]
    assert all(row["label_counts"] == {"0": 100, "1": 100} for row in security)
    assert all(row["raw_text_stored"] is False for row in rows)
    assert all(row["historical_logit_entries"] == row["buffer_size"] for row in rows if row["method"] == "derpp")
