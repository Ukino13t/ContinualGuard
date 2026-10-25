import json
from pathlib import Path

import pytest

ROOT = Path(r"E:\ContinualGuard")


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


@pytest.mark.skipif(not (ROOT / "results/loss_aware_artifact_validation.json").exists(), reason="control not summarized")
def test_loss_aware_artifact_validation():
    value=load("results/loss_aware_artifact_validation.json")
    assert value["status"]=="PASS"
    assert (value["new_runs"],value["new_checkpoints"],value["failed_runs"],value["resumed_runs"])==(9,36,0,0)
    assert value["checkpoint_files_hashed"]==252
    assert value["prediction_files"]==162
    assert value["future_stage_memory_entries"]==0


@pytest.mark.skipif(not (ROOT / "results/loss_aware_fairness_validation.json").exists(), reason="control not summarized")
def test_loss_aware_fairness():
    value=load("results/loss_aware_fairness_validation.json")
    assert value["status"]=="PASS"
    assert value["only_intended_difference"]=="pre-update memory selection score"
    assert value["fixed_common_factors"] is True
    assert len(value["capacity_checks"])==36 and all(x["size"]==200 for x in value["capacity_checks"])
    assert len(value["exposure_checks"])==36 and all(x["pass"] for x in value["exposure_checks"])


@pytest.mark.skipif(not (ROOT / "results/loss_aware_control_aggregate.json").exists(), reason="control not summarized")
def test_complete_loss_aware_aggregate():
    value=load("results/loss_aware_control_aggregate.json")
    assert value["method"]=="loss_aware"
    assert len(value["run_summary_per_seed"])==9
    assert len(value["retention_per_seed"])==27
    assert len(value["acquisition_per_seed"])==36
    assert len(value["acquisition_gates"])==12
    assert len(value["key_b_short_af3"])==3


@pytest.mark.skipif(not (ROOT / "results/loss_aware_memory_composition.json").exists(), reason="control not summarized")
def test_loss_aware_memory_and_score_outputs():
    rows=load("results/loss_aware_memory_composition.json")["rows"]
    assert len(rows)==36
    assert all(x["buffer_size"]==200 and x["label_counts"]=={"0":100,"1":100} for x in rows)
    assert all(x["raw_text_stored"] is False and x["historical_logit_entries"]==0 for x in rows)
    relationship=load("results/loss_aware_score_relationship.json")
    assert len(relationship["score_correlations"])==108
    assert len(relationship["selected_memory_overlap"])==108
    assert len(relationship["selected_memory_overlap_by_task_and_label"])==270
    pairwise=load("results/loss_aware_security_pairwise.json")
    assert len(pairwise["retention"])==27 and len(pairwise["acquisition"])==36
