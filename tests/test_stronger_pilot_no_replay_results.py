import json
from pathlib import Path

import pytest

from continualguard.stronger_pilot_summary import verify_artifacts

ROOT = Path(r"E:\ContinualGuard")
RESULT = ROOT / "results/stronger_pilot_no_replay"


@pytest.mark.skipif(not (RESULT / "hard/seed2026/stage4/stage_metadata.json").exists(), reason="Candidate B run is not complete")
def test_completed_candidate_b_artifact_contract():
    validation = verify_artifacts(ROOT)
    assert validation["status"] == "PASS"
    assert validation["runs"] == 9
    assert validation["checkpoints"] == 36
    assert validation["failed_runs"] == 0
    assert all(all(checks.values()) for checks in validation["stage1_three_arm_equivalence"].values())
    assert all(all(checks.values()) for checks in validation["stage2_short_hard_equivalence"].values())


@pytest.mark.skipif(not (ROOT / "results/stronger_pilot_no_replay_readiness.json").exists(), reason="Candidate B summary is not complete")
def test_summary_outputs_and_no_replay_decision_contract():
    readiness = json.loads((ROOT / "results/stronger_pilot_no_replay_readiness.json").read_text(encoding="utf-8"))
    aggregate = json.loads((ROOT / "results/stronger_pilot_no_replay_aggregate.json").read_text(encoding="utf-8"))
    assert readiness["classification"] in {"R0", "R1", "R2"}
    assert readiness["replay_implemented"] is False
    assert readiness["numerical_memory_budget"] is None
    assert aggregate["replay"] is False
    assert aggregate["selective_adaptation"] is False
    assert aggregate["numerical_memory_budget"] is None
    tables = RESULT / "tables"
    for name in (
        "attack_tpr_trajectory.csv", "benign_fpr_trajectory.csv", "af5_compositional_transfer.csv",
        "b_short_minus_b_doc.csv", "b_hard_minus_b_short.csv", "notinject_trajectory.csv",
        "bshort_trajectory.csv", "acquisition_gate.csv", "retention_by_task.csv",
        "score_margin.csv", "readiness_evidence.csv",
    ):
        assert (tables / name).is_file()
