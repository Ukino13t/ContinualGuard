"""Append the independently established Revision 3 selector-order correction."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
HISTORY = ROOT / "history"


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def append_unique(path: Path, field: str, record: dict) -> None:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(row.get(field) == record[field] for row in records):
        raise ValueError(f"Duplicate {field}: {record[field]}")
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    v1 = ROOT / "results/revision3/order_audit_v1_doc_seed2026_s1_s2.json"
    v2 = ROOT / "results/revision3/order_audit_v2_smoke_doc_seed2026_s1_s2.json"
    old, new = (json.loads(p.read_text(encoding="utf-8")) for p in (v1, v2))
    assert old["buffer_same_id_set"] and old["replay_position_delta"] == 400
    assert new["buffer_same_id_set"] and new["replay_position_delta"] == 0
    assert new["model_weights_sha256_a"] == new["model_weights_sha256_b"]
    stamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    append_unique(HISTORY / "experiments.jsonl", "run_id", {
        "run_id": "route-a-revision3-a0-v1-order-confound-20261018",
        "date": stamp, "project_root": str(ROOT),
        "experiment_regime": "Candidate_B_A0_v1_historical_confounded",
        "completed_A0_v1_M200_trajectories": 9,
        "checkpoint_validation_status": "PASS_but_did_not_test_replay_index_parity",
        "class_quota_causal_interpretation": "withdrawn",
        "same_selected_ID_set": True, "ordered_buffer_position_delta": 200,
        "next_stage_replay_position_delta": 400,
        "old_order_audit_sha256": sha(v1),
        "corrected_protocol_sha256": sha(ROOT / "results/revision3/protocol_v2.json"),
        "corrected_smoke_parity_sha256": sha(v2),
        "raw_content_redistributed": False,
    })
    append_unique(HISTORY / "decisions.jsonl", "id", {
        "id": "D058", "date": stamp,
        "decision": "Withdraw causal interpretation of A0-v1 versus A1 and freeze canonical-order A0-v2",
        "evidence": "same AF2 set but 400/400 AF3 replay IDs differ in v1; v2 smoke restores 400/400 replay and model-weight parity",
        "a1_a2_interpretation": "descriptive selector contrast because stage quotas and buffer order both change",
        "full_v2_results": "pending_validation_and_human_review",
        "archival_backup_touched": False,
    })
    append_unique(HISTORY / "changes.jsonl", "id", {
        "id": "C045", "date": stamp, "decision_id": "D058",
        "reason": "Version the corrected A0 protocol, preserve v1 artifacts, and update ACL causal boundary",
        "corrected_protocol_sha256": sha(ROOT / "results/revision3/protocol_v2.json"),
        "v1_results_deleted_or_overwritten": False,
        "archival_backup_touched": False,
        "raw_content_redistributed": False,
    })
    print("PASS: correction provenance recorded as experiment, D058, C045")


if __name__ == "__main__":
    main()
