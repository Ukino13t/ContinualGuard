"""Append idempotent Revision 3 provenance records after final validation."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(r"E:\ContinualGuard")
HISTORY = ROOT / "history"
ANALYSIS = ROOT / "results/revision3/analysis"


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def append_unique(file: Path, key: str, record: dict):
    existing = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line]
    matches = [r for r in existing if r.get(key) == record[key]]
    if matches:
        comparable = lambda value: {k: v for k, v in value.items() if k != "date"}
        if len(matches) != 1 or comparable(matches[0]) != comparable(record):
            raise RuntimeError(f"Existing {key} record differs: {record[key]}")
        return
    with file.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main():
    p0 = json.loads((ANALYSIS / "validation_M200.json").read_text(encoding="utf-8"))
    p1 = json.loads((ANALYSIS / "capacity_validation.json").read_text(encoding="utf-8"))
    package = json.loads((ROOT / "reports/revision3/overleaf_package_manifest.json").read_text(encoding="utf-8"))
    assert p0["status"] == p1["status"] == "PASS"
    assert p0["runs_checked"] == 9 and p1["runs_checked"] == 36
    assert p0["full_checkpoint_hashes"] and p1["full_checkpoint_hashes"]
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    common = {"date": timestamp, "project_root": str(ROOT),
              "model": "microsoft/deberta-v3-base", "seeds": [17, 42, 2026],
              "stage_order": ["AF2", "AF3", "AF4", "AF5"],
              "dataset_manifest_sha256": "81962e092e1f8502a537d675582549df94f128f5ebf493b1ed0b59540ec3e8f4"}
    append_unique(HISTORY / "experiments.jsonl", "run_id", {
        "run_id": "route-a-revision3-p0-matched-a0-20261018", **common,
        "experiment_regime": "Candidate_B_A0_A1_shared_rank_class_quota_isolation",
        "new_A0_trajectories": 9, "reused_frozen_A1_trajectories": 9,
        "selector_quota_affected_cells": 32, "selector_cells": 36,
        "protocol_sha256": sha(ROOT / "results/revision3/protocol_v1.json"),
        "validation_sha256": sha(ANALYSIS / "validation_M200.json"),
        "validation_runs": 9, "full_checkpoint_hashes": True,
        "interpretation": "No consistent attack-retention benefit from class quotas; B-SHORT benign FPR drift lower in two of three seeds; original online Reservoir contextual only",
        "status": "validated_pending_human_result_review"})
    append_unique(HISTORY / "experiments.jsonl", "run_id", {
        "run_id": "route-a-revision3-p1-capacity-20261018", **common,
        "experiment_regime": "Candidate_B_two_arm_matched_A0_A1_capacity_grid",
        "arms": ["B-SHORT", "B-HARD"], "capacities": [50, 200, 400],
        "matched_cells": 36, "new_trajectories": 22, "reused_compatible_trajectories": 14,
        "protocol_sha256": sha(ROOT / "results/revision3/capacity_protocol_v1.json"),
        "validation_sha256": sha(ANALYSIS / "capacity_validation.json"),
        "validation_runs": 36, "matched_pairs": 18, "full_checkpoint_hashes": True,
        "original_online_reservoir_in_grid": False,
        "independent_confirmation": "unexecuted_no_audited_independent_source",
        "status": "validated_pending_human_result_review"})
    append_unique(HISTORY / "decisions.jsonl", "id", {
        "id": "D058", "date": timestamp,
        "decision_type": "research_interpretation_record",
        "decision": "Keep original Reservoir distinct from A0-matched; interpret class and capacity contrasts only under the frozen shared-rank AMP-256 protocol; do not claim independent replication",
        "human_result_review": "pending", "independent_confirmation": "unexecuted",
        "external_validity": "Candidate_B_specific", "new_methods": "none"})
    append_unique(HISTORY / "changes.jsonl", "id", {
        "id": "C045", "date": timestamp, "decision_id": "D058",
        "reason": "Complete Route A Revision 3 validation and ACL submission polish",
        "new_p0_trajectories": 9, "new_p1_trajectories": 22,
        "p0_full_hash_runs": 9, "p1_full_hash_cells": 36,
        "pdf_sha256": package["pdf_sha256"], "overleaf_zip": package["package"],
        "overleaf_zip_sha256": package["sha256"],
        "raw_content_redistributed": False, "archival_backup_touched": False,
        "storage_cleanup": False})
    print(json.dumps({"status": "PASS", "records": ["P0", "P1", "D058", "C045"]}))


if __name__ == "__main__":
    main()
