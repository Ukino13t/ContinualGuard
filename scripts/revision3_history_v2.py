"""Append final corrected Revision 3 provenance after all validation and packaging."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
ANALYSIS = ROOT / "results/revision3/analysis_v2"
HISTORY = ROOT / "history"


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def append_unique(path: Path, field: str, record: dict) -> None:
    old = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(row.get(field) == record[field] for row in old):
        raise ValueError(f"Duplicate {field}: {record[field]}")
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    primary = load(ANALYSIS / "validation_M200.json")
    capacity = load(ANALYSIS / "capacity_validation.json")
    package = load(ROOT / "reports/revision3/overleaf_package_manifest.json")
    assert primary["status"] == capacity["status"] == "PASS"
    assert primary["runs_checked"] == 9 and primary["full_checkpoint_hashes"]
    assert capacity["runs_checked"] == 36 and capacity["full_checkpoint_hashes"]
    assert len(capacity["matched_pairs"]) == 18
    with (ANALYSIS / "paired_aggregate.csv").open(encoding="utf-8", newline="") as handle:
        paired = list(csv.DictReader(handle))
    assert {row["arm"] for row in paired} == {"doc", "short", "hard"}
    queue = [json.loads(line) for line in (ROOT / "results/revision3/v2_queue.jsonl")
             .read_text(encoding="utf-8").splitlines() if line]
    assert queue[-1]["event"] == "queue_complete"
    new_a1 = sum(row.get("event") == "complete" and
                 str(row.get("key", "")).startswith("a1_matched/") for row in queue)
    effects = {row["arm"]: {
        metric: {"mean_pp": float(row[f"a1_minus_a0_{metric}_mean"]),
                 "sample_sd_pp": float(row[f"a1_minus_a0_{metric}_sample_sd"])}
        for metric in ("tpr_bwt_pp", "macro_f1_bwt_pp", "benign_fpr_increase_pp")}
        for row in paired}
    stamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    common = {"date": stamp, "project_root": str(ROOT),
              "model": "microsoft/deberta-v3-base", "seeds": [17, 42, 2026],
              "stage_order": ["AF2", "AF3", "AF4", "AF5"],
              "dataset_manifest_sha256": "81962e092e1f8502a537d675582549df94f128f5ebf493b1ed0b59540ec3e8f4"}
    append_unique(HISTORY / "experiments.jsonl", "run_id", {
        "run_id": "route-a-revision3-p0-a0-v2-20261018", **common,
        "experiment_regime": "Candidate_B_corrected_canonical_order_A0_v2_vs_frozen_A1",
        "new_A0_v2_trajectories": 9, "frozen_A1_trajectories": 9,
        "protocol_sha256": sha(ROOT / "results/revision3/protocol_v2.json"),
        "validation_sha256": sha(ANALYSIS / "validation_M200.json"),
        "full_checkpoint_hashes": True, "paired_seed_effects_pp": effects,
        "v1_class_quota_effect_excluded": True,
        "independent_confirmation": "unexecuted_no_audited_independent_source",
        "human_result_review": "pending",
    })
    append_unique(HISTORY / "experiments.jsonl", "run_id", {
        "run_id": "route-a-revision3-p1-capacity-v2-20261018", **common,
        "experiment_regime": "Candidate_B_two_arm_corrected_A0_v2_A1_capacity_grid",
        "arms": ["B-SHORT", "B-HARD"], "capacities": [50, 200, 400],
        "matched_cells": 36, "matched_pairs": 18,
        "A0_v2_M50_M400_new_trajectories": 12,
        "A1_trajectories_frozen_protocol": 18,
        "A1_capacity_trajectories_new_in_v2_queue": new_a1,
        "A1_trajectories_preexisting_before_v2_queue": 18 - new_a1,
        "protocol_sha256": sha(ROOT / "results/revision3/capacity_protocol_v2.json"),
        "validation_sha256": sha(ANALYSIS / "capacity_validation.json"),
        "aggregate_sha256": sha(ANALYSIS / "capacity_aggregate.csv"),
        "full_checkpoint_hashes": True,
        "original_online_reservoir_in_grid": False,
        "human_result_review": "pending",
    })
    append_unique(HISTORY / "decisions.jsonl", "id", {
        "id": "D059", "date": stamp,
        "decision": "Record fully validated corrected Route A Revision 3 analyses and ACL package for human review",
        "class_quota_evidence": "A0-v2 versus frozen A1; three paired seeds; canonical replay ordering",
        "stage_quota_evidence": "A1/A2 remains descriptive stage-quota-plus-order contrast",
        "independent_confirmation": "unexecuted",
        "human_result_review": "pending",
        "new_method_proposed": False,
    })
    append_unique(HISTORY / "changes.jsonl", "id", {
        "id": "C046", "date": stamp, "decision_id": "D059",
        "reason": "Validate corrected P0/P1, regenerate ACL figures and tables, compile PDF, and package Overleaf sources",
        "p0_full_hash_runs": 9, "p1_full_hash_cells": 36,
        "pdf_sha256": package["pdf_sha256"],
        "overleaf_zip": package["package"], "overleaf_zip_sha256": package["sha256"],
        "raw_content_redistributed": False,
        "archival_backup_touched": False, "storage_cleanup": False,
    })
    print("PASS: corrected P0/P1, D059, and C046 provenance recorded")


if __name__ == "__main__":
    main()
