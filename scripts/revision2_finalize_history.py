"""Append idempotent, hash-linked Revision 2 history after packaging."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
ANALYSIS = ROOT / "results/revision2/analysis"


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def append_once(path: Path, key: str, entry: dict) -> None:
    existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    matches = [row for row in existing if row.get(key) == entry[key]]
    if matches:
        if len(matches) != 1 or matches[0] != entry:
            raise RuntimeError(f"Conflicting history record: {path} / {entry[key]}")
        return
    with path.open("ab") as file:
        if path.stat().st_size and not path.read_bytes().endswith(b"\n"):
            file.write(b"\n")
        file.write((json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))


def main() -> None:
    p0 = load(ANALYSIS / "validation.json")
    p1 = load(ANALYSIS / "capacity_validation.json")
    pair = load(ANALYSIS / "capacity_pair_controls.json")
    package = load(ROOT / "reports/revision2/overleaf_package_manifest.json")
    if (p0["status"], p0["runs_checked"], p0["full_checkpoint_hashes"],
            len(p0["matched_pair_controls"])) != ("PASS", 24, True, 9):
        raise RuntimeError("P0 incomplete")
    if (p1["status"], p1["runs_checked"], p1["full_checkpoint_hashes"],
            pair["status"], pair["runs_checked"]) != ("PASS", 4, True, "PASS", 5):
        raise RuntimeError("P1 incomplete")
    zip_path = ROOT / "paper/ContinualGuard_RouteA_Revision2_Overleaf.zip"
    if package["sha256"] != sha(zip_path):
        raise RuntimeError("Overleaf package manifest hash mismatch")
    with (ANALYSIS / "aggregate.csv").open(encoding="utf-8", newline="") as file:
        agg = {(r["method"], r["arm"]): r for r in csv.DictReader(file)}
    now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    history = ROOT / "history"
    manifest_hash = sha(ROOT / "data/processed/stronger_pilot_candidate_b/materialization_manifest.json")
    p0_entry = {
        "run_id": "route-a-revision2-p0-20261018", "date": now,
        "experiment_regime": "Candidate_B_shared_rank_ablation_and_three_seed_current_only",
        "project_root": str(ROOT), "model": "microsoft/deberta-v3-base",
        "dataset_manifest_sha256": manifest_hash,
        "arms": ["B-DOC", "B-SHORT", "B-HARD"], "seeds": [17, 42, 2026],
        "stage_order": ["AF2", "AF3", "AF4", "AF5"],
        "new_runs": {"A1_matched": 9, "A2_matched": 9,
                     "current_only_added_seeds_17_2026": 6},
        "reused_seed42_current_only_runs": 3,
        "amp_grad_scaler_initial_scale_matched": 256,
        "excluded_failed_amp1024_pilot": True,
        "validation": {"path": "results/revision2/analysis/validation.json",
                       "sha256": sha(ANALYSIS / "validation.json"),
                       "runs": 24, "checkpoint_files": 672,
                       "prediction_rows": 61848, "matched_pairs": 9},
        "findings_pp": {
            "b_short_a1_tpr_bwt_mean": float(agg[("A1_matched", "short")]["tpr_bwt_pp_mean"]),
            "b_short_a2_tpr_bwt_mean": float(agg[("A2_matched", "short")]["tpr_bwt_pp_mean"]),
            "b_hard_current_only_final_fpr_mean": float(agg[("compute_current_only", "hard")]["final_benign_fpr_pct_mean"]),
            "b_hard_reservoir_final_fpr_mean": float(agg[("A0_Reservoir", "hard")]["final_benign_fpr_pct_mean"]),
        },
        "coverage_nonbinding_all_actual_pools": True,
        "status": "validated_pending_human_result_review",
    }
    p1_entry = {
        "run_id": "route-a-revision2-p1-capacity-20261018", "date": now,
        "experiment_regime": "predeclared_B_HARD_seed42_exploratory_capacity_subset",
        "project_root": str(ROOT), "model": "microsoft/deberta-v3-base",
        "dataset_manifest_sha256": manifest_hash,
        "arm": "B-HARD", "seed": 42, "capacities": [50, 400],
        "new_runs": {"A1_matched": 2, "reservoir_capacity": 2},
        "contextual_same_policy_A1_M200": 1,
        "validation": {"path": "results/revision2/analysis/capacity_validation.json",
                       "sha256": sha(ANALYSIS / "capacity_validation.json"),
                       "runs": 4, "checkpoint_files": 112},
        "pair_control_sha256": sha(ANALYSIS / "capacity_pair_controls.json"),
        "full_capacity_sweep": "not run; 36 trajectories exceed remaining E-drive capacity",
        "single_seed_exploratory": True,
        "status": "validated_pending_human_result_review",
    }
    decision = {
        "id": "D057", "date": now,
        "decision": "Record the authorized Route A Revision 2 controlled repair and restrict interpretation pending human result review",
        "interpretation": "A1/A2 shared-rank stage-quota effect mixed; coverage nonbinding; three-seed current-only control does not reproduce principal replay endpoints; P1 capacity is single-seed exploratory",
        "human_result_review": "pending", "further_experiments_authorized": False,
        "no_selector_superiority_or_capacity_frontier_claim": True,
    }
    change = {
        "id": "C044", "date": now, "decision_id": "D057",
        "reason": "Complete validated Route A Revision 2 and update ACL manuscript",
        "new_runs": 28, "p0_checkpoint_files_sha256_pass": 672,
        "p1_checkpoint_files_sha256_pass": 112,
        "overleaf_zip": str(zip_path.relative_to(ROOT)).replace("\\", "/"),
        "overleaf_zip_sha256": package["sha256"],
        "overleaf_zip_members": package["members"],
        "raw_content_redistributed": False, "archival_backup_touched": False,
        "local_pdf_compile": "not run; Overleaf pdfLaTeX/BibTeX review required",
    }
    append_once(history / "experiments.jsonl", "run_id", p0_entry)
    append_once(history / "experiments.jsonl", "run_id", p1_entry)
    append_once(history / "decisions.jsonl", "id", decision)
    append_once(history / "changes.jsonl", "id", change)
    print(json.dumps({"status": "PASS", "experiments": 2, "decisions": 1, "changes": 1}))


if __name__ == "__main__":
    main()
