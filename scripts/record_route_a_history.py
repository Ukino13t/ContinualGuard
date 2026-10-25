"""Append one idempotent Route A record set; preserve earlier project history."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
DATE = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
MARKER = "ROUTE_A_P0_20261017"


def append_jsonl(relative: str, key: str, item: dict) -> None:
    path = ROOT / relative
    old = path.read_text(encoding="utf-8")
    if any(json.loads(line).get(key) == item[key] for line in old.splitlines() if line.strip()):
        return
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_doc(relative: str, title: str, body: str) -> None:
    path = ROOT / relative
    old = path.read_text(encoding="utf-8")
    if f"<!-- {MARKER}:BEGIN -->" in old:
        return
    addition = f"\n<!-- {MARKER}:BEGIN -->\n## {title}\n\n{body.strip()}\n<!-- {MARKER}:END -->\n"
    path.write_text(old + addition, encoding="utf-8")


append_jsonl("history/experiments.jsonl", "run_id", {
    "run_id": "route-a-p0-20261017", "date": DATE,
    "experiment_regime": "Candidate_B_Route_A_structured_ablation_and_offline_controls",
    "project_root": str(ROOT), "model": "microsoft/deberta-v3-base",
    "seeds": [17, 42, 2026], "stages": ["AF2", "AF3", "AF4", "AF5"],
    "arms": ["B-DOC", "B-SHORT", "B-HARD"],
    "new_completed_runs": {"a1_a2_a3_four_stage": 27, "joint_one_epoch": 9,
                           "compute_matched_seed42": 3},
    "reused_frozen_methods": ["M0", "A0/M1", "M2", "A4/M4", "A5/M3"],
    "primary_findings_pp": {"m0_b_short_attack_tpr_bwt_mean": -12.593,
                            "m0_b_hard_benign_fpr_increase_mean": 20.370,
                            "a0_b_short_attack_tpr_bwt_mean": 0.370,
                            "a0_b_hard_benign_fpr_increase_mean": 1.481},
    "coverage_rule_nonbinding": True,
    "validation": "full-hash validation in results/extended/analysis/validation.json",
    "artifacts": ["results/extended/analysis", "reports/RESEARCH_COMPLETION_REPORT.md",
                  "paper/acl/continualguard_acl.tex"],
    "p1_training": "unexecuted_due_to_disk_and_compute_budget",
    "status": "P0_complete_with_declared_RQ3_identification_limit",
})

append_jsonl("history/decisions.jsonl", "id", {
    "id": "D056", "date": DATE,
    "decision": "Interpret Route A as a controlled benchmark study; do not attribute A2/A3 differences to coverage",
    "reason": "All Candidate B TRAIN arm-stage coverage keys are unique; A1-A3 also use variant-specific rank streams. Three-seed selector differences are small and not uniformly directional.",
    "original_frozen_artifacts_preserved": True,
    "p1_status": "pending; capacity, ratio, new seeds and new order not run",
    "local_pdf_status": "unavailable_due_to_missing_TinyTeX_lineno; Overleaf source package provided",
})

append_jsonl("history/changes.jsonl", "id", {
    "id": "C042", "date": DATE, "decision_id": "D056",
    "reason": "Execute and document authorized Route A P0 extension",
    "files_changed": ["scripts/structured_ablation.py", "scripts/offline_controls.py",
                      "scripts/aggregate_route_a.py", "scripts/validate_route_a.py",
                      "scripts/write_route_a_reports.py", "scripts/check_coverage.py",
                      "scripts/plot_route_a.py", "scripts/plot_route_a_memory.py",
                      "scripts/plot_route_a_pairs.py", "tests/test_route_a_ablation.py",
                      "reports/00_repository_audit.md", "reports/01_ablation_results.md",
                      "reports/02_joint_and_compute_controls.md",
                      "reports/RESEARCH_COMPLETION_REPORT.md", "paper/acl/continualguard_acl.tex"],
    "validation": ["27 new A1-A3 runs complete", "9 joint runs complete",
                   "3 compute controls complete", "independent Route A validation",
                   "12/12 arm-stage TRAIN coverage keys unique per record"],
    "raw_content_redistributed": False,
})

append_doc("docs/experiment_history.md", "2026-10-17 - Route A P0 empirical extension",
           "The previously frozen 45-run Candidate B comparison was preserved. New Route A work completed 27 A1-A3 four-stage ablations, nine one-epoch joint references, and three seed-42 compute-matched current-only controls. Paired and per-stage results are under `results/extended/analysis/`; validation is recorded in `validation.json`. No capacity, ratio, added-seed, or new-order sweep was run. The earlier 2026-10-06 experiment-freeze entry accurately described the earlier milestone and is not overwritten.")
append_doc("docs/decision_log.md", "2026-10-17 - Route A interpretation and resource decision",
           "Interpret bounded replay as effective on the declared Candidate B retention failures, but do not claim a superior security-specific ranking formula. A2/A3 coverage selection is nonbinding because every arm-stage TRAIN item has a unique `coverage_key`; adjacent A1-A3 variants also change deterministic hash ranks. Joint is a finite-optimization offline reference, not an upper bound. Compute-matched current-only completed only seed 42. Preserve all frozen results and stop checkpoint-heavy P1 training with about 10 GiB free on E:; no storage cleanup or archival-backup access occurred.")
append_doc("docs/change_log.md", "2026-10-17 - Route A scripts, results, reports and ACL source",
           "Added controlled A1-A3 runner, joint/current-only controls, independent validation, aggregation, diagnostic and coverage audits, vector plots, and regenerated ACL tables. Added 39 new P0 runs under `results/extended/` and revised the ACL paper around RQ1-RQ3. No raw source prompt or reviewer note was copied into the paper package. Local TeX compilation remains blocked by a missing package; an Overleaf-ready package is supplied.")
append_doc("docs/assumptions_and_risks.md", "2026-10-17 - Route A remaining validity limits",
           "Candidate B's coverage key is unique per TRAIN record, so the coverage preference cannot be assessed in this benchmark. A1-A3 use distinct deterministic rank streams, preventing clean causal attribution of small adjacent differences. Only one compute-matched seed per arm was run; offline joint uses future-stage TRAIN records by design. Arms bundle source, language, format, length, and semantics. P1 sensitivity and new-order robustness are unexecuted; three-seed results are descriptive, and diagnostic banks are not deployment tests.")
append_doc("docs/open_questions.md", "2026-10-17 - Route A follow-up questions",
           "Before a causal coverage claim, can a new predeclared grouping yield repeated eligible keys without violating split independence? Can common random ranks and additional seeds distinguish class/stage balance from random selection noise? Do replay benefits persist at capacities 50/100/400, other replay exposures, and independently justified stream orders? Why does in-stream B-DOC retention coexist with high external NotInject false-positive rate? These are open; no result is imputed.")

print("Route A history and documentation records present")
