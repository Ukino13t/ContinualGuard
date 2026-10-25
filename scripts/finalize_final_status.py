from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(r"E:\ContinualGuard")
DOCS = ROOT / "docs"
HISTORY = ROOT / "history"


def prepend_once(path: Path, marker: str, text: str) -> None:
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in current:
        return
    path.write_text(text.rstrip() + "\n\n" + current, encoding="utf-8")


def append_once(path: Path, marker: str, text: str) -> None:
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in current:
        return
    sep = "" if not current or current.endswith("\n") else "\n"
    path.write_text(current + sep + "\n" + text.rstrip() + "\n", encoding="utf-8")


def append_jsonl_once(path: Path, record: dict) -> None:
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    for line in current.splitlines():
        try:
            if json.loads(line).get("id") == record["id"]:
                return
        except (json.JSONDecodeError, AttributeError):
            pass
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


STATUS = """<!-- FINAL_ANALYSIS_STATUS_20261006:BEGIN -->
## Final project status — 2026-10-06

**EXPERIMENTS COMPLETE / FINAL ANALYSIS / REPORT WRITING**

H2-mini remained **R0**. Candidate B No Replay reached **R2** under the frozen replay-entry criteria. The primary replay comparison and Generic Loss-aware control are complete and validated. All experimental artifacts are frozen, and no further experiment is authorized.

The current interpretation is in `docs/final_project_summary.md` and `docs/final_claims_audit.md`. Dated material below remains a historical record of the state at the time.
<!-- FINAL_ANALYSIS_STATUS_20261006:END -->"""


def main() -> None:
    if ROOT.resolve() != Path.cwd().resolve():
        raise SystemExit(f"run from canonical root {ROOT}")

    for name in [
        "replay_readiness_review.md",
        "two_week_experiment_plan.md",
        "open_questions.md",
        "project_completion_options.md",
        "assumptions_and_risks.md",
    ]:
        prepend_once(DOCS / name, "FINAL_ANALYSIS_STATUS_20261006:BEGIN", STATUS)

    decision = """<!-- FINAL_ANALYSIS_D054:BEGIN -->
## D054 — Freeze the five-method study and enter final writing (2026-10-06)

Accept the validated M0–M4 Candidate B corpus as the final experiment set. Freeze the detector, benchmark, seeds, order, optimization, exposure, and M=200 replay budget. Record H2-mini as R0, Candidate B No Replay as R2, the primary replay comparison as complete, and the Generic Loss-aware control as complete. No further experiment is authorized. Final work is limited to figures, tables, descriptive statistics, claims audit, and report writing.
<!-- FINAL_ANALYSIS_D054:END -->"""
    append_once(DOCS / "decision_log.md", "FINAL_ANALYSIS_D054:BEGIN", decision)

    change = """<!-- FINAL_ANALYSIS_C040:BEGIN -->
## C040 — Final analysis package (2026-10-06)

Generated the five-method master table, reproducibility table, Appendix Tables A–I, six PNG/PDF figure pairs, final descriptive statistics, A–J claims audit, report-ready Results/Discussion/Method/Limitations, contribution statement, experiment timeline, consistency audit, and final project summary. Inputs were frozen aggregate/validation artifacts only; no model, checkpoint, benchmark, seed, or prediction was rerun.
<!-- FINAL_ANALYSIS_C040:END -->"""
    append_once(DOCS / "change_log.md", "FINAL_ANALYSIS_C040:BEGIN", change)

    experiment = """<!-- FINAL_ANALYSIS_HISTORY_20261006:BEGIN -->
## Final analysis and experiment freeze — 2026-10-06

All experiments are complete. The final five-method study comprises No Replay, Reservoir ER, DER++, Security-aware Replay, and Generic Loss-aware Replay on frozen Candidate B. Final analysis used validated aggregate JSON only. Project state: **EXPERIMENTS COMPLETE / FINAL ANALYSIS / REPORT WRITING**.
<!-- FINAL_ANALYSIS_HISTORY_20261006:END -->"""
    append_once(DOCS / "experiment_history.md", "FINAL_ANALYSIS_HISTORY_20261006:BEGIN", experiment)

    now = "2026-10-06T12:00:00+08:00"
    append_jsonl_once(HISTORY / "decisions.jsonl", {
        "date": now,
        "id": "D054",
        "decision": "Freeze the final five-method Candidate B study and enter final analysis/report writing",
        "methods": ["no_replay", "reservoir", "derpp", "security_aware", "loss_aware"],
        "readiness": {"H2-mini": "R0", "Candidate B no-replay": "R2", "replay_comparison": "complete", "loss_aware_control": "complete"},
        "project_status": "EXPERIMENTS COMPLETE / FINAL ANALYSIS / REPORT WRITING",
        "no_further_experiments_authorized": True,
    })
    append_jsonl_once(HISTORY / "changes.jsonl", {
        "date": now,
        "id": "C040",
        "decision_id": "D054",
        "reason": "Generate final analysis and report-writing package from frozen validated artifacts only",
        "files_changed": [
            "scripts/build_final_analysis_package.py",
            "results/final_tables/**",
            "results/final_figures/**",
            "docs/final_*.md",
            "docs/replay_readiness_review.md",
            "docs/two_week_experiment_plan.md",
            "docs/open_questions.md",
            "docs/project_completion_options.md",
            "docs/assumptions_and_risks.md",
            "docs/decision_log.md",
            "docs/change_log.md",
            "docs/experiment_history.md",
        ],
        "validation": ["31 core artifacts initially validated", "final tables and figures regenerated from frozen aggregate JSON", "no experiment executed"],
        "scope": "final analysis, figures, tables, descriptive statistics, consistency audit, and report writing only",
    })

    print(json.dumps({"status": "PASS", "decision": "D054", "change": "C040"}, indent=2))


if __name__ == "__main__":
    main()
