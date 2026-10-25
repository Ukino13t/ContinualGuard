from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(r"E:\ContinualGuard")
DOCS = ROOT / "docs"
HISTORY = ROOT / "history"
OLD = "EXPERIMENTS COMPLETE / FINAL ANALYSIS / REPORT WRITING"
NEW = "EXPERIMENTS COMPLETE / FINAL ANALYSIS COMPLETE / REPORT WRITING ONLY"
MARKER = "FINAL_REPORT_DRAFT_STATUS_20261006:BEGIN"


def prepend_once(path: Path, block: str) -> None:
    current = path.read_text(encoding="utf-8")
    if MARKER not in current:
        path.write_text(block.rstrip() + "\n\n" + current, encoding="utf-8")


def append_once(path: Path, marker: str, block: str) -> None:
    current = path.read_text(encoding="utf-8")
    if marker not in current:
        path.write_text(current.rstrip() + "\n\n" + block.rstrip() + "\n", encoding="utf-8")


def append_jsonl(path: Path, record: dict) -> None:
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    for line in current.splitlines():
        try:
            if json.loads(line).get("id") == record["id"]:
                return
        except (json.JSONDecodeError, AttributeError):
            pass
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    if Path.cwd().resolve() != ROOT.resolve():
        raise SystemExit(f"run from {ROOT}")

    for name in [
        "final_project_summary.md",
        "final_descriptive_statistics.md",
        "final_claims_audit.md",
        "final_consistency_audit.md",
    ]:
        path = DOCS / name
        text = path.read_text(encoding="utf-8").replace(OLD, NEW)
        path.write_text(text, encoding="utf-8")

    summary = DOCS / "final_project_summary.md"
    report_block = """## Final report-writing package

- `docs/final_report_draft_en.md`
- `docs/final_report_draft_zh.md`
- `docs/final_citation_audit.md`
- `docs/final_appendix_plan.md`
- `docs/final_report_reduction_plan.md`

The English draft is approximately 4,007 words. Fifteen literature-dependent claims were audited. No experiment, checkpoint evaluation, benchmark generation, or numerical result changed during report assembly."""
    append_once(summary, "## Final report-writing package", report_block)

    status = """<!-- FINAL_REPORT_DRAFT_STATUS_20261006:BEGIN -->
## Current report-writing status — 2026-10-06

**EXPERIMENTS COMPLETE / FINAL ANALYSIS COMPLETE / REPORT WRITING ONLY**

The English and Chinese report drafts, citation audit, appendix plan, and reduction plan are complete. All experiments, benchmarks, checkpoints, aggregate results, and numerical claims remain frozen. No further experimental work is authorized.
<!-- FINAL_REPORT_DRAFT_STATUS_20261006:END -->"""
    for name in [
        "replay_readiness_review.md",
        "two_week_experiment_plan.md",
        "open_questions.md",
        "project_completion_options.md",
        "assumptions_and_risks.md",
    ]:
        prepend_once(DOCS / name, status)

    decision = """<!-- FINAL_REPORT_D055:BEGIN -->
## D055 — Assemble the final course-project report draft (2026-10-06)

Accept the frozen experimental and final-analysis corpus as the sole quantitative source. Produce matched English and Chinese academic drafts, audit every literature-dependent claim against reviewed project sources, and move operational detail to a planned appendix. Status becomes **EXPERIMENTS COMPLETE / FINAL ANALYSIS COMPLETE / REPORT WRITING ONLY**. No new experiment, method, benchmark, seed, checkpoint evaluation, or numerical change is authorized.
<!-- FINAL_REPORT_D055:END -->"""
    append_once(DOCS / "decision_log.md", "FINAL_REPORT_D055:BEGIN", decision)

    change = """<!-- FINAL_REPORT_C041:BEGIN -->
## C041 — Final bilingual report-writing package (2026-10-06)

Created `final_report_draft_en.md`, `final_report_draft_zh.md`, `final_citation_audit.md`, `final_appendix_plan.md`, and `final_report_reduction_plan.md`. The English draft is approximately 4,007 words, embeds six finalized figures, includes the five-method primary table and compact reproducibility settings, and preserves eight reviewed reference entries. Fifteen citation claims and all primary frozen numbers were validated. No experiment was run.
<!-- FINAL_REPORT_C041:END -->"""
    append_once(DOCS / "change_log.md", "FINAL_REPORT_C041:BEGIN", change)

    history = """<!-- FINAL_REPORT_HISTORY_20261006:BEGIN -->
## Final report assembly — 2026-10-06

The bilingual course-project report draft and its citation/appendix/reduction support documents were assembled from frozen final artifacts. Project status is **EXPERIMENTS COMPLETE / FINAL ANALYSIS COMPLETE / REPORT WRITING ONLY**. This is a writing milestone and adds no experimental evidence.
<!-- FINAL_REPORT_HISTORY_20261006:END -->"""
    append_once(DOCS / "experiment_history.md", "FINAL_REPORT_HISTORY_20261006:BEGIN", history)

    date = "2026-10-06T18:00:00+08:00"
    append_jsonl(HISTORY / "decisions.jsonl", {
        "date": date,
        "id": "D055",
        "decision": "Assemble the final bilingual course-project report from frozen artifacts only",
        "project_status": NEW,
        "new_experiments": 0,
        "numerical_changes": 0,
    })
    append_jsonl(HISTORY / "changes.jsonl", {
        "date": date,
        "id": "C041",
        "decision_id": "D055",
        "reason": "Create final English/Chinese report drafts and writing-support audits",
        "files_changed": [
            "scripts/assemble_final_report.py",
            "docs/final_report_draft_en.md",
            "docs/final_report_draft_zh.md",
            "docs/final_citation_audit.md",
            "docs/final_appendix_plan.md",
            "docs/final_report_reduction_plan.md",
            "results/final_tables/final_report_metadata.json",
            "current status/history documents",
        ],
        "validation": [
            "English approximate word count 4007",
            "15 literature claims audited",
            "all primary values matched frozen master table",
            "UTF-8/control-character PASS",
            "0 new experiments",
        ],
    })
    print(json.dumps({"status": "PASS", "decision": "D055", "change": "C041", "project_status": NEW}, indent=2))


if __name__ == "__main__":
    main()
