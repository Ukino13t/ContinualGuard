"""Append the observed review-provenance clarification without changing submitted evidence."""
import json,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from h2_mini_b0_review_common import *
def main():
    analysis=load(ROOT/"results/h2_mini_b0_replenishment_analysis.json")
    gate=analysis["review_provenance_gate"];flags=gate["flags"]
    assert len(flags)==609 and {f["reviewer"] for f in flags}=={"A"}
    assert len({f["note_sha256"] for f in flags})==1
    audit=load(ROOT/"data/audits/hard_benign/b0_replenishment_pre_review_audit.json")
    stop_path=ROOT/"results/h2_mini_b0_replenishment_integrity_stop.json";stop=load(stop_path)
    def verify():
        for path,entry in {**stop["source_submission_and_frozen_metadata_snapshots"],**audit["historical_evidence_snapshot"]}.items():
            p=ROOT/path;assert file_sha(p)==entry["sha256"] and p.stat().st_mtime_ns==entry["mtime_ns"],path
    verify();now=datetime.now(timezone.utc).isoformat()
    next_prompt="Resolve the B0 review provenance and timing blockers first. I have supplied evidence clarifying whether Reviewer A independently completed all609 judgments despite the AI-preliminary-review notes, and authentic per-record timing for all1219 assignments (or an explicit human decision on timing that was not collected). Preserve all submitted evidence and do not silently reinterpret preliminary AI suggestions as human judgments. Re-check all review gates and stop on any unresolved issue. Only after the gates pass may the previously approved hash-only assignment workflow resume; no materialization, training, replay or numerical memory-budget selection."
    body="""## Additional current blocker — independent-human-review provenance (2026-10-04)

**Two unresolved blockers remain:**1219 missing review_seconds values, and609 conflicting review-status notes in Reviewer A's submissions.

All609 A optional_note fields explicitly describe preliminary AI suggestions and state that independent human review and real timing are still needed. Their contents share one SHA256, recorded in the metadata-only analysis. This is evidence of an unresolved provenance conflict with the claimed completed-human-review status; it does **not** by itself prove whether a person later reviewed the entries. No corresponding explicit preliminary-review marker was found in B's notes; that absence alone does not establish human independence.

Adding timing alone does not resolve A's status. The human must clarify whether all609 A judgments were independently reviewed. If they remain AI suggestions, complete the approved independent human review; do not relabel their provenance automatically. If humans did review them, document that fact and the retained preliminary notes with traceable evidence. Preserve originals and do not silently erase notes or invent timing.

The earlier timing-only report below remains a record of the first detected blocker. Its conditional next prompt is superseded by this provenance-aware action. The statement about populated1219 decisions describes nonempty fields, not verified human authorship. Canonical freezing, capacity/assignment, final leakage/length/count verification and materialization remain blocked. No finalized acceptance/agreement/safety-conflict result is claimed.

See [analysis JSON](../results/h2_mini_b0_replenishment_analysis.json) for609 exact filename/review_id/field provenance flags, and the [missing-field list](../results/h2_mini_b0_replenishment_missing_fields.csv) for1219 timing omissions. Actual person-hours remain unavailable.

**R0 active; R1/R2 not entered.** The approved nonordinal B0/BH taxonomy and B-short Option B remain unchanged. No final IDs, raw-text research JSONL, training, replay or numerical memory budget were produced.

Recommended next prompt, only after the human has actually supplied the stated evidence or explicit policy decision:

> """+next_prompt+"""
"""
    marker="H2_MINI_B0_PROVENANCE_STOP_20261004"
    def prepend(name,text):
        path=ROOT/"docs"/name;old=path.read_text(encoding="utf-8");assert marker not in old
        title,rest=old.split("\n",1)
        path.write_text(title+f"\n\n<!-- {marker}:BEGIN -->\n"+text+f"\n<!-- {marker}:END -->\n"+rest,encoding="utf-8")
    names=["h2_mini_b0_replenishment_results.md","h2_mini_taxonomy_amendment.md","h2_mini_simplified_design.md","h2_mini_design.md",
      "h2_mini_final_materialization_plan.md","h2_mini_no_replay_protocol.md","replay_comparison_protocol.md",
      "replay_readiness_review.md","assumptions_and_risks.md","open_questions.md"]
    for name in names:prepend(name,body)
    decision={"date":now,"id":"D036","decision":"Keep B0 finalization stopped for conflicting independent-human-review provenance in addition to missing timing",
      "evidence":{"reviewer":"A","flagged_judgments":609,"field":"optional_note",
        "status_marker":"Explicitly identifies AI preliminary suggestions and requests independent human review with real timing",
        "note_sha256":flags[0]["note_sha256"]},
      "interpretation":"This creates a provenance conflict; it does not establish whether a human later reviewed the decisions. Require human clarification and evidence.",
      "missing_review_seconds":1219,"human_confirmation_required":True,"timing_only_repair_sufficient":False,
      "supersedes":"D035 timing-only next-action wording; timing stop itself remains valid",
      "original_review_files_unchanged":True,"canonical_manifest_created":False,"capacity_finalized":False,
      "assignment_proposals_created":False,"training":False,"replay":False,"replay_readiness":"R0"}
    change={"date":now,"id":"C023","reason":"Notes inspected for possible recorded timing reveal609 explicit preliminary-AI-review markers",
      "files_changed":["scripts/analyze_h2_mini_b0_replenishment.py","tests/test_h2_mini_b0_replenishment.py",
        "scripts/report_h2_mini_b0_provenance_stop.py","results/h2_mini_b0_replenishment_analysis.json",
        "results/h2_mini_b0_replenishment_integrity_stop.json",*["docs/"+name for name in [*names,"decision_log.md","change_log.md"]],
        "history/decisions.jsonl","history/changes.jsonl"],
      "effect":"Add exact metadata-only provenance flags; keep finalization blocked even if timing is subsequently populated without resolving provenance",
      "tests_run":["17 synthetic tests passed, including full timing with preliminary-AI provenance and non-flagging of ordinary AI-topic notes",
        "Original14 submissions and3 frozen package metadata files plus92 historical files retain hashes and modification times"],
      "human_fields_changed":False,"scientific_results_finalized":False}
    for kind,event in [("decisions",decision),("changes",change)]:
        p=ROOT/"history"/(kind+".jsonl")
        assert not any(json.loads(line)["id"]==event["id"] for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
        with p.open("a",encoding="utf-8",newline="\n") as f:f.write(json.dumps(event,ensure_ascii=False)+"\n")
    prepend("decision_log.md","## D036 — Independent-review provenance unresolved\n\n~~~json\n"+json.dumps(decision,ensure_ascii=False,indent=2)+"\n~~~\n")
    prepend("change_log.md","## C023 — Explicit preliminary-review provenance checks\n\n~~~json\n"+json.dumps(change,ensure_ascii=False,indent=2)+"\n~~~\n")
    stop.update(status="STOP_INCOMPLETE_TIMING_AND_UNRESOLVED_HUMAN_PROVENANCE",updated_at=now,
      analysis_sha256=file_sha(ROOT/"results/h2_mini_b0_replenishment_analysis.json"),
      review_provenance_gate={"status":gate["status"],"blocking_flag_count":609,"reviewer":"A",
        "note_sha256":flags[0]["note_sha256"],"exact_flags_reference":"results/h2_mini_b0_replenishment_analysis.json#review_provenance_gate.flags",
        "actual_human_review_completion":"UNRESOLVED; note conflicts with submitted status","timing_only_fix_sufficient":False},
      tests_passed=17)
    save(stop_path,stop);verify()
    print(json.dumps({"status":stop["status"],"missing_timing":1219,"unresolved_provenance":609,"reviewer":"A",
      "originals_unchanged":True,"decision":"D036","change":"C023","tests_passed":17},indent=2))
if __name__=="__main__":main()
