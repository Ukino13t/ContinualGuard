"""Report user-confirmed review evidence and the unchanged training audit stop."""
import json,sys
from datetime import datetime,timezone
from pathlib import Path
from h2_mini_b0_review_common import *
MARKER="H2_MINI_B0_CONFIRMED_AUDIT_STOP_20261004"
def prepend(name,body):
    p=ROOT/"docs"/name;s=p.read_text(encoding="utf-8");assert MARKER not in s
    title,rest=s.split("\n",1)
    p.write_text(title+f"\n\n<!-- {MARKER}:BEGIN -->\n"+body+f"\n<!-- {MARKER}:END -->\n"+rest,encoding="utf-8")
def append(path,event):
    old=path.read_text(encoding="utf-8")
    assert not any(json.loads(line)["id"]==event["id"] for line in old.splitlines() if line.strip())
    with path.open("a",encoding="utf-8",newline="\n") as f:f.write(json.dumps(event,ensure_ascii=False)+"\n")
def main():
    rp=ROOT/"results/h2_mini_b0_replenishment_analysis.json";r=load(rp)
    assert r["status"]=="STOP_TRAINING_AUDIT_DISAGREEMENT" and r["completed_judgments"]==1219
    canp=ROOT/"review/b0_replenishment_canonical_manifest.json";can=load(canp)
    assert r["review_provenance_gate"]["status"]=="RESOLVED_BY_CURRENT_USER_CONFIRMATION"
    for f in can["files"].values():
        p=ROOT/f["source_file"];assert file_sha(p)==f["source_sha256"] and p.stat().st_mtime_ns==f["source_mtime_ns"]
        assert file_sha(ROOT/f["canonical_file"])==f["canonical_sha256"]
    prior=load(ROOT/"data/audits/hard_benign/b0_replenishment_pre_review_audit.json")
    for p,e in prior["historical_evidence_snapshot"].items():
        path=ROOT/p;assert file_sha(path)==e["sha256"] and path.stat().st_mtime_ns==e["mtime_ns"]
    now=datetime.now(timezone.utc).isoformat()
    head="""## Current result after human clarification — 2026-10-04

**STOP_TRAINING_AUDIT_DISAGREEMENT.** The human confirmed that A's609 judgments were manually reviewed and the preliminary-AI notes were accidentally retained. The human also reported a mean25 seconds per judgment. These clarifications resolve the preceding provenance/timing reporting blockers for this submission; they do not relax the frozen review-agreement gate.

The [hash-bound human confirmation](../review/b0_replenishment_human_confirmation.json) covers the exact14 submitted files and assignment manifest. All1219 label judgments are complete under this clarification (A609/B610;600 primary,75 independent seconds,544 evaluation-double judgments). Original decisions, B0 confirmations, reason codes, confidence, notes and timing cells remain unchanged. UTF-8 copies and source/canonical hashes are recorded in the [canonical manifest](../review/b0_replenishment_canonical_manifest.json).

25s is a **human-reported aggregate mean**, not1219 measured row times. Individual review_seconds cells remain blank. Derived total time is approximately **30475 seconds =8.46528 person-hours**. Median, p90 and reviewer-specific timing are unavailable from that mean; none was fabricated. This is an explicit post-review reporting clarification, recorded before H2-mini training.

**The current gate failure is B0-confirmation disagreement in training audits:8/75 =10.6667%, above the frozen10% threshold.** Decision disagreement is2/75=2.6667%. The two criteria are reported separately. The excess is0.6667 percentage points; with75 pairs,8 disagreements cross the threshold whereas7 would not. This is a protocol stop, not a claim of broad label unreliability or continual forgetting.

Section4 of the approved workflow requires stopping here. No old/new accepted pool merge, post-review grouping audit, final capacity certification, final B0/BH/B-short assignment or matched-control/final leakage/length/count validation has been performed.
"""
    detailed=head+"""
## Completed review statistics

| Metric | Result |
| --- | ---: |
| Expected / complete judgments under current clarification |1219 /1219 |
| Reviewer A / B judgment counts |609 /610 |
| Missing label/confirmation/confidence/required rejection-reason fields |0 |
| Malformed rows / source-text or immutable-metadata errors |0 |
| Individual timing observations |0; aggregate25s only |
| Training primary candidates |600 |
| Primary ACCEPT_BENIGN |578/600 =96.3333% |
| Primary ACCEPT_BENIGN + B0 YES |550/600 =91.6667% |
| Primary rejected decisions |22/600 =3.6667% |
| Accepted benign but not confirmed B0 |28 |
| Total primary records not B0-eligible |50/600 =8.3333% |

| Independent paired metric | Training seconds (75 records) | Evaluation doubles (272 records) |
| --- | ---: | ---: |
| Decision agreements |73/75 =97.3333% |269/272 =98.8971% |
| Decision disagreements |2/75 =2.6667% |3/272 =1.1029% |
| Decision Cohen kappa |0.655963 |0.876401 |
| B0-confirmation agreements |67/75 =89.3333% |259/272 =95.2206% |
| B0-confirmation disagreements |8/75 =10.6667% |13/272 =4.7794% |
| B0-confirmation Cohen kappa |0.383350 |0.690232 |
| Both ACCEPT_BENIGN |71 |258 |
| Both B0 YES |65 |245 |
| Both ACCEPT_BENIGN and B0 YES |65 |245 |
| ACCEPT_BENIGN versus REJECT_TRUE_INJECTION conflicts |0 |0 |

The8 training B0-confirmation disagreements include the2 decision disagreements. The13 evaluation-confirmation disagreements include the3 decision disagreements. These overlap and must not be added as distinct records. The [paired-disagreement report](../results/h2_mini_b0_replenishment_disagreements.json) lists exact review IDs, fields and reviewer evidence without raw text, free-form notes or adjudications.

Across1219 judgments there are1170 ACCEPT_BENIGN and49 rejected decisions. Rejection counts: ambiguous intent6, true injection0, harmful/unsafe18, missing context19, privacy/rights3, quality3. Recorded reason codes match these categories; no other reason code occurs. These are judgment counts, not deduplicated record counts.

## Conditional label-only capacity; not a final accepted pool

Before a fresh group audit and conditional on resolving the training-audit gate, individually concordant label evidence would contribute792 new records:547 training candidates (65 audited) plus245 evaluation-double candidates. This does not certify the primary-only training pool while its audit gate is paused.

| Review lane | Existing frozen pool | New label-qualified candidates | Conditional total | Target | At least this shortage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training |234 |547 |781 |800 |19 |
| Independently audited training, included above |20 |65 |85 |80 |0 |
| Double-reviewed validation/test reserve |105 |245 |350 |360 |10 |
| Unique total |339 |792 |1131 |1160 |29 |

85 audits are a subset of781 training candidates, not additional records. The same350 evaluation candidates cannot fill120 validation plus240 test simultaneously. Individual marginal capacity flags do not establish joint feasibility. Four per-stage training audits of at least20 have not been assigned or certified.

These figures are an upper bound before post-review exclusions, not final group-safe counts. Even that bound cannot meet1160; a future group audit can only reduce capacity. No supplementation was automatically prepared or conducted. The original accepted B0 pool339 remains unchanged.

## Status of requested final outputs

| Requested output / decision | Status |
| --- | --- |
| Final group-safe B0 /800-120-240-1160 certification |Not reached: training-audit gate stops first. Conditional minimum deficit29 reported separately. |
| BH evidence |Frozen999:691 training including68 audits,104 validation,204 test. No600/90/180 concrete allocation. |
| Historical B1/B2/B3 |Descriptive only; not restored as mandatory labels or ordinal difficulty. |
| B-short Option B |Approved for later120 at42 English/78 multilingual; no final manifest. Unmatched descriptive external short-form benign control; aggregate NotInject minus B-short FPR is not a pure trigger effect. |
| H2m-C0 / H2m-F proposals |Not created. |
| Shared malicious assignment / matched-control check |Not created or run. |
| Final leakage/group/contamination/length/count validation |Not run; no final records selected and no PASS claimed. |
| Materialization approval |Not ready. |
| No-replay training proposal |Only after all review, capacity, assignment and materialization gates and human approvals. |
| Replay readiness |R0 active;R1/R2 not entered. No numerical memory budget. |

## Remaining human decision

Review the8 training B0-confirmation disagreements and decide the next review action under the unchanged rubric and threshold. The existing protocol pauses affected training evidence and calls for expanded independent review to at least30% when its disagreement gate fails. For600 replenishment training candidates,30% is180 independent second reviews;75 are present, so an extension would involve105 additional judgments if that scope is approved. This is a planning count, not an authorized or executed package.

Any adjudication must be explicit, preserve both original judgments and document its rationale. Do not remove disagreements from the denominator to manufacture a pass or reinterpret B0 NO/UNSURE as YES. Do not change the ordinary-B0 definition or10% threshold automatically. Resolve this review gate before finalizing a top-up size; conditional shortage is at least19 training plus10 double-reviewed evaluation records and may grow after grouping or review.

Recommended next prompt:

> Review the8 training B0-confirmation disagreements and prepare an independent-review expansion plan under the frozen ordinary-B0 rubric and10% gate, including coverage to at least30% of the600 training candidates and the conditional19-training/10-double-evaluation shortage. Preserve original evidence and my A-review/aggregate25s clarification. Prepare a plan for human review only; do not infer judgments, change the threshold, select final IDs, materialize research JSONL, train or implement replay.

## Reproducible analysis

~~~powershell
Set-Location E:\\ContinualGuard
& .\\.venv\\Scripts\\python.exe .\\scripts\\analyze_h2_mini_b0_replenishment.py --human-confirmation .\\review\\b0_replenishment_human_confirmation.json --output .\\results\\h2_mini_b0_replenishment_recheck.json
~~~

The explicit confirmation is bound to all original submitted hashes. Later file edits invalidate it and require renewed provenance handling. The script does not infer missing label fields or change CSVs. Without this explicit confirmation, it continues to flag the historical blank timing and preliminary-review notes. The command correctly exits nonzero at the current training-audit gate.

21 synthetic tests passed: exact-file confirmation binding, no timing imputation, missing-label protection and the8/75 stop after timing clarification. Canonical cell strings match originals exactly. All14 submitted files and92 protected prior-review/BH files retain hashes and modification times. The preceding blocked analysis is preserved in results/h2_mini_b0_review_history/20261004-v1_before_human_confirmation_analysis.json.

All operations use E:\\ContinualGuard and E:\\ContinualGuard\\.venv\\Scripts\\python.exe. The archival project was untouched. No storage audit/cleanup, raw-text research materialization, model training or replay occurred.
"""
    prepend("h2_mini_b0_replenishment_results.md",detailed)
    status=head+"""
See the [full review results and conditional capacity](h2_mini_b0_replenishment_results.md) and [paired disagreements](../results/h2_mini_b0_replenishment_disagreements.json). Earlier timing/provenance stop sections below are historical and superseded by the current human clarification; the unresolved issue is now the frozen training B0-confirmation gate.

The approved nonordinal B0/BH design, shared Stage-1 B0 anchor, matched malicious halves and B-short Option B remain unchanged. BH suffixes are partition indices, not difficulty levels. Future model/revision/initialization, data/order, seed, epoch, learning-rate, optimizer-reset and evaluation controls stay fixed. No numerical replay budget was chosen. R0 remains active;R1/R2 have not been entered.
"""
    docs=["h2_mini_taxonomy_amendment.md","h2_mini_simplified_design.md","h2_mini_design.md","h2_mini_final_materialization_plan.md",
      "h2_mini_no_replay_protocol.md","replay_comparison_protocol.md","replay_readiness_review.md","assumptions_and_risks.md","open_questions.md"]
    for name in docs:
        extra=""
        if name=="open_questions.md":
            extra="\n## Current human decision\n\nA-review provenance is resolved by human confirmation; aggregate timing is reported without row-level imputation. Decide the independent-review expansion/adjudication plan for8/75 B0-confirmation disagreements under the frozen10% gate, then re-evaluate the conditional shortage of at least29. No automatic expansion, supplementation, assignment or threshold change is authorized.\n"
        if name=="assumptions_and_risks.md":
            extra="\n8.46528 person-hours derive from the human-reported25s mean; per-reviewer duration, timing precision, median and p90 cannot be recovered. Manual review is confirmed by human attestation, not machine proof.10.6667% B0-confirmation disagreement is a small-sample protocol stop, distinct from2.6667% decision disagreement. Conditional1131 capacity is not a final group-safe result.\n"
        prepend(name,status+extra)
    decision={"date":now,"id":"D037","decision":"Accept A manual-review confirmation and aggregate25s timing; preserve original evidence and stop at unchanged training B0-confirmation gate",
      "human_confirmation":"review/b0_replenishment_human_confirmation.json","human_confirmation_sha256":file_sha(ROOT/"review/b0_replenishment_human_confirmation.json"),
      "reviewer_A_manual_review_confirmed":True,"stale_notes_preserved":True,
      "timing":{"source":"human-reported aggregate mean","mean_seconds":25,"judgments":1219,"derived_person_hours":1219*25/3600,"per_record_imputation":False,"median":None,"p90":None},
      "review_complete_under_clarification":True,"threshold_unchanged":.10,"training_B0_disagreements":8,"training_second_pairs":75,
      "training_B0_disagreement_rate":8/75,"status":"STOP_TRAINING_AUDIT_DISAGREEMENT",
      "conditional_label_only_counts":{"training":781,"audited_training":85,"double_evaluation":350,"unique":1131},
      "conditional_minimum_shortages":{"training":19,"double_evaluation":10,"unique":29},
      "final_group_safe_capacity_certified":False,"post_review_audit_run":False,"final_assignment":False,
      "materialization":False,"training":False,"replay":False,"numerical_memory_budget":None,"replay_readiness":"R0",
      "supersedes":"D035/D036 provenance/timing blockers only; scientific review and capacity gates unchanged"}
    files=["scripts/analyze_h2_mini_b0_replenishment.py","scripts/freeze_h2_mini_b0_confirmed_review.py","scripts/report_h2_mini_b0_confirmed_results.py",
      "tests/test_h2_mini_b0_replenishment.py",".gitignore","review/b0_replenishment_human_confirmation.json","review/b0_replenishment_canonical_manifest.json",
      "review/b0_replenishment_canonical/20261004-v1/*.csv (local ignored)","results/h2_mini_b0_review_history/20261004-v1_before_human_confirmation_analysis.json",
      "results/h2_mini_b0_replenishment_analysis.json","results/h2_mini_b0_replenishment_disagreements.json",
      *["docs/"+name for name in ["h2_mini_b0_replenishment_results.md",*docs,"decision_log.md","change_log.md"]],
      "history/decisions.jsonl","history/changes.jsonl"]
    change={"date":now,"id":"C024","reason":"Human supplies average timing and confirms A's actual manual review",
      "files_changed":files,"effect":"Hash-bound clarification, exact-value canonical evidence, complete label analysis and unchanged audit-gate stop; no imputation/adjudication",
      "tests_run":["21 synthetic tests passed","14 canonical CSVs preserve original cells and immutable source hashes",
        "14 original hashes/mtimes and92 historical evidence hashes/mtimes unchanged","8 training and13 evaluation paired disagreements reported without raw text/notes or adjudication"],
      "remaining":"Human decision on training-audit expansion/adjudication; conditional shortage remains before grouping; no final assignments"}
    prepend("decision_log.md","## D037 — Clarification accepted; training audit paused\n\n~~~json\n"+json.dumps(decision,ensure_ascii=False,indent=2)+"\n~~~\n")
    prepend("change_log.md","## C024 — Canonical clarified review and gate results\n\n~~~json\n"+json.dumps(change,ensure_ascii=False,indent=2)+"\n~~~\n")
    append(ROOT/"history/decisions.jsonl",decision);append(ROOT/"history/changes.jsonl",change)
    print(json.dumps({"status":r["status"],"docs_updated":12,"decision":"D037","change":"C024","canonical_manifest":rel(canp),
      "estimated_hours_from_human_mean":1219*25/3600,"training_B0_disagreement":8/75,
      "conditional_minimum_shortage":29,"post_review_group_audit_performed":False},indent=2))
if __name__=="__main__":main()
