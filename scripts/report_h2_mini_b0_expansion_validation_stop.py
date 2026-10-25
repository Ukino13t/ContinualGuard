"""Record an expansion submission-integrity stop; never repair or analyze labels."""
import csv, io, json, sys
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from h2_mini_b0_expansion_common import *
import analyze_h2_mini_b0_expansion as analyzer

MARKER = "H2_MINI_B0_EXPANSION_INTEGRITY_STOP_20261004"
REPORT = ROOT / "results/h2_mini_b0_expansion_analysis.json"
DOC = ROOT / "docs/h2_mini_b0_expansion_results.md"
OTHER_DOCS = [
    "h2_mini_b0_replenishment_results.md", "h2_mini_simplified_design.md",
    "h2_mini_design.md", "h2_mini_final_materialization_plan.md",
    "h2_mini_no_replay_protocol.md", "replay_comparison_protocol.md",
    "replay_readiness_review.md", "assumptions_and_risks.md", "open_questions.md",
    "decision_log.md", "change_log.md",
]

def snapshot(paths):
    return {str(p.relative_to(ROOT)).replace("\\", "/"):
            {"sha256": file_sha(p), "mtime_ns": p.stat().st_mtime_ns} for p in paths}

def historical_check(m):
    failures = []
    unchanged = 0
    amendment = m["authorized_historical_document_update"]
    for rel, expected in m["historical_evidence_snapshot"].items():
        if rel == amendment["path"]:
            p = ROOT / amendment["preserve_prior_copy_at"]
            if file_sha(p) != expected["sha256"]: failures.append(rel)
        else:
            p = ROOT / rel
            if file_sha(p) != expected["sha256"] or p.stat().st_mtime_ns != expected["mtime_ns"]:
                failures.append(rel)
            else: unchanged += 1
    if failures: raise ValueError(f"Historical evidence changed: {failures}")
    return {"unchanged_hash_and_mtime_files": unchanged,
            "authorized_prior_protocol_copy_sha256_verified": True, "failures": failures}

def append_event(path, event):
    old = path.read_bytes()
    records = [json.loads(line) for line in old.decode("utf-8").splitlines() if line.strip()]
    assert event["id"] not in {r["id"] for r in records}
    with path.open("ab") as f:
        if old and not old.endswith(b"\n"): f.write(b"\n")
        f.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
    assert path.read_bytes().startswith(old)

def main():
    assert Path.cwd().resolve() == ROOT.resolve()
    assert Path(sys.executable).resolve() == (ROOT / ".venv/Scripts/python.exe").resolve()
    assert not REPORT.exists() and not DOC.exists(), "Preserve prior reports; do not overwrite."
    for name in OTHER_DOCS:
        assert MARKER not in (ROOT / "docs" / name).read_text(encoding="utf-8")
    for name, event_id in [("decisions", "D039"), ("changes", "C026")]:
        assert not any(json.loads(x)["id"] == event_id for x in
                       (ROOT / "history" / (name + ".jsonl")).read_text(encoding="utf-8").splitlines() if x.strip())
    data = analyzer.inspect_expansion()
    assert data["errors"], "This reporter is only for a submission-integrity stop."
    m = data["manifest"]
    historical = historical_check(m)
    paths = [ROOT / rel for rel in m["historical_evidence_snapshot"]]
    paths += [EXP_MANIFEST, *sorted(EXP_DIR.glob("*.csv")),
              ROOT / m["authorized_historical_document_update"]["preserve_prior_copy_at"]]
    before = snapshot(paths)
    filename = "reviewer_A_batch_01.csv"
    raw = (EXP_DIR / filename).read_bytes()
    try:
        raw.decode("utf-8-sig", errors="strict")
        raise AssertionError("Expected the observed UTF-8 failure")
    except UnicodeDecodeError as exc:
        offset = exc.start + (3 if raw.startswith(b"\xef\xbb\xbf") else 0)
        encoding_failure = {"file": "review/b0_replenishment_expansion/" + filename,
                            "byte_offset_zero_based": offset, "byte_hex": hex(raw[offset]),
                            "review_id": "H2B0-8F8372AC03556AB5", "field": "text"}
    decoded = raw.decode("gb18030", errors="strict")
    assert decoded == raw.decode("cp936", errors="strict")
    assert decoded.encode("gb18030") == raw and decoded.encode("cp936") == raw
    rows = list(csv.DictReader(io.StringIO(decoded, newline=""), strict=True))
    expected = {r["review_id"]: r for r in m["records"] if r["second_reviewer"] == "A"}
    assert len(rows) == 53 and Counter(r["review_id"] for r in rows) == Counter(expected.keys())
    records = data["records"]
    cache = {}
    mismatches = []
    for row in rows:
        rid = row["review_id"]; r = records[rid]
        if sha(row["text"]) == r["raw_text_sha256"]: continue
        assignment = next(a for a in r["reviewer_assignments"] if a["review_role"] == "PRIMARY")
        fn = assignment["batch_filename"]
        if fn not in cache:
            with (ROOT / "review/b0_replenishment" / fn).open(encoding="utf-8-sig", newline="") as f:
                cache[fn] = {x["review_id"]: x for x in csv.DictReader(f)}
        intact = cache[fn][rid]["text"]
        assert sha(intact) == r["raw_text_sha256"]
        operations = []
        for tag, a, b, c, d in SequenceMatcher(None, intact, row["text"], autojunk=False).get_opcodes():
            if tag == "equal": continue
            operations.append({"operation": tag, "original_character_count": b-a,
                               "submitted_character_count": d-c,
                               "submitted_all_question_marks": bool(row["text"][c:d]) and set(row["text"][c:d]) == {"?"}})
        assert all(x["operation"] == "replace" and x["submitted_all_question_marks"] for x in operations)
        mismatches.append({"file": encoding_failure["file"], "review_id": rid, "field": "text",
                           "language": r["language"], "expected_sha256": r["raw_text_sha256"],
                           "observed_sha256_under_both_diagnostic_codecs": sha(row["text"]),
                           "difference_operations": operations})
    assert len(mismatches) == 17
    assert sum(s["status"] == "completed" for s in data["states"].values()) == 52
    assert all(s["where"]["reviewer"] == "B" for s in data["states"].values())
    report = analyzer.analyze_expansion(data)
    assert report["conditional_label_only_capacity"] is None
    now = datetime.now(timezone.utc).isoformat()
    prompt = (
        "Proceed with A-submission evidence recovery only. Preserve the current submitted files and hashes. "
        "Use an intact completed-review workbook or original export to provide a separate UTF-8 submission "
        "with all 53 assigned IDs and unchanged human judgments. If no intact reviewed copy exists, stop and "
        "prepare a separate original-text verification packet for Reviewer A covering the 17 IDs in the "
        "integrity report; require explicit human reconfirmation before accepting those judgments. Do not "
        "silently replace damaged text, inherit disputed judgments, adjudicate, or calculate the 180-pair gate. "
        "After recovery, rerun validation and stop for review of the integrity result."
    )
    report.update({
        "generated_at": now, "status": "STOP_EXPANSION_SUBMISSION_INTEGRITY",
        "expected_new_judgments": 105, "human_reported_complete": 105,
        "validated_new_judgments": 52, "reviewer_validated": {"A": 0, "B": 52},
        "reviewer_A_status": "53 rows present under identical strict GB18030/CP936 diagnostic decoding; entire submission blocked",
        "reviewer_A_rows_with_matching_source_hash_under_diagnostic_decoding": 36,
        "reviewer_A_source_hash_mismatches": 17,
        "reviewer_A_all_differences_replace_original_characters_with_question_marks": True,
        "encoding_failure": encoding_failure,
        "diagnostic_decoding_policy": "Read-only diagnosis, not accepted canonicalization or evidence repair; exact original encoding is not asserted.",
        "source_text_mismatches": mismatches,
        "source_mismatch_language_counts": dict(sorted(Counter(x["language"] for x in mismatches).items())),
        "raw_inspector_missing_row_errors_are_cascades_of_unreadable_file": True,
        "judgments_not_claimed_absent": True, "worksheet_sha256": data["hashes"],
        "transport_integrity": data["transport"],
        "expanded_audit": {"expected_full_denominator": 180, "old_pairs_preserved": 75,
                           "required_new_pairs": 105, "all180_complete_and_validated": False,
                           "decision_metrics": None, "B0_confirmation_metrics": None,
                           "safety_critical_conflicts": None, "gate_threshold": 0.1,
                           "B0_confirmation_gate_pass": None, "gate_status": "NOT_RUN_INTEGRITY_BLOCKED"},
        "canonical_manifest_created": False, "canonical_copies_created": False,
        "final_capacity": None, "final_grouping_audit": "NOT_RUN",
        "BH_historical_capacity": {"total": 999, "train": 691, "train_audited": 68, "validation": 104, "test": 204,
                                    "required_allocation": 870, "new_allocation_run": False},
        "B_short": {"policy": "Option B unmatched descriptive external control", "target": 120,
                    "target_english": 42, "target_multilingual": 78, "final_manifest_created": False},
        "reserve": {"training": 40, "evaluation": 25, "review_authorized": False, "worksheets_created": False},
        "C0_proposal": "NOT_CREATED", "F_proposal": "NOT_CREATED", "shared_malicious_manifest": "NOT_CREATED",
        "matched_control_verification": "NOT_RUN", "final_token_length_validation": "NOT_RUN",
        "final_exact_count_validation": "NOT_RUN", "final_assignment_validation": "NOT_RUN_BLOCKED",
        "final_pool_merge_performed": False, "materialization": False, "training": False, "replay": False,
        "numerical_memory_budget": None, "readiness": {"R0": "active", "R1": "not_entered", "R2": "not_entered"},
        "historical_evidence_verification": historical,
        "protected_files_before": before, "original_submissions_modified": False, "archive_accessed": False,
        "project_root": str(ROOT), "python": sys.executable, "storage_audit_or_cleanup": False,
        "next_step": prompt, "recommended_next_prompt": prompt,
    })
    assert snapshot(paths) == before
    report["protected_files_unchanged_during_validation"] = True
    save(REPORT, report)
    table = "\n".join("| " + x["review_id"] + " | " + x["language"] + " | text |" for x in mismatches)
    body = f"""# H2-mini B0 expanded-audit results

## Submission-integrity stop — {now}

**STOP_EXPANSION_SUBMISSION_INTEGRITY.** User-reported completion is 105 judgments.
Only B's 52 have passed submission validation; A's 53-row file is blocked.
This is a transport/source-integrity failure, not a failed disagreement gate or evidence of an unstable B0 definition.

Active root: E:\\ContinualGuard. Interpreter: E:\\ContinualGuard\\.venv\\Scripts\\python.exe.
The archived project was not accessed. No storage audit or cleanup ran.

A: review/b0_replenishment_expansion/reviewer_A_batch_01.csv.
SHA-256: {data['hashes'][filename]}.
The first UTF-8 error is at zero-based byte offset {offset}, field text,
review_id H2B0-8F8372AC03556AB5.
Strict GB18030 and CP936 decoding produce the same 53 rows and round-trip the submitted bytes.
Nevertheless, only 36 source-text hashes match; **17 do not**. Every differing span contains
literal question marks in place of original characters, including Polish and Thai letters.
Merely exporting this damaged file as UTF-8 cannot restore them.
The original encoding and time of corruption are not established; no claim is made about what A saw during review.

B: review/b0_replenishment_expansion/reviewer_B_batch_01.csv.
SHA-256: {data['hashes']['reviewer_B_batch_01.csv']}.
All 52 rows pass UTF-8, assignments, opposite-reviewer roles, immutable text/metadata hashes,
required decision/B0/confidence fields and required rejection reasons.
Both files have the permitted optional timing column; no values were inferred or rewritten.

## Exact affected records

Every row below belongs to the A file named above. Hashes and diagnostic character-change counts
are in the [machine-readable report](../results/h2_mini_b0_expansion_analysis.json).
The analyzer's 53 missing-row messages are cascading parse failures, not evidence that human judgments are absent.

| review_id | Language | Field |
| --- | --- | --- |
{table}

## Requested A–AB status

| Items | Status |
| --- | --- |
| A: new reviews | 105 reported complete; B52 validated, A53 blocked |
| B: training audit pairs | Intended 180 = old75 + new105; complete validated 180-pair analysis not run |
| C–F: decision/B0 agreement, disagreement, safety conflicts | Not computed; unknown, not zero |
| G: <=10% gate | NOT_RUN_INTEGRITY_BLOCKED; neither statistical PASS nor FAIL |
| H–M: final B0 capacities, quota pass, minimum top-up | Not certified or recomputed; earlier conditional counts are not final capacity |
| N: BH | Preserved historical group-safe999: train691 (audited68), validation104, test204; required600/90/180,870 total; no allocation run |
| O: B-short | Option B preserved; final120 target42 English/78 multilingual; no final manifest |
| P–R: C0/F/shared-malicious proposals | Not created |
| S–V: matched control, leakage/grouping, final token lengths/counts | Not run |
| W: final assignment validation | NOT_RUN_BLOCKED; not ready for human assignment approval |
| X: replay readiness | R0 active; R1/R2 not entered |
| Y: approve materialization now | No |
| Z: training after materialization | Remains conditional on evidence recovery, audit/capacity/assignment approval, materialization and validation, then human training authorization |
| AA: remaining risks | Corrupted submitted source text; unknown corruption timing; future review/capacity/grouping gates unresolved |
| AB: next action | Recover valid review evidence as specified below |

The original75 audited pairs (including all eight previous B0 disagreements), all primary/evaluation
judgments, human notes, confidence and labels remain unchanged. The previous manual-review
confirmation and aggregate25s timing clarification remain valid for their hash-bound original files.
No per-row timing is invented, and that historical clarification does not establish the integrity of this new A file.

127 protected historical files retain their hashes and modification times. The one previously
authorized protocol-document change retains its exact earlier bytes in the historical copy.
Original expansion submissions are unchanged. No canonical expansion manifest or accepted
canonical copy was created, because the prerequisite validation has failed.
No accepted-pool merge, final capacity audit, final assignments, materialization planning advance,
raw-text research JSONL, training, replay, reserve worksheets or memory budget was produced.

Taxonomy v2, historical B1/B2/B3 metadata, BH evidence, B-short policy, labels0/1 and future
no-replay controls remain unchanged: pinned DeBERTa-v3-base revision
8ccc9b6f36199bec6961081d44eb72fb3f7353f3, max_length1280, seeds17/42/2026,
current-stage-only training, optimizer reset between stages, carried model weights.
No test/NotInject/B-short tuning. Future replay comparisons retain identical detector/revision,
initialization, data/order, seeds, epochs, learning rate, optimizer-reset/evaluation policy and
a common as-yet-unspecified memory budget; only replay strategy may vary.

## Required human recovery and exact next prompt

Recover an intact completed-review workbook/export if one exists, preserve the damaged submission,
and provide a separate UTF-8 export without modifying judgments. If no intact reviewed copy exists,
the 17 affected IDs need original-text verification and explicit human reconfirmation; simply restoring
text from the source cannot prove the reviewer saw that text. Do not drop these IDs or infer confirmations.

> {prompt}
"""
    DOC.write_text(body, encoding="utf-8")
    status = f"""## Current expanded-audit status — {now}

**STOP_EXPANSION_SUBMISSION_INTEGRITY.** A's new53-row submission fails UTF-8 validation;
read-only strict GB18030/CP936 diagnosis finds17 source-text hash mismatches with original
characters replaced by question marks. B's52 new judgments pass validation. User-reported
completion is105; validated completeness has not been established.

The full180-pair gate is **not run**, rather than failed or passed. Do not calculate on a partial
denominator. No B0 merge, post-review capacity certification, final assignment or materialization
planning advance is authorized past this stop. Current submissions and all prior75 pairs remain
unchanged. Prior manual-review/timing confirmation is preserved and is not reopened.

See [exact affected IDs, integrity evidence and recovery prompt](h2_mini_b0_expansion_results.md).
Recover an intact separate UTF-8 review export, or obtain human original-text reconfirmation for
the17 affected IDs before accepting repaired evidence. No automatic text/label repair or adjudication.
The40train+25eval reserve remains metadata-only; no reserve worksheets are authorized.

R0 active; R1/R2 not entered. Taxonomy v2, BH999, B-short Option B and all planned no-replay/future
comparison controls remain unchanged. No numerical replay budget, materialization, training,
replay or selective adaptation. This dated block supersedes earlier next-action instructions only;
earlier evidence and protocol decisions below remain historical.

"""
    for name in OTHER_DOCS:
        p = ROOT / "docs" / name
        original = p.read_bytes()
        first = original.find(b"\n")
        assert first >= 0
        inserted = (f"\n<!-- {MARKER}:BEGIN -->\n" + status + f"<!-- {MARKER}:END -->\n").encode("utf-8")
        p.write_bytes(original[:first+1] + inserted + original[first+1:])
    event = {"date": now, "id": "D039",
             "decision": "Stop expanded B0 workflow at submission-integrity validation; no statistical gate decision",
             "reason": "A CSV invalid UTF-8; strict identical GB18030/CP936 diagnostic decoding has17 immutable text hash mismatches",
             "validated_new_judgments": 52, "A_blocked_rows": 53, "B_validated_rows": 52,
             "report": str(REPORT.relative_to(ROOT)).replace("\\", "/"),
             "original75_preserved": True, "full180_gate_run": False, "canonical_freeze": False,
             "final_capacity_certified": False, "assignment": False, "materialization": False,
             "training": False, "replay": False, "reserve_worksheets": False,
             "numerical_memory_budget": None, "replay_readiness": "R0", "next_step": prompt}
    append_event(ROOT / "history/decisions.jsonl", event)
    changed = ["scripts/report_h2_mini_b0_expansion_validation_stop.py",
               "results/h2_mini_b0_expansion_analysis.json", "docs/h2_mini_b0_expansion_results.md",
               *["docs/" + x for x in OTHER_DOCS], "history/decisions.jsonl", "history/changes.jsonl"]
    append_event(ROOT / "history/changes.jsonl", {
        "date": now, "id": "C026", "reason": "Completed submissions fail immutable-text validation",
        "files_changed": changed, "decision_id": "D039",
        "validation": ["B52 exact source/assignment/required-field checks pass",
                       "A53 rows decoded for diagnosis only;17 text hashes differ, all changed spans are question marks",
                       "127 historical hashes/mtimes unchanged and authorized prior protocol bytes preserved",
                       "Submitted originals and all protected evidence unchanged after report/doc writes"],
        "raw_text_or_notes_in_report": False, "analysis_or_training_performed": False,
    })
    assert snapshot(paths) == before, "Protected evidence changed during report generation"
    for name in OTHER_DOCS:
        assert (ROOT / "docs" / name).read_text(encoding="utf-8").count(MARKER + ":BEGIN") == 1
    assert load(REPORT)["expanded_audit"]["B0_confirmation_gate_pass"] is None
    print(json.dumps({"status": report["status"], "validated_new": 52, "A_text_mismatches": 17,
                      "historical": historical, "originals_unchanged": True,
                      "documents_updated": 12, "decision": "D039", "change": "C026",
                      "report": str(REPORT)}, indent=2))

if __name__ == "__main__":
    main()

