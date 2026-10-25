"""Freeze user-confirmed review evidence without changing original cells or review gates."""
import csv,io,json,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from h2_mini_b0_review_common import *
import analyze_h2_mini_b0_replenishment as analyzer

VERSION="20261004-v1"
CONFIRM=ROOT/"review/b0_replenishment_human_confirmation.json"
CAN=ROOT/"review/b0_replenishment_canonical_manifest.json"
OUT=ROOT/"review/b0_replenishment_canonical"/VERSION
RESULT=ROOT/"results/h2_mini_b0_replenishment_analysis.json"
def main():
    if CONFIRM.exists() or CAN.exists() or OUT.exists():raise ValueError("Confirmed evidence already exists; refuse to overwrite.")
    d=analyzer.inspect();assert not d["errors"]
    assert len(d["states"])==1219 and d["state_counts"]=={"partial":1219}
    assert all(s.get("missing")==["review_seconds"] for s in d["states"].values())
    assert len(d["provenance_flags"])==609 and {f["reviewer"] for f in d["provenance_flags"]}=={"A"}
    m=d["manifest"];before={fn:{"sha256":file_sha(ROOT/"review/b0_replenishment"/fn),
      "mtime_ns":(ROOT/"review/b0_replenishment"/fn).stat().st_mtime_ns} for fn in m["batches"]}
    audit=load(ROOT/m["audit_path"])
    for path,e in audit["historical_evidence_snapshot"].items():
        p=ROOT/path;assert file_sha(p)==e["sha256"] and p.stat().st_mtime_ns==e["mtime_ns"]
    now=datetime.now(timezone.utc).isoformat()
    statement="review_seconds平均为25s，A的备注是忘记删去，实际为人工复核后的"
    confirmation={"version":"h2mini-b0-human-confirmation-v1-20261004","recorded_at":now,
      "human_statement_source":"CURRENT_USER_MESSAGE","human_statement":statement,"human_statement_sha256":sha(statement),
      "package_id":m["package_id"],"assignment_manifest_sha256":d["manifest_sha256"],
      "submitted_worksheet_sha256":d["hashes"],
      "reviewer_A_confirmation":{"manual_review_completed":True,"notes_are_stale":True,"reviewer":"A","judgments":609,
        "stale_note_sha256":sorted({f["note_sha256"] for f in d["provenance_flags"]}),
        "original_notes_retained":True,"interpretation":"The human confirms actual manual review after the preliminary suggestions. The earlier notes were accidentally retained. No machine proof of review behavior or retrospective rewriting is claimed."},
      "timing_clarification":{"evidence_type":"HUMAN_REPORTED_AGGREGATE_MEAN","mean_seconds_per_judgment":25,
        "judgments_in_scope":1219,"scope_interpretation":"Current completed targeted-B0 replenishment review, both reviewers combined",
        "estimated_total_seconds":30475,"estimated_total_person_hours":1219*25/3600,"per_record_imputation":False,
        "individual_review_seconds_observations":0,"median_seconds":None,"p90_seconds":None,
        "reviewer_specific_mean_or_total":None,
        "interpretation":"Accept the user's current average-time clarification for analysis. Preserve blank individual time cells. This is not a claim of per-record measured timing or a recoverable timing distribution."},
      "protocol_effect":"Resolve A provenance marker by current human confirmation and accept aggregate timing in place of unavailable row-level timing for this submission; do not relax the decision/B0-confirmation disagreement gate.",
      "review_thresholds_unchanged":True,"no_new_human_judgments_by_agent":True}
    # Preserve the immediately preceding blocked analysis before updating its current-result path.
    old_analysis=ROOT/"results/h2_mini_b0_review_history"/f"{VERSION}_before_human_confirmation_analysis.json"
    if old_analysis.exists():raise ValueError("Historical analysis snapshot already exists")
    old_analysis.parent.mkdir(parents=True,exist_ok=True);old_analysis.write_bytes(RESULT.read_bytes())
    old_hash=file_sha(old_analysis);assert old_hash==file_sha(RESULT)
    save(CONFIRM,confirmation)
    ignore=ROOT/".gitignore";text=ignore.read_text(encoding="utf-8")
    if "review/b0_replenishment_canonical/" not in text.splitlines():
        with ignore.open("a",encoding="utf-8",newline="\n") as f:f.write("\nreview/b0_replenishment_canonical/\n")
    OUT.mkdir(parents=True);files={}
    for fn,batch in m["batches"].items():
        src=ROOT/"review/b0_replenishment"/fn;dst=OUT/fn
        original=src.read_bytes();canonical=original.decode("utf-8-sig",errors="strict").encode("utf-8")
        dst.write_bytes(canonical)
        orig_rows=list(csv.DictReader(io.StringIO(original.decode("utf-8-sig"),newline=""),strict=True))
        canon_rows=list(csv.DictReader(io.StringIO(canonical.decode("utf-8"),newline=""),strict=True))
        assert orig_rows==canon_rows and len(orig_rows)==batch["rows"]
        for row in canon_rows:
            assert sha(row["text"])==d["records"][row["review_id"]]["raw_text_sha256"]
            assert row["review_seconds"]==""
        files[fn]={"source_file":rel(src),"canonical_file":rel(dst),"source_sha256":sha(original),"canonical_sha256":sha(canonical),
          "source_mtime_ns":before[fn]["mtime_ns"],"canonical_mtime_ns":dst.stat().st_mtime_ns,
          "reviewer":batch["reviewer"],"expected_judgments":batch["rows"],"actual_judgments":len(canon_rows),
          "source_encoding":"utf-8-sig" if original.startswith(b"\xef\xbb\xbf") else "utf-8","canonical_encoding":"utf-8",
          "canonical_transformation":"Remove UTF-8 BOM if present; preserve all Unicode content, line endings, row order and CSV cell strings",
          "all_human_fields_exactly_preserved":True,"source_text_hash_validation":"PASS","immutable_metadata_validation":"PASS",
          "per_record_review_seconds_populated":0,"review_roles":dict(Counter(row["review_role"] for row in canon_rows))}
    can={"version":VERSION,"status":"CANONICAL_HUMAN_EVIDENCE_FROZEN_WITH_CURRENT_USER_CLARIFICATIONS",
      "created_at":now,"assignment_manifest":"review/b0_replenishment_manifest.json","assignment_manifest_sha256":d["manifest_sha256"],
      "human_confirmation":rel(CONFIRM),"human_confirmation_sha256":file_sha(CONFIRM),
      "expected_judgments":1219,"actual_judgments":1219,"reviewer_judgments":{"A":609,"B":610},
      "files":files,"human_field_values_preserved":True,"original_source_files_unchanged":True,
      "timing_evidence":"Human-reported aggregate mean25s; original row-level times remain blank",
      "previous_blocked_analysis":{"path":rel(old_analysis),"sha256":old_hash},
      "review_gate_pass_not_implied":True,"final_assignment_created":False}
    save(CAN,can)
    d=analyzer.apply_human_confirmation(d,CONFIRM)
    result=analyzer.analyze(d);assert result["status"]=="STOP_TRAINING_AUDIT_DISAGREEMENT"
    result.update(canonical_manifest=rel(CAN),canonical_manifest_sha256=file_sha(CAN),
      prior_analysis_snapshot=can["previous_blocked_analysis"])
    # Metadata only: paired differences, not adjudications or replacement judgments.
    pairs=[]
    for row in m["records"]:
        if len(row["reviewer_assignments"])!=2:continue
        evidence=[]
        for assignment in row["reviewer_assignments"]:
            who=assignment["reviewer"];v=d["states"][(who,row["review_id"])]["values"]
            evidence.append({"reviewer":who,"review_role":assignment["review_role"],"batch":assignment["batch_filename"],
              "reviewer_decision":v["reviewer_decision"],"b0_confirmed":v["b0_confirmed"],
              "reason_code":v["reason_code"],"confidence":v["confidence"],"optional_note_sha256":sha(v["optional_note"])})
        a,b=evidence
        changed=[f for f in ["reviewer_decision","b0_confirmed"] if a[f]!=b[f]]
        if changed:pairs.append({"review_id":row["review_id"],"review_lane":row["review_lane"],"differing_fields":changed,
          "reviewer_evidence":evidence,"adjudication":None})
    disagreement_path=ROOT/"results/h2_mini_b0_replenishment_disagreements.json"
    save(disagreement_path,{"status":"DESCRIPTIVE_DISAGREEMENTS_NO_ADJUDICATION","policy":POLICY,
      "training_audit_disagreements":sum(p["review_lane"]=="TRAIN_CANDIDATE" for p in pairs),
      "evaluation_double_disagreements":sum(p["review_lane"]=="EVALUATION_RESERVE" for p in pairs),"records":pairs})
    result.update(disagreement_report=rel(disagreement_path),disagreement_report_sha256=file_sha(disagreement_path),
      final_capacity_scope="NOT_RUN: section4 training-audit gate failed before merge/post-review audit. Capacity numbers are conditional label-only upper bounds, not final group-safe capacity.",
      assignment_proposal_status="NOT_CREATED_REVIEW_GATE_STOP",BH_partitions_status="FROZEN_CAPACITY_ONLY",
      B_short_status="OPTION_B_APPROVED_FINAL_IDS_NOT_SELECTED",post_review_group_audit_performed=False)
    save(RESULT,result)
    for fn,e in before.items():
        p=ROOT/"review/b0_replenishment"/fn;assert file_sha(p)==e["sha256"] and p.stat().st_mtime_ns==e["mtime_ns"]
    for path,e in audit["historical_evidence_snapshot"].items():
        p=ROOT/path;assert file_sha(p)==e["sha256"] and p.stat().st_mtime_ns==e["mtime_ns"]
    print(json.dumps({"status":result["status"],"completed":result["completed_judgments"],
      "mean_reported_seconds":25,"estimated_hours":1219*25/3600,"primary":result["training_primary_acceptance"],
      "training_audit":result["second_review_agreement"],"double_review":result["double_review_agreement"],
      "conditional_label_only_capacity":result["capacity"],"canonical_manifest":rel(CAN),
      "originals_unchanged":True},indent=2))
if __name__=="__main__":main()
