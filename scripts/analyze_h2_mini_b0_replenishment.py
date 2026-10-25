"""Validate/analyze human B0 worksheets without filling, correcting or adjudicating them."""
import argparse,csv,io,json,math,re,statistics,sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
from h2_mini_b0_review_common import *

DEFAULT_MANIFEST=ROOT/"review/b0_replenishment_manifest.json"
DEFAULT_DIRECTORY=ROOT/"review/b0_replenishment"

def rate(n,d):return n/d if d else None
def agreement(pairs,field):
    a=[x[field] for x,y in pairs];b=[y[field] for x,y in pairs];n=len(a)
    same=sum(x==y for x,y in zip(a,b));ca=Counter(a);cb=Counter(b)
    pe=sum(ca[k]*cb[k] for k in ca)/n**2 if n else None
    return {"completed_pairs":n,"agreements":same,"disagreements":n-same,
      "agreement_rate":rate(same,n),"disagreement_rate":rate(n-same,n),
      "cohen_kappa":((same/n-pe)/(1-pe) if n and pe<1 else None),
      "kappa_note":"Undefined for no pairs or degenerate identical marginals."}
def pair_summary(pairs):
    return {"decision":agreement(pairs,"reviewer_decision"),"B0_confirmation":agreement(pairs,"b0_confirmed"),
      "any_axis_disagreements":sum(x["reviewer_decision"]!=y["reviewer_decision"] or x["b0_confirmed"]!=y["b0_confirmed"] for x,y in pairs),
      "accept_vs_true_injection":sum({x["reviewer_decision"],y["reviewer_decision"]}=={"ACCEPT_BENIGN","REJECT_TRUE_INJECTION"} for x,y in pairs)}
def times(values):
    ordered=sorted(values);p90=None
    if ordered:
        position=.90*(len(ordered)-1);lo=math.floor(position);hi=math.ceil(position)
        p90=ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)
    return {"completed_judgments":len(values),"seconds":sum(values) if values else None,
      "person_hours":sum(values)/3600 if values else None,
      "mean_seconds":statistics.mean(values) if values else None,
      "median_seconds":statistics.median(values) if values else None,"p90_seconds":p90,
      "p90_method":"Linear interpolation at 0.90*(n-1), only over complete valid judgments",
      "min_seconds":min(values) if values else None,"max_seconds":max(values) if values else None,
      "missing_timing_is_not_zero":True}
def snapshot_counts(records):
    return {"unique":len(records),"training":sum(r["review_lane"]=="TRAIN_CANDIDATE" for r in records),
      "training_audited":sum(r["review_lane"]=="TRAIN_CANDIDATE" and len(r["reviewer_evidence"])==2 for r in records),
      "evaluation":sum(r["review_lane"] in {"VALIDATION_RESERVE","TEST_RESERVE"} for r in records)}
def inspect(manifest_path=DEFAULT_MANIFEST,directory=DEFAULT_DIRECTORY,root=ROOT):
    """Read only; immutable content and assignment failures are never repaired."""
    m=load(manifest_path);errors=[];warnings=[];provenance_flags=[];observed={};states={};hashes={};transport={}
    def error(message,**where):errors.append({"message":message,**where})
    if m.get("taxonomy_policy")!=POLICY:error("Unexpected taxonomy policy")
    if m.get("worksheet_columns")!=FIELDS or m.get("human_fields")!=HUMAN:error("Manifest worksheet schema mismatch")
    if m.get("candidate_source_revision")!=OA_REV:error("Pinned OASST1 revision mismatch")
    for path_key,hash_key in [("taxonomy_policy_path","taxonomy_policy_sha256"),("candidate_pool_path","candidate_pool_sha256"),("audit_path","audit_sha256")]:
        p=root/m[path_key]
        if not p.exists() or file_sha(p)!=m[hash_key]:error("Pinned supporting metadata changed or missing",path=str(p))
    old=m["old_B0_pool"];old_path=root/old["path"];old_records=[]
    if not old_path.exists() or file_sha(old_path)!=old["sha256"]:error("Historical B0 evidence changed or missing",path=str(old_path))
    else:
        old_records=load(old_path)["records"]
        if snapshot_counts(old_records)!=old["counts"]:error("Historical B0 counts differ from frozen evidence")
    expected={};records={};role_counts=Counter();by_reviewer=Counter();lanes=Counter()
    for r in m["records"]:
        rid=r["review_id"]
        if rid in records:error("Duplicate record in manifest",review_id=rid)
        records[rid]=r;lanes[r["review_lane"]]+=1
        assignments=r["reviewer_assignments"];roles=Counter(a["review_role"] for a in assignments)
        reviewers=[a["reviewer"] for a in assignments]
        if len(set(reviewers))!=len(reviewers) or any(w not in {"A","B"} for w in reviewers):error("Assignments are not independent A/B reviews",review_id=rid)
        if r["review_lane"]=="TRAIN_CANDIDATE":
            if roles not in [Counter(PRIMARY=1),Counter(PRIMARY=1,SECOND_REVIEW=1)]:error("Invalid training role contract",review_id=rid)
        elif r["review_lane"]=="EVALUATION_RESERVE":
            if roles!=Counter(DOUBLE_REVIEW=2) or set(reviewers)!={"A","B"}:error("Invalid double-review role contract",review_id=rid)
        else:error("Unknown review lane",review_id=rid)
        for a in assignments:
            fn=a["batch_filename"];k=(a["reviewer"],rid)
            if k in expected:error("Duplicate assignment",review_id=rid)
            expected[k]=(r,a);role_counts[a["review_role"]]+=1;by_reviewer[a["reviewer"]]+=1
            if fn not in m["batches"] or m["batches"][fn]["reviewer"]!=a["reviewer"]:error("Assignment batch mismatch",review_id=rid,batch=fn)
    if len(records)!=872 or len(expected)!=1219 or lanes!=Counter(TRAIN_CANDIDATE=600,EVALUATION_RESERVE=272):error("Approved package counts changed")
    if role_counts!=Counter(PRIMARY=600,SECOND_REVIEW=75,DOUBLE_REVIEW=544) or by_reviewer!=Counter(A=609,B=610):error("Approved role/reviewer counts changed")
    actual_files={p.name for p in directory.glob("*.csv")}
    for fn in sorted(actual_files-set(m["batches"])):error("Unrecognized CSV; do not silently combine packages",batch=fn)
    for fn,batch in m["batches"].items():
        path=directory/fn;who=batch["reviewer"]
        if not path.exists():error("Missing worksheet",batch=fn);continue
        raw=path.read_bytes();hashes[fn]=sha(raw)
        try:
            text=raw.decode("utf-8-sig",errors="strict")
            reader=csv.DictReader(io.StringIO(text,newline=""),strict=True)
            if reader.fieldnames!=FIELDS:error("Worksheet columns changed",batch=fn);continue
            rr=list(reader)
        except (UnicodeDecodeError,csv.Error) as exc:error("UTF-8/CSV decoding failed",batch=fn,detail=str(exc));continue
        transport[fn]={"encoding":"utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8","rows":len(rr),
          "unchanged_prepared_bytes":hashes[fn]==batch["prepared_sha256"]}
        if len(rr)!=batch["rows"]:error("Batch row count changed",batch=fn)
        if Counter(row.get("review_id") for row in rr)!=Counter(batch["review_ids"]):error("Batch record identities changed",batch=fn)
        for row in rr:
            rid=row.get("review_id","");k=(who,rid);where={"batch":fn,"reviewer":who,"review_id":rid}
            if None in row or any(row.get(f) is None for f in FIELDS):error("Malformed CSV row width",**where);continue
            if k not in expected:error("Unexpected reviewer/record assignment",**where);continue
            if k in observed:error("Duplicate reviewer judgment row",**where);continue
            r,a=expected[k];observed[k]=row
            immutable={"source":"OpenAssistant/oasst1","language":r["language"],"token_length":str(r["deberta_tokens"]),"review_role":a["review_role"]}
            if a["batch_filename"]!=fn:error("Record moved to another batch",**where)
            if sha(row["text"])!=r["raw_text_sha256"]:error("Source text changed",**where)
            for f,value in immutable.items():
                if row[f]!=value:error("Immutable metadata changed",field=f,**where)
            vals={f:row[f].strip() for f in HUMAN}
            # An explicit preliminary-AI marker is conflicting provenance, not proof of actual authorship.
            # Match the specific review-status phrase, never generic discussion of AI in a source prompt.
            if re.search(r"AI\s*\u9884\u5ba1\u5efa\u8bae",vals["optional_note"],re.I):
                provenance_flags.append({"reviewer":who,"review_id":rid,"batch":fn,"field":"optional_note",
                  "code":"EXPLICIT_AI_PRE_REVIEW_STATUS_REQUIRES_HUMAN_CONFIRMATION",
                  "note_sha256":sha(row["optional_note"]),
                  "resolution":"Clarify whether independent human review actually completed; do not silently reinterpret or remove the note."})
            if any(row[f]!=vals[f] for f in HUMAN if f!="optional_note"):
                warnings.append({"message":"Surrounding whitespace ignored for analysis only; original file unchanged",**where})
            states[k]={"status":"pending","values":vals,"row_sha256":sha(json.dumps({f:row[f] for f in HUMAN},ensure_ascii=False,sort_keys=True))}
            if not any(vals.values()):continue
            invalid=[]
            for f,allowed in [("reviewer_decision",DECISIONS),("b0_confirmed",B0_VALUES),("confidence",CONFIDENCE)]:
                if vals[f] and vals[f] not in allowed:invalid.append(f)
            seconds=None
            if vals["review_seconds"]:
                try:
                    seconds=float(vals["review_seconds"])
                    if not math.isfinite(seconds) or seconds<=0:raise ValueError
                except ValueError:invalid.append("review_seconds")
            if invalid:
                error("Invalid human-field value; human correction required",fields=invalid,**where);states[k]["status"]="invalid";continue
            missing=[f for f in ["reviewer_decision","b0_confirmed","confidence","review_seconds"] if not vals[f]]
            if vals["reviewer_decision"].startswith("REJECT_") and not vals["reason_code"]:missing.append("reason_code")
            if missing:states[k].update(status="partial",missing=missing);continue
            states[k].update(status="completed",seconds=seconds)
            if vals["reviewer_decision"]!="ACCEPT_BENIGN" and vals["b0_confirmed"]=="YES":
                warnings.append({"message":"Rejected record marked B0 YES; excluded, not auto-corrected",**where})
    for who,rid in expected:
        if (who,rid) not in observed:error("Missing assigned judgment row",reviewer=who,review_id=rid)
    counts=Counter(s["status"] for s in states.values())
    return {"manifest":m,"records":records,"expected":expected,"states":states,"errors":errors,"warnings":warnings,
      "hashes":hashes,"transport":transport,"state_counts":dict(counts),"manifest_sha256":file_sha(manifest_path),
      "provenance_flags":provenance_flags}
def apply_human_confirmation(data,path,root=ROOT):
    """Apply a hash-bound user clarification; preserve every submitted human field."""
    confirmation=load(path)
    if confirmation.get("package_id")!=data["manifest"]["package_id"]:raise ValueError("Human confirmation package mismatch")
    if confirmation.get("assignment_manifest_sha256")!=data["manifest_sha256"]:raise ValueError("Human confirmation manifest mismatch")
    if confirmation.get("submitted_worksheet_sha256")!=data["hashes"]:raise ValueError("Human confirmation does not cover these exact submitted files")
    if data["errors"]:raise ValueError("Cannot apply clarification over an integrity/field error")
    if confirmation.get("human_statement_source")!="CURRENT_USER_MESSAGE":raise ValueError("Missing human source")
    timing=confirmation["timing_clarification"]
    if timing.get("evidence_type")!="HUMAN_REPORTED_AGGREGATE_MEAN" or timing.get("per_record_imputation") is not False:raise ValueError("Unsupported timing clarification")
    mean=timing.get("mean_seconds_per_judgment")
    if not isinstance(mean,(int,float)) or not math.isfinite(mean) or mean<=0:raise ValueError("Invalid reported mean")
    if timing.get("judgments_in_scope")!=len(data["expected"]):raise ValueError("Reported mean scope mismatch")
    resolution=confirmation["reviewer_A_confirmation"]
    if resolution.get("manual_review_completed") is not True or resolution.get("notes_are_stale") is not True:raise ValueError("A review status remains unresolved")
    allowed_hashes=set(resolution["stale_note_sha256"])
    if any(f["reviewer"]!="A" or f["note_sha256"] not in allowed_hashes for f in data["provenance_flags"]):raise ValueError("Human confirmation does not resolve all flagged provenance")
    missing=[]
    for k,state in data["states"].items():
        if state["status"] not in {"partial","completed"} or set(state.get("missing",[]))-{"review_seconds"}:raise ValueError("Human clarification cannot infer another missing judgment field")
        if state["status"]=="partial":
            if state.get("missing")!=["review_seconds"]:raise ValueError("Unrecognized incomplete judgment")
            missing.append({"reviewer":k[0],"review_id":k[1],"batch":data["expected"][k][1]["batch_filename"],"missing_field":"review_seconds"})
            state["submitted_status"]="partial";state["status"]="completed";state["seconds"]=None
            state["completion_basis"]="User-confirmed human labels; aggregate timing only, no row-level value imputed"
    data["submitted_state_counts"]=data["state_counts"]
    data["submitted_missing_fields_resolved_as_aggregate_only"]=missing
    data["resolved_provenance_flags"]=data["provenance_flags"]
    data["provenance_flags"]=[]
    data["state_counts"]=dict(Counter(state["status"] for state in data["states"].values()))
    data["human_confirmation"]={"path":str(path),"sha256":file_sha(path),"record":confirmation}
    return data

def analyze(data,interim=False):
    m=data["manifest"];states=data["states"];completed={k:s["values"] for k,s in states.items() if s["status"]=="completed"}
    eligible=[];outcomes=[];audit_pairs=[];eval_pairs=[]
    for rid,r in data["records"].items():
        keys=[(a["reviewer"],rid) for a in r["reviewer_assignments"]]
        rows=[completed[k] for k in keys if k in completed]
        done=len(rows)==len(keys);ok=done and not data["errors"] and not data["provenance_flags"] and label_eligible(rows)
        if ok:eligible.append(r)
        if len(keys)==2 and done:(audit_pairs if r["review_lane"]=="TRAIN_CANDIDATE" else eval_pairs).append(tuple(rows))
        outcomes.append({"review_id":rid,"completed_judgments":len(rows),"required_judgments":len(keys),
          "review_lane":r["review_lane"],"all_required_complete":done,"review_eligible_B0":ok,
          "judgment_evidence":[{"reviewer":w,"status":states.get((w,rid),{}).get("status","missing"),
             "human_fields_sha256":states.get((w,rid),{}).get("row_sha256")} for w,_ in keys],
          "final_assignment":None})
    second=pair_summary(audit_pairs);double=pair_summary(eval_pairs)
    # The user-approved >10% threshold applies separately to decision and B0-confirmation disagreement.
    gate_exceeded=any(second[axis]["disagreement_rate"] is not None and second[axis]["disagreement_rate"]>.10 for axis in ["decision","B0_confirmation"])
    primary=[v for k,v in completed.items() if data["expected"][k][1]["review_role"]=="PRIMARY"]
    ca=coverage(eligible,m["old_B0_pool"]["counts"])
    all_done=len(completed)==m["expected_judgments"] and not data["errors"]
    def counts(rows):
        ac=sum(v["reviewer_decision"]=="ACCEPT_BENIGN" for v in rows)
        b0=sum(v["reviewer_decision"]=="ACCEPT_BENIGN" and v["b0_confirmed"]=="YES" for v in rows)
        return {"completed":len(rows),"ACCEPT_BENIGN":ac,"acceptance_rate":rate(ac,len(rows)),
          "ACCEPT_BENIGN_and_B0_YES":b0,"B0_confirmation_yield":rate(b0,len(rows)),
          "decisions":dict(Counter(v["reviewer_decision"] for v in rows)),
          "rejection_reason_codes":dict(Counter(v["reason_code"] for v in rows if v["reviewer_decision"]!="ACCEPT_BENIGN"))}
    incomplete=[{"reviewer":k[0],"review_id":k[1],"batch":data["expected"][k][1]["batch_filename"],"status":s["status"],"missing_fields":s.get("missing",[])}
      for k,s in states.items() if s["status"]!="completed"]
    missing_fields=Counter(field for row in incomplete for field in row["missing_fields"])
    per_batch=defaultdict(Counter)
    for row in incomplete:
        for field in row["missing_fields"]:per_batch[row["batch"]][field]+=1
    field_presence={field:sum(bool(s["values"][field]) for s in states.values()) for field in HUMAN}
    status=("INVALID_INTEGRITY_OR_FIELDS" if data["errors"] else "INTERIM_DESCRIPTIVE_ONLY" if interim else
      "INCOMPLETE_HUMAN_REVIEW" if not all_done else "STOP_UNVERIFIED_HUMAN_REVIEW" if data["provenance_flags"] else "STOP_TRAINING_AUDIT_DISAGREEMENT" if gate_exceeded else
      "REVIEW_COMPLETE_POST_REVIEW_AUDIT_AND_HUMAN_REVIEW_REQUIRED")
    report={"status":status,"generated_at":datetime.now(timezone.utc).isoformat(),"package_id":m["package_id"],
      "taxonomy_policy":POLICY,"manifest_sha256":data["manifest_sha256"],"worksheet_sha256":data["hashes"],
      "transport_integrity":data["transport"],"errors":data["errors"],"warnings":data["warnings"],
      "review_provenance_gate":{"status":"HUMAN_CONFIRMATION_REQUIRED" if data["provenance_flags"] else "NO_EXPLICIT_PRELIMINARY_MARKER_DETECTED",
        "blocking_flag_count":len(data["provenance_flags"]),"flags":data["provenance_flags"],
        "note":"An explicit AI preliminary-review marker conflicts with completed independent human evidence. Absence of this marker is not proof of human authorship. No annotation is changed automatically."},
      "completed_judgments":len(completed),"expected_judgments":m["expected_judgments"],"state_counts":data["state_counts"],
      "all_required_judgments_complete":all_done,"interim":interim,"interim_200_reached":len(completed)>=200,
      "completeness":{"expected_assignments":m["expected_judgments"],"observed_assignment_rows":len(states),
        "fully_complete_judgments":len(completed),"populated_human_fields":field_presence,
        "missing_fields":dict(missing_fields),"missing_fields_by_batch":{k:dict(v) for k,v in per_batch.items()},
        "note":"Filled decisions and B0 confirmations do not complete a judgment when required timing or other fields are missing. Do not infer absent human values."},
      "final_capacity_assessment":{"performed":False,"status":"BLOCKED_INCOMPLETE_OR_INVALID_REVIEW" if not all_done else
          "BLOCKED_HUMAN_REVIEW_PROVENANCE" if data["provenance_flags"] else "BLOCKED_TRAINING_AUDIT_DISAGREEMENT" if gate_exceeded else "POST_REVIEW_GROUP_AUDIT_REQUIRED",
        "final_capacity_pass":None,"note":"Capacity figures below are provisional review-qualified bookkeeping, not a final group-safe count or final shortage. No final assignment is authorized by this analyzer."},
      "interim_caution":"Optional descriptive checkpoint; small paired samples are uncertain. No automatic rubric changes, adjudication, final IDs or stage assignment.",
      "all_completed_judgments":counts(list(completed.values())),"training_primary_acceptance":counts(primary),
      "by_role":{role:counts([v for k,v in completed.items() if data["expected"][k][1]["review_role"]==role]) for role in ["PRIMARY","SECOND_REVIEW","DOUBLE_REVIEW"]},
      "by_language":{lang:counts([v for k,v in completed.items() if data["records"][k[1]]["language"]==lang]) for lang in sorted({r["language"] for r in m["records"]})},
      "second_review_agreement":second,"double_review_agreement":double,
      "training_audit_gate":{"decision_or_confirmation_threshold":.10,"exceeded":gate_exceeded,
         "state":"PROVISIONAL_INTERIM" if interim else "INCOMPLETE" if not all_done else "HUMAN_REVIEW_REQUIRED" if gate_exceeded else "WITHIN_THRESHOLD",
         "any_axis_union_rate":rate(second["any_axis_disagreements"],len(audit_pairs)),
         "no_automatic_adjudication":True},
      "timing":times([s["seconds"] for s in states.values() if s["status"]=="completed" and s.get("seconds") is not None]),
      "timing_by_reviewer":{w:times([s["seconds"] for k,s in states.items() if k[0]==w and s["status"]=="completed" and s.get("seconds") is not None]) for w in ["A","B"]},
      "capacity":ca,"capacity_is_provisional":not all_done or gate_exceeded or interim or bool(data["provenance_flags"]),
      "capacity_note":"Review-qualified counts only. No final splits or records assigned. Required new post-review group/contamination audit may reduce capacity. Each future stage still needs20 independent accepted training audits.",
      "incomplete_judgments":incomplete,"record_outcomes_metadata_only":outcomes,
      "scientific_finalization_authorized":False,"human_judgments_written":False,"final_IDs_selected":False,
      "research_JSONL_materialized":False,"training":False,"replay":False,"replay_readiness":"R0"}
    for name,pairs in [("second_review_agreement",audit_pairs),("double_review_agreement",eval_pairs)]:
        report[name]["both_ACCEPT_BENIGN"]=sum(all(v["reviewer_decision"]=="ACCEPT_BENIGN" for v in pair) for pair in pairs)
        report[name]["both_B0_YES"]=sum(all(v["b0_confirmed"]=="YES" for v in pair) for pair in pairs)
        report[name]["both_ACCEPT_and_B0_YES"]=sum(label_eligible(pair) for pair in pairs)
    report["training_primary_acceptance"]["not_B0_eligible_count"]=len(primary)-report["training_primary_acceptance"]["ACCEPT_BENIGN_and_B0_YES"]
    report["training_primary_acceptance"]["rejected_decision_count"]=len(primary)-report["training_primary_acceptance"]["ACCEPT_BENIGN"]
    if data.get("human_confirmation"):
        att=data["human_confirmation"];timing=att["record"]["timing_clarification"]
        n=timing["judgments_in_scope"];mean=timing["mean_seconds_per_judgment"]
        report["human_confirmation"]=att
        report["completeness"].update(submitted_state_counts=data["submitted_state_counts"],
          submitted_per_record_timing_missing=len(data["submitted_missing_fields_resolved_as_aggregate_only"]),
          complete_under_user_timing_clarification=True,per_record_timing_imputed=False,
          note="All judgment decisions and confirmations are complete under the current user's aggregate-timing clarification. Submitted per-record timing cells remain blank and are not represented as observed values.")
        report["review_provenance_gate"].update(status="RESOLVED_BY_CURRENT_USER_CONFIRMATION",
          resolved_flag_count=len(data["resolved_provenance_flags"]),resolved_flags=data["resolved_provenance_flags"],
          evidence_limit="The current human confirms A's judgments were manually reviewed and the notes were accidentally retained. Notes are preserved; no independent machine verification of review behavior is claimed.")
        report["timing"]["human_reported_aggregate"]={"evidence_type":"HUMAN_REPORTED_AGGREGATE_MEAN",
          "mean_seconds_per_judgment":mean,"judgments_in_scope":n,"estimated_total_seconds":n*mean,
          "estimated_total_person_hours":n*mean/3600,"median_seconds":None,"p90_seconds":None,
          "per_reviewer_means_or_totals":None,"note":"Total derived from user-reported mean; not independently measured per-record timing. Median/p90 and reviewer-specific timing cannot be recovered."}
        report["submitted_missing_timing_rows"]=data["submitted_missing_fields_resolved_as_aggregate_only"]
        report["reviewer_judgment_counts"]={w:sum(k[0]==w and s["status"]=="completed" for k,s in states.items()) for w in ["A","B"]}
        report["capacity_conditional_on_training_audit_pass"]=gate_exceeded

    return report
def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=DEFAULT_MANIFEST)
    parser.add_argument("--review-dir",type=Path,default=DEFAULT_DIRECTORY)
    mode=parser.add_mutually_exclusive_group();mode.add_argument("--interim",action="store_true");mode.add_argument("--check-package",action="store_true")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--human-confirmation",type=Path,help="Explicit hash-bound human provenance and aggregate-timing clarification")
    args=parser.parse_args(argv);data=inspect(args.manifest,args.review_dir)
    if args.human_confirmation:
        if args.check_package:parser.error("Human confirmation applies to completed review analysis, not a blank-package check.")
        data=apply_human_confirmation(data,args.human_confirmation)
    if args.check_package:
        changed=[fn for fn,t in data["transport"].items() if not t["unchanged_prepared_bytes"]]
        blank=len(data["states"])==1219 and data["state_counts"].get("pending")==1219
        report={"status":"PREPARED_BLANK_PACKAGE_VALID" if not data["errors"] and blank and not changed else "PACKAGE_CHECK_FAILED",
          "candidates":len(data["records"]),"assignments":len(data["states"]),"blank_human_judgments":data["state_counts"].get("pending",0),
          "errors":data["errors"],"changed_prepared_files":changed,"note":"This mode verifies the initially blank package; use --interim or full analysis after human edits."}
        print(json.dumps(report,ensure_ascii=False,indent=2));return 0 if report["status"]=="PREPARED_BLANK_PACKAGE_VALID" else 2
    report=analyze(data,args.interim)
    output=args.output or ROOT/("results/h2_mini_b0_replenishment_interim.json" if args.interim else "results/h2_mini_b0_replenishment_analysis.json")
    protected={args.manifest.resolve(),*(p.resolve() for p in args.review_dir.glob("*.csv"))}
    if output.resolve() in protected or output.suffix.lower()!=".json":parser.error("Output must be a separate JSON analysis artifact, never a worksheet or manifest.")
    if output.resolve().is_relative_to((ROOT/"review").resolve()) or output.resolve().is_relative_to((ROOT/"data").resolve()):parser.error("Write analysis reports under results, not evidence or data directories.")
    save(output,report)
    print(json.dumps({"status":report["status"],"completed_judgments":report["completed_judgments"],"expected_judgments":1219,
      "second_review_agreement":report["second_review_agreement"],"double_review_agreement":report["double_review_agreement"],
      "capacity":report["capacity"],"output":str(output)},ensure_ascii=False,indent=2))
    return 2 if data["errors"] else 3 if not args.interim and not report["all_required_judgments_complete"] else 5 if not args.interim and data["provenance_flags"] else 4 if not args.interim and report["training_audit_gate"]["exceeded"] else 0
if __name__=="__main__":sys.exit(main())
