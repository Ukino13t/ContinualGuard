"""Validate blank expansion worksheets or analyze all75+105 independent audit pairs."""
import argparse,csv,io,json,math,re,sys
from collections import Counter
from datetime import datetime,timezone
from h2_mini_b0_expansion_common import *

def inspect_expansion(manifest_path=EXP_MANIFEST,directory=EXP_DIR,root=ROOT,baseline=None):
    m=load(manifest_path);errors=[];warnings=[];states={};hashes={};transport={}
    def error(message,**where):errors.append({"message":message,**where})
    for path,digest in m["frozen_dependencies"].items():
        p=root/path
        if not p.exists() or file_sha(p)!=digest:error("Frozen dependency changed or missing",path=path)
    data,_,_=baseline or load_baseline()
    records=data["manifest"]["records"];by={r["review_id"]:r for r in records}
    expected_selection,_=expansion_selection(records)
    if m["records"]!=expected_selection:error("Deterministic answer-blind assignment differs from manifest")
    old=old_audit_records(records)
    if set(m["old_audited_training_ids"])!={r["review_id"] for r in old} or len(m["old_audited_training_ids"])!=75:
        error("Original75-pair denominator changed")
    if m["worksheet_columns"]!=EXP_FIELDS or m["new_second_review_assignments"]!=105:error("Expansion schema/count changed")
    expected={r["review_id"]:r for r in m["records"]}
    if len(expected)!=105:error("Duplicate or missing expansion identities")
    if Counter(r["second_reviewer"] for r in expected.values())!=Counter(A=53,B=52):error("Expected A53/B52 assignments")
    overrides={};auxiliary=set();recovery=None
    if root.resolve()==ROOT.resolve() and manifest_path.resolve()==EXP_MANIFEST.resolve():
        try:
            from h2_mini_b0_recovered_evidence import resolve_recovered
            overrides,auxiliary,recovery=resolve_recovered(m,root,directory)
        except (ValueError,KeyError,OSError) as exc:
            error("Recovered provenance integrity failure",detail=str(exc))
    for filename in sorted({p.name for p in directory.glob("*.csv")}-set(m["batches"])-auxiliary):
        error("Unexpected CSV file",batch=filename)
    for fn,batch in m["batches"].items():
        p=overrides.get(fn,directory/fn)
        if not p.exists():error("Missing expansion worksheet",batch=fn);continue
        raw=p.read_bytes();hashes[fn]=sha(raw)
        try:
            text=raw.decode("utf-8-sig",errors="strict");reader=csv.DictReader(io.StringIO(text,newline=""),strict=True)
            if reader.fieldnames not in [EXP_FIELDS,EXP_FIELDS+["review_seconds"]]:
                error("Expansion header differs from the11 frozen columns or allowed optional timing extension",batch=fn);continue
            fields=list(reader.fieldnames);rows=list(reader)
        except (UnicodeDecodeError,csv.Error) as exc:error("Invalid UTF-8/CSV",batch=fn,detail=str(exc));continue
        if len(rows)!=batch["rows"]:error("Worksheet row count changed",batch=fn)
        if Counter(r.get("review_id") for r in rows)!=Counter(batch["review_ids"]):error("Worksheet review IDs changed",batch=fn)
        transport[fn]={"encoding":"utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8","rows":len(rows),
          "unchanged_prepared_bytes":sha(raw)==batch["prepared_sha256"],"optional_timing_column":"review_seconds" in fields}
        for row in rows:
            rid=row.get("review_id","");who=batch["reviewer"];where={"batch":fn,"reviewer":who,"review_id":rid}
            if None in row or any(row.get(k) is None for k in fields):error("Malformed CSV row width",**where);continue
            if rid not in expected:error("Unexpected expansion review ID",**where);continue
            if rid in states:error("Duplicate independent second judgment",**where);continue
            meta=expected[rid]
            if meta["second_reviewer"]!=who or meta["batch_filename"]!=fn:error("Reviewer assignment changed",**where)
            if meta["primary_reviewer"]==who:error("Second reviewer equals primary reviewer",**where)
            if sha(row["text"])!=meta["raw_text_sha256"]:error("Immutable source text changed",**where)
            for key,value in {"source":"OpenAssistant/oasst1","language":meta["language"],"token_length":str(meta["deberta_tokens"]),"review_role":"SECOND_REVIEW"}.items():
                if row[key]!=value:error("Immutable metadata changed",field=key,**where)
            v={key:row[key].strip() for key in EXP_HUMAN}
            s={"values":v,"status":"pending","where":where,"seconds":None,
              "human_fields_sha256":sha(json.dumps({k:row[k] for k in fields if k in EXP_HUMAN or k=="review_seconds"},ensure_ascii=False,sort_keys=True))}
            states[rid]=s
            timing=row.get("review_seconds","").strip()
            if not any(v.values()) and not timing:continue
            invalid=[]
            for field,allowed in [("reviewer_decision",DECISIONS),("b0_confirmed",B0_VALUES),("confidence",CONFIDENCE)]:
                if v[field] and v[field] not in allowed:invalid.append(field)
            if timing:
                try:
                    seconds=float(timing)
                    if not math.isfinite(seconds) or seconds<=0:raise ValueError
                    s["seconds"]=seconds
                except ValueError:invalid.append("review_seconds")
            if invalid:error("Invalid entered field",fields=invalid,**where);s["status"]="invalid";continue
            if re.search(r"AI\s*\u9884\u5ba1\u5efa\u8bae",v["optional_note"],re.I):
                error("New preliminary-AI note is not covered by the old A human confirmation",field="optional_note",**where)
            missing=[key for key in ["reviewer_decision","b0_confirmed","confidence"] if not v[key]]
            if v["reviewer_decision"].startswith("REJECT_") and not v["reason_code"]:missing.append("reason_code")
            if missing:s.update(status="partial",missing=missing);continue
            s["status"]="completed"
    for rid in expected:
        if rid not in states:error("Missing assigned second-review row",review_id=rid,batch=expected[rid]["batch_filename"])
    return {"manifest":m,"base_data":data,"states":states,"errors":errors,"warnings":warnings,"hashes":hashes,"transport":transport,
      "manifest_sha256":file_sha(manifest_path),"records":by,"recovery_manifest":recovery}

def pair_diagnostics(data):
    baseline=data["base_data"];mapping={r["review_id"]:r for r in data["manifest"]["records"]}
    pairs=[]
    def length_bin(n):
        lower=1
        for upper in (16,24,32,48,64,128,256,1280):
            if n<=upper:return str(lower)+"-"+str(upper)
            lower=upper+1
        return ">1280"
    for rid in data["manifest"]["old_audited_training_ids"]+list(mapping):
        r=data["records"][rid]
        primary=next(a for a in r["reviewer_assignments"] if a["review_role"]=="PRIMARY")
        first=baseline["states"][(primary["reviewer"],rid)]["values"]
        if rid in mapping:
            if data["states"][rid]["status"]!="completed":continue
            second_who=mapping[rid]["second_reviewer"];second=data["states"][rid]["values"];cohort="new105"
        else:
            assignment=next(a for a in r["reviewer_assignments"] if a["review_role"]=="SECOND_REVIEW")
            second_who=assignment["reviewer"];second=baseline["states"][(second_who,rid)]["values"];cohort="original75"
        fields=["reviewer_decision","b0_confirmed","confidence"]
        pairs.append({"review_id":rid,"cohort":cohort,"language":r["language"],"token_length":r["deberta_tokens"],
            "token_length_bin":length_bin(r["deberta_tokens"]),"primary_reviewer":primary["reviewer"],"second_reviewer":second_who,
            "primary":{f:first[f] for f in fields},"second":{f:second[f] for f in fields},
            "decision_disagreement":first["reviewer_decision"]!=second["reviewer_decision"],
            "B0_disagreement":first["b0_confirmed"]!=second["b0_confirmed"],
            "safety_critical":{first["reviewer_decision"],second["reviewer_decision"]}=={"ACCEPT_BENIGN","REJECT_TRUE_INJECTION"}})
    axes={
      "language":lambda r:r["language"],"token_length_bin":lambda r:r["token_length_bin"],
      "primary_reviewer":lambda r:r["primary_reviewer"],"second_reviewer":lambda r:r["second_reviewer"],
      "primary_confidence":lambda r:r["primary"]["confidence"],"second_confidence":lambda r:r["second"]["confidence"],
      "confidence_pair":lambda r:r["primary"]["confidence"]+"/"+r["second"]["confidence"]}
    breakdown={}
    for axis,key in axes.items():
        groups={}
        for r in pairs:groups.setdefault(key(r),[]).append(r)
        breakdown[axis]={k:base.pair_summary([(r["primary"],r["second"]) for r in rows]) for k,rows in sorted(groups.items())}
    return {"pair_count":len(pairs),"all_pair_evidence_metadata_only":pairs,"breakdown":breakdown,
      "disagreement_ids":[r["review_id"] for r in pairs if r["decision_disagreement"] or r["B0_disagreement"]],
      "safety_critical_ids":[r["review_id"] for r in pairs if r["safety_critical"]],
      "interpretation":"Descriptive denominators, not causal language/reviewer effects; primary and second reviewer identities are coupled by opposite-reviewer design."}

def analyze_expansion(data):
    m=data["manifest"];baseline=data["base_data"];states=data["states"];mapping={r["review_id"]:r for r in m["records"]}
    if data["errors"]:
        return {"package_id":EXP_PACKAGE,"status":"INVALID_EXPANSION_INTEGRITY_OR_FIELDS",
          "completed_new_judgments":sum(s["status"]=="completed" for s in states.values()),
          "expanded_audit":{"all180_complete":False,"expected_full_denominator":180,"B0_confirmation_gate_pass":None,"decision_gate_pass":None},
          "conditional_label_only_capacity":None,"errors":data["errors"],
          "next_step":"Resolve exact integrity errors without inferring judgments; no audit gate or finalization",
          "final_pool_merge_performed":False,"training":False,"replay":False}

    old_ids=m["old_audited_training_ids"];old_pairs=[];new_pairs=[];differences=[]
    for rid in old_ids:
        r=data["records"][rid];values=[baseline["states"][(a["reviewer"],rid)]["values"] for a in r["reviewer_assignments"]]
        old_pairs.append(tuple(values))
    for rid,s in states.items():
        if s["status"]!="completed":continue
        original=baseline["states"][(mapping[rid]["primary_reviewer"],rid)]["values"];new=s["values"]
        new_pairs.append((original,new))
        differing=[f for f in ["reviewer_decision","b0_confirmed"] if original[f]!=new[f]]
        if differing:differences.append({"review_id":rid,"differing_fields":differing,"adjudication":None})
    audit=summarize_pairs(old_pairs,new_pairs)
    if data["errors"]:
        audit.update(status="INVALID_EXPANSION_INTEGRITY_OR_FIELDS",B0_confirmation_gate_pass=None,decision_gate_pass=None)
    complete=audit["all180_complete"] and not data["errors"]
    # The audit never merges pools or computes capacity, including when its gate fails.
    outcomes=[];capacity=None
    audit["operative_gate"]="B0_confirmation disagreement <=10% on all180; current human instruction"
    audit["status"]=("INCOMPLETE_EXPANDED_REVIEW" if not complete else
        "EXPANDED_AUDIT_GATE_PASS_POST_REVIEW_AUDIT_REQUIRED" if audit["B0_confirmation_gate_pass"] else
        "STOP_EXPANDED_AUDIT_DISAGREEMENT")
    diagnostics=pair_diagnostics(data)
    timing=[s["seconds"] for s in states.values() if s["status"]=="completed" and s["seconds"] is not None]
    return {"package_id":EXP_PACKAGE,"generated_at":datetime.now(timezone.utc).isoformat(),
      "status":audit["status"],"expected_new_judgments":105,"completed_new_judgments":len(new_pairs),
      "reviewer_completed":dict(Counter(s["where"]["reviewer"] for s in states.values() if s["status"]=="completed")),
      "expanded_audit":audit,"diagnostics":diagnostics,"recovery_manifest":data.get("recovery_manifest"),"old75_metrics":base.pair_summary(old_pairs),"new105_metrics_descriptive_only":base.pair_summary(new_pairs),
      "baseline_expected_and_complete_judgments":1219,"total_complete_judgments":1219+len(new_pairs),
      "errors":data["errors"],"warnings":data["warnings"],"manifest_sha256":data["manifest_sha256"],
      "worksheet_sha256":data["hashes"],"transport_integrity":data["transport"],
      "missing_or_partial":[dict(s["where"],status=s["status"],missing_fields=s.get("missing",[])) for s in states.values() if s["status"]!="completed"],
      "new_disagreement_ids":differences,"new_true_injection_vs_accept_conflicts":base.pair_summary(new_pairs)["accept_vs_true_injection"],
      "conditional_label_only_capacity":capacity,"capacity_is_final_group_safe":False,
      "record_outcomes_metadata_only":outcomes,"optional_timing":base.times(timing),
      "timing_policy":"No timing required; only actual voluntary values summarized. Previous25s is a projection basis, never imputed for new rows.",
      "next_step":"Complete all105 new human judgments" if not complete else
        "Human review of full180-pair failure; no further automatic expansion" if audit["B0_confirmation_gate_pass"] is not True else
        "Proceed separately with the approved accepted-evidence merge and complete post-review grouping/contamination audit; then exact capacity; any reserve review requires separate human approval",
      "final_pool_merge_performed":False,"reserve_review_launched":False,"final_ids_selected":False,
      "materialization":False,"training":False,"replay":False,"replay_readiness":"R0"}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=EXP_MANIFEST);parser.add_argument("--review-dir",type=Path,default=EXP_DIR)
    parser.add_argument("--check-package",action="store_true")
    parser.add_argument("--output",type=Path,default=ROOT/"results/h2_mini_b0_expansion_analysis.json")
    args=parser.parse_args(argv);d=inspect_expansion(args.manifest,args.review_dir)
    if args.check_package:
        blank=len(d["states"])==105 and all(s["status"]=="pending" for s in d["states"].values())
        unchanged=all(t["unchanged_prepared_bytes"] for t in d["transport"].values())
        ok=not d["errors"] and blank and unchanged
        print(json.dumps({"status":"BLANK_EXPANSION_PACKAGE_VALID" if ok else "PACKAGE_CHECK_FAILED",
          "new_judgments":len(d["states"]),"blank":blank,"old_pairs_retained":75,"planned_coverage":"180/600",
          "reviewer_assignments":d["manifest"]["reviewer_assignments"],"errors":d["errors"]},indent=2))
        return 0 if ok else 2
    report=analyze_expansion(d)
    if args.output.suffix.lower()!=".json" or not args.output.resolve().is_relative_to((ROOT/"results").resolve()):
        parser.error("Analysis output must be a JSON file under the active project's results directory.")
    if args.output.resolve()==BASE_RESULT.resolve():parser.error("Never overwrite the frozen1219-judgment baseline analysis")
    if args.output.exists():
        previous=args.output.read_bytes()
        history=ROOT/"results/h2_mini_b0_review_history"
        history.mkdir(parents=True,exist_ok=True)
        snapshot=history/("expanded_analysis_before_"+sha(previous)[:16]+".json")
        if snapshot.exists():
            if snapshot.read_bytes()!=previous:raise ValueError("Historical snapshot mismatch")
        else:snapshot.write_bytes(previous)
        report["prior_report_snapshot"]=str(snapshot.relative_to(ROOT)).replace("\\","/")
    save(args.output,report)
    print(json.dumps({k:report[k] for k in ["status","completed_new_judgments","expanded_audit","conditional_label_only_capacity","next_step"]},indent=2))
    return 2 if d["errors"] else 3 if not report["expanded_audit"]["all180_complete"] else 4 if report["status"].startswith("STOP") else 0
if __name__=="__main__":sys.exit(main())
