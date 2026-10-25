"""Validate and analyze the approved final B0 top-up human reviews."""
import argparse,csv,io,json,math,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from h2_mini_b0_review_common import ROOT,load,save,sha,file_sha,DECISIONS,B0_VALUES,CONFIDENCE

MANIFEST=ROOT/"review/b0_final_topup_manifest.json"
DIRECTORY=ROOT/"review/b0_final_topup"
OUTPUT=ROOT/"results/h2_mini_b0_final_topup_analysis.json"
FIELDS=["review_id","text","source","language","token_length","review_role",
        "reviewer_decision","b0_confirmed","reason_code","confidence","optional_note"]
HUMAN=["reviewer_decision","b0_confirmed","reason_code","confidence","optional_note"]
ROLES={"PRIMARY","DOUBLE_REVIEW"}

def agreement(pairs,field):
    n=len(pairs);a=[x[field] for x,y in pairs];b=[y[field] for x,y in pairs]
    same=sum(x==y for x,y in zip(a,b));ca=Counter(a);cb=Counter(b)
    pe=sum(ca[k]*cb[k] for k in set(ca)|set(cb))/n**2 if n else None
    return {"pairs":n,"agreements":same,"disagreements":n-same,
      "agreement_rate":same/n if n else None,"disagreement_rate":(n-same)/n if n else None,
      "cohen_kappa":((same/n-pe)/(1-pe) if n and pe is not None and pe<1 else None)}

def inspect(manifest_path=MANIFEST,directory=DIRECTORY):
    m=load(manifest_path);errors=[];states={};hashes={};transport={}
    def error(message,**where):errors.append({"message":message,**where})
    if m.get("status")!="PREPARED_AWAITING_HUMAN_REVIEW":error("Unexpected manifest status")
    for rel,digest in m.get("frozen_dependencies",{}).items():
        p=ROOT/rel
        if not p.exists() or file_sha(p)!=digest:error("Frozen dependency changed or missing",path=rel)
    expected={r["review_id"]:r for r in m["records"]}
    if len(expected)!=65:error("Expected65 unique reserve candidates")
    pairs=set()
    for filename,batch in m["batches"].items():
        path=directory/filename
        if not path.exists():error("Missing worksheet",file=filename);continue
        raw=path.read_bytes();hashes[filename]=sha(raw)
        try:
            reader=csv.DictReader(io.StringIO(raw.decode("utf-8-sig",errors="strict"),newline=""),strict=True)
            fields=list(reader.fieldnames or []);rows=list(reader)
        except (UnicodeError,csv.Error) as exc:
            error("Invalid UTF-8/CSV",file=filename,detail=str(exc));continue
        if fields!=FIELDS:error("Worksheet header differs from frozen schema",file=filename);continue
        if len(rows)!=batch["rows"]:error("Worksheet row count changed",file=filename)
        if Counter(r.get("review_id") for r in rows)!=Counter(batch["review_ids"]):
            error("Worksheet review IDs changed",file=filename)
        transport[filename]={"encoding":"utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8",
            "rows":len(rows),"prepared_bytes_unchanged":sha(raw)==batch["prepared_sha256"]}
        for row in rows:
            rid=row.get("review_id","");who=batch["reviewer"];where={"file":filename,"review_id":rid,"reviewer":who}
            if None in row or any(row.get(k) is None for k in fields):error("Malformed row width",**where);continue
            if rid not in expected:error("Unexpected review_id",**where);continue
            key=(who,rid)
            if key in pairs:error("Duplicate reviewer/review_id judgment",**where);continue
            pairs.add(key);meta=expected[rid]
            assignment=[a for a in meta["reviewer_assignments"] if a["reviewer"]==who]
            if len(assignment)!=1:error("Reviewer is not assigned exactly once",**where);continue
            role=assignment[0]["review_role"]
            immutable={"source":"OpenAssistant/oasst1","language":meta["language"],
                "token_length":str(meta["deberta_tokens"]),"review_role":role}
            if sha(row["text"])!=meta["raw_text_sha256"]:error("Immutable source text changed",field="text",**where)
            for field,value in immutable.items():
                if row[field]!=value:error("Immutable worksheet metadata changed",field=field,**where)
            values={k:row[k].strip() for k in HUMAN}
            status="pending";missing=[];invalid=[]
            if any(values.values()):
                for field,allowed in [("reviewer_decision",DECISIONS),("b0_confirmed",B0_VALUES),("confidence",CONFIDENCE)]:
                    if values[field] and values[field] not in allowed:invalid.append(field)
                missing=[k for k in ["reviewer_decision","b0_confirmed","confidence"] if not values[k]]
                if values["reviewer_decision"].startswith("REJECT_") and not values["reason_code"]:missing.append("reason_code")
                status="invalid" if invalid else "partial" if missing else "completed"
                if invalid:error("Invalid human field",fields=invalid,**where)
            states[key]={"status":status,"values":values,"missing":missing,"role":role,
                "human_fields_sha256":sha(json.dumps({k:row[k] for k in HUMAN},ensure_ascii=False,sort_keys=True))}
    expected_pairs={(a["reviewer"],r["review_id"]) for r in m["records"] for a in r["reviewer_assignments"]}
    for who,rid in sorted(expected_pairs-pairs):error("Missing assigned judgment row",reviewer=who,review_id=rid)
    if pairs-expected_pairs:error("Unexpected reviewer/review_id pairs",pairs=sorted(pairs-expected_pairs))
    return {"manifest":m,"states":states,"errors":errors,"worksheet_sha256":hashes,
            "transport":transport,"expected_pairs":expected_pairs}

def analyze(data):
    m=data["manifest"];states=data["states"];complete=not data["errors"] and len(states)==90 and all(s["status"]=="completed" for s in states.values())
    pending=[{"reviewer":who,"review_id":rid,"status":s["status"],"missing_fields":s["missing"]}
             for (who,rid),s in states.items() if s["status"]!="completed"]
    if data["errors"]:
        return {"status":"STOP_TOPUP_INTEGRITY_OR_FIELD_ERROR","errors":data["errors"],
            "completed_judgments":sum(s["status"]=="completed" for s in states.values()),
            "capacity_decision":None,"model_training_performed":False,"replay":False}
    training=[];evaluation=[];decision_pairs=[];b0_pairs=[];safety=0
    training_rejection_decisions=Counter();training_rejection_reasons=Counter();evaluation_rejection_decisions=Counter();evaluation_rejection_reasons=Counter()
    for r in m["records"]:
        assignments=r["reviewer_assignments"];vals=[states[(a["reviewer"],r["review_id"])]["values"] for a in assignments
            if states[(a["reviewer"],r["review_id"])]["status"]=="completed"]
        rejection_decisions,rejection_reasons=(training_rejection_decisions,training_rejection_reasons) if r["reserve_purpose"]=="TRAINING_REVIEW_RESERVE" else (evaluation_rejection_decisions,evaluation_rejection_reasons)
        for v in vals:
            if v["reviewer_decision"]!="ACCEPT_BENIGN":
                rejection_decisions[v["reviewer_decision"]]+=1;rejection_reasons[v["reason_code"]]+=1
        if r["reserve_purpose"]=="TRAINING_REVIEW_RESERVE":
            eligible=len(vals)==1 and vals[0]["reviewer_decision"]=="ACCEPT_BENIGN" and vals[0]["b0_confirmed"]=="YES"
            training.append({"review_id":r["review_id"],"eligible":eligible})
        else:
            eligible=len(vals)==2 and all(v["reviewer_decision"]=="ACCEPT_BENIGN" and v["b0_confirmed"]=="YES" for v in vals)
            if len(vals)==2:
                decision_pairs.append((vals[0],vals[1]));b0_pairs.append((vals[0],vals[1]))
                safety+=int({vals[0]["reviewer_decision"],vals[1]["reviewer_decision"]}=={"ACCEPT_BENIGN","REJECT_TRUE_INJECTION"})
            evaluation.append({"review_id":r["review_id"],"eligible":eligible})
    t=sum(x["eligible"] for x in training);e=sum(x["eligible"] for x in evaluation)
    base=m["frozen_capacity"];final={"training":base["training"]+t,"evaluation":base["evaluation"]+e,"unique":base["unique"]+t+e}
    pass_capacity=complete and t>=m["required_topup"]["training"] and e>=m["required_topup"]["evaluation"] and final["training"]>=800 and final["evaluation"]>=360 and final["unique"]>=1160
    status=("PENDING_HUMAN_REVIEW" if not complete else
      "TOPUP_CAPACITY_PASS_CONTINUE_APPROVED_FINALIZATION" if pass_capacity else "STOP_TOPUP_CAPACITY_SHORTAGE")
    return {"status":status,"generated_at":datetime.now(timezone.utc).isoformat(),
      "expected_judgments":90,"completed_judgments":sum(s["status"]=="completed" for s in states.values()),
      "pending_or_partial":pending,"training":{"reviewed_candidates":40,"accepted_B0_YES":t,
        "rejection_decisions":dict(training_rejection_decisions),"rejection_reasons":dict(training_rejection_reasons),
        "additional_train_eligible":t,"required":20,"passes":t>=20 if complete else None},
      "evaluation":{"double_reviewed_candidates":25,"both_ACCEPT_and_both_YES":e,
        "decision_agreement":agreement(decision_pairs,"reviewer_decision"),
        "B0_confirmation_agreement":agreement(b0_pairs,"b0_confirmed"),
        "safety_critical_conflicts":safety,"rejection_decisions":dict(evaluation_rejection_decisions),
        "rejection_reasons":dict(evaluation_rejection_reasons),"additional_evaluation_eligible":e,
        "required":10,"passes":e>=10 if complete else None},
      "frozen_capacity":base,"final_capacity":final if complete else None,
      "capacity_decision":pass_capacity if complete else None,
      "remaining_shortage":None if not complete else {"training":max(0,800-final["training"]),
        "evaluation":max(0,360-final["evaluation"]),"unique":max(0,1160-final["unique"])},
      "eligible_record_ids_metadata_only":{"training":[x["review_id"] for x in training if x["eligible"]],
        "evaluation":[x["review_id"] for x in evaluation if x["eligible"]]} if complete else None,
      "worksheet_sha256":data["worksheet_sha256"],"transport":data["transport"],
      "next_step":"Humans complete all90 independent judgments" if not complete else
        "Continue approved final accepted-pool/grouping/assignment validation workflow; stop before materialization" if pass_capacity else
        "Stop and request explicit human approval for any additional top-up",
      "automatic_continuation_scope":"After complete PASS: final B0 merge/audit, B0/BH/B-short/malicious hash-only assignments and validations only; no JSONL, training or replay.",
      "human_judgments_inferred":False,"detector_outputs_used":False,"materialization":False,
      "model_training_performed":False,"replay":False,"numerical_memory_budget":None,"replay_readiness":"R0"}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=MANIFEST)
    parser.add_argument("--review-dir",type=Path,default=DIRECTORY)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--check-package",action="store_true")
    args=parser.parse_args(argv);data=inspect(args.manifest,args.review_dir);report=analyze(data)
    if args.check_package:
        ok=not data["errors"] and len(data["states"])==90 and all(s["status"]=="pending" for s in data["states"].values())
        unchanged=all(x["prepared_bytes_unchanged"] for x in data["transport"].values())
        result={"status":"BLANK_TOPUP_PACKAGE_VALID" if ok and unchanged else "PACKAGE_CHECK_FAILED",
            "A":sum(w=="A" for w,r in data["expected_pairs"]),"B":sum(w=="B" for w,r in data["expected_pairs"]),
            "total_judgments":len(data["expected_pairs"]),"all_human_fields_blank":ok,
            "prepared_bytes_unchanged":unchanged,"errors":data["errors"]}
        print(json.dumps(result,indent=2));return 0 if result["status"]=="BLANK_TOPUP_PACKAGE_VALID" else 2
    if not args.output.resolve().is_relative_to((ROOT/"results").resolve()) or args.output.suffix.lower()!=".json":
        parser.error("Output must be JSON under results")
    if args.output.exists():raise ValueError("Preserve prior top-up analysis; archive explicitly before rerun")
    save(args.output,report);print(json.dumps({k:report[k] for k in ["status","completed_judgments","training","evaluation","final_capacity","remaining_shortage","next_step"]},indent=2))
    return 2 if data["errors"] else 3 if report["status"]=="PENDING_HUMAN_REVIEW" else 4 if report["status"].startswith("STOP") else 0
if __name__=="__main__":raise SystemExit(main())

