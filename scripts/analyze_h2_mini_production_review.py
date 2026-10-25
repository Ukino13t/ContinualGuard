"""Validate, freeze and analyze completed H2-mini human production review.
No model inference, adjudication, relabeling, split assignment or raw JSONL.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math, statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = {"ACCEPT_BENIGN","REJECT_TRUE_INJECTION","REJECT_AMBIGUOUS_INTENT","REJECT_HARMFUL_OR_UNSAFE","REJECT_CONTEXT_MISSING","REJECT_PRIVACY_OR_RIGHTS","REJECT_QUALITY"}
STRATA = {"B0","B1","B2","B3","UNSURE","NOT_APPLICABLE"}
CONFIDENCE = {"HIGH","MEDIUM","LOW"}
COHORTS = ["B0","B1","B2","B3","B-short"]
REQUIRED = dict(zip(COHORTS, [290,290,290,290,120]))
HUMAN_FIELDS = ["reviewer_decision","reviewer_stratum","reason_code","confidence","optional_note","review_seconds"]
GROUP_FIELDS = ["source_native_id","source_parent_id","source_immediate_parent_id","exact_group","normalized_group","model_view_group","project_semantic_parent_group","project_template_group"]
HASH_FIELDS = ["raw_text_sha256","normalized_text_sha256","model_view_sha256"]

def sha(data):
    if isinstance(data,str): data=data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()

def file_sha(path): return sha(path.read_bytes())
def load(path): return json.loads(path.read_text(encoding="utf-8-sig"))
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def relative(path): return path.resolve().relative_to(ROOT).as_posix()
def evidence_hash(row):
    return sha(json.dumps([row[k] for k in HUMAN_FIELDS],ensure_ascii=False,separators=(",",":")))
def rate(a,b): return a/b if b else None

def read_csv(path,fields):
    b=path.read_bytes()
    if b.startswith((b"\xff\xfe",b"\xfe\xff")): encodings=["utf-16"]
    else: encodings=["utf-8-sig","gb18030"]
    for enc in encodings:
        try: content=b.decode(enc,errors="strict"); break
        except UnicodeError: continue
    else: raise ValueError("No lossless supported decoding")
    r=csv.DictReader(io.StringIO(content,newline=""))
    if r.fieldnames!=fields: raise ValueError("Header differs from assignment manifest")
    rows=list(r)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError("Malformed CSV column count")
    return rows,enc,b

def validate(manifest,review_dir):
    records=manifest["records"]; byid={x["review_id"]:x for x in records}
    issues=[]; warnings=[]; expected={}; expected_pairs=set()
    if len(records)!=len(byid): raise ValueError("Duplicate manifest record")
    for x in records:
        roles=sorted(a["review_role"] for a in x["reviewer_assignments"])
        reviewers=[a["reviewer"] for a in x["reviewer_assignments"]]
        if len(set(reviewers))!=len(reviewers): raise ValueError("Repeated manifest reviewer")
        if x["review_lane"]=="TRAIN_CANDIDATE":
            if roles not in [["PRIMARY"],["PRIMARY","SECOND_REVIEW"]]: raise ValueError("Invalid training role contract")
        elif roles!=["DOUBLE_REVIEW","DOUBLE_REVIEW"] or set(reviewers)!={"A","B"}:
            raise ValueError("Invalid double-review role contract")
        for a in x["reviewer_assignments"]:
            key=(a["batch_filename"],x["review_id"])
            if key in expected: raise ValueError("Duplicate assignment")
            expected[key]=a; expected_pairs.add((a["reviewer"],x["review_id"]))
    filenames=sorted({key[0] for key in expected})
    for p in review_dir.glob("*.csv"):
        if p.name not in filenames: issues.append({"filename":p.name,"error":"unexpected CSV batch"})
    seen=set(); pairs=set(); tables={}; judgments=defaultdict(list)
    for name in filenames:
        p=review_dir/name
        if not p.is_file():
            issues.append({"filename":name,"error":"missing expected file"});continue
        try: rows,enc,b=read_csv(p,manifest["worksheet_columns"])
        except (ValueError,UnicodeError) as e:
            issues.append({"filename":name,"error":str(e)});continue
        tables[name]={"rows":rows,"encoding":enc,"submitted_sha256":sha(b),"bytes":b}
        for lineno,row in enumerate(rows,2):
            rid=row["review_id"]; key=(name,rid)
            def issue(msg): issues.append({"filename":name,"csv_row":lineno,"review_id":rid,"error":msg})
            if key not in expected: issue("unexpected review ID or assignment");continue
            a=expected[key]; pair=(a["reviewer"],rid)
            if key in seen or pair in pairs: issue("duplicate reviewer/review_id")
            seen.add(key); pairs.add(pair); r=byid[rid]
            for field in ["source","source_category","language","proposed_stratum"]:
                if row[field]!=r[field]:issue("immutable metadata changed: "+field)
            if row["review_role"]!=a["review_role"]:issue("assigned role changed")
            if sha(row["text"])!=r["raw_text_sha256"]:issue("immutable text hash mismatch; source recovery required")
            for field,allowed in [("reviewer_decision",DECISIONS),("reviewer_stratum",STRATA),("confidence",CONFIDENCE)]:
                if row[field] not in allowed:issue("missing/invalid "+field)
            if row["reviewer_decision"]!="ACCEPT_BENIGN" and not row["reason_code"].strip():issue("missing rejection reason_code")
            if row["reviewer_decision"]=="ACCEPT_BENIGN" and row["reviewer_stratum"]=="NOT_APPLICABLE":issue("accepted record has NOT_APPLICABLE stratum")
            if row["reviewer_stratum"]=="UNSURE" and not row["optional_note"].strip():issue("UNSURE requires note")
            try:
                seconds=float(row["review_seconds"])
                if not math.isfinite(seconds) or seconds<=0:raise ValueError()
            except ValueError:issue("missing/invalid review_seconds")
            judgments[rid].append({**row,"reviewer":a["reviewer"],"batch_filename":name,"csv_row":lineno})
    for name,rid in sorted(set(expected)-seen):
        issues.append({"filename":name,"review_id":rid,"error":"missing expected judgment"})
    return tables,judgments,{"expected_judgments":len(expected),"present_judgments":len(seen),
      "missing_judgments":len(set(expected)-seen),"malformed_judgments":len({(x.get("filename"),x.get("review_id")) for x in issues if x["error"]!="missing expected judgment"}),
      "expected_batches":len(filenames),"found_batches":len(tables),"issues":issues,"warnings":warnings}

def kappa(a,b):
    if not a:return {"value":None,"reason":"No paired observations"}
    ca,cb=Counter(a),Counter(b);n=len(a);labels=set(a)|set(b)
    pe=sum(ca[k]*cb[k] for k in labels)/(n*n)
    if math.isclose(pe,1):return {"value":None,"reason":"Degenerate category marginals; expected agreement is one"}
    return {"value":(sum(x==y for x,y in zip(a,b))/n-pe)/(1-pe),"reason":None,
      "interpretation":"Descriptive unweighted Cohen kappa; no independence-based significance claim"}

def agreement(records,judgments):
    pairs=[sorted(judgments[r["review_id"]],key=lambda j:j["reviewer"]) for r in records if len(judgments[r["review_id"]])==2]
    a=[x[0]["reviewer_decision"] for x in pairs];b=[x[1]["reviewer_decision"] for x in pairs]
    sa=[x[0]["reviewer_stratum"] for x in pairs];sb=[x[1]["reviewer_stratum"] for x in pairs]
    n=len(pairs);d=sum(x!=y for x,y in zip(a,b));s=sum(x!=y for x,y in zip(sa,sb))
    safe=sum({x,y}=={"ACCEPT_BENIGN","REJECT_TRUE_INJECTION"} for x,y in zip(a,b))
    patterns=Counter()
    for (x,y),(sx,sy) in zip(zip(a,b),zip(sa,sb)):
        if x!=y:
            patterns["benign_vs_reject" if (x=="ACCEPT_BENIGN")!=(y=="ACCEPT_BENIGN") else "different_rejection_decisions"]+=1
        elif sx!=sy:patterns["stratum_only_disagreement"]+=1
    def matrix(xs,ys):
        c=Counter(zip(xs,ys))
        return {x:{y:c[x,y] for y in sorted(set(ys))} for x in sorted(set(xs))}
    return {"pairs":n,"decision_agreement_count":n-d,"decision_agreement":rate(n-d,n),
      "decision_disagreement_count":d,"decision_disagreement_rate":rate(d,n),
      "decision_kappa":kappa(a,b),"stratum_agreement_count":n-s,"stratum_agreement":rate(n-s,n),
      "stratum_disagreement_count":s,"stratum_kappa":kappa(sa,sb),
      "any_decision_or_stratum_disagreement":sum(x!=y or sx!=sy for x,y,sx,sy in zip(a,b,sa,sb)),
      "safety_critical_disagreements":safe,"patterns":dict(patterns),
      "decision_confusion":matrix(a,b),"stratum_confusion":matrix(sa,sb)}

def timing(values):
    return {"judgments":len(values),"mean_seconds":statistics.fmean(values) if values else None,
      "median_seconds":statistics.median(values) if values else None,
      "p90_seconds":sorted(values)[math.ceil(.9*len(values))-1] if values else None,
      "total_seconds":sum(values),"person_hours":sum(values)/3600}

def outcome(record,rows):
    benign=all(x["reviewer_decision"]=="ACCEPT_BENIGN" for x in rows)
    matched=all(x["reviewer_stratum"]==record["proposed_stratum"] for x in rows)
    reasons=[]
    if not benign:reasons.append("one_or_more_human_rejections")
    if benign and not matched:reasons.append("human_stratum_not_unanimously_proposed")
    if len({x["reviewer_decision"] for x in rows})>1:reasons.append("unresolved_decision_disagreement")
    if len({x["reviewer_stratum"] for x in rows})>1:reasons.append("unresolved_stratum_disagreement")
    if {x["reviewer_decision"] for x in rows}=={"ACCEPT_BENIGN","REJECT_TRUE_INJECTION"}:
        reasons.append("safety_critical_disagreement")
    return {"all_required_accept_benign":benign,"eligible_in_proposed_stratum":benign and matched,"exclusion_reasons":reasons}

def accepted_metadata(record,rows,freeze):
    keys=["candidate_id","review_id","source","source_revision","source_native_id","source_parent_id","source_immediate_parent_id","source_row_index","source_category","language","review_cohort","proposed_stratum","review_lane","deberta_tokens",*HASH_FIELDS,*GROUP_FIELDS]
    out={k:record.get(k) for k in dict.fromkeys(keys)}
    out.update(project_record_id="h2m-"+sha(record["source_revision"]+"|"+record["candidate_id"])[:24],
      label=0,accepted_stratum=record["proposed_stratum"],human_review_status="accepted_under_frozen_rubric_in_proposed_stratum",
      rubric_version="h2-mini-review-rubric-v1",post_review_audit_status="pending",
      reviewer_evidence=[{"reviewer":x["reviewer"],"review_role":x["review_role"],
        "reviewer_decision":x["reviewer_decision"],"reviewer_stratum":x["reviewer_stratum"],
        "reason_code":x["reason_code"],"confidence":x["confidence"],"review_seconds_recorded":x["review_seconds"],
        "note_sha256":sha(x["optional_note"]),"human_fields_sha256":evidence_hash(x),
        "canonical_file":freeze[x["batch_filename"]]["canonical_path"],"canonical_file_sha256":freeze[x["batch_filename"]]["canonical_sha256"],
        "csv_row":x["csv_row"]} for x in rows])
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,default=ROOT/"review/production_review_manifest.json")
    p.add_argument("--review-dir",type=Path,default=ROOT/"review/production")
    p.add_argument("--output",type=Path,default=ROOT/"results/h2_mini_production_review_analysis.json")
    p.add_argument("--freeze-id",default="20261003-v1")
    p.add_argument("--verify-only",action="store_true",help="Validate completed evidence without changing any artifact")
    args=p.parse_args();m=load(args.manifest)
    if m["rubric_version"]!="h2-mini-review-rubric-v1":raise ValueError("Unapproved rubric")
    tables,j,v=validate(m,args.review_dir)
    if v["issues"]:
        save(ROOT/"results/h2_mini_production_review_validation.json",{"status":"STOP_HUMAN_CORRECTION_REQUIRED",**v})
        print(json.dumps(v,ensure_ascii=True,indent=2));return 2
    frozen_path=ROOT/"review/production_review_canonical_manifest.json"
    if args.verify_only:
        frozen=load(frozen_path)
        if frozen["assignment_manifest_sha256"]!=file_sha(args.manifest):
            raise ValueError("Frozen assignment manifest hash changed")
        for name,t in tables.items():
            f=frozen["files"][name]
            if t["submitted_sha256"]!=f["submitted_sha256"]:
                raise ValueError("Submitted evidence changed: "+name)
            snapshot=ROOT/f["original_snapshot_path"];canonical=ROOT/f["canonical_path"]
            if file_sha(snapshot)!=f["original_snapshot_sha256"] or snapshot.read_bytes()!=t["bytes"]:
                raise ValueError("Original evidence snapshot changed: "+name)
            if file_sha(canonical)!=f["canonical_sha256"]:
                raise ValueError("Canonical evidence hash changed: "+name)
            canonical_rows,_,_=read_csv(canonical,m["worksheet_columns"])
            if canonical_rows!=t["rows"]:raise ValueError("Canonical decoded fields differ: "+name)
        print(json.dumps({"status":"FROZEN_EVIDENCE_VERIFIED_READ_ONLY","batches":len(tables),
          "judgments":v["present_judgments"],"human_field_changes":0},indent=2))
        return 0
    completed_path=ROOT/"results/h2_mini_production_review_analysis.json"
    if completed_path.exists() and load(completed_path).get("post_review_audit"):
        raise ValueError("Completed post-review analysis is frozen. Use --verify-only; do not overwrite pools or evidence hashes.")
    stamp=datetime.now(timezone.utc).isoformat();files={}
    for name,t in tables.items():
        original=ROOT/"review/production_submitted"/args.freeze_id/name
        canonical=ROOT/"review/production_canonical"/args.freeze_id/name
        text=io.StringIO(newline="");w=csv.DictWriter(text,fieldnames=m["worksheet_columns"],lineterminator="\r\n")
        w.writeheader();w.writerows(t["rows"]);encoded=text.getvalue().encode("utf-8-sig")
        for path,data in [(original,t["bytes"]),(canonical,encoded)]:
            if path.exists() and path.read_bytes()!=data:raise ValueError("Refusing to overwrite frozen evidence: "+str(path))
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        canonical_rows,_,_=read_csv(canonical,m["worksheet_columns"])
        assert all([a[k] for k in HUMAN_FIELDS]==[b[k] for k in HUMAN_FIELDS] for a,b in zip(t["rows"],canonical_rows))
        files[name]={"submitted_path":relative(args.review_dir/name),"submitted_sha256":t["submitted_sha256"],
          "original_snapshot_path":relative(original),"original_snapshot_sha256":file_sha(original),
          "submitted_encoding":t["encoding"],"canonical_path":relative(canonical),"canonical_sha256":file_sha(canonical),
          "rows":len(t["rows"]),"exact_decoded_fields_preserved":canonical_rows==t["rows"]}
    canonical_manifest={"freeze_id":args.freeze_id,"frozen_at":stamp,"assignment_manifest_sha256":file_sha(args.manifest),
      "rubric_version":m["rubric_version"],"human_field_changes":0,"transport_normalization":"UTF-8 BOM / CSV quoting / CRLF only",
      "validation":v,"files":files}
    save(ROOT/"review/production_review_canonical_manifest.json",canonical_manifest)
    records=m["records"];outcomes={r["review_id"]:outcome(r,j[r["review_id"]]) for r in records}
    def summarise(rr):
        benign=sum(outcomes[r["review_id"]]["all_required_accept_benign"] for r in rr)
        eligible=sum(outcomes[r["review_id"]]["eligible_in_proposed_stratum"] for r in rr)
        return {"candidates":len(rr),"all_required_accept_benign":benign,"benign_acceptance_rate":rate(benign,len(rr)),
          "eligible_in_proposed_stratum":eligible,"target_stratum_acceptance_rate":rate(eligible,len(rr)),
          "excluded_at_least_one_rejection":len(rr)-benign,"benign_but_stratum_not_confirmed":benign-eligible,
          "decision_counts_per_judgment":dict(Counter(x["reviewer_decision"] for r in rr for x in j[r["review_id"]])),
          "rejection_reason_counts_per_judgment":dict(Counter(x["reason_code"] for r in rr for x in j[r["review_id"]] if x["reviewer_decision"]!="ACCEPT_BENIGN")),
          "dominant_exclusions":dict(Counter(reason for r in rr for reason in outcomes[r["review_id"]]["exclusion_reasons"]))}
    cohorts={c:summarise([r for r in records if r["review_cohort"]==c]) for c in COHORTS}
    paired=[r for r in records if len(j[r["review_id"]])==2]
    double=[r for r in paired if r["review_lane"]!="TRAIN_CANDIDATE"]
    audits={c:agreement([r for r in paired if r["review_lane"]=="TRAIN_CANDIDATE" and r["review_cohort"]==c],j) for c in COHORTS[:-1]}
    audit_failures=[c for c,a in audits.items() if a["decision_disagreement_rate"] is None or a["decision_disagreement_rate"]>.10]
    unresolved_stratum=[c for c,a in audits.items() if a["stratum_agreement"] is not None and a["stratum_agreement"]<.9]
    pools={}
    for c in COHORTS:
        rr=[accepted_metadata(r,j[r["review_id"]],files) for r in records if r["review_cohort"]==c and outcomes[r["review_id"]]["eligible_in_proposed_stratum"]]
        dest=ROOT/"data/manifests/hard_benign"/(c.replace("-","_")+"_accepted.json")
        save(dest,{"status":"human_accepted_proposed_stratum_post_review_audit_pending","rubric_version":m["rubric_version"],
          "canonical_manifest_sha256":file_sha(ROOT/"review/production_review_canonical_manifest.json"),"records":rr,
          "count":len(rr),"raw_text_in_manifest":False,"final_split_assigned":False})
        pools[c]={"path":relative(dest),"count":len(rr)}
    save(ROOT/"data/audits/hard_benign/h2_mini_review_outcomes.json",{"outcomes":[
      {"review_id":r["review_id"],"review_cohort":r["review_cohort"],"review_lane":r["review_lane"],
       **outcomes[r["review_id"]],"reviewer_labels":[{k:x[k] for k in ["reviewer","reviewer_decision","reviewer_stratum","confidence","reason_code"]} for x in j[r["review_id"]]]}
      for r in records],"automatic_adjudication":False,"automatic_stratum_reassignment":False})
    times={who:timing([float(x["review_seconds"]) for rows in j.values() for x in rows if x["reviewer"]==who]) for who in ["A","B"]}
    times["combined"]=timing([float(x["review_seconds"]) for rows in j.values() for x in rows])
    result={"status":"REVIEW_ANALYSIS_COMPLETE_POST_AUDIT_PENDING","analyzed_at":stamp,"rubric_version":m["rubric_version"],
      "package_id":m["package_id"],"validation":v,"completed_judgments":sum(len(x) for x in j.values()),
      "candidate_counts":m["candidate_counts"],"by_cohort":cohorts,
      "by_source":{s:summarise([r for r in records if r["source"]==s]) for s in sorted({r["source"] for r in records})},
      "by_source_category":{s:summarise([r for r in records if r["source_category"]==s]) for s in sorted({r["source_category"] for r in records if r["source_category"]})},
      "double_review":agreement(double,j),"all_paired_review":agreement(paired,j),
      "double_review_by_lane":{lane:agreement([r for r in double if r["review_lane"]==lane],j) for lane in ["VALIDATION_RESERVE","TEST_RESERVE","B_SHORT"]},
      "training_second_review_audit":audits,"training_decision_audit_failures":audit_failures,
      "training_stratum_disagreement_flags":unresolved_stratum,
      "gate_interpretation":"Predeclared >10% gate uses decision disagreement. Stratum disagreements are separately reported, excluded at record level, and cannot establish target-stratum reliability even when the decision gate passes.",
      "accepted_pools":pools,"timing":times,"minimum_required":REQUIRED,
      "input_sha256":{relative(args.review_dir/n):t["submitted_sha256"] for n,t in tables.items()},
      "canonical_manifest":"review/production_review_canonical_manifest.json",
      "automatic_adjudication":False,"final_split_assigned":False,"replay_readiness":"R0"}
    save(args.output,result)
    print(json.dumps({"completed":result["completed_judgments"],"eligible_counts":{c:pools[c]["count"] for c in COHORTS},
      "training_decision_audit_failures":audit_failures,"stratum_flags":unresolved_stratum,"person_hours":times["combined"]["person_hours"]},indent=2))
    return 0

if __name__=="__main__":raise SystemExit(main())
