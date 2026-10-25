"""Merge eligible B0 metadata and audit frozen grouping rules; stop on capacity shortage."""
import csv,gzip,json,sys,time,contextlib
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import analyze_h2_mini_b0_expansion as analyzer
from h2_mini_b0_expansion_common import *
from h2_mini_b0_review_common import rel
import audit_h2_mini_sources as src
import audit_h2_mini_post_review as post
from transformers import AutoTokenizer

PRE=ROOT/"data/manifests/hard_benign/B0_expanded_review_eligible.json"
FINAL=ROOT/"data/manifests/hard_benign/B0_expanded_group_safe.json"
AUDIT=ROOT/"data/audits/hard_benign/h2_mini_b0_expanded_post_review_audit.json"
CAP=ROOT/"results/h2_mini_b0_expanded_capacity.json"
class QuietProgress:
    def write(self,text):return len(text)
    def flush(self):pass

def counts(records):
    train=[r for r in records if r["review_lane"]=="TRAIN_CANDIDATE"]
    evaluation=[r for r in records if r["review_lane"] in {"VALIDATION_RESERVE","TEST_RESERVE","EVALUATION_RESERVE"}]
    return {"training":len(train),"audited_training":sum(len(r["reviewer_evidence"])==2 for r in train),
        "double_evaluation":len(evaluation),"unique":len(records),
        "by_review_lane":dict(Counter(r["review_lane"] for r in records))}

def snapshot(paths):
    return {rel(p):{"sha256":file_sha(p),"mtime_ns":p.stat().st_mtime_ns} for p in paths}

def main():
    started=time.monotonic()
    assert Path.cwd().resolve()==ROOT.resolve()
    assert not any(p.exists() for p in [FINAL,AUDIT,CAP]),"Preserve completed audit artifacts"
    expanded=load(ROOT/"results/h2_mini_b0_expansion_analysis.json")
    assert expanded["expanded_audit"]["all180_complete"] and expanded["expanded_audit"]["B0_confirmation_gate_pass"]
    assert expanded["expanded_audit"]["observed_pairs"]==180
    d=analyzer.inspect_expansion();assert not d["errors"]
    baseline=d["base_data"];selected={r["review_id"]:r for r in d["manifest"]["records"]}
    recovery=load(ROOT/"review/b0_replenishment_expansion_integrity_recovery_manifest.json")
    provenance={r["review_id"]:r for r in recovery["row_provenance"]}
    old_path=ROOT/"data/manifests/hard_benign/B0_accepted.json";old=load(old_path)["records"]
    bh_path=ROOT/"results/h2_mini_bh_capacity.json";bh=load(bh_path)["proposed_BH_records_metadata_only"]
    old_prep=load(ROOT/recovery["preparation_history"]["path"])
    protected={r for r in old_prep["protected_file_snapshots"]}
    protected.update([rel(old_path),rel(bh_path),rel(EXP_MANIFEST),
        "review/b0_replenishment_expansion_integrity_recovery_manifest.json",
        "results/h2_mini_b0_expansion_integrity_recovery.json","results/h2_mini_b0_expansion_analysis.json",
        rel(RESERVE),rel(RESERVE_AUDIT)])
    protected.update(recovery[k]["path"] for k in ["original_damaged_A","Reviewer_B","reconfirmation_17","canonical_A"])
    before=snapshot([ROOT/x for x in protected])
    records=[dict(r,ordinary_B0_evidence_policy="previously_frozen_B0_pool",final_assignment=None) for r in old]
    review_exclusions=[];new_eligible=[]
    def evidence(assignment,state,rid):
        who=assignment["reviewer"];fn=assignment["batch_filename"]
        expansion=rid in selected and assignment["review_role"]=="SECOND_REVIEW" and who==selected[rid]["second_reviewer"]
        path=(ROOT/recovery["canonical_A"]["path"] if who=="A" else ROOT/recovery["Reviewer_B"]["path"]) if expansion else ROOT/"review/b0_replenishment"/fn
        v=state["values"]
        ev={"reviewer":who,"review_role":assignment["review_role"],"reviewer_decision":v["reviewer_decision"],
            "b0_confirmed":v["b0_confirmed"],"confidence":v["confidence"],"reason_code":v["reason_code"],
            "human_fields_sha256":state.get("human_fields_sha256",state.get("row_sha256")),"evidence_file":rel(path),"evidence_file_sha256":file_sha(path)}
        if expansion and who=="A":ev["recovery_provenance"]=provenance[rid]["provenance"]
        return ev
    for original in baseline["manifest"]["records"]:
        r=dict(original);rid=r["review_id"];ev=[]
        for a in r["reviewer_assignments"]:ev.append(evidence(a,baseline["states"][(a["reviewer"],rid)],rid))
        if rid in selected:
            a={"reviewer":selected[rid]["second_reviewer"],"review_role":"SECOND_REVIEW","batch_filename":selected[rid]["batch_filename"]}
            ev.append(evidence(a,d["states"][rid],rid))
        ok=all(x["reviewer_decision"]=="ACCEPT_BENIGN" and x["b0_confirmed"]=="YES" for x in ev)
        if not ok:
            review_exclusions.append({"review_id":rid,"review_lane":r["review_lane"],
                "reasons":sorted({x["reviewer_decision"] for x in ev if x["reviewer_decision"]!="ACCEPT_BENIGN"}
                    |({"B0_NOT_CONFIRMED_BY_ALL_ASSIGNED_REVIEWERS"} if any(x["b0_confirmed"]!="YES" for x in ev) else set())
                    |({"UNRESOLVED_DECISION_CONFLICT"} if len({x["reviewer_decision"] for x in ev})>1 else set())
                    |({"UNRESOLVED_B0_CONFIRMATION_CONFLICT"} if len({x["b0_confirmed"] for x in ev})>1 else set())),
                "judgments":ev})
            continue
        r.update(project_record_id="CGHB-"+rid,label=0,accepted_stratum="B0",rubric_version=POLICY,
            reviewer_evidence=ev,human_review_status="accepted_ordinary_B0",independent_audit_status=
            ("accepted_independent_second_review" if r["review_lane"]=="TRAIN_CANDIDATE" and len(ev)==2 else
             "primary_reviewed_not_sampled" if r["review_lane"]=="TRAIN_CANDIDATE" else "accepted_double_review"),
            ordinary_B0_evidence_policy="primary_ACCEPT_BENIGN_and_YES_and_no_conflict_in_assigned_review_evidence",
            final_assignment=None,post_review_audit_status="pending")
        new_eligible.append(r)
    records+=new_eligible
    assert len({r["review_id"] for r in records})==len(records)
    assert all(r["label"]==0 and r["source"]=="OASST1" for r in records)
    # No single-primary record is rejected merely for not being selected in the independent audit.
    assert all(r["review_lane"]=="TRAIN_CANDIDATE" or len(r["reviewer_evidence"])==2 for r in records)
    pre_counts=counts(records)
    pre_report={"status":"REVIEW_ELIGIBLE_PENDING_GROUPING","taxonomy":POLICY,"records":records,"counts":pre_counts,
        "review_exclusions":review_exclusions,"raw_text_present":False,"final_assignment":False,
        "eligibility_policy":"Retain primary-qualified unsampled training records; retain assigned audit evidence. Quarantine unresolved decision or B0 conflicts under the frozen no-policy-conflict rule; all evaluation records must be double ACCEPT_BENIGN/YES.",
        "historical_pool_unchanged":rel(old_path),"full180_gate_report_sha256":file_sha(ROOT/"results/h2_mini_b0_expansion_analysis.json")}
    if PRE.exists():
        assert load(PRE)==pre_report,"Existing review-eligible metadata differs from deterministic reconstruction"
    else:
        save(PRE,pre_report)
    print("Review-qualified B0 metadata",json.dumps(pre_counts),flush=True)
    source_hashes={}
    def verify(path,digest):
        assert file_sha(path)==digest,rel(path)
        source_hashes[rel(path)]=digest
    for fn,data_name in [("oasst1_full_audit.json","2023-04-12_oasst_ready.messages.jsonl.gz"),("or_bench_full_audit.json","or-bench-80k.csv")]:
        meta=load(ROOT/"data/audits/hard_benign"/fn);entry=next(x for x in meta["source"]["files"] if x["file_name"]==data_name)
        verify(ROOT/entry["local_path"],entry["sha256"])
    with gzip.open(src.OAP/"2023-04-12_oasst_ready.messages.jsonl.gz","rt",encoding="utf-8") as f:oa={i:json.loads(line) for i,line in enumerate(f) if line.strip()}
    with (src.ORP/"or-bench-80k.csv").open(encoding="utf-8-sig",newline="") as f:ore=list(csv.DictReader(f))
    oa_meta={r["candidate_id"]:r for r in load(ROOT/"data/manifests/hard_benign/b_short_mini_candidates.json")["records"]}
    or_meta={r["candidate_id"]:r for r in load(ROOT/"data/manifests/hard_benign/h2_mini_candidates.json")["records"]}
    historical=[dict(r) for r in load(ROOT/"review/production_review_manifest.json")["records"]]
    for original in load(ROOT/"review/calibration_manifest.json")["records"]:
        r=dict(original)
        r["review_cohort"]="CALIBRATION"
        if r["source"]=="OASST1":r.update(oa_metadata(oa_meta[r["candidate_id"]]))
        else:
            source=or_meta.get(r["candidate_id"],{})
            r.update({k:source.get(k,r.get(k)) for k in HASH_FIELDS+GROUP_FIELDS})
            r["raw_text_sha256"]=original["text_sha256"]
        historical.append(r)
    historical+=[dict(r) for r in baseline["manifest"]["records"]]
    assert len(historical)==2512 and len({r["review_id"] for r in historical})==2512
    hby={r["review_id"]:r for r in historical}
    assert {r["review_id"] for r in bh}<=set(hby)
    bs=[r for r in historical if r["review_cohort"]=="B-short"];assert len(bs)==150
    accepted_ids={r["review_id"] for r in records}
    # Accepted records are audited internally, not treated as contamination of their own review provenance.
    external_history=[r for r in historical if r["review_id"] not in accepted_ids]
    assert len(external_history)+len(accepted_ids)==2512
    def text_of(r):
        if r["source"]=="OASST1":
            assert r["source_revision"]==src.OAR
            row=oa[r["source_row_index"]]
            assert row["message_id"]==r["source_native_id"] and row["message_tree_id"]==r["source_parent_id"]
            text=row["text"]
        else:
            assert r["source_revision"]==src.ORR;text=ore[r["source_row_index"]]["prompt"]
        assert sha(text)==r["raw_text_sha256"],r["review_id"]
        return text
    ni=src.load_ni();ns=load(ROOT/"data/manifests/hard_benign/notinject/source.json")
    verify(ROOT/ns["records_manifest"],ns["records_manifest_sha256"]);niby={r["id"]:r for r in ni}
    for r in [json.loads(line) for line in (ROOT/ns["records_manifest"]).read_text(encoding="utf-8").splitlines()]:
        assert sha(niby[r["source_split"]+":"+str(r["source_row_idx"])]["text"])==r["prompt_sha256"]
    afmeta=load(ROOT/"data/manifests/attack_family_shift/construction.json")
    verify(ROOT/afmeta["processed_file"],afmeta["processed_sha256"]);af=[]
    for line in (ROOT/afmeta["processed_file"]).open(encoding="utf-8"):
        r=json.loads(line)
        if r["task_id"] in {"AF1","AF2","AF3","AF4"}:
            assert sha(r["text"])==r["text_sha256"]
            af.append({"id":r["example_id"],"text":r["text"],"family":r["task_id"],"label":r["label"]})
    assert len(ni)==339 and len(af)==14640
    tok=AutoTokenizer.from_pretrained(ROOT/"data/raw/huggingface/microsoft_deberta-v3-base"/src.TOKR,local_files_only=True,use_fast=False)
    print("Fresh source/model-view integrity checks",flush=True)
    ad=post.descriptions(records,[text_of(r) for r in records],tok)
    hd=post.descriptions(external_history,[text_of(r) for r in external_history],tok)
    nd=post.descriptions(ni,[r["text"] for r in ni],tok);fd=post.descriptions(af,[r["text"] for r in af],tok)
    for rr,dd in [(records,ad),(external_history,hd)]:
        for r,dsc in zip(rr,dd):
            assert (r["raw_text_sha256"],r["normalized_text_sha256"],r["deberta_tokens"])==(dsc["raw"],dsc["normalized_hash"],dsc["tokens"])
            if r.get("model_view_sha256"):assert r["model_view_sha256"]==dsc["model_view"]
            r["model_view_sha256"]=dsc["model_view"]
    print("Fit unchanged OR lexical/SVD basis, no detector",flush=True)
    hv,sv,_,_=src.fit_embed([src.norm(r["prompt"]) for r in ore])
    emb=lambda ds:src.xembed([d["norm"] for d in ds],hv,sv)
    ae=emb(ad)
    def compare(name,b,be,internal=False):
        print("Comparing",name,len(ad),"x",len(b),flush=True)
        with contextlib.redirect_stdout(QuietProgress()):out=post.compare(ad,ae,b,be,internal=internal)
        print("Completed",name,"affected",out["affected_queries"],flush=True)
        return out
    uf,stored=post.stored_group_audit(records);internal=compare("accepted internal",ad,ae,True)
    index={r["review_id"]:i for i,r in enumerate(records)}
    for pair in internal["pairs"]:uf.union(index[pair["query_id"]],index[pair["target_id"]])
    exclusions=[]
    for group in uf.groups():
        if len(group)<2:continue
        representative=min(group,key=lambda i:post.representative_key(records[i]))
        for i in group:
            if i!=representative:exclusions.append({"review_id":records[i]["review_id"],"reason":"prohibited_accepted_group",
                "retained_representative":records[representative]["review_id"],
                "members":[records[j]["review_id"] for j in group]})
    external={"all_nonaccepted_prior_review_and_reserves":compare("nonaccepted prior review and reserves",hd,emb(hd)),
        "NotInject":compare("NotInject",nd,emb(nd)),"AF1_AF4_all_labels":compare("AF1-AF4",fd,emb(fd))}
    # Native, tree, stored semantic/template and content hashes use unchanged source-scoped/global rules.
    blocked=defaultdict(list)
    for r in external_history:
        for key in group_keys(r):blocked[key].append(r["review_id"])
    stored_external=[]
    for r in records:
        for key in group_keys(r):
            for target in blocked.get(key,[]):
                stored_external.append({"query_id":r["review_id"],"target_id":target,"method":key[0] if len(key)==2 else key[2]})
    external["stored_prohibited_groups"]={"pairs":stored_external,"affected_queries":len({x["query_id"] for x in stored_external})}
    for bank,report in external.items():
        for rid in sorted({x["query_id"] for x in report["pairs"]}):
            exclusions.append({"review_id":rid,"reason":"external_overlap_quarantine","bank":bank})
    for r,dsc in zip(records,ad):
        if dsc["tokens"]>1280:exclusions.append({"review_id":r["review_id"],"reason":"above_max_length"})
    excluded={r["review_id"] for r in exclusions}
    final=[dict(r,post_review_audit_status="passed_with_documented_semantic_limits") for r in records if r["review_id"] not in excluded]
    c=counts(final)
    train_short=max(0,800-c["training"]);eval_short=max(0,360-c["double_evaluation"]);total_short=max(0,1160-c["unique"])
    audit_short=max(0,80-c["audited_training"])
    passes=not any([train_short,eval_short,total_short,audit_short])
    lane=c["by_review_lane"];fval=lane.get("VALIDATION_RESERVE",0);ftest=lane.get("TEST_RESERVE",0);flex=lane.get("EVALUATION_RESERVE",0)
    demand_val=max(0,120-fval);demand_test=max(0,240-ftest)
    capacity={"status":"CAPACITY_PASS_REQUIRES_ASSIGNMENT" if passes else "STOP_B0_CAPACITY_SHORTAGE",
        "counts":c,"targets":{"training":800,"audited_training":80,"validation":120,"test":240,"unique":1160},
        "shortages":{"training":train_short,"double_evaluation":eval_short,"unique":total_short,"audited_training":audit_short},
        "evaluation_joint_feasibility":{"existing_validation_lane":fval,"existing_test_lane":ftest,"flexible_double_evaluation":flex,
            "additional_validation_demand":demand_val,"additional_test_demand":demand_test,
            "combined_demand":demand_val+demand_test,"combined_shortage":max(0,demand_val+demand_test-flex),
            "no_split_assignment_made":True,"illustration_only":
            "Meeting120 validation first leaves"+str(max(0,flex-demand_val)+ftest)+" for test; the same flexible record cannot count twice."},
        "independent_audit_coverage":{"available":c["audited_training"],"required_total":80,"required_per_stage":20,
            "unassigned_count_feasible_for_four_stages":c["audited_training"]>=80,"stage_assignment_performed":False},
        "all_requirements_pass":passes,"minimum_accepted_topup":{"training":train_short,"double_evaluation":eval_short,"total":train_short+eval_short},
        "frozen_reserve":{"training_candidates":40,"evaluation_candidates":25,"candidate_counts_cover_minimum":
            train_short<=40 and eval_short<=25,"acceptance_not_guaranteed":True,
            "minimum_training_yield_required":train_short/40,"minimum_evaluation_yield_required":eval_short/25,
            "no_review_worksheets_created":True,"no_reserve_records_used":True},
        "group_audit":rel(AUDIT),"final_group_safe_pool":rel(FINAL),"thresholds_weakened":False,
        "training":False,"replay":False,"materialization":False,"final_assignment":False}
    subgroups={"BH_frozen999":{r["review_id"] for r in bh},"B_short_all150":{r["review_id"] for r in bs},
        "calibration50":{r["review_id"] for r in historical if r["review_id"] in {x["review_id"] for x in load(ROOT/"review/calibration_manifest.json")["records"]}},
        "other_nonaccepted_reviewed":{r["review_id"] for r in external_history}}
    scope={}
    for name,target_ids in subgroups.items():
        hits=[x for x in external["all_nonaccepted_prior_review_and_reserves"]["pairs"]+stored_external if x["target_id"] in target_ids]
        scope[name]={"target_count":len(target_ids),"affected_queries":len({x["query_id"] for x in hits}),"hits":hits}
    audit={"status":"POST_REVIEW_AUDIT_COMPLETE","created_at":datetime.now(timezone.utc).isoformat(),
        "source_hashes_verified":source_hashes,"review_eligible_before":pre_counts,"group_safe_after":c,
        "stored_group_findings":stored,"internal_accepted_comparison":internal,"external_comparison":external,
        "external_subgroup_coverage":scope,"historical_union":2512,"accepted_provenance_identities_screened_internally":len(records),
        "nonaccepted_historical_records":len(external_history),"NotInject":339,"AF1_AF4":14640,
        "exclusions":exclusions,"unique_post_review_excluded":len(excluded),
        "grouping_policy":{"representative":"source,source_revision,source_row_index,candidate_id ascending",
            "thresholds":"Unchanged audit_h2_mini_post_review.compare; frozen source-scoped group IDs retained",
            "audit_helper_sha256":file_sha(ROOT/"scripts/audit_h2_mini_post_review.py"),
            "source_helper_sha256":file_sha(ROOT/"scripts/audit_h2_mini_sources.py"),
            "semantic_limit":"Lexical/SVD/template approximation; full-source rare-phrase groups preserved; no guarantee of unobserved semantic lineage",
            "self_identity_policy":"All accepted records checked internally. Their own historical review identity is not an external contaminant; all other reviewed/rejected/reserved records screened externally."},
        "token_length_check":{"scope":"Complete review-eligible B0 pool only, not final assignments","tokenizer_revision":src.TOKR,
            "max_tokens":max(d["tokens"] for d in ad),"above1280":sum(d["tokens"]>1280 for d in ad),"truncation":False},
        "detector_inference":False,"training":False,"replay":False,"runtime_seconds":time.monotonic()-started}
    save(AUDIT,audit)
    save(FINAL,{"status":"GROUP_SAFE_METADATA_CAPACITY_STOP" if not passes else "GROUP_SAFE_METADATA",
        "records":final,"counts":c,"review_eligible_pool":rel(PRE),"post_review_audit":rel(AUDIT),
        "post_review_audit_sha256":file_sha(AUDIT),"raw_text_present":False,"final_split_or_stage_assigned":False})
    capacity["audit_sha256"]=file_sha(AUDIT);capacity["pool_sha256"]=file_sha(FINAL)
    save(CAP,capacity)
    assert snapshot([ROOT/x for x in protected])==before,"Protected evidence changed"
    print(json.dumps({"status":capacity["status"],"counts":c,"shortages":capacity["shortages"],
        "post_review_excluded":len(excluded),"runtime_seconds":audit["runtime_seconds"]},indent=2),flush=True)

if __name__=="__main__":main()

