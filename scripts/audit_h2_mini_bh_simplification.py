"""Audit a proposed BH taxonomy amendment without changing frozen review evidence."""
from __future__ import annotations
import csv, gzip, json, math, sys, time
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
import audit_h2_mini_sources as src
import audit_h2_mini_post_review as post
from analyze_h2_mini_production_review import ROOT, HASH_FIELDS, GROUP_FIELDS, accepted_metadata, agreement, file_sha, load, relative, save, sha, validate

POLICY="h2mini-BH-amendment-proposal-v1-20261003"
CAPACITY=ROOT/"results/h2_mini_bh_capacity.json"
FEASIBILITY=ROOT/"results/h2_mini_bh_partition_feasibility.json"

def benign_eligible(rows):
    return bool(rows) and all(r["reviewer_decision"]=="ACCEPT_BENIGN" for r in rows)

def capacities(records):
    train=[r for r in records if r["review_lane"]=="TRAIN_CANDIDATE"]
    evaluation=[r for r in records if r["review_lane"] in {"VALIDATION_RESERVE","TEST_RESERVE"}]
    return {"total_unique":len(records),"training_lane":len(train),
      "training_primary_only":sum(len(r["reviewer_evidence"])==1 for r in train),
      "training_audited_double":sum(len(r["reviewer_evidence"])==2 for r in train),
      "all_double_reviewed":sum(len(r["reviewer_evidence"])==2 for r in records),
      "validation_test_reserve_double":len(evaluation),
      "original_validation_reserve":sum(r["review_lane"]=="VALIDATION_RESERVE" for r in evaluation),
      "original_test_reserve":sum(r["review_lane"]=="TEST_RESERVE" for r in evaluation),
      "all_double_potential_validation_test":sum(len(r["reviewer_evidence"])==2 for r in records),
      "all_benign_potential_training":len(records),
      "original_proposed_stratum_counts":dict(Counter(r["proposed_stratum"] for r in records)),
      "categories":dict(Counter(r["source_category"] for r in records)),
      "languages":dict(Counter(r["language"] for r in records))}

def token_bin(n):
    for upper in (16,24,32,48,64,128,256,1280):
        if n<=upper:return "<="+str(upper)
    return ">1280"

def bucket_tuple(r):
    return (r["source_category"],token_bin(r["deberta_tokens"]),r["language"],r["proposed_stratum"],
      tuple(sorted(r.get("cue_family_ids",[]))))

def features(key):
    category,length,language,original,cues=key
    return [("category",category),("length_bin",length),("language",language),("original_proposed_stratum",original)]+[("cue_family",x) for x in cues]

def histogram_witness(records):
    """Allocate anonymous metadata-bucket counts only; never assign a review ID."""
    available={}
    for name,predicate in [
      ("train_audited",lambda r:r["review_lane"]=="TRAIN_CANDIDATE" and len(r["reviewer_evidence"])==2),
      ("train_primary",lambda r:r["review_lane"]=="TRAIN_CANDIDATE" and len(r["reviewer_evidence"])==1),
      ("validation_double",lambda r:r["review_lane"]=="VALIDATION_RESERVE"),
      ("test_double",lambda r:r["review_lane"]=="TEST_RESERVE")]:
        available[name]=Counter(bucket_tuple(r) for r in records if predicate(r))
    targets={"train_audited":20,"train_primary":180,"validation_double":30,"test_double":60}
    if any(sum(available[name].values())<3*n for name,n in targets.items()):
        return {"status":"NOT_CONSTRUCTED_WITH_ORIGINAL_REVIEW_RESERVES",
          "note":"Aggregate capacity may still suffice via approved review-lane reassignment; no record assignments made."}
    allocated={};overall={stage:defaultdict(Counter) for stage in ["BH-1","BH-2","BH-3"]}
    stage_split_counts={stage:Counter() for stage in overall}
    for name,per_stage in targets.items():
        buckets=available[name].copy();supply=sum(buckets.values());want=3*per_stage
        ideal=Counter()
        for key,n in buckets.items():
            for f in features(key):ideal[f]+=n*want/supply
        selected=Counter();counts={stage:Counter() for stage in overall};done=Counter();cell=defaultdict(Counter)
        for step in range(want):
            # Equal progress per stage avoids a first-stage access-to-data advantage.
            stage=min(overall,key=lambda t:(done[t]/per_stage,sha(POLICY+"|"+name+"|"+str(step)+"|"+t)))
            possible=[key for key,n in buckets.items() if n>0]
            def score(key):
                ff=features(key);progress=(step+1)/want
                weights={"category":4.0,"original_proposed_stratum":3.0,"length_bin":2.0,"language":1.0,"cue_family":.5}
                # Fill underrepresented marginals, with stable hash tie breaks; no content/model score.
                gap=sum(weights[k]*(ideal[(k,v)]*progress-selected[(k,v)]) for k,v in ff)
                imbalance=sum(weights[k]*(counts[stage][(k,v)]-sum(counts[t][(k,v)] for t in overall)/3) for k,v in ff)
                return (-gap+imbalance,sha(POLICY+"|"+name+"|"+json.dumps(key,ensure_ascii=True)))
            chosen=min(possible,key=score)
            buckets[chosen]-=1;done[stage]+=1;cell[stage][chosen]+=1
            split="train" if name.startswith("train_") else ("validation" if name=="validation_double" else "test")
            stage_split_counts[stage][split]+=1
            for f in features(chosen):selected[f]+=1;counts[stage][f]+=1;overall[stage][f[0]][f[1]]+=1
        allocated[name]={
          "target_per_stage":per_stage,"anonymous_allocated_counts":{t:sum(cell[t].values()) for t in overall},
          "marginals_by_stage":{t:{feature:dict(Counter({v:n for (f,v),n in counts[t].items() if f==feature})) for feature in ["category","length_bin","language","original_proposed_stratum","cue_family"]} for t in overall},
          "metadata_bucket_capacity_respected":all(sum(cell[t][key] for t in overall)<=available[name][key] for key in available[name])}
    diffs={}
    for feature in ["category","length_bin","language","original_proposed_stratum","cue_family"]:
        values={v for t in overall for v in overall[t][feature]}
        diffs[feature]={v:max(overall[t][feature][v] for t in overall)-min(overall[t][feature][v] for t in overall) for v in sorted(values)}
    return {"status":"ANONYMOUS_METADATA_COUNT_WITNESS_ONLY","policy_version":POLICY,
      "procedure":"Greedy equal-stage-progress deficit balancing over source category, fixed token bins, language, cue families and historical proposed stratum; SHA256(policy|scope|metadata) breaks ties.",
      "record_to_partition_mapping_created":False,"record_to_split_mapping_created":False,
      "unique_record_IDs_selected":False,"groups_already_single_representatives":True,
      "allocations":allocated,"stage_split_counts":{t:dict(n) for t,n in stage_split_counts.items()},
      "stage_overall_marginals":{t:{f:dict(n) for f,n in d.items()} for t,d in overall.items()},
      "max_between_partition_count_gaps":{f:max(d.values(),default=0) for f,d in diffs.items()},
      "between_partition_count_gaps":diffs,
      "future_ID_selection":"Only after approval, hash-order members within each metadata/review-coverage bucket using a frozen salt, verify whole-group isolation and all exact quotas, and present a hash-only assignment for review.",
      "caveat":"This is a count-feasibility witness, not a frozen or final partition. Finite-sample marginals may differ; no ordinal difficulty target is used."}

def main():
    start=time.monotonic()
    if CAPACITY.exists() or FEASIBILITY.exists():raise ValueError("Proposal audit outputs already exist; refuse silent replacement.")
    m=load(ROOT/"review/production_review_manifest.json")
    can=load(ROOT/"review/production_review_canonical_manifest.json")
    old=load(ROOT/"results/h2_mini_production_review_analysis.json")
    tables,j,validation=validate(m,ROOT/"review/production")
    if validation["issues"]:raise ValueError("Frozen production evidence validation failed")
    historical_paths=[ROOT/"review/production_review_manifest.json",ROOT/"review/production_review_canonical_manifest.json",
      ROOT/"results/h2_mini_production_review_analysis.json",ROOT/"data/audits/hard_benign/h2_mini_post_review_audit.json",
      ROOT/"data/audits/hard_benign/h2_mini_review_outcomes.json",ROOT/"docs/h2_mini_production_review_results.md",
      ROOT/"docs/h2_mini_calibration_results.md",ROOT/"review/h2_mini_calibration_analysis.json",
      ROOT/"docs/h2_mini_review_protocol.md"]
    historical_paths += list((ROOT/"data/manifests/hard_benign").glob("*_accepted.json"))
    historical_paths += list((ROOT/"review").glob("*.md"))
    historical_paths += [ROOT/f[k] for f in can["files"].values() for k in ["submitted_path","original_snapshot_path","canonical_path"]]
    protected={relative(p):{"sha256":file_sha(p),"mtime_ns":p.stat().st_mtime_ns} for p in historical_paths}
    for name,t in tables.items():
        f=can["files"][name]
        assert t["submitted_sha256"]==f["submitted_sha256"]==file_sha(ROOT/f["original_snapshot_path"])
        assert file_sha(ROOT/f["canonical_path"])==f["canonical_sha256"]
    assert file_sha(ROOT/"review/production_review_manifest.json")==can["assignment_manifest_sha256"]
    orrecords=[r for r in m["records"] if r["source"]=="OR-Bench"]
    audits={c:agreement([r for r in orrecords if r["proposed_stratum"]==c and r["review_lane"]=="TRAIN_CANDIDATE" and len(j[r["review_id"]])==2],j) for c in ["B1","B2","B3"]}
    if any(d["decision_disagreement_rate"]>.1 for d in audits.values()):
        raise ValueError("Existing benign-decision audit gate failed; primary-only BH cannot automatically proceed.")
    eligible=[r for r in orrecords if benign_eligible(j[r["review_id"]])]
    proposed=[]
    for r in eligible:
        item=accepted_metadata(r,j[r["review_id"]],can["files"])
        del item["accepted_stratum"]
        item.update(proposed_final_benign_taxonomy="BH",taxonomy_policy=POLICY,taxonomy_status="PROPOSAL_ONLY",
          cue_family_ids=r.get("cue_family_ids",[]),original_stratum_use="historical descriptive/candidate-sampling metadata only",
          human_review_status="all_required_benign_decisions_accepted_no_ordinal_agreement_claim")
        proposed.append(item)
    pre=capacities(proposed)
    source_hashes={}
    def verify(p,expected):
        assert file_sha(p)==expected,relative(p);source_hashes[relative(p)]=expected
    for an,filename in [("or_bench_full_audit.json","or-bench-80k.csv"),("oasst1_full_audit.json","2023-04-12_oasst_ready.messages.jsonl.gz")]:
        a=load(ROOT/"data/audits/hard_benign"/an);f=next(x for x in a["source"]["files"] if x["file_name"]==filename);verify(ROOT/f["local_path"],f["sha256"])
    with (src.ORP/"or-bench-80k.csv").open(encoding="utf-8-sig",newline="") as stream:orr=list(csv.DictReader(stream))
    with gzip.open(src.OAP/"2023-04-12_oasst_ready.messages.jsonl.gz","rt",encoding="utf-8") as stream:oar={i:json.loads(line) for i,line in enumerate(stream) if line.strip()}
    candidates={}
    for fn in ["h2_mini_candidates.json","b_short_mini_candidates.json"]:
        candidates.update({r["candidate_id"]:r for r in load(ROOT/"data/manifests/hard_benign"/fn)["records"]})
    def get_text(r):
        reference=candidates[r["candidate_id"]]
        for k in [*HASH_FIELDS,"deberta_tokens","source_row_index","exact_group","normalized_group","model_view_group","project_semantic_parent_group","project_template_group"]:
            assert r.get(k)==reference.get(k),(r["review_id"],k)
        if r["source"]=="OR-Bench":
            assert r["source_revision"]==src.ORR
            text=orr[r["source_row_index"]]["prompt"]
        else:
            assert r["source_revision"]==src.OAR
            row=oar[r["source_row_index"]];text=row["text"]
            assert (row["message_id"],row["message_tree_id"],row.get("parent_id"))==(r["source_native_id"],r["source_parent_id"],r["source_immediate_parent_id"])
        assert sha(text)==r["raw_text_sha256"];return text
    texts=[get_text(r) for r in proposed]
    b0=load(ROOT/"data/manifests/hard_benign/B0_accepted.json")["records"]
    bs=load(ROOT/"data/manifests/hard_benign/B_short_accepted.json")["records"]
    bsp=[r for r in m["records"] if r["review_cohort"]=="B-short"]
    ni=src.load_ni();nis=load(ROOT/"data/manifests/hard_benign/notinject/source.json")
    verify(ROOT/nis["records_manifest"],nis["records_manifest_sha256"])
    nimeta=[json.loads(x) for x in (ROOT/nis["records_manifest"]).read_text(encoding="utf-8").splitlines()]
    niby={r["id"]:r for r in ni}
    assert len(ni)==len(nimeta)==339
    for r in nimeta:assert sha(niby[r["source_split"]+":"+str(r["source_row_idx"])]["text"])==r["prompt_sha256"]
    afm=load(ROOT/"data/manifests/attack_family_shift/construction.json")
    verify(ROOT/afm["processed_file"],afm["processed_sha256"])
    af=[]
    for line in (ROOT/afm["processed_file"]).open(encoding="utf-8"):
        r=json.loads(line)
        if r["task_id"] in {"AF1","AF2","AF3","AF4"}:
            assert sha(r["text"])==r["text_sha256"]
            af.append({"id":r["example_id"],"text":r["text"],"family":r["task_id"],"label":r["label"]})
    assert len(af)==14640
    tok=AutoTokenizer.from_pretrained(ROOT/"data/raw/huggingface/microsoft_deberta-v3-base"/src.TOKR,local_files_only=True,use_fast=False)
    desc=post.descriptions(proposed,texts,tok)
    banks={"NotInject":(ni,[r["text"] for r in ni]),"B_short_all_150_proposed":(bsp,[get_text(r) for r in bsp]),
      "AF1_AF4_all_labels":(af,[r["text"] for r in af]),"existing_B0_control":(b0,[get_text(r) for r in b0])}
    bd={name:post.descriptions(rr,tt,tok) for name,(rr,tt) in banks.items()}
    for r,d in zip(proposed,desc):
        assert (r["raw_text_sha256"],r["normalized_text_sha256"],r["model_view_sha256"],r["deberta_tokens"])==(d["raw"],d["normalized_hash"],d["model_view"],d["tokens"])
    print("Fitting fixed lexical latent audit basis; no detector inference",flush=True)
    hv,sv,_,_=src.fit_embed([src.norm(r["prompt"]) for r in orr])
    embed=lambda dd:src.xembed([d["norm"] for d in dd],hv,sv)
    pe=embed(desc);uf,stored=post.stored_group_audit(proposed)
    internal=post.compare(desc,pe,desc,pe,internal=True);index={r["review_id"]:i for i,r in enumerate(proposed)}
    for pair in internal["pairs"]:uf.union(index[pair["query_id"]],index[pair["target_id"]])
    exclusions=[]
    for group in uf.groups():
        if len(group)<2:continue
        representative=min(group,key=lambda i:post.representative_key(proposed[i]))
        for i in group:
            if i!=representative:exclusions.append({"review_id":proposed[i]["review_id"],"reason":"duplicate_or_project_group","retained_representative":proposed[representative]["review_id"]})
    ext={name:post.compare(desc,pe,dd,embed(dd)) for name,dd in bd.items()}
    for name,d in ext.items():
        for rid in sorted({x["query_id"] for x in d["pairs"]}):exclusions.append({"review_id":rid,"reason":"external_overlap_candidate_quarantine","bank":name})
    # Cross-source project group strings are scoped; content hashes remain global.
    group_cross=[]
    for r in proposed:
        for b in b0+bsp:
            mm=[k for k in dict.fromkeys(post.FIELDS) if post.group_key(r,k) is not None and post.group_key(r,k)==post.group_key(b,k)]
            if mm:group_cross.append({"query_id":r["review_id"],"target_id":b["review_id"],"methods":mm})
    for pair in group_cross:exclusions.append({"review_id":pair["query_id"],"reason":"B0_or_Bshort_group_overlap","target_id":pair["target_id"]})
    for r in proposed:
        if r["deberta_tokens"]>1280:exclusions.append({"review_id":r["review_id"],"reason":"over_max_length"})
    excluded={e["review_id"] for e in exclusions}
    final=[dict(r,post_review_audit_status="passed_for_proposed_BH_with_lineage_caveats") for r in proposed if r["review_id"] not in excluded]
    count=capacities(final);witness=histogram_witness(final)
    checks={"unique_870":count["total_unique"]>=870,"training_lane_600":count["training_lane"]>=600,
      "training_audit_60":count["training_audited_double"]>=60,
      "validation_test_reserve_270":count["validation_test_reserve_double"]>=270,
      "original_validation_reserve_90":count["original_validation_reserve"]>=90,
      "original_test_reserve_180":count["original_test_reserve"]>=180,
      "all_groups_have_one_retained_representative":all(sum(proposed[i]["review_id"] not in excluded for i in g)<=1 for g in uf.groups())}
    feasible=all(checks.values()) and witness["status"]=="ANONYMOUS_METADATA_COUNT_WITNESS_ONLY"
    b0count=capacities(b0);times=old["timing"]
    b0seconds=[float(x["review_seconds"]) for r in m["records"] if r["review_cohort"]=="B0" for x in j[r["review_id"]]]
    bsseconds=[float(x["review_seconds"]) for r in m["records"] if r["review_cohort"]=="B-short" for x in j[r["review_id"]]]
    b0mean=sum(b0seconds)/len(b0seconds);bsmean=sum(bsseconds)/len(bsseconds)
    missing=1160-len(b0);train_add=800-b0count["training_lane"];eval_add=360-b0count["validation_test_reserve_double"]
    audit_add=80-b0count["training_audited_double"]
    min_judgments=train_add+2*eval_add+audit_add
    # Observed role-specific target-B0 yields. Workload projections are not guaranteed.
    b0r=[r for r in m["records"] if r["review_cohort"]=="B0"]
    train_reviewed=sum(r["review_lane"]=="TRAIN_CANDIDATE" for r in b0r)
    eval_reviewed=len(b0r)-train_reviewed
    train_yield=b0count["training_lane"]/train_reviewed
    eval_yield=b0count["validation_test_reserve_double"]/eval_reviewed
    audit_yield=b0count["training_audited_double"]/25
    plan_train=math.ceil(train_add/train_yield);plan_eval=math.ceil(eval_add/eval_yield)
    plan_audits=max(math.ceil(.1*plan_train),math.ceil(audit_add/audit_yield))
    b0plan={"current":b0count,"B0_rule":"Retain ordinary-B0 confirmation from OASST1 and prior exclusions. The proposed relaxation applies to OR-Bench BH only; do not silently reclassify OASST B3/UNSURE records.",
      "required_unique":1160,"required_train":800,"required_double_validation_test":360,"required_final_training_audit":80,
      "unique_shortage":missing,"double_validation_test_shortage_allowing_training_audit_promotion":360-b0count["all_double_reviewed"],
      "minimum_design_keep_review_lanes":{"new_training_candidates":train_add,"new_validation_test_candidates":eval_add,
        "additional_training_second_judgments":audit_add,"minimum_new_unique_candidates":missing,"minimum_new_judgments":min_judgments,
        "person_hours_at_observed_B0_mean":min_judgments*b0mean/3600,
        "condition":"All new candidates and added second judgments pass; all current accepted IDs remain group-safe."},
      "observed_B0_mean_seconds":b0mean,"observed_training_yield":train_yield,"observed_evaluation_yield":eval_yield,
      "observed_audit_target_B0_yield":audit_yield,
      "initial_workload_projection":{"training_candidates":plan_train,"double_validation_test_candidates":plan_eval,
        "training_second_judgments":plan_audits,"new_unique_candidates":plan_train+plan_eval,
        "judgments":plan_train+2*plan_eval+plan_audits,"person_hours":(plan_train+2*plan_eval+plan_audits)*b0mean/3600,
        "adaptive_topups_may_be_needed":True,"new_review_started":False},
      "reserved_Bshort_not_reused":True}
    english=sum(r["language"]=="en" for r in bs);eng_reviewed=sum(r["language"]=="en" for r in bsp)
    shortage=max(0,90-english);eng_yield=english/eng_reviewed;plan_eng=math.ceil(shortage/eng_yield)
    bsplan={"current_unique":len(bs),"English":english,"multilingual":len(bs)-english,
      "observed_Bshort_mean_seconds":bsmean,"English_reviewed":eng_reviewed,"English_accepted_yield":eng_yield,
      "option_A":{"target_English":90,"target_multilingual":30,"additional_English_accepted_required":shortage,
        "minimum_new_candidates":shortage,"minimum_judgments":shortage*2,"minimum_person_hours":shortage*2*bsmean/3600,
        "projected_candidates":plan_eng,"projected_judgments":plan_eng*2,"projected_person_hours":plan_eng*2*bsmean/3600,
        "methodological_consequence":"Preserves planned approximate NotInject language proportions; still requires token-length matching and cannot remove source/domain confounds."},
      "option_B":{"new_judgments":0,"new_review_person_hours":0,"proposed_English":english,"proposed_multilingual":120-english,
        "English_fraction":english/120,"English_fraction_gap_vs_planned_pp":100*(english/120-.75),
        "methodological_consequence":"Lowest review cost; descriptive ordinary-benign diagnostic only. NotInject-vs-Bshort overall FPR is confounded by language; report language-stratified denominators/FPR and do not attribute the aggregate gap to trigger bias."},
      "recommendation":"OPTION_B_FOR_COURSE_SCALE_DESCRIPTIVE_CONTROL_ONLY_HUMAN_APPROVAL_REQUIRED",
      "recommendation_condition":"If preserving an approximately language-matched aggregate diagnostic is a required primary aim, choose A instead. No option selected or bank frozen.",
      "final_bank_created":False,"any_new_review_started":False}
    protected_after={p:{"sha256":file_sha(ROOT/p),"mtime_ns":(ROOT/p).stat().st_mtime_ns} for p in protected}
    assert protected==protected_after,"Historical evidence changed"
    cap={"status":"PROPOSED_TAXONOMY_AUDIT_COMPLETE_NO_FINAL_ASSIGNMENTS","policy_version":POLICY,
      "created_at":datetime.now(timezone.utc).isoformat(),"root":str(ROOT),"interpreter":sys.executable,
      "not_predeclared":True,"human_final_amendment_approval_required":True,
      "eligibility":"All required ACCEPT_BENIGN judgments; primary-only training allowed by preserved decision audit; historical stratum disagreement neither adjudicated nor used as final eligibility.",
      "production_candidates":len(orrecords),"pre_grouping":pre,"post_grouping":count,
      "training_decision_audits_by_historical_cohort":audits,
      "historical_evidence_before_and_after":protected,"historical_evidence_bytes_and_mtime_unchanged":True,
      "source_file_hashes_verified":source_hashes,
      "audit":{"stored_group_conflicts":stored,"accepted_pair_screen":internal,"external_banks":ext,
        "B0_Bshort_group_overlap":group_cross,"exclusions":exclusions,"excluded_unique_records":len(excluded),
        "model_view_tokenizer_revision":src.TOKR,"max_length":1280,"max_observed_tokens":max(r["deberta_tokens"] for r in proposed),
        "source_native_parent":"OR-Bench source/native/official rewrite-parent IDs unavailable; pinned row/candidate identity and frozen approximate groups retained.",
        "grouping":"Same exact/normalized/model-view and source-scoped full-source semantic/template groups, plus exhaustive accepted-pair and external lexical/SVD/template comparisons; no thresholds weakened.",
        "semantic_caveat":"Lexical latent approximation, not full official lineage or proof of absent paraphrase siblings.",
        "detector_predictions_used":False,"NotInject_scoring_used":False},
      "proposed_BH_records_metadata_only":final,"B0_replenishment":b0plan,"Bshort_options":bsplan,
      "human_rejection_count":len(orrecords)-len(proposed),
      "records_recovered_relative_to_stratum_mandatory_pool":len(final)-sum(old["capacity"][c]["group_safe_available"] for c in ["B1","B2","B3"]),
      "replay_readiness":"R0","new_review_started":False,"raw_JSONL_materialized":False,"training":False,"replay":False,
      "runtime_seconds":time.monotonic()-start}
    feasibility={"status":"FEASIBLE_AS_PROPOSAL" if feasible else "NOT_FEASIBLE_UNDER_CURRENT_CAPACITY",
      "policy_version":POLICY,"required":{"train":600,"validation":90,"test":180,"unique":870,"training_second_review":60},
      "available":count,"checks":checks,"BH_three_partition_feasible":feasible,
      "validation_test_eligible_pools_overlap":"The total double-reviewed count includes training audits; evaluation reserves and the training lane are disjoint. Feasibility uses original reserve lanes, so no record or review capacity is double-counted.",
      "anonymous_metadata_count_witness":witness,
      "revised_order":[{"stage":1,"attack_family":"AF1","C0":"B0-1","F":"same B0-1"},
        {"stage":2,"attack_family":"AF2","C0":"B0-2","F":"BH-1"},
        {"stage":3,"attack_family":"AF3","C0":"B0-3","F":"BH-2"},
        {"stage":4,"attack_family":"AF4","C0":"B0-4","F":"BH-3"}],
      "interpretation":"Persistent OR-Bench hard-benign pressure; BH-1/2/3 are balanced disjoint partitions, not ordinal difficulty.",
      "B0_shortage":missing,"Bshort_option_not_selected":True,"whole_H2mini_ready":False,
      "final_partition_or_split_IDs_assigned":False,"raw_JSONL_created":False,"training":False,"replay_readiness":"R0"}
    save(CAPACITY,cap);feasibility["capacity_audit_sha256"]=file_sha(CAPACITY);save(FEASIBILITY,feasibility)
    print(json.dumps({"pre_grouping":pre,"post_grouping":count,"excluded":len(excluded),
      "external_affected":{k:v["affected_queries"] for k,v in ext.items()},"BH_feasible":feasible,
      "witness_max_marginal_gaps":witness.get("max_between_partition_count_gaps"),"B0_plan":b0plan,"Bshort_options":bsplan,
      "historical_evidence_unchanged":True,"runtime_seconds":cap["runtime_seconds"]},indent=2))

if __name__=="__main__":main()
