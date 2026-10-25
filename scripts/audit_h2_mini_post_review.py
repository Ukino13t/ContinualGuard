"""Post-review metadata audit. No detector inference, assignments, training or replay."""
from __future__ import annotations
import csv, gzip, json, math, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from transformers import AutoTokenizer
import audit_h2_mini_sources as src
from analyze_h2_mini_production_review import ROOT, COHORTS, REQUIRED, HASH_FIELDS, GROUP_FIELDS, load, save, file_sha, sha, relative

AUD=ROOT/"data/audits/hard_benign/h2_mini_post_review_audit.json"
FIELDS=[*HASH_FIELDS,*GROUP_FIELDS]
CONTENT=set(HASH_FIELDS)

def group_key(r,k):
    value=r.get(k)
    if value in (None,""):return None
    # Project group labels are source-local hashes of row indices, NOT global IDs.
    return (k,value) if k in CONTENT else (k,r["source"],r["source_revision"],value)

def stored_group_audit(records):
    uf=src.UF(len(records)); findings=[]
    for k in dict.fromkeys(FIELDS):
        buckets=defaultdict(list)
        for i,r in enumerate(records):
            key=group_key(r,k)
            if key is not None:buckets[key].append(i)
        for key,g in buckets.items():
            if len(g)<2:continue
            for i in g[1:]:uf.union(g[0],i)
            findings.append({"method":k,"members":[records[i]["review_id"] for i in g],
              "cross_source":len({records[i]["source"] for i in g})>1,
              "cross_cohort":len({records[i]["review_cohort"] for i in g})>1})
    return uf,findings

def descriptions(records,texts,tok):
    hh,ll=src.mviews(texts,tok)
    return [{"id":r.get("review_id",r.get("id")),"raw":sha(s),"norm":src.norm(s),
      "normalized_hash":sha(src.norm(s)),"model_view":h,"tokens":n,
      "template":src.template(s,src.classify(s)[1]),"words":src.tset(s)}
      for r,s,h,n in zip(records,texts,hh,ll)]

def sparse_sets(a,b):
    vocab={}
    for d in a+b:
        for w in d["words"]:
            if w not in vocab:vocab[w]=len(vocab)
    def matrix(ds):
        indices=[];ptr=[0]
        for d in ds:
            indices.extend(sorted(vocab[w] for w in d["words"]));ptr.append(len(indices))
        return csr_matrix((np.ones(len(indices),dtype=np.float64),indices,ptr),shape=(len(ds),len(vocab)))
    return matrix(a),matrix(b)

def compare(a,ae,b,be,internal=False):
    """All pairs, no LSH pruning. Thresholds retained from frozen source audit."""
    ax,bx=sparse_sets(a,b);bs=np.array([len(d["words"]) for d in b])
    hits=[];ncomp=0
    for start in range(0,len(a),64):
        aa=a[start:start+64];inter=(ax[start:start+64]@bx.T).toarray()
        den=np.array([len(d["words"]) for d in aa])[:,None]+bs[None,:]-inter
        jac=np.divide(inter,den,out=np.zeros_like(inter),where=den>0)
        cos=ae[start:start+64]@be.T
        for ii,q in enumerate(aa):
            i=start+ii
            for j,t in enumerate(b):
                if q["id"]==t["id"] or (internal and j<=i):continue
                ncomp+=1;methods=[]
                if q["raw"]==t["raw"]:methods.append("raw_exact")
                if q["normalized_hash"]==t["normalized_hash"]:methods.append("normalized_exact")
                if q["model_view"]==t["model_view"]:methods.append("model_view_exact")
                if internal:
                    size_ratio=min(len(q["words"]),len(t["words"]))/max(1,len(q["words"]),len(t["words"]))
                    if size_ratio>=.45 and jac[ii,j]>=.82:methods.append("project_lexical_parent_candidate")
                    if size_ratio>=.45 and cos[ii,j]>=.97 and jac[ii,j]>=.35:methods.append("project_semantic_parent_candidate")
                    if q["template"] and q["template"]==t["template"] and jac[ii,j]>=.55:methods.append("project_template_candidate")
                else:
                    if (len(q["norm"])>=25 and q["norm"] in t["norm"]) or (len(t["norm"])>=25 and t["norm"] in q["norm"]):methods.append("substring")
                    if jac[ii,j]>=.85:methods.append("high_lexical_neighbor")
                    if cos[ii,j]>=.97 and jac[ii,j]>=.30:methods.append("semantic_neighbor")
                    if q["template"] and q["template"]==t["template"]:methods.append("template")
                if methods:
                    hits.append({"query_id":q["id"],"target_id":t["id"],"methods":methods,
                      "latent_cosine":float(cos[ii,j]),"token_set_jaccard":float(jac[ii,j])})
        print("compared",min(start+64,len(a)),len(a),"against",len(b),flush=True)
    return {"query_count":len(a),"target_count":len(b),"nonself_pairs_checked":ncomp,"pairs":hits,
      "affected_queries":len({x["query_id"] for x in hits}),
      "method_pair_counts":dict(Counter(m for x in hits for m in x["methods"]))}

def representative_key(r):
    return (r["source"],r["source_revision"],r["source_row_index"],r["candidate_id"])

def capacity_report(records,analysis):
    out={}
    for c in COHORTS:
        rr=[r for r in records if r["review_cohort"]==c];n=len(rr);total=analysis["by_cohort"][c]["candidates"];deficit=max(0,REQUIRED[c]-n)
        eligible_rate=n/total
        lane=Counter(r["review_lane"] for r in rr)
        double_qualified=sum(len(r["reviewer_evidence"])==2 for r in rr)
        out[c]={"candidates_reviewed":total,"group_safe_available":n,"required":REQUIRED[c],"shortage":deficit,
          "passes_minimum_count":n>=REQUIRED[c],"usable_yield":eligible_rate,"not_usable_rate":1-eligible_rate,
          "human_benign_rejection_rate":1-analysis["by_cohort"][c]["benign_acceptance_rate"],
          "eligible_by_review_lane":dict(lane),"eligible_with_two_independent_judgments":double_qualified,
          "minimum_additional_double_qualified_for_30val_60test":max(0,90-double_qualified) if c!="B-short" else None,
          "additional_candidates_at_observed_pooled_yield":math.ceil(deficit/eligible_rate) if deficit and eligible_rate else (0 if not deficit else None),
          "projection_caveat":"Arithmetic workload illustration only. Nonexchangeable review roles and unresolved stratum reliability invalidate a reliable extrapolation; requires human review before any sampling."}
    out["C0_all_four_B0_pools"]={"required":1160,"available":out["B0"]["group_safe_available"],
      "shortage":max(0,1160-out["B0"]["group_safe_available"]),
      "additional_candidates_at_observed_pooled_yield":math.ceil(max(0,1160-out["B0"]["group_safe_available"])/out["B0"]["usable_yield"]),
      "minimum_two_judgment_records_required":360,
      "two_judgment_records_available":out["B0"]["eligible_with_two_independent_judgments"],
      "review_scope_note":"The completed 360-candidate B0 package covered one anchor only. The current request retains four independent 290-record C0 stages."}
    return out

def main():
    started=time.monotonic();hashes={}
    def verify(p,expected):
        actual=file_sha(p)
        if actual!=expected:raise ValueError("Source integrity failure: "+relative(p))
        hashes[relative(p)]=actual
    for audit_name,filename in [("or_bench_full_audit.json","or-bench-80k.csv"),("oasst1_full_audit.json","2023-04-12_oasst_ready.messages.jsonl.gz")]:
        meta=load(ROOT/"data/audits/hard_benign"/audit_name)
        entry=next(x for x in meta["source"]["files"] if x["file_name"]==filename)
        verify(ROOT/entry["local_path"],entry["sha256"])
    assignment=load(ROOT/"review/production_review_manifest.json")
    canonical=load(ROOT/"review/production_review_canonical_manifest.json")
    verify(ROOT/"review/production_review_manifest.json",canonical["assignment_manifest_sha256"])
    for f in canonical["files"].values():
        for pathkey,hashkey in [("submitted_path","submitted_sha256"),("original_snapshot_path","original_snapshot_sha256"),("canonical_path","canonical_sha256")]:
            verify(ROOT/f[pathkey],f[hashkey])
    poolpaths={c:ROOT/"data/manifests/hard_benign"/(c.replace("-","_")+"_accepted.json") for c in COHORTS}
    pools={c:load(p) for c,p in poolpaths.items()};records=[r for c in COHORTS for r in pools[c]["records"]]
    if any(p.get("post_review_audit") for p in pools.values()):
        raise ValueError("Accepted pools already audited; do not overwrite frozen analysis without explicit rerun design.")
    with (src.ORP/"or-bench-80k.csv").open(encoding="utf-8-sig",newline="") as f:orr=list(csv.DictReader(f))
    with gzip.open(src.OAP/"2023-04-12_oasst_ready.messages.jsonl.gz","rt",encoding="utf-8") as f:
        oar={i:json.loads(line) for i,line in enumerate(f) if line.strip()}
    candidates={}
    for name in ["h2_mini_candidates.json","b_short_mini_candidates.json"]:
        for r in load(ROOT/"data/manifests/hard_benign"/name)["records"]:candidates[r["candidate_id"]]=r
    def get_text(r):
        original=candidates[r["candidate_id"]]
        for k in [*HASH_FIELDS,"deberta_tokens","source_row_index","exact_group","normalized_group","model_view_group","project_semantic_parent_group","project_template_group"]:
            if r.get(k)!=original.get(k):raise ValueError("Manifest provenance mismatch: "+r["review_id"]+" "+k)
        if r["source"]=="OR-Bench":
            if r["source_revision"]!=src.ORR:raise ValueError("OR revision mismatch")
            text=orr[r["source_row_index"]]["prompt"]
        else:
            if r["source_revision"]!=src.OAR:raise ValueError("OA revision mismatch")
            a=oar[r["source_row_index"]];text=a["text"]
            if (a["message_id"],a["message_tree_id"],a.get("parent_id"))!=(r["source_native_id"],r["source_parent_id"],r["source_immediate_parent_id"]):raise ValueError("OA native/tree lineage mismatch")
        if sha(text)!=r["raw_text_sha256"]:raise ValueError("Source text integrity failure: "+r["review_id"])
        return text
    texts=[get_text(r) for r in records]
    bsrecords=[r for r in assignment["records"] if r["review_cohort"]=="B-short"]
    bstexts=[get_text(r) for r in bsrecords]
    nisource=load(ROOT/"data/manifests/hard_benign/notinject/source.json")
    verify(ROOT/nisource["records_manifest"],nisource["records_manifest_sha256"])
    nimeta=[json.loads(x) for x in (ROOT/nisource["records_manifest"]).read_text(encoding="utf-8").splitlines()]
    ni=src.load_ni();niby={x["id"]:x for x in ni}
    if len(ni)!=339 or len(nimeta)!=339:raise ValueError("NotInject count mismatch")
    for r in nimeta:
        k=r["source_split"]+":"+str(r["source_row_idx"])
        if sha(niby[k]["text"])!=r["prompt_sha256"]:raise ValueError("Frozen NotInject prompt hash mismatch")
    afmeta=load(ROOT/"data/manifests/attack_family_shift/construction.json")
    verify(ROOT/afmeta["processed_file"],afmeta["processed_sha256"])
    af=[]
    for line in (ROOT/afmeta["processed_file"]).open(encoding="utf-8"):
        r=json.loads(line)
        if r["task_id"] in {"AF1","AF2","AF3","AF4"}:
            if sha(r["text"])!=r["text_sha256"]:raise ValueError("AF record hash mismatch")
            af.append({"id":r["example_id"],"text":r["text"],"family":r["task_id"],"label":r["label"]})
    assert len(af)==14640 and len(records)==sum(p["count"] for p in pools.values())
    tok=AutoTokenizer.from_pretrained(ROOT/"data/raw/huggingface/microsoft_deberta-v3-base"/src.TOKR,local_files_only=True,use_fast=False)
    ad=descriptions(records,texts,tok);bd=descriptions(bsrecords,bstexts,tok)
    nd=descriptions(ni,[r["text"] for r in ni],tok);fd=descriptions(af,[r["text"] for r in af],tok)
    for r,d in zip(records,ad):
        if (r["raw_text_sha256"],r["normalized_text_sha256"],r["model_view_sha256"],r["deberta_tokens"])!=(d["raw"],d["normalized_hash"],d["model_view"],d["tokens"]):raise ValueError("Fresh tokenizer/hash mismatch: "+r["review_id"])
    print("Fit frozen deterministic OR lexical latent basis (no detector)",flush=True)
    hv,sv,_,planes=src.fit_embed([src.norm(r["prompt"]) for r in orr])
    ae=src.xembed([d["norm"] for d in ad],hv,sv)
    be=src.xembed([d["norm"] for d in bd],hv,sv)
    ne=src.xembed([d["norm"] for d in nd],hv,sv)
    fe=src.xembed([d["norm"] for d in fd],hv,sv)
    uf,stored=stored_group_audit(records)
    internal=compare(ad,ae,ad,ae,internal=True);index={r["review_id"]:i for i,r in enumerate(records)}
    for pair in internal["pairs"]:uf.union(index[pair["query_id"]],index[pair["target_id"]])
    exclusions=[]
    for g in uf.groups():
        if len(g)<2:continue
        rep=min(g,key=lambda i:representative_key(records[i]))
        for i in g:
            if i!=rep:exclusions.append({"review_id":records[i]["review_id"],"reason":"duplicate_or_project_group","retained_representative":records[rep]["review_id"],
              "component_members":[records[j]["review_id"] for j in g]})
    external={"NotInject":compare(ad,ae,nd,ne),"B_short_all_proposed":compare(ad,ae,bd,be),
      "AF1_AF4_all_labels":compare(ad,ae,fd,fe)}
    # Audit source-native/tree/project groups against the entire B-short reserve, even records later rejected.
    group_pairs=[]
    for r in records:
        for b in bsrecords:
            if r["review_id"]==b["review_id"]:continue
            methods=[k for k in dict.fromkeys(FIELDS) if group_key(r,k) is not None and group_key(r,k)==group_key(b,k)]
            if methods:group_pairs.append({"query_id":r["review_id"],"target_id":b["review_id"],"methods":methods})
    external["B_short_prohibited_groups"]={"pairs":group_pairs,"affected_queries":len({p["query_id"] for p in group_pairs})}
    for name,report in external.items():
        for q in sorted({p["query_id"] for p in report["pairs"]}):
            exclusions.append({"review_id":q,"reason":"external_overlap_or_neighbor_quarantine","target_bank":name})
    for r,d in zip(records,ad):
        if d["tokens"]>1280:exclusions.append({"review_id":r["review_id"],"reason":"over_max_length"})
    excluded={e["review_id"] for e in exclusions};final=[r for r in records if r["review_id"] not in excluded]
    analysis=load(ROOT/"results/h2_mini_production_review_analysis.json")
    # Failed predeclared decision gates disable an entire stratum, irrespective of individual labels.
    for r in final:
        if r["review_cohort"] in analysis["training_decision_audit_failures"]:
            exclusions.append({"review_id":r["review_id"],"reason":"stratum_decision_audit_gate_failure"})
    excluded={e["review_id"] for e in exclusions};final=[r for r in records if r["review_id"] not in excluded]
    capacity=capacity_report(final,analysis)
    capacity_pass=all(capacity[c]["passes_minimum_count"] for c in COHORTS)
    status="STOP_CAPACITY_SHORTAGE" if not capacity_pass else "CAPACITY_ONLY_COMPLETE_HUMAN_REVIEW_REQUIRED"
    result={"status":status,"audited_at":datetime.now(timezone.utc).isoformat(),"canonical_root":str(ROOT),
      "interpreter":sys.executable,"source_hashes_verified":hashes,"tokenizer_revision":src.TOKR,
      "review_eligible_before_post_audit":len(records),"group_safe_after_post_audit":len(final),
      "stored_group_findings":stored,"accepted_only_all_pair_screen":internal,"external_contamination":external,
      "AF_coverage":{"total_records":len(af),"by_family":dict(Counter(r["family"] for r in af)),
        "by_label":dict(Counter(r["label"] for r in af)),"scope":"All AF1-AF4 malicious records AND companion benign documents; AF5 excluded."},
      "B_short_scope":{"all_proposed":len(bsrecords),"accepted":len([r for r in records if r["review_cohort"]=="B-short"]),"final_frozen_120":False,"self_identity_pairs_excluded":True},
      "NotInject_records":339,"exclusions":exclusions,"unique_excluded":len(excluded),"capacity":capacity,
      "all_minimum_targets_pass":capacity_pass,"matched_C0_capacity_pass":capacity["C0_all_four_B0_pools"]["shortage"]==0,
      "max_length_check":{"limit":1280,"verified_accepted_records":len(ad),"max_tokens":max(d["tokens"] for d in ad),
        "over_limit":sum(d["tokens"]>1280 for d in ad),"scope":"Accepted benign pools only; complete uncreated assignments are NOT verified."},
      "group_policy":{"representative":"Lowest source row index within each source/revision; deterministic source/revision prefix resolves cross-source ties",
        "scope":"Full-source frozen group IDs retained; exhaustive accepted-pair lexical/latent/template check added without weakening thresholds",
        "source_scoped_project_ids":True,"official_OR_parent_or_generator_ids":"Unavailable; no claim of official lineage recovery",
        "OA_lineage":"Native message, root conversation tree, immediate parent verified against pinned source",
        "semantic_method":"Frozen HashingVectorizer8192 unigram/bigram + TruncatedSVD48 seed42, fitted on all 80359 OR prompts; cosine/Jaccard thresholds from source audit; exhaustive pair comparison replaces approximate LSH candidate search",
        "rare_phrase_policy":"Existing full-source semantic groups retain rare-six-word links; do not redefine rarity in the small accepted subset",
        "caveat":"Lexical latent grouping is an approximation, not a pretrained semantic encoder or proof of absent semantic siblings. New external candidates conservatively quarantine without changing human labels."},
      "assignment_proposal_created":False,"B_short_final_hash_manifest_created":False,"raw_JSONL_created":False,
      "detector_predictions_used":False,"training":False,"replay":False,"runtime_seconds":time.monotonic()-started}
    save(AUD,result)
    for c,pool in pools.items():
        rr=[dict(r,post_review_audit_status="passed_with_documented_lineage_limits") for r in final if r["review_cohort"]==c]
        pool.update(status="accepted_group_safe_metadata_capacity_stop" if not capacity_pass else "accepted_group_safe_metadata",
          count=len(rr),records=rr,review_eligible_before_post_audit=sum(r["review_cohort"]==c for r in records),
          excluded_review_ids=[r["review_id"] for r in records if r["review_cohort"]==c and r["review_id"] in excluded],
          post_review_audit=relative(AUD),post_review_audit_sha256=file_sha(AUD),final_split_assigned=False)
        save(poolpaths[c],pool)
    analysis.update(status=status,capacity=capacity,post_review_audit=relative(AUD),post_review_audit_sha256=file_sha(AUD),
      assignment_proposal_status="NOT_CREATED_CAPACITY_STOP",final_B_short_status="DEFERRED_AT_SECTION_8_CAPACITY_STOP",
      final_materialization_ready=False,no_replay_training_ready=False)
    for c in COHORTS:
        analysis["accepted_pools"][c].update(count=pools[c]["count"],sha256=file_sha(poolpaths[c]))
    save(ROOT/"results/h2_mini_production_review_analysis.json",analysis)
    print(json.dumps({"status":status,"group_safe_counts":{c:capacity[c]["group_safe_available"] for c in COHORTS},
      "exclusions":len(excluded),"external_affected":{k:v["affected_queries"] for k,v in external.items()},
      "runtime_seconds":result["runtime_seconds"]},indent=2))
    return 0

if __name__=="__main__":raise SystemExit(main())
