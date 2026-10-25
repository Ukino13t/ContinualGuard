"""Shared stdlib schema and sampling rules for the approved B0 review package."""
from collections import Counter,defaultdict
from pathlib import Path
import hashlib,json,math
ROOT=Path(__file__).resolve().parents[1]
POLICY="h2-mini-benign-taxonomy-v2"
PACKAGE="h2mini-b0-replenishment-v1-20261004"
SEED=2026100401
OA_REV="fdf72ae0827c1cda404aff25b6603abec9e3399b"
FIELDS=["review_id","text","source","language","token_length","review_role","reviewer_decision","b0_confirmed","reason_code","confidence","optional_note","review_seconds"]
HUMAN=["reviewer_decision","b0_confirmed","reason_code","confidence","optional_note","review_seconds"]
DECISIONS={"ACCEPT_BENIGN","REJECT_TRUE_INJECTION","REJECT_AMBIGUOUS_INTENT","REJECT_HARMFUL_OR_UNSAFE","REJECT_CONTEXT_MISSING","REJECT_PRIVACY_OR_RIGHTS","REJECT_QUALITY"}
B0_VALUES={"YES","NO","UNSURE"}
CONFIDENCE={"HIGH","MEDIUM","LOW"}
HASH_FIELDS=["raw_text_sha256","normalized_text_sha256","model_view_sha256"]
GROUP_FIELDS=["source_native_id","source_parent_id","source_immediate_parent_id","exact_group","normalized_group","model_view_group","project_semantic_parent_group","project_template_group"]
def sha(s):return hashlib.sha256(s.encode("utf-8") if isinstance(s,str) else s).hexdigest()
def key(*parts):return sha("|".join(map(str,parts)))
def file_sha(p):return sha(p.read_bytes())
def load(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def rel(p):return p.resolve().relative_to(ROOT).as_posix()
def group_keys(r):
    keys={(k,r[k]) for k in HASH_FIELDS if r.get(k)}
    keys.update((r["source"],r["source_revision"],k,r[k]) for k in GROUP_FIELDS if r.get(k))
    return keys
def oa_metadata(r):
    return {"candidate_id":r["candidate_id"],"source":"OASST1","source_revision":OA_REV,
      "source_native_id":r["message_id"],"source_parent_id":r["message_tree_id"],
      "source_immediate_parent_id":r.get("parent_id"),"source_row_index":r["source_row_index"],
      "language":r["lang"],"deberta_tokens":r["deberta_tokens"],
      **{k:r.get(k) for k in HASH_FIELDS+GROUP_FIELDS if k not in ["source_native_id","source_parent_id","source_immediate_parent_id"]}}
def choose_diverse(rows,n,salt):
    """Language-proportional quotas, length-quartile coverage, deterministic hashes."""
    if not 0<=n<=len(rows):raise ValueError("Insufficient candidates")
    if not n:return []
    by=defaultdict(list)
    for r in rows:by[r["language"]].append(r)
    counts={k:len(v) for k,v in by.items()};base={k:int(v>=5) for k,v in counts.items()}
    if sum(base.values())>n:base={k:0 for k in counts}
    remaining=n-sum(base.values());raw={k:remaining*v/len(rows) for k,v in counts.items()}
    quotas={k:min(counts[k],base[k]+math.floor(raw[k])) for k in counts}
    while sum(quotas.values())<n:
        k=min((k for k in counts if quotas[k]<counts[k]),key=lambda k:(-(base[k]+raw[k]-quotas[k]),key(salt,k)))
        quotas[k]+=1
    chosen=[]
    for language in sorted(by):
        rr=by[language];lengths=sorted(r["deberta_tokens"] for r in rr)
        cuts=[lengths[max(0,math.ceil(q*len(lengths))-1)] for q in [.25,.5,.75]]
        buckets=defaultdict(list)
        for r in rr:buckets[sum(r["deberta_tokens"]>cut for cut in cuts)].append(r)
        for b in buckets:buckets[b].sort(key=lambda r:key(salt,language,r["candidate_id"]))
        used=Counter()
        for _ in range(quotas[language]):
            b=min((b for b in buckets if buckets[b]),key=lambda b:(used[b],key(salt,language,b)))
            chosen.append(buckets[b].pop(0));used[b]+=1
    assert len(chosen)==n and len({r["candidate_id"] for r in chosen})==n
    return chosen
def assign_reviews(selected):
    evaluation=choose_diverse(selected,272,f"{SEED}:evaluation-reserve")
    ev={r["candidate_id"] for r in evaluation};train=[r for r in selected if r["candidate_id"] not in ev]
    primary_a=choose_diverse(train,300,f"{SEED}:primary-A")
    pa={r["candidate_id"] for r in primary_a};primary_b=[r for r in train if r["candidate_id"] not in pa]
    second_a=choose_diverse(primary_b,37,f"{SEED}:second-A")
    second_b=choose_diverse(primary_a,38,f"{SEED}:second-B")
    sa={r["candidate_id"] for r in second_a};sb={r["candidate_id"] for r in second_b}
    records=[]
    for original in selected:
        r=dict(original);cid=r["candidate_id"];r["review_id"]="H2B0-"+key(PACKAGE,cid)[:16].upper()
        r["review_lane"]="EVALUATION_RESERVE" if cid in ev else "TRAIN_CANDIDATE"
        r["reviewer_assignments"]=[{"reviewer":w,"review_role":"DOUBLE_REVIEW"} for w in ["A","B"]] if cid in ev else [{"reviewer":"A" if cid in pa else "B","review_role":"PRIMARY"}]
        if cid in sa:r["reviewer_assignments"].append({"reviewer":"A","review_role":"SECOND_REVIEW"})
        if cid in sb:r["reviewer_assignments"].append({"reviewer":"B","review_role":"SECOND_REVIEW"})
        r.update(human_review_status="not_started",final_assignment=None)
        records.append(r)
    # Blinded paired coverage in the first two 100-judgment files supports the optional interim check.
    first_pair=choose_diverse(evaluation,24,f"{SEED}:interim-double")
    first_pair+=choose_diverse(second_a,4,f"{SEED}:interim-audit-A")
    first_pair+=choose_diverse(second_b,4,f"{SEED}:interim-audit-B")
    return records,{r["candidate_id"] for r in first_pair}
def batch_sizes(total):
    if total not in [609,610]:raise ValueError("Unexpected reviewer workload")
    rest=total-100;q,r=divmod(rest,6)
    return [100]+[q+int(i<r) for i in range(6)]
def label_eligible(rows):
    return bool(rows) and all(r["reviewer_decision"]=="ACCEPT_BENIGN" and r["b0_confirmed"]=="YES" for r in rows)
def coverage(eligible,old_counts):
    train=[r for r in eligible if r["review_lane"]=="TRAIN_CANDIDATE"]
    evaluation=[r for r in eligible if r["review_lane"]=="EVALUATION_RESERVE"]
    audit=sum(len(r["reviewer_assignments"])==2 for r in train)
    t=old_counts["training"]+len(train);a=old_counts["training_audited"]+audit;e=old_counts["evaluation"]+len(evaluation)
    total=t+e;d=a+e
    return {"new_accepted_unique":len(eligible),"new_training":len(train),"new_training_audited":audit,"new_double_evaluation":len(evaluation),
      "total_B0":total,"training_lane_total":t,"independently_audited_training_total":a,"double_evaluation_reserve_total":e,
      "all_double_reviewed_total":d,"total_1160_capacity_available":total>=1160,"training_800_capacity_available":t>=800,
      "audited_training_80_capacity_available":a>=80,"joint_lane_preserving_feasible":t>=800 and a>=80 and e>=360 and total>=1160,
      "joint_review_coverage_feasible_if_lanes_reallocated":total>=1160 and d>=440,
      "lane_shortages":{"unique":max(0,1160-total),"training":max(0,800-t),"training_audited":max(0,80-a),"double_evaluation":max(0,360-e)},
      "individual_evaluation_shortages_nonadditive":{"validation":max(0,120-e),"test":max(0,240-e)},
      "targets":{"unique":1160,"training":800,"audited_training":80,"validation":120,"test":240},
      "validation_120_capacity_available":e>=120,"test_240_capacity_available":e>=240,
      "validation120_and_test240_jointly_available":e>=360,
      "warning":"Individual validation/test booleans reuse the same pool; only their joint360 condition establishes both quotas. Counts are review-qualified capacity, not final assignment or post-review scientific approval."}
