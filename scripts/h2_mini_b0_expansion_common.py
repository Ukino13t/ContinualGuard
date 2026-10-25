"""Frozen metadata-only sampling and baseline access for the 105-review expansion."""
from collections import Counter
from h2_mini_b0_review_common import *
import analyze_h2_mini_b0_replenishment as base

EXP_PACKAGE="h2mini-b0-expansion-v1-20261004"
EXP_SEED=2026100402
EXP_FIELDS=[f for f in FIELDS if f!="review_seconds"]
EXP_HUMAN=[f for f in HUMAN if f!="review_seconds"]
EXP_MANIFEST=ROOT/"review/b0_replenishment_expansion_manifest.json"
EXP_DIR=ROOT/"review/b0_replenishment_expansion"
RESERVE=ROOT/"data/manifests/hard_benign/b0_small_topup_reserve.json"
RESERVE_AUDIT=ROOT/"data/audits/hard_benign/b0_small_topup_reserve_audit.json"
BASE_MANIFEST=ROOT/"review/b0_replenishment_manifest.json"
BASE_CANONICAL=ROOT/"review/b0_replenishment_canonical_manifest.json"
BASE_CONFIRMATION=ROOT/"review/b0_replenishment_human_confirmation.json"
BASE_RESULT=ROOT/"results/h2_mini_b0_replenishment_analysis.json"

def load_baseline():
    data=base.inspect()
    if data["errors"]:raise ValueError("Baseline evidence integrity failed")
    data=base.apply_human_confirmation(data,BASE_CONFIRMATION)
    result=base.analyze(data)
    if result["completed_judgments"]!=1219:raise ValueError("Baseline judgments incomplete")
    canonical=load(BASE_CANONICAL)
    if canonical["assignment_manifest_sha256"]!=file_sha(BASE_MANIFEST):raise ValueError("Baseline assignment hash changed")
    for entry in canonical["files"].values():
        for path_field,hash_field in [("source_file","source_sha256"),("canonical_file","canonical_sha256")]:
            if file_sha(ROOT/entry[path_field])!=entry[hash_field]:raise ValueError("Frozen submitted/canonical evidence changed")
    return data,result,canonical

def expansion_selection(records):
    """Uses role assignment, IDs, language and length only. No primary judgment input."""
    frame=[]
    allowed=["review_id","candidate_id","source","source_revision","source_native_id","source_parent_id",
      "source_immediate_parent_id","source_row_index","language","deberta_tokens","review_cohort",
      *HASH_FIELDS,*GROUP_FIELDS]
    for r in records:
        assignments=r["reviewer_assignments"]
        if r["review_lane"]!="TRAIN_CANDIDATE" or len(assignments)!=1:continue
        if assignments[0]["review_role"]!="PRIMARY":raise ValueError("Invalid single-primary frame")
        clean={k:r.get(k) for k in dict.fromkeys(allowed)}
        clean["primary_reviewer"]=assignments[0]["reviewer"]
        clean["primary_batch_filename"]=assignments[0]["batch_filename"]
        frame.append(clean)
    if len(frame)!=525:raise ValueError("Expected525 singly reviewed training candidates")
    selected=[]
    for second,n in [("A",53),("B",52)]:
        first="B" if second=="A" else "A"
        pool=[r for r in frame if r["primary_reviewer"]==first]
        chosen=choose_diverse(pool,n,f"{EXP_PACKAGE}|{EXP_SEED}|second-{second}")
        for r in chosen:
            rr=dict(r,second_reviewer=second,review_role="SECOND_REVIEW",
              batch_filename=f"reviewer_{second}_batch_01.csv",human_review_status="not_started",final_assignment=None)
            selected.append(rr)
    if len(selected)!=105 or len({r["review_id"] for r in selected})!=105:raise ValueError("Expansion count/identity failure")
    return selected,frame

def old_audit_records(records):
    return [r for r in records if r["review_lane"]=="TRAIN_CANDIDATE" and len(r["reviewer_assignments"])==2]

def summarize_pairs(old_pairs,new_pairs,expected_new=105):
    """Never drops old pairs; incomplete expansion cannot yield a gate decision."""
    if len(old_pairs)!=75:raise ValueError("Exactly75 frozen old audit pairs are required")
    if len(new_pairs)>expected_new:raise ValueError("Too many expansion audit pairs")
    all_pairs=list(old_pairs)+list(new_pairs)
    full=len(new_pairs)==expected_new
    metrics=base.pair_summary(all_pairs)
    b0=metrics["B0_confirmation"]["disagreement_rate"]
    decision=metrics["decision"]["disagreement_rate"]
    return {"expected_full_denominator":180,"observed_pairs":len(all_pairs),
      "old_pairs_retained":75,"new_completed_pairs":len(new_pairs),"all180_complete":full,
      "full_audit_metrics":metrics,"gate_threshold":.10,
      "B0_confirmation_gate_pass":b0<=.10 if full else None,
      "decision_gate_pass":decision<=.10 if full else None,
      "status":"INCOMPLETE_EXPANDED_REVIEW" if not full else "STOP_EXPANDED_AUDIT_DISAGREEMENT" if b0>.10 or decision>.10 else "EXPANDED_AUDIT_GATE_PASS_POST_REVIEW_AUDIT_REQUIRED",
      "denominator_policy":"All75 original and105 new independent pairs, regardless of acceptance or disagreement; no old pair replaced or dropped."}
