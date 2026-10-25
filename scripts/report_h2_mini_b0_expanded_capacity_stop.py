import json,sys
from datetime import datetime,timezone
from pathlib import Path
from h2_mini_b0_expansion_common import ROOT,load

MARKER="H2_MINI_B0_EXPANDED_CAPACITY_STOP_20261004"
DOCS=["h2_mini_b0_expansion_results.md","h2_mini_b0_replenishment_results.md","h2_mini_simplified_design.md","h2_mini_design.md","h2_mini_final_materialization_plan.md","h2_mini_no_replay_protocol.md","replay_comparison_protocol.md","replay_readiness_review.md","decision_log.md","change_log.md","assumptions_and_risks.md","open_questions.md"]

def append(path,event):
    before=path.read_bytes()
    assert event["id"] not in {json.loads(x)["id"] for x in before.decode("utf-8").splitlines() if x.strip()}
    with path.open("ab") as f:
        if before and not before.endswith(b"\n"):f.write(b"\n")
        f.write((json.dumps(event,ensure_ascii=False)+"\n").encode("utf-8"))
    assert path.read_bytes().startswith(before)

def prepend(path,body):
    old=path.read_bytes();assert MARKER.encode() not in old
    pos=old.index(b"\n")+1
    block=("\n<!-- "+MARKER+":BEGIN -->\n"+body+"\n<!-- "+MARKER+":END -->\n").encode("utf-8")
    path.write_bytes(old[:pos]+block+old[pos:])

def main():
    assert Path.cwd().resolve()==ROOT.resolve()
    assert Path(sys.executable).resolve()==(ROOT/".venv/Scripts/python.exe").resolve()
    cap=load(ROOT/"results/h2_mini_b0_expanded_capacity.json")
    exp=load(ROOT/"results/h2_mini_b0_expansion_analysis.json")
    rec=load(ROOT/"results/h2_mini_b0_expansion_integrity_recovery.json")
    aud=load(ROOT/"data/audits/hard_benign/h2_mini_b0_expanded_post_review_audit.json")
    pool=load(ROOT/"data/manifests/hard_benign/B0_expanded_group_safe.json")
    rm=load(ROOT/"review/b0_replenishment_expansion_integrity_recovery_manifest.json")
    assert rec["integrity_status"]=="PASS" and rm["status"]=="PASS"
    metric=exp["expanded_audit"]["full_audit_metrics"]
    assert exp["expanded_audit"]["B0_confirmation_gate_pass"] is True
    assert metric["B0_confirmation"]["disagreements"]==12 and metric["decision"]["disagreements"]==4
    assert cap["status"]=="STOP_B0_CAPACITY_SHORTAGE" and not cap["all_requirements_pass"]
    assert aud["unique_post_review_excluded"]==0 and pool["counts"]==cap["counts"]
    absent=["data/manifests/hard_benign/b_short_final_hash_manifest.json",
      "data/manifests/h2_mini/h2m_c0_final_assignment_proposal.json",
      "data/manifests/h2_mini/h2m_f_final_assignment_proposal.json",
      "data/manifests/h2_mini/h2m_shared_malicious_assignment.json",
      "results/h2_mini_matched_control_verification.json",
      "data/manifests/h2_mini/h2m_final_assignment_validation.json"]
    assert all(not (ROOT/x).exists() for x in absent)
    now=datetime.now(timezone.utc).isoformat()
    next_prompt=("Approve the minimum B0 top-up review phase only. Use the existing frozen metadata-only reserve; "
      "do not add candidates or alter thresholds. Prepare human review worksheets for enough reserve records to "
      "cover at least20 accepted training B0 and10 accepted double-reviewed evaluation B0, while accounting for "
      "rejection yield. Keep the40 training/25 evaluation reserve roles fixed, use opposite independent reviewers "
      "for any required training audits and two independent reviewers for evaluation, preserve all prior evidence, "
      "and stop after human-review package preparation. Do not create final H2-mini assignments, materialize JSONL, "
      "train, implement replay, or choose a replay memory budget.")
    body=f"""## Current result — {now}

**STOP_B0_CAPACITY_SHORTAGE.** Recovery integrity passed: Reviewer A53 (36 original-intact plus17 explicit reconfirmations), Reviewer B52, and all105 new second reviews validate against the fixed assignment and immutable source hashes. The original damaged A file, B file, reconfirmation file and all prior judgments remain unchanged. The separate canonical A file and row-level provenance are recorded in [the recovery manifest](../review/b0_replenishment_expansion_integrity_recovery_manifest.json).

The complete audit retained all75 original pairs and105 new pairs. Over180 pairs, decisions agree176/180 (97.78%; kappa0.6586), with4 disagreements; B0 confirmation agrees168/180 (93.33%; kappa0.5393), with12 disagreements =6.67%. The frozen <=10% B0 gate passes. One ACCEPT_BENIGN-versus-REJECT_TRUE_INJECTION conflict is retained and reported; it was not adjudicated and does not change the gate denominator.

B0 disagreements are5/59 English,5/66 Spanish,1/5 Portuguese-Brazil and1/14 Russian. By paired reviewer direction they are9/90 where B was primary/A second, and3/90 where A was primary/B second; these roles are coupled by design. The129–256 bin has3/5 disagreements and the257–1280 bin1/3, but both denominators are small. Medium-confidence subgroups also have small denominators. These descriptive concentrations do not establish language, length or reviewer causality. The aggregate gate supports proceeding to capacity audit, while moderate kappa and the retained safety conflict remain limitations.

The review-eligible metadata pool contains780 training records,181 independently audited accepted training records,350 independently double-reviewed evaluation records and1130 unique records. The fixed duplicate/group/contamination audit removed zero additional records. It checked native/tree, exact, normalized, model-view, semantic-parent and template groups; nonaccepted/rejected/calibration/BH/B-short history; NotInject339; and all14640 AF1–AF4 records. No affected B0 query was found. Thresholds were unchanged. Fresh pinned-tokenizer validation gives maximum399 tokens and zero above1280.

Final requirements fail: training needs800 (short20), double-reviewed evaluation needs360 for120 validation +240 test (short10), and unique total needs1160 (short30). Independent-audit coverage passes181>=80 and is count-feasible for20 per future stage, but no stage allocation was made. Existing fixed lanes provide33 validation,72 test and245 flexible evaluation records; combined remaining demand is255, so the same flexible record cannot satisfy both quotas.

The frozen metadata-only reserve has40 training candidates and25 evaluation candidates. Candidate counts can cover the minimum accepted shortage only if at least20/40 training (50%) and10/25 evaluation (40%) qualify. Acceptance is unknown, so the reserve is not certified sufficient accepted capacity. No reserve record was used and no worksheet was created.

BH evidence remains frozen at999 total (691 training,68 audited training,104 validation,204 test), sufficient in count for the planned870 allocation, but no concrete BH partition was resolved because the B0 capacity stop occurs first. B-short Option B remains approved in principle; no final120-record manifest was created.

No C0/F proposal, shared malicious assignment, matched-control verification, final leakage/count/assignment validation, research JSONL, materialization, training, replay or numerical memory budget was produced. R0 remains active; R1/R2 are not entered. Earlier status blocks below are historical.

Exact recommended next prompt:

> {next_prompt}
"""
    for name in DOCS:prepend(ROOT/"docs"/name,body)
    append(ROOT/"history/decisions.jsonl",{"date":now,"id":"D041",
      "decision":"Accept recovered105-review integrity and full180-pair B0 gate; stop at exact final B0 capacity shortage",
      "recovery_integrity":"PASS","new_second_reviews":105,"audit_pairs":180,
      "decision_agreement":{"count":176,"rate":176/180,"kappa":metric["decision"]["cohen_kappa"]},
      "B0_confirmation_agreement":{"count":168,"rate":168/180,"kappa":metric["B0_confirmation"]["cohen_kappa"]},
      "B0_disagreements":12,"B0_disagreement_rate":12/180,"gate_threshold":0.10,"gate_pass":True,
      "safety_critical_conflicts":1,"capacity":cap["counts"],"shortages":cap["shortages"],
      "post_review_exclusions":0,"reserve_used":False,"reserve_worksheets_created":False,
      "final_assignment":False,"materialization":False,"training":False,"replay":False,
      "numerical_memory_budget":None,"replay_readiness":"R0","next_prompt":next_prompt})
    changed=["scripts/recover_h2_mini_b0_expansion.py","scripts/h2_mini_b0_recovered_evidence.py",
      "scripts/patch_h2_mini_b0_recovered_analyzer.py","scripts/analyze_h2_mini_b0_expansion.py",
      "scripts/audit_h2_mini_b0_expanded_pool.py","scripts/report_h2_mini_b0_expanded_capacity_stop.py",
      "review/b0_replenishment_expansion/reviewer_A_batch_01_canonical_utf8.csv",
      "review/b0_replenishment_expansion_integrity_recovery_manifest.json",
      "results/h2_mini_b0_expansion_integrity_recovery.json","results/h2_mini_b0_expansion_analysis.json",
      "data/manifests/hard_benign/B0_expanded_review_eligible.json",
      "data/manifests/hard_benign/B0_expanded_group_safe.json",
      "data/audits/hard_benign/h2_mini_b0_expanded_post_review_audit.json",
      "results/h2_mini_b0_expanded_capacity.json"]+[f"docs/{x}" for x in DOCS]+[
      "history/decisions.jsonl","history/changes.jsonl"]
    append(ROOT/"history/changes.jsonl",{"date":now,"id":"C028","decision_id":"D041",
      "files_changed":changed,"effect":"Recovered-evidence PASS, full180-pair gate PASS and metadata-only final B0 capacity STOP",
      "validation":["10 synthetic expansion tests pass","A53/B52 canonical evidence and row provenance pass",
        "180 denominator retains old75+new105","Fixed grouping thresholds find zero additional B0 exclusions",
        "NotInject339 and AF1-AF4 14640 screened","Pinned tokenizer max399 and zero above1280",
        "All original submissions and protected evidence unchanged"],
      "topup_launched":False,"assignments_created":False,"training":False,"replay":False})
    for name in DOCS:
        assert (ROOT/"docs"/name).read_text(encoding="utf-8").count(MARKER+":BEGIN")==1
    for path,eid in [(ROOT/"history/decisions.jsonl","D041"),(ROOT/"history/changes.jsonl","C028")]:
        assert sum(json.loads(x)["id"]==eid for x in path.read_text(encoding="utf-8").splitlines() if x.strip())==1
    print(json.dumps({"status":cap["status"],"gate_pass":True,"capacity":cap["counts"],
      "shortages":cap["shortages"],"docs_updated":len(DOCS),"history":["D041","C028"],
      "assignments_created":False},indent=2))
if __name__=="__main__":main()

