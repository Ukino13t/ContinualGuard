"""Publish metadata-only review results and synchronize project status documents."""
import csv, json, math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from analyze_h2_mini_production_review import ROOT, COHORTS, load, save, file_sha, agreement

TAG="H2_MINI_PRODUCTION_RESULTS_20261003"
def block(path,text,top=False):
    old=path.read_text(encoding="utf-8-sig") if path.exists() else "# "+path.stem.replace("_"," ")+"\n"
    start=f"<!-- {TAG}:BEGIN -->";end=f"<!-- {TAG}:END -->"
    new=start+"\n"+text.rstrip()+"\n"+end
    if start in old:
        a,b=old.split(start,1);_,c=b.split(end,1);out=a+new+c
    elif top:
        first,sep,rest=old.partition("\n");out=first+"\n\n"+new+"\n\n"+rest.lstrip()
    else:out=old.rstrip()+"\n\n"+new+"\n"
    path.write_text(out,encoding="utf-8")
def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]+
      ["| "+" | ".join(map(str,r))+" |" for r in rows])+"\n"
def pct(x):return f"{100*x:.2f}%"
def main():
    result_path=ROOT/"results/h2_mini_production_review_analysis.json"
    a=load(result_path);audit=load(ROOT/a["post_review_audit"]);m=load(ROOT/"review/production_review_manifest.json")
    canonical=load(ROOT/a["canonical_manifest"]);byid={r["review_id"]:r for r in m["records"]}
    judgments=defaultdict(list)
    for name,f in canonical["files"].items():
        who=name.split("_")[1]
        with (ROOT/f["canonical_path"]).open(encoding="utf-8-sig",newline="") as stream:
            for row in csv.DictReader(stream):judgments[row["review_id"]].append({**row,"reviewer":who})
    paired={c:agreement([r for r in m["records"] if r["review_cohort"]==c],judgments) for c in COHORTS}
    reviewer={}
    for c in COHORTS:
        reviewer[c]={}
        for who in ["A","B"]:
            jj=[x for rid,rr in judgments.items() if byid[rid]["review_cohort"]==c for x in rr if x["reviewer"]==who]
            reviewer[c][who]={"judgments":len(jj),"stratum_counts":dict(Counter(x["reviewer_stratum"] for x in jj)),
              "decision_counts":dict(Counter(x["reviewer_decision"] for x in jj))}
    combined_audit={}
    for c in COHORTS[:-1]:
        d=a["training_second_review_audit"][c]
        combined_audit[c]={"audited":d["pairs"],"any_decision_or_stratum_disagreement":d["any_decision_or_stratum_disagreement"],
          "rate":d["any_decision_or_stratum_disagreement"]/d["pairs"],
          "exceeds_10_percent":d["any_decision_or_stratum_disagreement"]/d["pairs"]>.10}
    paused=[c for c,d in combined_audit.items() if d["exceeds_10_percent"]]
    short=load(ROOT/a["accepted_pools"]["B-short"]["path"])["records"];langs=Counter(r["language"] for r in short)
    matching={"eligible_count":len(short),"eligible_language_counts":dict(langs),"English_available":langs["en"],
      "prospective_English_target":90,"prospective_non_English_target":30,"English_shortage":max(0,90-langs["en"]),
      "max_English_fraction_in_120":min(langs["en"],120)/120,
      "target_English_fraction":.75,"freeze_executed":False,
      "status":"COUNT_PASSES_BUT_90_30_LANGUAGE_MATCH_NOT_FEASIBLE_IN_THIS_REVIEW_POOL"}
    for c in COHORTS:
        path=ROOT/a["accepted_pools"][c]["path"];pool=load(path)
        pool["stratum_finalization_status"]="PAUSED_FOR_STRATUM_RELIABILITY_REVIEW" if c in paused else "BLOCKED_GLOBAL_CAPACITY_GATE"
        pool["counts_are_frozen_label_evidence_not_authorization_to_train"]=True
        save(path,pool);a["accepted_pools"][c]["sha256"]=file_sha(path)
    a.update(by_cohort_reviewer=reviewer,paired_review_by_cohort=paired,combined_label_training_audit=combined_audit,
      paused_strata=paused,overall_review_gate="NOT_PASSED_FOR_FINALIZATION",
      stratum_gate_note="Preserve the prior analyzer's decision-only >10% statistic. The user requires unresolved stratum conflicts to block finalization: combined-label disagreement exceeds 10% for B1/B2/B3, which remain paused; accepted-pool counts freeze individually matching label evidence only.",
      b_short_matching_feasibility=matching)
    save(result_path,a)
    cap=a["capacity"];c0=cap["C0_all_four_B0_pools"]
    next_prompt=("Review the H2-mini production-review findings and keep final assignments, materialization, training and replay blocked. "
      "Approve preparation of a focused human review-resolution and re-calibration plan only, without changing original rubric-v1 judgments or performing automated adjudication. "
      "Diagnose B0 versus B1/B2/B3 disagreement, propose any new rubric version for explicit human approval, and plan independent second review of at least 75 of the 248 existing training candidates per hard-benign stratum (25 already reviewed; at least 50 more each), plus adequate double-reviewed validation/test reserves. "
      "Retain the complete matched H2m-C0 design; propose additional B0 review addressing its 821-record accepted-count shortage. "
      "Address B-short's 48-record English deficit relative to the planned 90/30 language mix. "
      "Report revised workload and replacement targets before preparing new production batches. "
      "Do not reuse disputed records as ground truth, relabel from source/model outputs, weaken grouping, select final IDs, create research JSONL, train, choose a numerical memory budget, or implement replay. Stop for human review.")
    report=["# H2-mini production-review results\n",
      "Status: **STOP_CAPACITY_SHORTAGE; overall finalization gate NOT PASSED.** Analysis and post-review audit are complete. The user's section-8 stop takes precedence over downstream B-short freezing and assignment steps. B1/B2/B3 are also paused for severe stratum disagreement. No final split proposal, matched bank of 120, raw research JSONL, training or replay was created.\n",
      "## Evidence integrity and timing\n",
      f"All {a['completed_judgments']:,}/{a['validation']['expected_judgments']:,} judgments on 1,590 candidates in 24 batches are present. Missing=0; malformed=0; unexpected/duplicate assignments=0; all immutable source text hashes match. All submissions decode as UTF-8. Canonical UTF-8-BOM/CRLF transport copies preserve every decoded field, including decisions, strata, reasons, confidence, notes and exact recorded timing strings. Original submissions and byte-identical snapshots are untouched. Original/snapshot/canonical hashes are in review/production_review_canonical_manifest.json; raw CSVs and free-form notes remain local and Git-ignored.\n",
      table(["Reviewer","Judgments","Mean seconds","Median seconds","p90 seconds","Recorded person-hours"],
        [[w,v["judgments"],f'{v["mean_seconds"]:.4f}',v["median_seconds"],v["p90_seconds"],f'{v["person_hours"]:.6f}'] for w,v in a["timing"].items()]),
      "Recorded total is 34,881 seconds = **9.689167 person-hours** (A 5.073889; B 4.615278). These are submitted per-item estimates, not independent stopwatch validation. No timing is imputed.\n",
      "## Human acceptance versus target-stratum eligibility\n",
      "'Benign accepted' means every required reviewer chose ACCEPT_BENIGN (one primary for unaudited training records; both for audited/double-reviewed records). 'Target-confirmed' also requires every required reviewer to confirm the original proposed stratum. B-short's proposed/accepted stratum is B0. No source-based, model-based or automatic human-label reassignment occurs.\n",
      table(["Cohort","Reviewed","Benign accepted","Benign acceptance","Target-confirmed before audit","Target yield","Final group-safe evidence"],
        [[c,a["by_cohort"][c]["candidates"],a["by_cohort"][c]["all_required_accept_benign"],pct(a["by_cohort"][c]["benign_acceptance_rate"]),a["by_cohort"][c]["eligible_in_proposed_stratum"],pct(a["by_cohort"][c]["target_stratum_acceptance_rate"]),cap[c]["group_safe_available"]] for c in COHORTS]),
      "These are frozen review-evidence counts, not authorization to finalize the paused strata or train. In particular, all-required benign acceptance does not establish the proposed hard-benign level.\n",
      table(["Cohort","At least one human rejection","Benign but target not confirmed","Post-audit exclusion","Overall unusable rate"],
        [[c,a["by_cohort"][c]["excluded_at_least_one_rejection"],a["by_cohort"][c]["benign_but_stratum_not_confirmed"],a["by_cohort"][c]["eligible_in_proposed_stratum"]-cap[c]["group_safe_available"],pct(cap[c]["not_usable_rate"])] for c in COHORTS]),
      "For B1/B2/B3 the dominant capacity loss is unconfirmed target stratum (231/243/241 records), not malicious detection or duplicate filtering. Actual human rejection rates are 8.61%/7.22%/6.67%; overall unusable rates are 72.78%/74.72%/73.61%.\n",
      "Per-judgment decision counts (paired records can contribute twice):\n",
      table(["Cohort","Accept","Ambiguous","Harmful/unsafe","Missing context","Privacy/rights","True injection","Quality"],
        [[c,*[a["by_cohort"][c]["decision_counts_per_judgment"].get(k,0) for k in ["ACCEPT_BENIGN","REJECT_AMBIGUOUS_INTENT","REJECT_HARMFUL_OR_UNSAFE","REJECT_CONTEXT_MISSING","REJECT_PRIVACY_OR_RIGHTS","REJECT_TRUE_INJECTION","REJECT_QUALITY"]]] for c in COHORTS]),
      "Dominant rejection decisions are missing context for B0 (5 judgments); ambiguous intent for B1 (20) and B2 (16); harmful/unsafe for B3 (13); privacy/rights and missing context tied for B-short (2 each). Exact recorded reason_code counts, without collapsing different human codes, are retained in the analysis JSON.\n",
      "## Paired agreement and reviewer asymmetry\n",
      table(["Scope","Pairs","Decision agreement","Decision kappa","Stratum agreement","Stratum kappa","Safety conflicts"],
        [[name,d["pairs"],pct(d["decision_agreement"]),f'{d["decision_kappa"]["value"]:.6f}',pct(d["stratum_agreement"]),f'{d["stratum_kappa"]["value"]:.6f}',d["safety_critical_disagreements"]] for name,d in [("Val/test/B-short double review",a["double_review"]),("All pairs including training audits",a["all_paired_review"])]]),
      "Among all 698 pairs: 22 benign-versus-reject conflicts, 7 disagreements between rejection decisions, and 357 stratum-only disagreements. There are 0 ACCEPT_BENIGN versus REJECT_TRUE_INJECTION conflicts. One unaudited primary judgment rejects TRUE_INJECTION; it is excluded, and zero paired safety conflicts must not be interpreted as universal safety validation. Kappas are descriptive and prevalence-sensitive.\n",
      table(["Proposed cohort","A: judgments","A: stratum counts","B: judgments","B: stratum counts"],
        [[c,reviewer[c]["A"]["judgments"],json.dumps(reviewer[c]["A"]["stratum_counts"],sort_keys=True),reviewer[c]["B"]["judgments"],json.dumps(reviewer[c]["B"]["stratum_counts"],sort_keys=True)] for c in COHORTS]),
      "Reviewer B assigns B0 to 225/248, 235/249 and 235/248 judgments in proposed B1/B2/B3; reviewer A confirms the corresponding target in 207/249, 210/248 and 212/249. This is a substantial reviewer-dependent operational difference. It does not identify which reviewer is correct and is not evidence of misconduct. Source labels and cue heuristics cannot resolve it. The earlier accepted calibration PASS remains historical evidence; it does not establish reliable production strata or continual-learning retention.\n",
      "## Training second-review gate\n",
      table(["Cohort","Audited","Decision disagreement","Stratum disagreement","Any decision/stratum disagreement","Safety conflicts"],
        [[c,d["pairs"],f'{d["decision_disagreement_count"]}/{d["pairs"]} ({pct(d["decision_disagreement_rate"])})',f'{d["stratum_disagreement_count"]}/{d["pairs"]} ({pct(1-d["stratum_agreement"])})',f'{d["any_decision_or_stratum_disagreement"]}/{d["pairs"]} ({pct(combined_audit[c]["rate"])})',d["safety_critical_disagreements"]] for c,d in a["training_second_review_audit"].items()]),
      "The previous analyzer's decision-only 10% statistic passes in each stratum. It is not a blanket review PASS: the combined decision-or-stratum audit exceeds 10% for B1/B2/B3, which are paused and must not be finalized. Preserve rubric v1; request human resolution and a separately approved expanded independent audit (at least 30% =75 of 248 per hard stratum, 50 additional to the existing 25), with re-calibration or a new rubric version only after explicit approval. No new judgments or adjudication were inferred.\n",
      "## Source/category slices\n",
      table(["Source","Candidates","Benign accepted","Benign rate","Target-confirmed"],[[s,d["candidates"],d["all_required_accept_benign"],pct(d["benign_acceptance_rate"]),d["eligible_in_proposed_stratum"]] for s,d in a["by_source"].items()]),
      table(["OR category","Candidates","Benign accepted","Benign rate","Target-confirmed"],[[s,d["candidates"],d["all_required_accept_benign"],pct(d["benign_acceptance_rate"]),d["eligible_in_proposed_stratum"]] for s,d in a["by_source_category"].items()]),
      "## Post-review grouping and contamination\n",
      f"Freshly checked all 768 individually eligible records against pinned source hashes, native/tree metadata, normalization and untruncated token hashes. Stored-group conflicts=0; accepted-only exhaustive lexical/latent/template pairs crossing thresholds=0. Source-scoped project IDs are never treated as globally unique. Cross-stratum and cross-source identity checks are included.\n",
      table(["External bank","Records checked","Affected accepted queries"],[[k,v["target_count"],v["affected_queries"]] for k,v in audit["external_contamination"].items() if "target_count" in v]),
      "AF coverage includes all 7,320 malicious records and all 7,320 companion benign document records from AF1-AF4 (3,660 per family). NotInject covers all 339 frozen hashes. B-short coverage is the full proposed 150, including later rejected candidates; identity self-comparisons are excluded. Exact/normalized/model-view matches, normalized substrings of >=25 characters, high lexical Jaccard>=.85, latent cosine>=.97 with Jaccard>=.30, and project template matches were checked. No predictions or detector inference were used.\n",
      "One B0 record, H2PR-032898BDE74A8A07, is quarantined for a B-short semantic-neighbor candidate against H2PR-4EE6BB5ABE4414EE (latent cosine 0.97470659; token-set Jaccard 0.30). It is not an exact duplicate or a new judgment about benignness. The original human labels remain frozen. All retained pools have zero remaining candidates under the checked thresholds; the exclusion and target IDs are fully recorded.\n",
      "The audit reuses the deterministic full-OR hashed lexical/SVD basis with existing thresholds and compares all relevant pairs, avoiding the older LSH shortlist's missed candidate. It is a lexical semantic approximation, not a pretrained semantic encoder. Official OR seed/rewrite-parent mapping is unavailable; neither this audit nor the existing approximate groups prove all semantic siblings absent. Rare-phrase links remain those of the frozen full-source groups, not redefined subset rarity.\n",
      f"All 768 pre-exclusion accepted texts fit max_length=1280 with special tokens included; maximum observed length={audit['max_length_check']['max_tokens']}; truncations=0. This verifies benign pool inputs only. Full assignment size/label/length/leakage verification is **not run**, because final assignments do not exist. Audit runtime={audit['runtime_seconds']:.2f} seconds, excluding library import; no GPU detector was loaded.\n",
      "## Capacity stop and workload implications\n",
      table(["Pool","Available","Required","Exact shortage","Additional candidates at observed pooled yield"],
        [[c,cap[c]["group_safe_available"],cap[c]["required"],cap[c]["shortage"],cap[c]["additional_candidates_at_observed_pooled_yield"]] for c in COHORTS]+[["Full C0 B0 pools",c0["available"],c0["required"],c0["shortage"],c0["additional_candidates_at_observed_pooled_yield"]]]),
      "Additional-candidate figures use ceil(shortage / observed usable yield). They are arithmetic planning illustrations, not reliable success forecasts: reviewer/role-dependent yield and stratum disagreement invalidate simple extrapolation. Do not launch another large review queue using these estimates alone.\n",
      table(["Cohort","Train review lane","Validation reserve","Test reserve","Total double-qualified","Additional double-qualified needed for 30val+60test"],
        [[c,cap[c]["eligible_by_review_lane"].get("TRAIN_CANDIDATE",0),cap[c]["eligible_by_review_lane"].get("VALIDATION_RESERVE",0),cap[c]["eligible_by_review_lane"].get("TEST_RESERVE",0),cap[c]["eligible_with_two_independent_judgments"],cap[c]["minimum_additional_double_qualified_for_30val_60test"]] for c in COHORTS[:-1]]),
      "B1 has only 2 double-qualified records; B2/B3 have none. Primary-only records cannot fill double-reviewed validation/test quotas. The provisional lanes are not fixed final splits, but moving a record cannot waive review coverage. B1's double-review reserve yield is 2/112; naively finding 88 more at that rate would require 4,928 double-reviewed candidates. B2/B3 have zero observed double-reserve yield, so no finite empirical replacement estimate is supportable before resolving reviewer interpretation. No such sampling is authorized.\n",
      "C0 requires four independent B0 sets (1,160), not four copies of the same 290: with 339 available it lacks 821 records and needs at least 235 more double-qualified B0 records for its combined 360 validation/test requirement. The shared Stage-1 anchor does not waive independence of C0 stages 2-4.\n",
      f"B-short count passes (144>=120), but only {langs['en']} accepted records are English. The predeclared approximately 90 English/30 multilingual mix lacks {matching['English_shortage']} English records. The maximum possible English share of a 120-record selection is 35%, versus planned 75%. No claim of matched language composition is made. Human approval is needed for additional English review or an explicitly documented matching limitation; no bank is frozen at this stop.\n",
      "## Artifact status and human actions\n",
      table(["Requested output/check","Status"],[
        ["Original/canonical evidence and accepted metadata pools","Completed; labels unchanged, raw text local"],
        ["H2m-C0 hash-only proposal","NOT CREATED: capacity/review stop"],
        ["H2m-F hash-only proposal","NOT CREATED: capacity/review stop"],
        ["Shared malicious assignment/equivalence verification","DEFERRED; frozen source integrity checked only"],
        ["Matched-control size/label/split/position verification","DEFERRED; no final assignment to validate"],
        ["Final B-short bank of 120","DEFERRED; global capacity stop and matching limitation"],
        ["Finalization/materialization readiness","NO"],
        ["No-replay training execution proposal readiness","NO; prospective protocol documented only"],
        ["Replay readiness","R0 active; R1/R2 not entered"],
        ["Numerical replay memory budget","Not selected"]]),
      "Required next human actions: review the stratum disagreement and choose an approved resolution/re-calibration process; approve any expanded independent review and new rubric version before use; approve replenishment covering both hard-benign strata and all four C0 B0 pools; resolve B-short language matching; then re-audit before final hash-only assignments. Later assignment approval, raw materialization approval, training approval and any replay approval remain separate.\n",
      "Active root is E:\\ContinualGuard; interpreter is E:\\ContinualGuard\\.venv\\Scripts\\python.exe (Python 3.14.5 venv). The archival copy was not accessed. No storage auditing, cleanup, deduplication on disk, training, selective adaptation or replay occurred. The scientific R0 interpretation and prospective R1/R2 criteria are unchanged. PIGuard motivates over-defense; CAPTURE remains contextual design inspiration only; Chen-derived AF construction is not claimed as an exact reproduction.\n",
      "## Exact recommended next prompt\n\n"+next_prompt+"\n"]
    (ROOT/"docs/h2_mini_production_review_results.md").write_text("\n".join(report),encoding="utf-8")
    status=("## Current production-review status (2026-10-03)\n\n"
      "**STOP_CAPACITY_SHORTAGE; R0 remains active.** All 2,288 judgments are frozen and complete (9.689167 recorded person-hours). "
      "After post-review audit, individually target-confirmed group-safe evidence is B0=339, B1=98, B2=91, B3=95, B-short=144. "
      "B1/B2/B3 remain paused for stratum reliability; decision agreement alone is not an overall PASS. "
      "Full four-stage C0 requires 1,160 B0 (shortage 821); B-short has only 42 English candidates against the planned 90.\n\n"
      "No final B-short bank, assignment proposal, research JSONL, H2 training or replay exists. "
      "See [production review results](h2_mini_production_review_results.md), "
      "[future materialization plan](h2_mini_final_materialization_plan.md) and [prospective no-replay protocol](h2_mini_no_replay_protocol.md). "
      "Earlier dated sections below describe historical states and must not be read as current authorization.\n")
    for name in ["h2_mini_design.md","h2_mini_review_protocol.md","security_bidirectional_stream.md","replay_readiness_review.md","h2_mini_capacity_report.md","h2_mini_materialization_plan.md","b_short_control_mini.md"]:
        block(ROOT/"docs"/name,status,top=True)
    # Update unqualified stale status sentences; preserve dated history and frozen rubric content.
    replacements={
      "h2_mini_design.md":[("Status: source acquisition, audit, capacity estimation, and review planning complete. Human record review, final assignment, final JSONL materialization, training, replay, and selective adaptation have not started.","Historical source-design status (2026-09-13): source audit and review planning were complete; human record review had not yet started. Current production-review results are above.")],
      "h2_mini_review_protocol.md":[("No review timing has been measured, so person-hours are not projected yet.","Historical pre-calibration planning assumption: timing had not yet been measured. Production timing is now reported above.")],
      "security_bidirectional_stream.md":[("This update covers discovery and design only.","The full-scale sections below preserve the earlier source-discovery design; the current reduced H2-mini production-review status is above.")],
      "replay_readiness_review.md":[("Current approved usable counts are 0 for B-short, H2-C0, B0, B1, B2, and B3.","At the historical full-scale source-discovery stage, approved usable counts were 0; the current H2-mini evidence pools and gates are reported above.")],
      "replay_comparison_protocol.md":[("after the H2-mini baseline size and review outcome are known.","after complete H2-mini no-replay results justify replay entry and a human approves the numerical budget.")]}
    for name,pairs in replacements.items():
        path=ROOT/"docs"/name;text=path.read_text(encoding="utf-8")
        for before,after in pairs:text=text.replace(before,after)
        path.write_text(text,encoding="utf-8")
    block(ROOT/"docs/replay_comparison_protocol.md","## Production-review stop and fixed controls\n\nR0 is unchanged after the 2026-10-03 data-quality audit. No replay strategy or numerical budget is authorized. Numerical memory selection occurs only after completed no-replay results justify entry, followed by human review. Preserve the same detector, initialization, stage records/order, seeds, epochs, learning rate, optimizer-reset policy, evaluation and historical-example cap for every strategy. Logit/metadata bytes are reported separately; no method gains extra examples.",top=True)
    block(ROOT/"docs/assumptions_and_risks.md",
      "## Production-review risks (2026-10-03)\n\n"+table(["ID","Risk","Response/status"],[
       ["R076","High decision agreement masks severe reviewer-stratum disagreement","Double-review stratum agreement48.49%; B1/B2/B3 training stratum disagreement88/96/92%. Pause finalization; human resolution and expanded independent audit."],
       ["R077","Pooled acceptance yield masks almost zero double-qualified hard-benign capacity","Only2/0/0 double-qualified B1/B2/B3; numerical replacement estimates are not forecasts."],
       ["R078","B-short review selection underrepresents English","42 English accepted versus planned 90; do not claim90/30 match or use detector outputs to repair it."],
       ["R079","Approximate source audit missed a semantic-neighbor candidate","Exhaustive post-review comparison quarantines one B0 against B-short; official OR parents remain unavailable."],
       ["R080","Single reviewed B0 anchor is mistaken for complete matched control","C0 needs1160 independent B0, has339, lacks821; no repeated-stage reuse."],
       ["R081","Review success may be misrepresented as continual-learning evidence","R0 unchanged; no H2-mini training or replay."],
      ]),top=True)
    block(ROOT/"docs/open_questions.md",
      "## Current decisions needed (2026-10-03)\n\n"+table(["ID","Question","Status/recommendation"],[
       ["Q042","Did completed production review pass?","Resolved: complete evidence, but NOT passed for finalization; stratum disagreement and capacity stop."],
       ["Q043","Retain complete four-stage C0?","Resolved by current user request: YES. Additional B0 review plan still needs approval."],
       ["Q044","Choose numerical replay memory budget?","Deferred until no-replay evidence justifies entry and human approval."],
       ["Q045","Create final hash-only proposals now?","Blocked: B1/B2/B3 deficits192/199/195; C0 B0 deficit821."],
       ["Q046","Materialize or train now?","NO. Assignment, materialization and training approvals remain separate."],
       ["Q047","How should reviewer-dependent B0/B1/B2/B3 interpretation be resolved?","Human review; preserve v1 evidence, separately approve re-calibration/version change and >=30% independent training audit."],
       ["Q048","What replacement review workload and double-review coverage are approved?","Plan after interpretation is resolved; do not rely on pooled yield."],
       ["Q049","How to address B-short's48-English-record matching deficit?","Approve additional independent English review or explicitly accept a documented limitation; no score-based selection."],
      ])+"\nExact recommended next prompt:\n\n"+next_prompt,top=True)
    stamp=datetime.now(timezone.utc).isoformat()
    decision={"date":stamp,"id":"D032","decision":"Freeze completed H2-mini production review and stop at the mandated capacity gate; pause hard-benign stratum finalization",
      "source_evidence":["User production-analysis request sections4 and8","results/h2_mini_production_review_analysis.json","data/audits/hard_benign/h2_mini_post_review_audit.json"],
      "motivation":"All2288 judgments are complete, but B1/B2/B3 stratum reliability and required capacities fail; complete matched C0 and B-short language match are also unsupported",
      "consequences":"Preserve all human labels and accepted metadata; no automatic adjudication or reassignment; no final assignments,120-bank,JSONL,training,replay or numerical memory budget",
      "counts":{"B0":339,"B1":98,"B2":91,"B3":95,"B-short":144},"replay_readiness":"R0",
      "status":"User-mandated stop applied; human review-resolution/replenishment approval pending"}
    changed=[".gitignore","scripts/analyze_h2_mini_production_review.py","scripts/audit_h2_mini_post_review.py","scripts/report_h2_mini_production_review.py","tests/test_h2_mini_review_audit.py",
      "review/production_review_canonical_manifest.json","review/production_submitted/20261003-v1/ (local ignored)","review/production_canonical/20261003-v1/ (local ignored)",
      "results/h2_mini_production_review_analysis.json","data/audits/hard_benign/h2_mini_review_outcomes.json","data/audits/hard_benign/h2_mini_post_review_audit.json",
      "data/manifests/hard_benign/{B0,B1,B2,B3,B_short}_accepted.json"]
    changed += ["docs/"+n for n in ["h2_mini_production_review_results.md","h2_mini_design.md","h2_mini_review_protocol.md","h2_mini_final_materialization_plan.md","h2_mini_no_replay_protocol.md","security_bidirectional_stream.md","replay_comparison_protocol.md","replay_readiness_review.md","decision_log.md","change_log.md","assumptions_and_risks.md","open_questions.md","h2_mini_capacity_report.md","h2_mini_materialization_plan.md","b_short_control_mini.md"]]
    changed+=["history/decisions.jsonl","history/changes.jsonl"]
    change={"date":stamp,"id":"C019","files_changed":changed,"reason":"Analyze completed human production review with provenance-safe freezing and accepted-only post-review audit",
      "effect":"Freeze1590 candidate outcomes/2288 judgments; verify all external banks; quarantine1 B0 semantic-neighbor candidate; record exact capacity stop and future protocols",
      "tests_run":["7 synthetic review/audit unit tests passed","24 completed batches/2288 expected judgments validated; no missing/malformed/changed-text rows",
        "Canonical fields and submitted/snapshot hashes verified","768 accepted records freshly tokenized; max399, none over1280",
        "768 accepted-only all-pair checks;339 NotInject;150 B-short;14640 AF1-AF4 external coverage",
        "No final assignments/raw training JSONL or model runs created"],
      "scientific_behavior_changed":False,
      "limitations":["Severe stratum disagreement; individually accepted labels do not authorize paused strata","OR official parent mapping unavailable; semantic grouping approximate",
        "B1/B2/B3 shortages192/199/195; full C0 B0 shortage 821","B-short English shortage48","No H2-mini retention evidence;R0 unchanged"]}
    for path,item in [(ROOT/"history/decisions.jsonl",decision),(ROOT/"history/changes.jsonl",change)]:
        previous=[json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        existing=next((x for x in previous if x.get("id")==item["id"]),None)
        if existing is None:
            with path.open("a",encoding="utf-8") as f:f.write(json.dumps(item,ensure_ascii=False)+"\n")
        else:item.clear();item.update(existing)
    block(ROOT/"docs/decision_log.md","## D032 - Production-review capacity stop\n\n"+"\n\x60\x60\x60json\n"+json.dumps(decision,ensure_ascii=False,indent=2)+"\n\x60\x60\x60",top=False)
    block(ROOT/"docs/change_log.md","## C019 - Complete production-review evidence and audit\n\n"+"\n\x60\x60\x60json\n"+json.dumps(change,ensure_ascii=False,indent=2)+"\n\x60\x60\x60",top=False)
    print(json.dumps({"status":a["status"],"paused_strata":paused,"b_short_matching":matching,
      "report":"docs/h2_mini_production_review_results.md","decision":"D032","change":"C019"},indent=2))
if __name__=="__main__":main()
