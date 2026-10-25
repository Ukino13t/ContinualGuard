"""Record approved B0 preparation and current policy; preserve dated proposal evidence."""
import json,sys,statistics
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from h2_mini_b0_review_common import *
import analyze_h2_mini_b0_replenishment as analysis
MARKER="H2_MINI_B0_PREPARATION_20261004"
def update_doc(name,body,archive=False):
    path=ROOT/"docs"/name;old=path.read_text(encoding="utf-8")
    begin=f"<!-- {MARKER}:BEGIN -->";end=f"<!-- {MARKER}:END -->"
    if begin in old:
        prefix,tail=old.split(begin,1);_,suffix=tail.split(end,1)
        path.write_text(prefix+begin+"\n"+body.strip()+"\n"+end+suffix,encoding="utf-8");return
    title,rest=old.split("\n",1)
    if archive:
        rest="\n<details>\n<summary>Historical pre-adoption proposal; superseded by the approved current policy above</summary>\n\n"+old+"\n</details>\n"
        title=title.replace(" proposal","").replace("Proposal","Adopted policy")
    else:rest="\nEarlier dated sections below are preserved historical states; the current approved policy above takes precedence.\n"+rest
    path.write_text(title+"\n\n"+begin+"\n"+body.strip()+"\n"+end+"\n"+rest,encoding="utf-8")
def append_event(path,event):
    existing=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(x["id"]==event["id"] for x in existing):return
    with path.open("a",encoding="utf-8",newline="\n") as f:f.write(json.dumps(event,ensure_ascii=False)+"\n")
def main():
    mp=ROOT/"review/b0_replenishment_manifest.json";m=load(mp);audit=load(ROOT/m["audit_path"])
    checked=analysis.inspect();assert not checked["errors"],checked["errors"]
    assert checked["state_counts"]=={"pending":1219}
    assert all(t["unchanged_prepared_bytes"] for t in checked["transport"].values())
    for p,s in audit["historical_evidence_snapshot"].items():
        assert file_sha(ROOT/p)==s["sha256"] and (ROOT/p).stat().st_mtime_ns==s["mtime_ns"],p
    now=datetime.now(timezone.utc).isoformat()
    common="""## Current approved status — 2026-10-04

The human adopted **h2-mini-benign-taxonomy-v2**, the two-arm B0/BH design and **B-short Option B**. Authorization for this execution is **B0 review-package preparation only**. No human judgments were produced.

B0 means human-reviewed ordinary benign OASST1 prompts. BH means human-reviewed OR-Bench hard-benign prompts under the accepted source-defined policy. Historical B1/B2/B3 proposed/reviewer strata are descriptive metadata only, never mandatory final labels or a difficulty ranking. This amendment is transparently **post-review and pre-H2-mini-training**. All original rubric-v1 decisions, strata, notes, confidence, timing and calibration/production results remain unchanged.

The revised operational question is:

> Under sequential attack-family adaptation, can the detector retain correct acceptance of independently reviewed hard-benign inputs compared with an ordinary-benign matched control?

The [frozen adoption policy](../review/h2_mini_benign_taxonomy_v2.json) records the human request hash. The [review manifest](../review/b0_replenishment_manifest.json) binds that policy, source pool, candidate/assignment metadata, source hashes and prepared worksheet hashes. Current readiness is **R0 active; R1/R2 not entered**. Existing numerical replay-entry criteria are unchanged. No final B0/BH/B-short IDs or final splits, research JSONL, training, replay or numerical memory budget has been selected.
"""
    arms="""## Adopted two-arm structure

| Stage | H2m-C0 | H2m-F |
| --- | --- | --- |
| 1 | AF1 Naive + B0-1 | AF1 Naive + exact same B0-1 |
| 2 | AF2 Ignore + B0-2 | AF2 Ignore + BH-1 |
| 3 | AF3 Escape-Separation + B0-3 | AF3 Escape-Separation + BH-2 |
| 4 | AF4 Fake Completion + B0-4 | AF4 Fake Completion + BH-3 |

Each future stage requires benign200 train/30 validation/60 test, matched with the same malicious counts. C0 requires **1160 distinct B0 =800/120/240** across four independent stages; F reuses the exact C0 Stage-1 anchor and creates no extra B0 requirement. Malicious IDs, splits, positions and all training/evaluation controls will be shared exactly across arms after separate approval.

BH-1/2/3 are disjoint, composition-balanced partitions. Their suffix is an index, not increasing benign difficulty, easy/medium/hard strata or monotonic benign evolution. Source/language/format differences remain potential confounds. Each future training stage requires at least20 independently audited accepted benign records.

The accepted BH capacity remains **999**, comprising691 training-lane records (68 independent training second reviews included),104 validation reserves and204 test reserves. Later BH needs600/90/180=870. The previous [capacity audit](../results/h2_mini_bh_capacity.json) and [anonymous count witness](../results/h2_mini_bh_partition_feasibility.json) are unchanged proposal-level artifacts; no final record-to-partition mapping was created. No further BH/B1/B2/B3 review is requested.
"""
    short="""## B-short Option B adopted

No new B-short review. Later choose120 from the already-reviewed eligible pool144 (42 English,102 multilingual), targeting42 English +78 multilingual. **No final IDs have been selected.** The full original150-record candidate reserve remains excluded from B0 replenishment.

B-short is an **unmatched descriptive external control**: the planned English fraction is35%, compared with255/339=75.22% in frozen NotInject. Aggregate NotInject minus B-short FPR is descriptive and is **not a pure trigger effect**. Report English and multilingual FPR separately, with per-language counts where feasible; source/length/format differences remain even within language. External banks remain excluded from training, validation, hyperparameter/threshold/epoch selection, replay selection and method ranking. No new review is authorized for Option A.
"""
    table=["| Reviewer | Primary | Independent second | Double review | Total | Batch sizes |","| --- | ---: | ---: | ---: | ---: | --- |"]
    for who in ["A","B"]:
        counts=Counter(a["review_role"] for r in m["records"] for a in r["reviewer_assignments"] if a["reviewer"]==who)
        sizes=[str(b["rows"]) for b in m["batches"].values() if b["reviewer"]==who]
        table.append(f"| {who} | {counts['PRIMARY']} | {counts['SECOND_REVIEW']} | {counts['DOUBLE_REVIEW']} | {sum(counts.values())} | {', '.join(sizes)} |")
    batches="\n".join(table)
    paths="\n".join(f"- [review/b0_replenishment/{fn}](../review/b0_replenishment/{fn}) — {b['rows']} judgments." for fn,b in m["batches"].items())
    tokens=sorted(r["deberta_tokens"] for r in m["records"])
    package=f"""## Prepared review package

**872 unique candidates;1219 planned judgments;0 human judgments completed by the agent.** These are review candidates, not accepted records or final research IDs.

{batches}

All600 training candidates receive a primary review. The75 second reviews are assigned to the other reviewer independently (A37/B38). All272 evaluation-reserve candidates receive both independent reviews, producing544 judgments. No primary-only record may fill evaluation quotas.

Each reviewer has seven local UTF-8-BOM CSV files; the first batch has100 rows, and the rest84-85. Both first batches together expose35 independently paired records (9 training audits and26 evaluation doubles) once both humans finish. The manifest guarantees a minimum32 paired records (8/24); random completion of the remaining first-batch slots produced three additional pairs. This supports the optional approximately200-judgment interim check without selecting by answers.

{paths}

The exact worksheet columns are review_id, text, source, language, token_length, review_role, reviewer_decision, b0_confirmed, reason_code, confidence, optional_note and review_seconds. Roles are PRIMARY, SECOND_REVIEW and DOUBLE_REVIEW. No ordinal-stratum field is present. All six human fields are initially empty.
Source is pinned OpenAssistant/oasst1 revision **{OA_REV}**, from the existing9134-record audited pool. Excluding520 previously reviewed/calibration OASST identities and prohibited groups left8614 metadata-eligible candidates. A deterministic language/length sample of1000 underwent the full pre-review audit; all1000 passed, and872 were selected without detector scores. The selected language counts are {json.dumps(audit["selected_language_counts"],sort_keys=True)}. Token lengths are min{tokens[0]}, median{statistics.median(tokens):g}, p95{tokens[int(.95*(len(tokens)-1))]}, max{tokens[-1]} (all<=1280). A long tail exists; human ordinary-B0 eligibility and later length balance are still required.

The [pre-review audit](../data/audits/hard_benign/b0_replenishment_pre_review_audit.json) checked source status, native/message/tree IDs, exact/normalized/model-view hashes, source-scoped project semantic/template groups and exhaustive lexical/latent/template comparisons. Coverage includes all1590 prior production candidates plus50 calibration records (accepted/rejected/disputed, current B0339, B-short reserve150 and BH999), NotInject339 and all14640 AF1-AF4 records. These overlapping subset counts are not additive. No selected candidate had a detected prohibited overlap. Approximate grouping is not proof that all paraphrase siblings are absent.

All92 protected evidence files retained their SHA256 and modification timestamps, including rubric-v1 review evidence and both previous BH audit artifacts. Raw review CSVs are local and ignored under review/b0_replenishment/. Metadata-only manifests contain no raw text or free-form reviewer notes.
"""
    commands=r"""## Human work and later analysis

Read the [Chinese review guide](../review/B0补充人工审核说明.md). Each reviewer fills only their own seven CSVs independently. Keep all immutable source/identity fields unchanged. Only ACCEPT_BENIGN plus B0 confirmation YES qualifies; all assigned independent judgments must agree on eligibility. Rejection/ambiguity, ACCEPT with NO/UNSURE and missing second judgments are not silently accepted.

~~~powershell
Set-Location E:\ContinualGuard
# Initial blank-package check only, before reviewers edit:
& .\.venv\Scripts\python.exe .\scripts\analyze_h2_mini_b0_replenishment.py --check-package
# Optional after approximately200 combined completed judgments:
& .\.venv\Scripts\python.exe .\scripts\analyze_h2_mini_b0_replenishment.py --interim
# After all1219 assigned judgments are completed:
& .\.venv\Scripts\python.exe .\scripts\analyze_h2_mini_b0_replenishment.py
~~~

The analysis script checks exact headers, UTF-8/CSV transport, frozen manifest dependencies, row identities, source hashes, immutable metadata, complete assignments, allowed human values and positive finite timing. Human answer edits change file hashes as expected; they do not authorize changing source text. Each run records the actual input hashes and produces metadata-only reports under results, without changing worksheets or inferring answers.

Later reports include completed/missing judgments, acceptance/rejection reasons, primary acceptance, separate decision/B0-confirmation agreement and kappa for training second and evaluation double reviews, conflict counts, reviewer timing, eligible B0 counts and joint quota shortages. More than10% completed training-audit disagreement on either decision or B0 confirmation requires human review; interim small-sample rates are provisional. Integrity errors never pass. Neither interim nor full analysis changes the rubric, selects final IDs or assigns splits.

Current accepted B0 remains339, with234 training (20 audited) and105 double-reviewed evaluation reserves. New candidates do not change that number. The remaining accepted-record deficit is821. Final targets are800 train,120 validation,240 test and at least80 independently audited accepted training records, with20 available for each future stage.

The oversample is not a guarantee. If every candidate eventually passes both human review and subsequent grouping, capacity would be1211 total,834 training (95 audited),377 double-reviewed evaluation; this is **a hypothetical upper bound, not an observed result**. Shared evaluation capacity must jointly cover360 records; the same records cannot fill both120 validation and240 test. An optional aggregate lane-reallocation feasibility check is hypothetical only and does not authorize moving or assigning records. Post-review group/contamination audit and human approval remain required.

Projected work is **4.49324 person-hours**, using the observed prior B0 mean13.2696 seconds/judgment (A2.24478/B2.24846). Multilingual/long-tail content, coordination, fatigue, uncertainty and any later top-up can increase this. No automatic top-up is approved.
"""
    fixed="""## Scientific and execution limits

All active operations use E:\\ContinualGuard and E:\\ContinualGuard\\.venv\\Scripts\\python.exe. The archival project was not accessed or modified. No storage audit, size gate or storage cleanup ran.

Future comparisons retain microsoft/deberta-v3-base, pinned model revision/initialization, fixed benign0/IPI1 labels, max_length1280, identical stage data/order, seeds17/42/2026, epochs, learning rate, optimizer resets and evaluation protocol. Any eventual method comparison also fixes a separately approved replay memory budget. No numerical budget was chosen here.

This package supplies no acquisition, retention, forgetting or replay-effectiveness result. R0 remains active; R1/R2 have not been entered. Reservoir ER, DER++, LARS/BRS, MIR, GSS, GEM, UPGD, security-aware replay and selective adaptation remain blocked. Stop for the humans to perform the new B0 review.
"""
    update_doc("h2_mini_taxonomy_amendment.md",common+arms+short+fixed,True)
    update_doc("h2_mini_simplified_design.md",common+arms+short+fixed,True)
    update_doc("h2_mini_b0_replenishment_plan.md",common+package+commands+fixed,True)
    update_doc("b_short_resolution_options.md",common+short+fixed,True)
    update_doc("h2_mini_design.md",common+arms+short+fixed)
    risk="""## Current risks and obligations

- Review preparation is not accepted capacity. B0 remains339/1160 until human answers and a new post-review audit pass; the872-candidate package may still leave shortages.
- The v2 amendment follows observed ordinal-label failure; preserve that history and do not retrospectively claim a predeclared ordinal design. BH is source-defined, not calibrated detector hardness or monotonic difficulty.
- BH999 is sufficient proposal-level capacity, with691 training (68 audited) and308 double-evaluation records. Anonymous counts do not establish final composition-balanced IDs, per-stage independent audit coverage or absence of all missing-lineage paraphrases.
- B-short Option B is adopted but language-unmatched (planned35% English versus75.22% in NotInject). Aggregate FPR gaps are descriptive, not a pure trigger effect. Stratification does not remove all source, length or format confounds.
- B0 candidates span18 languages and token lengths7-1149 (median25,p95103). Humans need adequate language competence; automated source/quality screening does not establish benignness, ordinary-B0 membership or comparable difficulty. Record unknown-language issues without guessing.
- The deterministic language/length-balanced training audit is prospective. First200 judgments provide a small provisional quality check, not global reliability or permission to change labels/rubric. Primary-only training and double-reviewed evaluation remain distinct.
- Lexical/SVD semantic groups are approximate, and official OR rewrite-parent lineage is unavailable. Native/tree/hash/project groups and external contamination checks stay mandatory; do not relax thresholds to meet quotas.
- Raw review text and free-form notes stay local/ignored. The canonical root currently has no .git directory, so ignore patterns were checked as configuration; no Git commit or publication was performed.
- Original OASST1/OR-Bench provenance and license/rights caveats remain those in the accepted source audit. This task neither changed sources nor expands redistribution rights.
"""
    update_doc("assumptions_and_risks.md",common+risk+fixed)
    update_doc("replay_readiness_review.md",common+arms+short+fixed)
    questions="""## Current human decisions and next action

| Item | Status |
| --- | --- |
| Q050 taxonomy amendment | CLOSED: adopted h2-mini-benign-taxonomy-v2. |
| Q051 targeted B0 preparation | CLOSED:872 candidates/1219 blank judgments prepared and validated. Human judgment work remains. |
| Q052 B-short option | CLOSED: Option B, later120 at42 English/78 multilingual; no new review, no final IDs. |
| Q054 operational research question | CLOSED: revised nonordinal ordinary-control versus hard-benign retention question adopted. |
| Q053 final assignments | OPEN: no final B0/BH/B-short IDs or train/validation/test assignments. Complete B0 review and post-review audit first. |
| B0 capacity after review | OPEN: determine accepted yield, independent training audits, joint800/120/240 coverage and any shortage; no automatic top-up. |
| Final materialization and H2 training | OPEN: separate human approval after review/audit and proposed assignment review. |
| Q044 numerical replay memory budget | DEFERRED: R0; no replay implementation or budget selection. |

**Exact next human action:** A and B independently fill only their own seven CSVs using the Chinese guide. Start with each batch01 (100 judgments each) for the optional combined200-judgment interim check; then complete the remaining assigned batches. Do not expose another reviewer's answers.

After both batch01 files are complete, the optional next prompt is:

> The two human reviewers have independently completed reviewer_A_batch_01.csv and reviewer_B_batch_01.csv in E:\\ContinualGuard\\review\\b0_replenishment. Run the B0 interim integrity, acceptance, decision/B0-confirmation disagreement and timing check only. Preserve all human answers and the frozen v2 rubric. Do not select final IDs, assign splits, materialize research JSONL, train or implement replay.

After all14 files are complete:

> The two human reviewers have completed all1219 B0 replenishment judgments. Analyze completeness, integrity, acceptance, independent-review agreement and timing; report accepted B0 capacity, training-audit coverage and joint800/120/240 shortages under h2-mini-benign-taxonomy-v2. Preserve original evidence and stop for human review before final IDs, splits, materialization, training or replay.
"""
    update_doc("open_questions.md",common+questions)
    decision={"date":now,"id":"D034","decision":"Adopt h2-mini-benign-taxonomy-v2, nonordinal two-arm B0/BH structure and B-short Option B; prepare targeted B0 review only",
      "authorization":"Explicit human approval in request bound by review/h2_mini_benign_taxonomy_v2.json",
      "approved_request_sha256":load(ROOT/m["taxonomy_policy_path"])["approved_request_sha256"],
      "approved_taxonomy":POLICY,"B1_B2_B3":"Historical/descriptive only, never mandatory final labels or ordinal difficulty",
      "BH_capacity_accepted":999,"BH_final_assignment":False,
      "B_short":{"option":"B","future_total":120,"English":42,"multilingual":78,"unmatched_descriptive_only":True,"new_review":False,"final_IDs":False},
      "B0_review":{"new_candidates":872,"primary":600,"independent_second":75,"double_candidates":272,"double_judgments":544,"total_judgments":1219,"A":609,"B":610},
      "status":"Package prepared; human judgments not started by agent; stop for human review work",
      "historical_evidence_unchanged":True,"final_splits":False,"materialization":False,"training":False,"replay":False,"replay_budget":None,"replay_readiness":"R0"}
    files=["scripts/h2_mini_b0_review_common.py","scripts/prepare_h2_mini_b0_replenishment.py","scripts/analyze_h2_mini_b0_replenishment.py",
      "scripts/report_h2_mini_b0_preparation.py","tests/test_h2_mini_b0_replenishment.py",".gitignore",
      "review/B0补充人工审核说明.md","review/h2_mini_benign_taxonomy_v2.json","review/b0_replenishment_manifest.json",
      "review/b0_replenishment/reviewer_A_batch_01.csv through07.csv (local ignored)",
      "review/b0_replenishment/reviewer_B_batch_01.csv through07.csv (local ignored)",
      "data/audits/hard_benign/b0_replenishment_pre_review_audit.json","results/h2_mini_b0_preparation_validation.json",
      *["docs/"+n for n in ["h2_mini_taxonomy_amendment.md","h2_mini_simplified_design.md","h2_mini_b0_replenishment_plan.md","b_short_resolution_options.md",
        "h2_mini_design.md","assumptions_and_risks.md","replay_readiness_review.md","open_questions.md","decision_log.md","change_log.md"]],
      "history/decisions.jsonl","history/changes.jsonl"]
    change={"date":now,"id":"C021","reason":"Human-approved targeted B0 package preparation after taxonomy simplification adoption",
      "files_changed":files,"effect":"Pinned-source/group-safe review candidates, independent blinded assignments, Chinese guide, metadata manifest and future analysis script",
      "tests_run":["12 synthetic review-integrity/coverage tests passed","Prepared blank package validation passed:872 candidates/1219 rows/14 batches/0 human values",
        "All1000 prospectively sampled audit candidates passed fixed contamination/group checks;872 prepared","92 protected evidence files retained identical SHA256 and mtime"],
      "project_root":str(ROOT),"python":sys.executable,"scientific_behavior_changed":"Adopted nonordinal pre-training protocol; no model experiment",
      "limitations":["B0 remains339 accepted, short821; new candidate yield unknown","BH capacity accepted but final partitions not assigned",
        "B-short unmatched by language; descriptive-only gap","Approximate semantic grouping and source rights caveats unchanged","Raw CSVs ignored; canonical root has no Git metadata; no commit"],
      "replay_readiness":"R0","human_judgments_performed":False,"training":False,"replay":False}
    update_doc("decision_log.md","## D034 — Adopted taxonomy and B0 preparation\n\n~~~json\n"+json.dumps(decision,ensure_ascii=False,indent=2)+"\n~~~")
    update_doc("change_log.md","## C021 — B0 reviewer package and analysis\n\n~~~json\n"+json.dumps(change,ensure_ascii=False,indent=2)+"\n~~~")
    append_event(ROOT/"history/decisions.jsonl",decision);append_event(ROOT/"history/changes.jsonl",change)
    for p,s in audit["historical_evidence_snapshot"].items():
        assert file_sha(ROOT/p)==s["sha256"] and (ROOT/p).stat().st_mtime_ns==s["mtime_ns"],p
    validation={"status":"PREPARATION_VALIDATED_HUMAN_REVIEW_NOT_STARTED","at":now,"project_root":str(ROOT),"python":sys.executable,
      "manifest_sha256":file_sha(mp),"policy_sha256":m["taxonomy_policy_sha256"],"candidates":872,"judgment_rows":1219,
      "human_fields_all_blank":True,"reviewer_judgments":m["reviewer_judgments"],"batch_count":14,"worksheet_sha256":checked["hashes"],
      "historical_protected_files":len(audit["historical_evidence_snapshot"]),"historical_hashes_and_mtimes_unchanged":True,
      "selected_overlap_findings":audit["selected_overlap_findings"],"audit_runtime_seconds":audit["runtime_seconds"],
      "synthetic_tests_passed":12,"synthetic_test_command":".venv/Scripts/python.exe -m unittest discover -s tests -p test_h2_mini_b0_replenishment.py -v",
      "source_text_modified":False,"raw_worksheets_ignore_rule":"review/b0_replenishment/","git_metadata_present":(ROOT/".git").exists(),
      "archival_copy_accessed":False,"storage_audit_or_cleanup":False,"BH_artifacts_frozen":True,"final_IDs_selected":False,
      "final_splits_assigned":False,"research_JSONL_materialized":False,"training":False,"replay":False,"replay_readiness":"R0"}
    save(ROOT/"results/h2_mini_b0_preparation_validation.json",validation)
    print(json.dumps({"status":validation["status"],"decision":"D034","change":"C021","historical_files_unchanged":92,
      "manifest":rel(mp),"validation":"results/h2_mini_b0_preparation_validation.json","docs_updated":10},indent=2))
if __name__=="__main__":main()
