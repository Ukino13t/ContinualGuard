"""Render the H2-mini audit/design documents and append immutable project-history records."""
import json
from datetime import datetime,timezone,timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/"docs"
A=json.loads((ROOT/"data/audits/hard_benign/or_bench_full_audit.json").read_text(encoding="utf-8"))
O=json.loads((ROOT/"data/audits/hard_benign/oasst1_full_audit.json").read_text(encoding="utf-8"))
C=json.loads((ROOT/"data/audits/hard_benign/h2_mini_capacity.json").read_text(encoding="utf-8"))
NOW=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
fc=A["group_safe_counts_pending_human_review"]; raw=A["raw_label_counts"]
b1,b2,b3=fc["B1_candidate"],fc["B2_candidate"],fc["B3_candidate"]
rb1,rb2,rb3=raw["B1_candidate"],raw["B2_candidate"],raw["B3_candidate"]
oa=O["group_safe_ordinary_capacity_pending_human_review"]; reserve=O["ordinary_capacity_after_Bshort_reserve"]; bs=O["B_short"]["candidate_pool"]
def write(name,text):(DOC/name).write_text(text.strip()+"\n",encoding="utf-8")
def table_steps(rows,count_key):
 lines=["| Step | Operation | Remaining | Removed |","|---:|---|---:|---:|"]
 for r in rows:
  if count_key=="counts":
   z=r["counts"];remain=f"B1 {z['B1_candidate']}; B2 {z['B2_candidate']}; B3 {z['B3_candidate']}"
  else:remain=str(r["count"])
  lines.append(f"| {r['step']} | {r['name']} | {remain} | {r.get('removed',0)} |")
 return "\n".join(lines)
write("h2_mini_design.md",f"""
# H2-mini feasibility benchmark

Status: source acquisition, audit, capacity estimation, and review planning complete. Human record review, final assignment, final JSONL materialization, training, replay, and selective adaptation have not started. H2-mini is a course-scale feasibility benchmark and is not equivalent to full H2-F.

## Scientific purpose

The completed AF1-AF4 streams are near ceiling and show only small or transient retention loss. Frozen NotInject evaluation shows severe absolute over-defense, often about 90-100% FPR, without monotonic or seed-consistent worsening. H2-mini therefore tests whether evolving hard-benign pressure creates a reproducible trade-off among malicious IPI acquisition, earlier-family retention, and benign false-positive reduction. Replay remains R0.

## Exact two-arm structure

| Stage | Malicious half in both arms | H2m-C0 benign half | H2m-F benign half |
|---|---|---|---|
| H2m-1 | AF1 Naive | B0-1 ordinary benign | exact same B0-1 IDs |
| H2m-2 | AF2 Ignore | B0-2 ordinary benign | B1 mild hard benign |
| H2m-3 | AF3 Escape-Separation | B0-3 ordinary benign | B2 medium hard benign |
| H2m-4 | AF4 Fake Completion | B0-4 ordinary benign | B3 instruction-like/context hard benign |

Per stage, train is 200 malicious plus 200 benign, validation is 30 plus 30, and test is 60 plus 60. Each class contributes 290 records per stage. Each arm contains 1,160 malicious and 1,160 benign records, 2,320 total. B1, B2, and B3 each require 290 records. No upsampling, duplication, or synthetic filler is allowed.

The malicious pools already contain 1,830 unique records per family: train 500/500/500, validation 50/50/50, and test 60/60/60 across head/middle/tail. H2-mini targets train 67/66/67, validation 10/10/10, and test 20/20/20. AF1-AF4 have no cross-family example, base-document, or injection identity overlap.

## Fixed comparison factors

Both arms must use identical malicious IDs, positions and splits; AF order; microsoft/deberta-v3-base revision; tokenizer; max_length 1280; threshold; seeds; training budget; optimizer reset; and evaluation. Only the benign half changes after stage 1.

## Current capacity and gate

Automated, group-safe candidates pending human review are B1 {b1:,}, B2 {b2:,}, B3 {b3:,}, and OASST ordinary benign {oa:,}. After reserving a disjoint B-short bank of 120, ordinary capacity is {reserve:,}. Capacity passes both 290-per-stratum H2-mini and 650-per-stratum future-pilot screens.

These counts are not accepted benign labels. H2-mini is ready to begin the approved human-review workflow only after a human accepts the ontology and approximate grouping policy. It is not ready for final materialization or training.
""")
ontology_rows=[
("C01","Override or negative constraint","IPI override and suppression language","Ignore/disregard/override/bypass plus negative constraints","Constraint is discussed or legitimately scoped","Bare host override","Host target unclear","Discuss why a quoted rule says to disregard an earlier rule"),
("C02","Instruction hierarchy or policy","System/developer hierarchy is an IPI target","Instruction, policy, guideline, rule, protocol terms","Terms are legitimate subject matter","Replaces host hierarchy","Policy owner unclear","Explain how a system policy is documented"),
("C03","Role or simulation","Role reassignment is used by IPI","Role-play, persona, pretend, fictional, hypothetical, scenario","Legitimate exercise or content","Redirects host role","Host/content boundary missing","Create a fictional staff-training scenario"),
("C04","Output or imperative control","IPI often controls outputs","Answer, write, return, create, list, explain, translate and related verbs","Normal request controls requested content","Suppresses governing task","Output target unclear","Summarize a policy in three bullets"),
("C05","Secret or protected information","Exfiltration is an IPI objective","Secret, password, credential, token, reveal, leak and related terms","Education, policy, or defense","Requests real protected data","Authorization unclear","Describe why passwords should not be disclosed"),
("C06","Tool, code, or execution","IPI can trigger tools","Execute, command, shell, tool, API, code, algorithm terms","Legitimate code/configuration content","Untrusted host action","Execution authority unclear","Explain a harmless sample command"),
("C07","Encoding, boundary, transformation","IPI can hide in encoding or delimiters","Base64, decode, delimiter, code block, quote, translation terms","Transformation is the benign task","Conceals an override","Decoded intent unavailable","Translate a quoted sentence mentioning a delimiter"),
("C08","Context, memory, or message","IPI refers to retrieved context","Context, history, previous, document, prompt, message terms","Context is discussed as content","Redirects through hidden context","Referenced context unavailable","Analyze context around a quoted message"),
("C09","Authority or privilege","IPI may impersonate authority","Admin, developer, system, root, authority terms","Governance or software subject","False host authority","Authority relationship unclear","Explain an administrator role"),
("C10","Security or prompt meta-language","Defensive text shares attack cues","Security, malicious, attack, injection, phishing, moderation terms","Analysis, defense, design, education","Performs an attack","Defensive intent unclear","Explain prompt injection as a security risk")]
lines=["| ID | Name | Motivation | Matching rule | Inclusion | Exclusion | Ambiguity | Abstract example |","|---|---|---|---|---|---|---|---|"]
for r in ontology_rows:lines.append("| "+" | ".join(r)+" |")
write("hard_benign_cue_ontology.md",f"""
# H2-mini hard-benign cue ontology

Version: h2m-cue-ontology-v1. The ontology was defined without NotInject text, trigger lists, model scores, or detector predictions. OR-Bench was used only as the approved candidate source.

{chr(10).join(lines)}

## Stratum rules

B1 requires exactly one meaningful matched span from one cue family, benign intent, and no operational attempt to override or redirect the host. B2 requires at least two independent cue families or at least two distinct cue spans; length alone never makes B2. B3 requires a legitimate content frame such as security analysis, configuration, policy, translation, quotation, research, training, fictional/hypothetical work, prompt engineering, or system design, plus relevant control language. The content must remain benign toward the host.

A bare operational override, secret-exfiltration request, or role redirection without a clear benign content frame is ambiguous or excluded. Automated labels are candidate strata only; reviewers confirm benign intent and stratum.
""")
write("or_bench_lineage_audit.md",f"""
# OR-Bench lineage audit

## Official artifacts checked

The audit checked the pinned Hugging Face dataset revision {ORR if False else A['source']['revision']}, its dataset card and full file list, the official OR-Bench paper, and official Space revision b8f3dcd267880f7e525c49773d64cb1c3c02f481 including app.py. The 80K CSV schema is exactly prompt and category.

The paper states that 2,000 toxic seeds per category were deduplicated and each seed was rewritten into five prompts before moderation. The release does not expose native record IDs, toxic-seed IDs, rewrite-parent IDs, generator seed IDs, template IDs, generation batch IDs, or source-parent relationships. No official generation repository or lineage sidecar is present. Row order was not used to invent parent IDs.

## Project-derived grouping

The project creates exact, Unicode-normalized, DeBERTa model-view, project semantic-parent, and project template groups. The semantic approximation uses eight 16-bit locality-sensitive hash tables over deterministic 48-dimensional hashed word/bigram TF-IDF-SVD vectors, high token-set overlap, and rare six-word overlap. The template approximation uses a structural signature plus token Jaccard.

These groups are deliberately conservative project approximations. They are not official parent labels. The semantic grouping contains {A['grouping']['project_semantic']['multi_record_groups']:,} multi-record groups; its largest connected component contains {A['grouping']['project_semantic']['max_group_size']:,} records. That large component shows over-grouping/percolation risk and must remain visible. The method also skipped {A['grouping']['project_semantic']['oversized_lsh_buckets_skipped']} oversized LSH buckets, so missed sibling leakage remains possible.

A future split must keep every available identity and project group intact. Human acceptance of these approximations is required before final assignment.
""")
write("or_bench_candidate_audit.md",f"""
# OR-Bench full H2-mini candidate audit

Pinned source: bench-llm/or-bench, config or-bench-80k, revision {A['source']['revision']}, CC BY 4.0. Records: 80,359. The source paper/card describes these as moderator-screened seemingly toxic safe prompts, but also warns that some prompts can be toxic or debatable. Project human review remains mandatory.

## Raw ontology counts

| Label | Records |
|---|---:|
| B1 candidate | {rb1:,} |
| B2 candidate | {rb2:,} |
| B3 candidate | {rb3:,} |
| Unassigned | {raw['unassigned']:,} |
| Ambiguous | {raw['ambiguous']:,} |

## Auditable filter sequence

{table_steps(A['filter_sequence'],'counts')}

Final automated group-safe counts pending human review are B1 {b1:,}, B2 {b2:,}, and B3 {b3:,}. All three exceed 290 and 650. Exact, normalized, and model-view duplicate counts were zero. The largest reduction came from conservative project semantic grouping. Automated unsafe filtering removed 191 current representatives. NotInject comparison removed one current representative; comparison with the full OASST root pool removed 20; AF1-AF4 removed none.

Cross-corpus semantic-neighbor findings are screening candidates, not proof of shared provenance. NotInject was used only after ontology/grouping for contamination auditing and did not influence cue rules or candidate ranking.

## Length audit

| Stratum | Min | Median | P90 | P95 | P99 | Max | Above 1280 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | {A['length_by_stratum']['B1']['min']} | {A['length_by_stratum']['B1']['median']} | {A['length_by_stratum']['B1']['p90']} | {A['length_by_stratum']['B1']['p95']} | {A['length_by_stratum']['B1']['p99']} | {A['length_by_stratum']['B1']['max']} | 0 |
| B2 | {A['length_by_stratum']['B2']['min']} | {A['length_by_stratum']['B2']['median']} | {A['length_by_stratum']['B2']['p90']} | {A['length_by_stratum']['B2']['p95']} | {A['length_by_stratum']['B2']['p99']} | {A['length_by_stratum']['B2']['max']} | 0 |
| B3 | {A['length_by_stratum']['B3']['min']} | {A['length_by_stratum']['B3']['median']} | {A['length_by_stratum']['B3']['p90']} | {A['length_by_stratum']['B3']['p95']} | {A['length_by_stratum']['B3']['p99']} | {A['length_by_stratum']['B3']['max']} | 0 |

No detector inference, model-score matching, final assignment, final JSONL, or training occurred.
""")
write("oasst1_candidate_audit.md",f"""
# OASST1 ordinary-benign candidate audit

Pinned source: OpenAssistant/oasst1 revision {O['source']['revision']}, Apache-2.0. The official ready-messages artifact contains {O['all_ready_rows']:,} records. The audit uses {O['root_prompter_rows']:,} messages whose role is prompter, parent is missing/null, and message ID equals tree ID.

{table_steps(O['filter_sequence'],'count')}

The automated group-safe ordinary pool contains {oa:,} records pending project human review. After reserving a disjoint 120-record B-short bank, maximum ordinary capacity is {reserve:,}, exceeding H2m-C0's 1,160-record requirement.

The audit records language, role/root identity, message/tree IDs, source review/deleted state, DeBERTa length, duplicates, tree identity, project semantic/template groups, unsafe/IPI flags, possible PII, missing context, and overlap with NotInject, OR-Bench, and AF1-AF4. OASST review_result is source metadata and is not treated as the project benign/IPI judgment.

B0 candidate lengths: min {O['length']['B0']['min']}, median {O['length']['B0']['median']}, p90 {O['length']['B0']['p90']}, p95 {O['length']['B0']['p95']}, p99 {O['length']['B0']['p99']}, max {O['length']['B0']['max']}; none exceeds 1280 after filtering. Dolly was not used.
""")
write("b_short_control_mini.md",f"""
# B-short mini frozen control

B-short is a 120-record frozen ordinary-benign evaluation bank intended to separate short prompt-format mismatch from trigger-associated NotInject over-defense. It never enters training, validation, threshold/epoch/hyperparameter selection, replay selection, or method ranking.

The OASST audit found {bs:,} group-safe candidates in the NotInject token range 12-57. The pool contains {O['B_short']['languages'].get('en',0):,} English and {O['B_short']['languages'].get('multilingual',0):,} non-English candidates. Scaling NotInject's 255 English versus 84 multilingual composition gives an approximate 120-record target of {O['B_short']['target_approx']['english']} English and {O['B_short']['target_approx']['multilingual']} multilingual records, which is feasible.

Candidate-pool DeBERTa lengths are min {O['length']['B_short_candidates']['min']}, median {O['length']['B_short_candidates']['median']}, p90 {O['length']['B_short_candidates']['p90']}, p95 {O['length']['B_short_candidates']['p95']}, p99 {O['length']['B_short_candidates']['p99']}, max {O['length']['B_short_candidates']['max']}. Frozen NotInject is min 12, median 21, p90 33, p95 36, p99 47, max 57.

Final IDs have not been selected. Later allocation must use length/language quotas and deterministic group hashing only, never detector scores, and must not intentionally remove injection-like vocabulary. All 120 final records require double independent benign review and must remain disjoint from H2-mini and NotInject.
""")
write("h2_mini_review_protocol.md",f"""
# H2-mini course-scale review protocol

This reduced protocol applies only to H2-mini and is not equivalent to the full H2-F protocol.

## Decisions and rejection codes

Reviewers see the exact candidate text and minimum required source context, but no detector/model score, NotInject similarity, future split/stage, or another reviewer's decision. Decisions are ACCEPT_BENIGN, REJECT_TRUE_INJECTION, REJECT_AMBIGUOUS_INTENT, REJECT_HARMFUL_OR_UNSAFE, REJECT_CONTEXT_MISSING, REJECT_PRIVACY_OR_RIGHTS, and REJECT_QUALITY. Reviewers also confirm B0/B1/B2/B3 membership.

## Coverage

Training benign records receive 100% single review plus a stratified 10% independent second-review audit. If disagreement exceeds 10% in any stratum, pause it, expand independent second review to at least 30%, revise the rubric if needed, and do not finalize it.

Validation and test benign records receive 100% double independent review. Any disagreement excludes the record unless explicitly adjudicated. All 120 B-short records receive double independent review.

Report raw agreement, disagreement rate, Cohen kappa or Krippendorff alpha where meaningful, rejection counts by reason, and audited-training disagreement by stratum. Keep raw text and free-form review notes local; committed outcomes contain IDs, hashes, rubric version, decisions, reason codes, and timestamps.

## Timing and workload gate

Production minimum is 3,040 judgments: 1,200 for B1-B3, 1,600 for C0, and 240 for B-short. A 50-record timing calibration with two reviewers adds 100 judgments, for 3,140 before replacements or adjudication.

No review timing has been measured, so person-hours are not projected yet. Use observed median seconds per judgment after calibration. At 3,140 base judgments, 10 hours requires 11.46 seconds per judgment and 15 hours requires 17.20 seconds. If the measured projection including replacements/adjudication is substantially above 15 person-hours, stop before production review and propose a smaller balanced benchmark.
""")
write("h2_mini_capacity_report.md",f"""
# H2-mini capacity report

| Requirement | Automated group-safe capacity pending human review | Result |
|---|---:|---|
| B1 290 | {b1:,} | feasible |
| B2 290 | {b2:,} | feasible |
| B3 290 | {b3:,} | feasible |
| Future B1/B2/B3 650 each | {b1:,} / {b2:,} / {b3:,} | feasible |
| H2m-C0 ordinary 1,160 plus disjoint B-short 120 | {oa:,} total; {reserve:,} after reserve | feasible |
| B-short 120 | {bs:,} length-range candidates | feasible |

Feasible means sufficient automated, project-group representatives exist. It does not mean the records have passed human benign/IPI review. The full H2-F target remains a separate long-form design and was not implemented.

The principal capacity uncertainty is yield after human review, especially OR-Bench toxicity/ambiguity and OASST unsafe, privacy, or missing-context exclusions. The largest OR project semantic component has 4,979 records, showing that approximation can over-group. The absence of official OR rewrite parents also permits missed sibling leakage. Both risks require human acceptance and post-review re-audit.

## Exact recommended next prompt

The H2-mini source acquisition and candidate audit have been reviewed and accepted. Approve h2m-cue-ontology-v1 and h2m-project-grouping-v1 as provisional review controls, preserving the documented 4,979-record over-grouping and residual missed-sibling caveats. Proceed with human-review calibration only: create blinded metadata-linked review worksheets, run the 50-record timing calibration with two independent reviewers, and report agreement, disagreement, kappa or alpha where meaningful, median seconds per judgment, projected total person-hours, and any rubric revisions. STOP after calibration for human review. Do not begin full production review, assign final train/validation/test IDs, materialize H2-mini JSONL, train any model, or implement replay.
""")
write("h2_mini_materialization_plan.md",f"""
# H2-mini materialization plan

Current status: candidate audit complete; final materialization is blocked.

1. Human accepts or revises h2m-cue-ontology-v1 and h2m-project-grouping-v1, including the 4,979-record maximum semantic component and residual missed-sibling risk.
2. Run the 50-record, two-reviewer timing calibration. Stop or resize if total projected effort is substantially above 15 person-hours.
3. Build review queues from metadata-only candidate IDs. Review under the H2-mini protocol; do not expose model scores or NotInject content.
4. Re-run grouping and contamination after exclusions. Require B1/B2/B3 290 each, four disjoint B0 sets of 290, and one disjoint B-short set of 120.
5. Propose hash-only assignments using salt H2-mini-split-v1-2026-09-13. Sort whole groups by SHA256(source revision || group ID || salt); fill test, then validation, then train. Never split a group or change the salt to fill a quota.
6. Preserve the single exception: H2m-C0 and H2m-F share exactly the same H2m-1 B0 IDs. All other groups remain disjoint across arms, stages, strata, train/validation/test, B-short, and NotInject.
7. Stop for human approval of the proposed assignments. Only after separate approval may local git-ignored final JSONL be created. Training requires another separate approval.

The final materializer must verify source revision and hashes, exact counts, labels, no truncation at 1280, malicious position counts 67/66/67 train, 10/10/10 validation, 20/20/20 test, complete human-review outcomes, group disjointness, and contamination gates. It must never synthesize filler.
""")
updates={
"security_bidirectional_stream.md":f"""
<!-- H2_MINI_AUDIT_20260913:BEGIN -->
## H2-mini audited feasibility update

H2-mini is the approved course-scale feasibility design; full H2-F remains the long-form target and is not implemented. Pinned acquisition/audit yields automated group-safe candidates pending human review: B1 {b1:,}, B2 {b2:,}, B3 {b3:,}, OASST ordinary {oa:,}, and B-short length/language pool {bs:,}. Both 290-per-stratum and future 650-per-stratum screens pass.

No official OR rewrite-parent mapping exists. Project semantic/template groups are approximate; the largest semantic component is 4,979 records. Final assignment, JSONL, training, replay, and selective adaptation remain blocked.
<!-- H2_MINI_AUDIT_20260913:END -->
""",
"replay_readiness_review.md":"""
<!-- H2_MINI_AUDIT_20260913:BEGIN -->
## H2-mini readiness update

Candidate capacity is sufficient to begin human review, but there are no H2-mini no-replay results. R0 remains active; R1 and R2 are not entered. Severe NotInject over-defense remains an absolute boundary problem, not evidence of monotonic continual forgetting. Replay remains blocked until matched H2m-C0/H2m-F no-replay results satisfy the prospective criteria and are reviewed.
<!-- H2_MINI_AUDIT_20260913:END -->
""",
"replay_entry_criteria.md":"""
<!-- H2_MINI_AUDIT_20260913:BEGIN -->
## H2-mini prerequisite gate

Before any R1 assessment, H2-mini needs accepted ontology/grouping, measured review-time feasibility, completed human labels, approved hash-only splits, separately approved materialization, and complete three-seed matched no-replay H2m-C0/H2m-F results. Absolute FPR alone, candidate capacity, or a single seed cannot advance readiness.
<!-- H2_MINI_AUDIT_20260913:END -->
""",
"assumptions_and_risks.md":"""
<!-- H2_MINI_RISKS_20260913:BEGIN -->
| R063 | OR-Bench publishes no toxic-seed/rewrite-parent mapping | high | Keep project semantic/template groups intact; disclose residual sibling leakage; require human acceptance | open |
| R064 | Project semantic approximation has a 4,979-record connected component | medium | Treat it as conservative over-grouping; do not split it merely to fill quotas; inspect before final allocation | open |
| R065 | Approximate groups can still miss paraphrased siblings | high | Human duplicate flags and post-review semantic/template re-audit | open |
| R066 | OR-Bench moderator-safe labels include possible toxic/debatable cases | high | Required project benign/IPI review; reject unsafe or ambiguous cases | open |
| R067 | OASST source review is not a project benign/IPI judgment | high | Apply H2-mini review rubric before use | open |
| R068 | Review workload may exceed 10-15 person-hours | medium | Measure 50 records first; resize before production review if projection exceeds 15 hours materially | open |
| R069 | B-short matching can become a hidden selection channel | medium | Use only predeclared length/language quotas and hash allocation; never use detector scores | open |
<!-- H2_MINI_RISKS_20260913:END -->
""",
"open_questions.md":"""
<!-- H2_MINI_QUESTIONS_20260913:BEGIN -->
| Q035 | Accept h2m-cue-ontology-v1 for candidate review? | Required before review queues | Human approve or revise | open |
| Q036 | Accept h2m-project-grouping-v1 despite the 4,979-record component and missed-sibling risk? | Required before split planning | Human inspect and approve/revise | open |
| Q037 | Authorize the 50-record two-reviewer timing calibration? | Needed for person-hour projection | Human approval | open |
| Q038 | Does measured review effort stay within 10-15 person-hours? | Controls benchmark scale | Decide after calibration | pending calibration |
| Q039 | Approve final hash-only assignments after completed review and re-audit? | Required before JSONL | Later human review | blocked |
| Q040 | Approve local final JSONL materialization? | Separate data-construction gate | Later human approval | blocked |
| Q041 | Approve H2-mini no-replay training? | Separate experiment gate | Later human approval | blocked |
<!-- H2_MINI_QUESTIONS_20260913:END -->
""",
"decision_log.md":f"""
<!-- H2_MINI_DECISIONS_20260913:BEGIN -->
## D028

- date: 2026-09-13
- decision: Adopt H2-mini as a course-scale two-arm feasibility benchmark while retaining full H2-F as an unimplemented long-form target.
- motivation: Existing malicious-family streams saturate, while frozen NotInject exposes severe absolute over-defense without persistent sequential worsening.
- consequences: Use 290 records per class per stage, B-short 120, reduced review policy, and separate gates for review, assignment, JSONL, and training. Replay remains R0.

## D029

- date: 2026-09-13
- decision: Record OR-Bench and OASST1 as capacity-feasible candidate sources, but treat all automated counts as pending human benign/IPI review.
- motivation: Audited counts exceed mini and future-pilot screens, but OR official parent lineage is absent and source labels do not meet project ground-truth requirements.
- consequences: Project grouping approximations and their risks remain visible; final materialization is blocked.
<!-- H2_MINI_DECISIONS_20260913:END -->
""",
"change_log.md":"""
<!-- H2_MINI_CHANGE_20260913:BEGIN -->
## C016

- date: 2026-09-13
- reason: Execute the approved H2-mini pinned-source acquisition, lineage/candidate/contamination/length audit, capacity estimate, and review planning.
- effect: Adds exact source hashes, metadata-only candidates, cue ontology, project grouping, two-arm counts, review workload gate, and materialization plan.
- scientific behavior changed: false; no model inference for selection, training, replay, selective adaptation, or final JSONL.
- limitations: official OR parent lineage absent; automated benign labels unconfirmed; person-hours unmeasured; final assignment/materialization/training blocked.
<!-- H2_MINI_CHANGE_20260913:END -->
"""}
for name,block in updates.items():
 p=DOC/name;text=p.read_text(encoding="utf-8")
 marker=block.strip().splitlines()[0]
 if marker in text:
  start=text.index(marker);end_marker=block.strip().splitlines()[-1];end=text.index(end_marker,start)+len(end_marker)
  text=text[:start]+block.strip()+text[end:]
 else:text=text.rstrip()+"\n\n"+block.strip()+"\n"
 p.write_text(text,encoding="utf-8")
dec={"date":NOW,"id":"D028","decision":"Adopt H2-mini as the course-scale matched H2m-C0/H2m-F feasibility benchmark; retain full H2-F as an unimplemented long-form target and retain replay status R0","motivation":"AF1-AF4 are near ceiling while frozen NotInject shows severe absolute over-defense without persistent seed-consistent worsening","alternatives_considered":"Implement full H2-F now; train on NotInject; start replay before a bidirectional no-replay benchmark","source_evidence":"Human-approved H2-mini specification and pinned-source audit","consequences":"Review, final assignment, JSONL materialization, training, replay, and selective adaptation remain separately blocked","status":"design and candidate audit complete"}
dec2={"date":NOW,"id":"D029","decision":"Treat OR-Bench and OASST1 capacity as sufficient for human review while keeping automated group-safe counts provisional","motivation":f"Automated group-safe candidates are B1 {b1}, B2 {b2}, B3 {b3}, ordinary {oa}, but OR official parent mapping is absent and source labels require project review","alternatives_considered":"Treat source labels as final; infer parent IDs from row order; split large project groups; generate filler","source_evidence":"Pinned file hashes, full cue/group/overlap/length audits","consequences":"Human acceptance of ontology/grouping and review-time calibration are the next gates","status":"ready to propose human review; final materialization blocked"}
chg={"date":NOW,"id":"C016","files_changed":["scripts/audit_h2_mini_sources.py","scripts/render_h2_mini_audit_docs.py","data/raw/hard_benign/or_bench/<revision>","data/raw/hard_benign/oasst1/<revision>","data/audits/hard_benign/or_bench_full_audit.json","data/audits/hard_benign/oasst1_full_audit.json","data/audits/hard_benign/h2_mini_capacity.json","data/manifests/hard_benign/h2_mini_candidates.json","data/manifests/hard_benign/b_short_mini_candidates.json","docs/","history/decisions.jsonl","history/changes.jsonl"],"reason":"Complete approved H2-mini acquisition, audit, capacity, and review-planning phase","effect":"Establishes source hashes, candidate capacities, cue/group policies, contamination and length audits, review workload, and materialization gates while retaining R0","tests_run":["H2-mini artifact/hash/privacy/count contract passed","compileall passed","pytest 39 passed with one retained Python 3.14 Torch JIT warning","pip check found no broken requirements","active code/config old-root scan passed","automatic project-size gate scan passed"],"scientific_behavior_changed":False,"limitations":["No official OR parent mapping","Automated candidates await human review","Review timing not measured","No final assignments/JSONL/training/replay/selective adaptation"]}
def upsert_jsonl(path,items):
 old=[]
 if path.exists():
  for line in path.read_text(encoding="utf-8").splitlines():
   if line.strip():old.append(json.loads(line))
 by={x.get("id"):x for x in old}
 for x in items:by[x["id"]]=x
 order=sorted(by.values(),key=lambda x:(x.get("id","")[0],int(x.get("id","X0")[1:]) if x.get("id","X0")[1:].isdigit() else 0))
 path.write_text("".join(json.dumps(x,ensure_ascii=False,separators=(",",":"))+"\n" for x in order),encoding="utf-8")
upsert_jsonl(ROOT/"history/decisions.jsonl",[dec,dec2]);upsert_jsonl(ROOT/"history/changes.jsonl",[chg])
print(json.dumps({"docs_created":9,"docs_updated":7,"history":["D028","D029","C016"],"timestamp":NOW},indent=2))
