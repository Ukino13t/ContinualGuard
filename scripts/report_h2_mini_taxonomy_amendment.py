"""Write proposal-only BH amendment documents; preserve rubric-v1 evidence."""
from collections import Counter
from datetime import datetime,timezone,timedelta
import hashlib,json
from pathlib import Path
from analyze_h2_mini_production_review import ROOT,load,save,file_sha,relative

TAG="H2_MINI_BH_TAXONOMY_PROPOSAL_20261004"
def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join("---" for _ in headers)+" |"]+
      ["| "+" | ".join(map(str,row))+" |" for row in rows])+"\n"
def block(path,text):
    content=path.read_text(encoding="utf-8-sig")
    begin=f"<!-- {TAG}:BEGIN -->";end=f"<!-- {TAG}:END -->";new=begin+"\n"+text.rstrip()+"\n"+end
    if begin in content:
        a,b=content.split(begin,1);_,c=b.split(end,1);content=a+new+c
    else:
        a,_,b=content.partition("\n");content=a+"\n\n"+new+"\n\n"+b.lstrip()
    path.write_text(content,encoding="utf-8")
def main():
    cap_path=ROOT/"results/h2_mini_bh_capacity.json";feas_path=ROOT/"results/h2_mini_bh_partition_feasibility.json"
    c=load(cap_path);f=load(feas_path)
    for name,state in c["historical_evidence_before_and_after"].items():
        p=ROOT/name
        assert file_sha(p)==state["sha256"] and p.stat().st_mtime_ns==state["mtime_ns"],"Historical evidence changed: "+name
    n=c["post_grouping"];b0=c["B0_replenishment"];minimum=b0["minimum_design_keep_review_lanes"];projection=b0["initial_workload_projection"]
    bs=c["Bshort_options"];A=bs["option_A"];B=bs["option_B"];w=f["anonymous_metadata_count_witness"]
    day=datetime.now(timezone(timedelta(hours=8))).date().isoformat()
    stamp=datetime.now(timezone.utc).isoformat()
    status=f"Proposal dated {day}; audit began 2026-10-03. **Not adopted as the final benchmark.** No new review, final partition/split IDs, research JSONL, model training or replay is authorized by this document.\n"
    question="Under sequential attack-family adaptation, can the detector retain correct acceptance of independently reviewed hard-benign inputs compared with an ordinary-benign matched control?"
    next_prompt=("Approve the proposed B0/BH taxonomy amendment for H2-mini, preserving all rubric-v1 and production-review evidence. "
      "BH-1/BH-2/BH-3 are disjoint balanced partitions under persistent hard-benign pressure, with no ordinal difficulty claim. "
      "Approve B-short Option B as a descriptive external diagnostic using a proposed 42 English/78 multilingual composition; explicitly retain the language-mismatch limitation and language-stratified reporting. Do not freeze its IDs yet. "
      "Prepare the targeted OASST1 B0 review package only: initially 600 primary-reviewed training candidates with 75 independent second reviews, plus 272 independently double-reviewed validation/test candidates (872 new candidates, 1219 judgments). "
      "Keep both reviewer workloads balanced, use the existing ordinary-benign rubric, exclude all previously reviewed/rejected/disputed/reserved identities and prohibited groups, and audit contamination before worksheets. "
      "Preserve enough accepted double reviews for 360 C0 validation/test records and 80 training-audit records overall; report failures and any needed top-ups before expanding. "
      "Do not replenish B1/B2/B3, infer human judgments, assign final BH partitions or train/validation/test IDs, materialize research JSONL, train, choose a numerical replay budget, or implement replay. Stop after preparing the review package for the human reviewers.")
    documents={}
    documents["h2_mini_taxonomy_amendment.md"]="# H2-mini benign-taxonomy amendment proposal\n\n"+status+"""
## Decision under consideration

Propose replacing mandatory final B1/B2/B3 difficulty labels with two source-defined benign groups:

| Proposed group | Operational definition | Claim supported |
| --- | --- | --- |
| B0 | Ordinary short-form OASST1 prompts satisfying the existing ordinary-benign review rule | Human-confirmed ordinary benign, subject to the approved review coverage |
| BH | OR-Bench production candidates satisfying the existing benign ACCEPT_BENIGN policy, regardless of B1/B2/B3 stratum disagreement | Human-confirmed hard benign in the source-defined over-defense sense; no calibrated ordinal or detector-specific hardness |

The fixed detector labels remain benign=0 and malicious indirect prompt injection=1. B0/BH are benchmark metadata, not new classifier classes. Historical proposed strata, both reviewer strata, reviewer decisions, confidence, notes, timing and all rubric-v1 results remain unchanged. Original B1/B2/B3 metadata may balance samples and describe results, but cannot be asserted as final ground-truth difficulty.

This is explicitly a **protocol amendment motivated by failed production-level stratum reliability**. It was not predeclared. The previous rubric-v1 failure/stop remains a valid historical result; this proposal does not rewrite it as a PASS.

## Why the amendment is defensible

All 2,288 production judgments are complete. The accepted decision evidence is 96.32% double-review decision agreement, decision kappa 0.5644, zero ACCEPT_BENIGN-versus-TRUE_INJECTION paired conflicts and approximately 91-93% benign acceptance among OR candidates. In contrast, double-review stratum agreement is 48.49%, stratum kappa 0.0852, and the reviewers systematically differ on B0 versus B1/B2/B3.

The change uses the better-supported benign decision axis and removes the poorly supported ordinal axis. It does not improve the original agreement numerically, prove every label correct, or imply that all training records have double review. It avoids automated adjudication: an ACCEPT/REJECT conflict still excludes a record; only stratum disagreement ceases to be a BH exclusion under the proposed new rule.

The amendment is being evaluated before any H2-mini training, final assignment or materialization. Earlier AF/NotInject experiments exist, but neither their predictions nor test performance formed BH eligibility, ranked candidates or balanced the partitions. NotInject text/hashes are used only for contamination checking; no detector or NotInject score enters the audit.

BH hardness is source-defined by OR-Bench's over-refusal setting. Humans confirmed benignness under the course-scale review policy; they did not establish an ordinal ranking or prove the pinned detector will fail on these prompts. OR source rights, missing official rewrite-parent IDs, approximate semantic groups and residual unsafe-label risk remain disclosed.

## Review eligibility

Both required reviewers must ACCEPT_BENIGN for a double-reviewed record. Primary-only training candidates retain the approved 100% primary screening plus stratified independent training-audit policy. The decision audit rates in the three historical sampling cohorts remain 4%, 8%, 8%, each below the existing 10% decision-disagreement gate.

True injection, ambiguous intent, unsafe/harmful content, missing context, privacy/rights problems, quality failures and any unresolved benign-versus-reject conflict remain excluded. Existing independent validation/test double-review coverage is mandatory. No primary-only record may be used to fill a double-reviewed quota. Stratum disagreements remain visible as evidence, not silently adjudicated.

The relaxation is specific to proposed OR-Bench BH membership. Current OASST B0 and B-short evidence retain the ordinary-B0 requirement and existing exclusions; no OASST B3/UNSURE record is silently converted to B0.

## Scientific claim and scope change

Remove the claims that B1 -> B2 -> B3 or BH-1 -> BH-2 -> BH-3 represent increasingly difficult benign inputs, calibrated easy/medium/hard strata, or a monotonic benign-difficulty evolution. Do not report an ordinal-difficulty retention result from these partitions.

The revised question is:

> """+question+"""

H2m-C0 studies sequential attack-family learning with ordinary-benign controls. H2m-F studies the same sequence under persistent OR-Bench hard-benign pressure after the Stage-1 ordinary-benign anchor. BH-1/2/3 have similar marginal composition and new disjoint records. There is no designed benign difficulty progression across stages 2-4; repeated similar BH exposure may strengthen retention and may reveal little forgetting. That is a possible result, not a design failure or reason to relax replay criteria.

This preserves the broader over-defense/continual-adaptation motivation, but narrows the operational claim. The arms are matched for malicious IDs, splits, positions, training and evaluation; their benign source, language and format can still differ. Arm differences cannot by themselves establish a pure causal effect of benign difficulty. Retention is measured from first acquisition and distinguished from acquisition, source/domain shift and optimization effects.

## Status

Scientifically defensible as a transparent, pre-H2-training amendment proposal with these limits. Human adoption is still required. No large B1/B2/B3 replenishment or forced ordinal adjudication is recommended. R0 remains active; R1/R2 criteria and all replay approvals remain unchanged.

See [capacity audit](h2_mini_bh_capacity_audit.md), [simplified design](h2_mini_simplified_design.md), [B0 plan](h2_mini_b0_replenishment_plan.md) and [B-short options](b_short_resolution_options.md).
"""
    checks=table(["Resource","Post-audit available","Required","Feasible"],
      [["Unique BH",n["total_unique"],870,"yes"],["Training review lane",n["training_lane"],600,"yes"],
       ["Training independent second reviews",n["training_audited_double"],60,"yes"],
       ["Validation double-review reserve",n["original_validation_reserve"],90,"yes"],
       ["Test double-review reserve",n["original_test_reserve"],180,"yes"],
       ["Validation+test disjoint reserve",n["validation_test_reserve_double"],270,"yes"]])
    audit_table=table(["External bank","Target records","Affected BH records"],[[name,v["target_count"],v["affected_queries"]] for name,v in c["audit"]["external_banks"].items()])
    documents["h2_mini_bh_capacity_audit.md"]="# H2-mini proposed BH capacity audit\n\n"+status+f"""
## Exact results

All 1,080 OR-Bench production candidates were evaluated against the frozen human decisions. **{c['pre_grouping']['total_unique']} unique records qualify before grouping; {n['total_unique']} remain after all checked grouping and contamination rules.** The other {c['human_rejection_count']} have at least one required non-ACCEPT decision and remain excluded. Historical proposed B1/B2/B3 contributions are 329/334/336; these are descriptive sampling metadata only.

| Eligible review coverage | Count |
| --- | ---: |
| Training lane total | {n['training_lane']} |
| Training primary only | {n['training_primary_only']} |
| Training independently double-reviewed audit | {n['training_audited_double']} |
| Validation reserve, double reviewed | {n['original_validation_reserve']} |
| Test reserve, double reviewed | {n['original_test_reserve']} |
| Validation/test reserves combined | {n['validation_test_reserve_double']} |
| All double-reviewed records, including training audits | {n['all_double_reviewed']} |
| Total unique BH | {n['total_unique']} |

These categories are not all additive: 376 includes 68 training audits plus 308 evaluation reserves; 691 plus 308 equals 999. The conservative feasible construction keeps the training and evaluation reserve lanes disjoint. All 376 double-reviewed records could meet the review-coverage aspect of evaluation eligibility, but moving training audits would consume training-review capacity; the witness needs no such movement.

{checks}
The original mandatory-stratum pools had 284 combined B1/B2/B3 records. The proposed rule makes {c['records_recovered_relative_to_stratum_mandatory_pool']} additional already-benign-accepted records eligible without adding or changing any human judgment. It does not rescue any of the 81 decision-rejected/conflicting candidates.

## Accepted-only audit

Exact, normalized and model-view duplicates; native/source identity and available parent lineage; source-scoped project semantic-parent and template groups; exhaustive lexical/latent/template accepted-pair checks; and cross-source checks against existing B0 all produced zero BH exclusions. Each retained prohibited group has one representative. The deterministic rule remains lowest pinned source-row representative, with stable source/revision ordering for ties; grouping thresholds were not weakened.

{audit_table}
AF coverage includes all 7,320 malicious and 7,320 companion benign records in AF1-AF4; AF5 is excluded. The full proposed B-short reserve of 150 is checked, not only a favorable future subset. Existing B0 capacity remains 339; its prior one-record B-short-neighbor quarantine is untouched.

The overlap checks include exact/normalized/model-view hashes, normalized substrings of at least 25 characters, lexical Jaccard >=0.85, latent cosine >=0.97 with Jaccard >=0.30, and project template matches. Internal grouping retains the frozen full-source group IDs and internal lexical/latent criteria, including rare-phrase links already present in full-source groups. Recomputing rarity on the small BH subset is not used.

The semantic screen is the existing deterministic hashed word/bigram + 48-dimensional SVD approximation fitted on pinned OR text, with exhaustive candidate comparisons. It is an audit representation, not detector training or a pretrained semantic encoder. OR's official generator/rewrite-parent IDs are unavailable; a zero result is not proof that no unknown paraphrase siblings exist.

Pinned sources: OR-Bench e36d8b80e81837c8a8f264bbb2a49f1b32c7e272 (CC BY 4.0); OASST1 fdf72ae0827c1cda404aff25b6603abec9e3399b (Apache-2.0); NotInject 847ae76cf8fea5ed325429e569ae8cfef022d2e0 (dataset-card MIT). These preserve the accepted source-level provenance/rights caveats; raw text is not redistributed.

Fresh tokenizer hashes use DeBERTa revision 8ccc9b6f36199bec6961081d44eb72fb3f7353f3, special tokens included, truncation disabled. All 999 BH texts fit 1280; maximum observed length is {c['audit']['max_observed_tokens']}. Audit runtime was {c['runtime_seconds']:.2f} seconds excluding library import; no detector predictions or NotInject scores were used.

## Evidence and artifacts

All {len(c['historical_evidence_before_and_after'])} protected historical files retained both their SHA-256 and modification timestamp across this audit, including all original/snapshot/canonical review files, rubric guides, previous production analysis, prior accepted pools and historical reports. No historical proposed/reviewer stratum, decision, confidence, note, timing or rubric result changed.

results/h2_mini_bh_capacity.json contains the exact counts, exclusions, coverage, input hashes and metadata-only proposed BH eligibility records. Each record retains original proposed-stratum metadata and references the original canonical reviewer evidence; its proposed_final_benign_taxonomy=BH is explicitly a new proposal field. No stage or final split field is assigned.

results/h2_mini_bh_partition_feasibility.json contains a deterministic anonymous metadata-count witness. It selects no record IDs. The result supports capacity feasibility, not adoption or final assignment approval.
"""
    designs=[["1","AF1 Naive","B0-1","exact same B0-1"],["2","AF2 Ignore","independent B0-2","BH-1"],["3","AF3 Escape-Separation","independent B0-3","BH-2"],["4","AF4 Fake Completion","independent B0-4","BH-3"]]
    gaps=table(["Balancing feature","Largest count difference between anonymous BH partitions"],
      [[k,v] for k,v in w["max_between_partition_count_gaps"].items()])
    marg=w["stage_overall_marginals"]
    original_table=table(["Anonymous partition","Historical B1","Historical B2","Historical B3"],
      [[stage,*[marg[stage]["original_proposed_stratum"].get(k,0) for k in ["B1","B2","B3"]]] for stage in ["BH-1","BH-2","BH-3"]])
    documents["h2_mini_simplified_design.md"]="# H2-mini simplified two-arm design proposal\n\n"+status+"""
## Revised stream

"""+table(["Stage","Malicious half shared between arms","H2m-C0 benign","H2m-F benign"],designs)+"""
C0 measures sequential attack-family adaptation under ordinary-benign controls. F measures the same process under persistent hard-benign pressure after stage 1. BH-1/BH-2/BH-3 are three disjoint, composition-balanced OR-Bench partitions; the suffix is an index, not a difficulty level.

Each stage remains train 200 malicious +200 benign, validation 30+30 and test 60+60: 580 records per stage, 2320 per arm. The three BH partitions together require 600 train, 90 validation and 180 test benign records. C0 needs 1160 independent B0 records; the same B0-1 anchor is shared with F, so the anchor is not counted twice. No B0 record may cross C0 stages.

The malicious half will be selected once from the already-audited AF1-AF4 pools and shared exactly across arms, including labels, IDs, split, family, positions and source hashes. Keep malicious head/middle/tail counts 67/66/67 train, 10/10/10 validation and 20/20/20 test. Nothing is selected in the present task.

## Deterministic partition feasibility without final IDs

The anonymous count witness uses only native OR category, fixed token-length bins (<=16,17-24,25-32,33-48,49-64,65-128,129-256,257-1280), language, cue-family metadata and original proposed-stratum metadata. Multi-cue memberships are nonexclusive. These metadata describe composition; none establishes final difficulty.

Four disjoint coverage pools support each anonymous partition: 20 independently audited training records, 180 primary-only training records, 30 double-reviewed validation records and 60 double-reviewed test records. Group-safe records are already single representatives. Within each coverage pool, the procedure fills the least-filled partition, minimizes metadata marginal deficits/imbalances, and uses SHA-256(policy/scope/metadata) ties. Weights are category4, historical proposed stratum3, length2, language1 and cue family0.5. The algorithm and amendment policy ID are in scripts/audit_h2_mini_bh_simplification.py. The ordering does not inspect text, detector predictions/confidence, NotInject scores or future outcomes.

The witness contains metadata-bucket counts and marginals only: no review-ID-to-partition or review-ID-to-final-split mapping was created. Each partition has exactly 200/30/60 benign capacity and 20 training second reviews. All bucket supply constraints pass. The three anonymous partitions have these small descriptive differences:

"""+gaps+"\n"+original_table+"""
All selected-count capacity is English because the human-eligible OR production pool is English. The partitions have similar composition, not identical every-feature marginals or proven equal empirical difficulty. The cue-family maximum difference is 6/290 =2.07 percentage points; cue memberships overlap. Historical B1/B2/B3 counts are deliberately mixed across every partition rather than used as stage labels.

After explicit amendment and assignment-proposal approval, a separate step may resolve anonymous bucket counts into source/hash IDs using a frozen deterministic salt and reverify all quotas, retained groups, lineage, external contamination and reviewer coverage. It must stop again for review of concrete hash-only assignments. No salt search or detector-guided selection is allowed.

## Interpretation and controls

Operational question:

> """+question+"""

The proposed comparison matches malicious data and optimization/evaluation factors. B0 is OASST1 while BH is OR-Bench, and current B0 is multilingual while BH is English. Residual source, language, topic and format differences prevent attribution of an arm gap solely to hard-benign semantics. Report these distributions and acquisition/retention separately; the term matched control does not imply matched benign-language/source composition.

Keep microsoft/deberta-v3-base revision 8ccc9b6f36199bec6961081d44eb72fb3f7353f3, binary labels0/1, max_length1280, seeds17/42/2026, identical initialization protocol, stage order, epochs, learning rate, optimizer reset each stage, carried model weights, fixed threshold and existing evaluation protocol. Use current-stage training only. The existing prospective no-replay protocol is not executed or retuned by this proposal.

At future approved stage endpoints evaluate every acquired AF test and corresponding benign partition, frozen NotInject and an approved B-short bank. Keep attack TPR/FNR and benign FPR/TNR separate; report paired-stage accuracy/macro-F1, first-acquisition retention, signed BWT, peak forgetting, position diagnostics, runtime and memory per seed and mean/sample SD. Single-class macro-F1 remains N/A under the project metric convention. External banks never select thresholds, epochs, records, methods or replay samples.

The future replay comparison still fixes detector/revision, initialization, stage datasets/order, seeds, epochs, learning rate, reset policy, evaluation and historical-example budget; replay strategy is the main changed variable. Numerical memory budget remains unresolved. R0 remains active and all predeclared R1/R2 gates remain unchanged.

## Remaining gates

BH capacity passes as a proposal. B0 still lacks 821 accepted records; B-short language policy needs a human choice. A targeted review can address the B0 deficit without another large B1/B2/B3 review, but it is not a trivial handful of records. After adequate labels, independently re-audit, propose final hash-only IDs, obtain materialization approval and then obtain separate training approval. No final H2-mini dataset or experiment is ready today.
"""
    workload=table(["Additional work (all accepted lower bound)","Candidates","Judgments"],
      [["New B0 training primary screening",566,566],["Independent second review of training candidates","subset of above",60],["New B0 validation/test double review",255,510],["Total unique new candidates / total judgments",821,1136]])
    documents["h2_mini_b0_replenishment_plan.md"]="# H2-mini targeted ordinary-B0 replenishment proposal\n\n"+status+f"""
## Exact deficit and existing review credit

Four independent C0 stages need 4x290=1160 B0: 800 train, 120 validation and 240 test. F shares C0's Stage-1 anchor, so it creates no fifth B0 requirement.

Current group-safe ordinary-B0 capacity remains **339**, including 234 training-lane records (214 primary-only and 20 independently audited) and 105 validation/test reserves (33/72, all double reviewed). In total 125 have independent double review. The previous B-short-neighbor exclusion remains excluded. The proposed BH rule does not reinterpret OASST stratum disagreements or reclaim rejected records.

The exact unique-record deficit is **821**. Using all current double-reviewed records for validation/test would leave 235 additional double-reviewed evaluation records needed, but would move the 20 training audits and require replacing their audit coverage. Preserving current review lanes instead needs 255 new evaluation records and 60 additional training second judgments. Both accounting approaches give the same minimum workload; do not count the 20 existing audits twice.

## Smallest workload under the existing reserved-pool policy

Retain all currently eligible B0 evidence; do not review it again. Continue ordinary-benign screening, not the failed hard-benign ordinal taxonomy. Every training candidate gets a human primary decision; final training requires at least 10% independent review (80 of 800, with at least20 in each future200-record stage). Every validation/test candidate requires two independent ACCEPT/B0 confirmations.

{workload}
This lower bound is **{minimum['person_hours_at_observed_B0_mean']:.3f} person-hours**, using the observed B0 mean {b0['observed_B0_mean_seconds']:.4f} seconds/judgment. It assumes every added primary/second judgment is acceptable and every selected record remains group-safe.

Why 1136 is a lower bound: final evidence needs 1160 primary judgments +360 evaluation second judgments +80 training-audit second judgments =1600 usable judgments. The current339 records provide339+125=464 usable judgments. The difference is1136. This is a minimum within the present policy that keeps B-short reserved and retains the ordinary-B0 definition. Releasing unused external-control candidates or changing that definition would be a separate protocol change, not an assumed saving.

## Efficient initial package to propose, not execute

Observed accepted training yield is234/248=94.35%; evaluation reserve yield105/112=93.75%; usable audited-training yield20/25=80% (a small and uncertain sample). A modest initial oversample is:

| Review role | Initial proposed candidates/judgments |
| --- | ---: |
| New ordinary-B0 training primary candidates | {projection['training_candidates']} |
| Independent second reviews among those training candidates | {projection['training_second_judgments']} |
| New independent double-review validation/test candidates | {projection['double_validation_test_candidates']} |
| Unique new candidates | {projection['new_unique_candidates']} |
| Total new judgments | {projection['judgments']} |

Projected time is **{projection['person_hours']:.3f} person-hours** at the observed B0 mean. This uses a600-candidate primary pool,272 double-review candidates and75 stratified second reviews (12.5% of new training candidates); the extra audits aim to leave60 accepted second-reviewed records after the observed losses. These are expected-yield planning numbers, not a confidence bound or guarantee. Re-audit and report any shortage before adding an explicitly approved small top-up.

The complete production review already used9.689 person-hours. This initial package adds about4.493, for14.182 before prior calibration, administration, any top-ups or B-short option A. Reviewer fatigue and batch overhead may increase elapsed and person time.

## Review safeguards and proposed logistics

Use the same approved ordinary-benign rubric: standalone, context-complete OASST1 root prompts, clear benign intent and ordinary-B0 membership; reject injection, ambiguity, unsafe content, privacy/rights problems and poor quality. Do not ask reviewers to distinguish B1/B2/B3, and do not show source safety labels as ground truth.

Future candidate preparation uses pinned OASST1 revision fdf72ae0827c1cda404aff25b6603abec9e3399b and the already audited candidate pool. Exclude all previously reviewed IDs, calibration records, rejected/disputed IDs, current accepted pools, the reserved B-short pool and their prohibited native/tree/content/semantic/template groups. Audit candidate overlap with NotInject, all AF1-AF4 records and proposed BH; do not rank by detector outputs or external scores. Preserve source language/length metadata and report remaining composition differences.

Balance training primaries300/300 between reviewers; each reviewer sees all272 evaluation candidates, plus37/38 of the other reviewer's training candidates as independently selected second reviews. This gives609/610 judgments per reviewer. Use manageable independent batches with metadata-only assignment manifests and local ignored raw worksheets. Prospective audit samples are stratified using source language/length metadata and sampling seeds, never selected by the other reviewer's answers.

Only independently double-approved records may fill validation/test; training audit quotas cannot be lost when final subsets are chosen. Any decision audit exceeding10% triggers the existing pause/escalation policy. No unapproved top-up, automatic adjudication, ordinal relabeling, B0 duplication across stages or relaxation of grouping is allowed.

**No new candidate IDs, batches, worksheets or human judgments were created in this task.** Human approval is required before even preparing the review package. Final assignment, materialization and training remain later separate approvals.
"""
    documents["b_short_resolution_options.md"]="# B-short language-resolution options\n\n"+status+f"""
## Existing evidence and unresolved choice

The reserved accepted B-short pool contains144 records:42 English and102 multilingual. The intended120-record bank aimed for approximately90 English/30 multilingual, reflecting the frozen NotInject composition255/84. All selected B-short records must remain double-reviewed ordinary benign, independent of H2 train/validation/test and NotInject, and excluded from every model/method/replay-selection decision.

No bank or option is selected by this report. All figures below are marginal review work, separate from the B0 replenishment.

| Option | Additional review | Estimated person-hours | Methodological effect |
| --- | --- | ---: | --- |
| A: retain90 English/30 multilingual | At least48 new accepted English candidates,96 judgments; initial projection53 candidates,106 judgments | {A['minimum_person_hours']:.3f} minimum; {A['projected_person_hours']:.3f} projection | Preserves planned approximate language match; length matching and residual source differences still need reporting |
| B: use achievable reviewed composition | 0 new judgments; proposed42 English/78 multilingual from existing evidence | 0 new human-review hours | Descriptive short-form ordinary-benign diagnostic; substantial language mismatch remains |

Timing uses observed B-short mean12.7 seconds/judgment. Option A's projected53 candidates use the observed42/46=91.30% accepted-English yield; yields and token-length coverage may differ in a new sample. Neither estimate includes review coordination or additional failures/group exclusions.

## Option A

Review only additional English OASST1 candidates, preserving the validated ordinary-benign rubric and independent double review. At least48 must pass. Do not repeat the full hard-benign review procedure or inspect detector scores. Candidate sampling must maintain source/native/tree and semantic/template safeguards; match only token-length/language metadata and recheck the entire approved external bank's independence.

The planned90/30 split is75% English, approximating NotInject's75.22%. This makes an aggregate diagnostic contrast less confounded by language, but does not eliminate source/format/topic differences or establish causal trigger bias. Select exactly120 and freeze IDs only in a later explicitly approved step.

## Option B

A feasible proposed quota is42 English plus78 multilingual, retaining all available English capacity to approach the original target as closely as this pool permits. This is a quota proposal only; no specific120 IDs are chosen. It is35% English versus the planned75%, a40-percentage-point gap (40.22 points versus NotInject's255/339).

Use deterministic length/language balancing and group hashing later, never model outputs. Call it an **unmatched descriptive external control**, not the originally planned language-matched diagnostic. Report English and non-English FPR and denominators separately; where feasible report individual-language slices, with small-support caveats. Do not subtract aggregate NotInject and B-short FPR and interpret the difference as a pure trigger-word or prompt-format effect. Language-stratified reporting mitigates interpretation but does not recreate matching or remove all residual confounds.

The unused reviewed reserve remains reserved in this proposal; it is not automatically recycled into B0 training. Any future release would need a separate recorded decision before final assignments.

## Recommendation requiring human approval

**Recommend Option B for the reduced course-scale study only if B-short's role is explicitly limited to a descriptive external diagnostic.** It is the lowest-cost scientifically defensible option for that narrower purpose: no further human review, and the limitation remains visible.

If preserving the original approximately language-matched aggregate diagnostic is a required scientific aim, recommend Option A instead. Its estimated incremental cost is only about22.4 minutes of person time; Option B is not an equivalent substitute for that aim. The human must choose the objective and option. Neither has been adopted or frozen.

With the projected B0 review, Option B adds about4.493 person-hours total; Option A about4.867. B-short itself never authorizes training or advancement from R0.
"""
    for name,text in documents.items():
        (ROOT/"docs"/name).write_text(text,encoding="utf-8")
    summary=f"""## Latest proposal: simplify benign taxonomy ({day})

**Proposal only; human adoption pending.** The user requested an audit of B0 plus pooled BH after failed production-level B1/B2/B3 reliability. Historical stratum labels and rubric-v1 outcomes remain untouched; the previous capacity stop remains valid for that original design.

The proposed BH rule yields999 group-safe human-benign-accepted OR records:691 training-lane and308 independent double-review evaluation reserves (104 validation,204 test), plus68 training audits included in the691. Anonymous metadata counts support three200/30/60 BH partitions with20 training second reviews each. This is persistent hard-benign pressure, not increasing benign difficulty. No final record-to-partition mapping exists.

Current ordinary B0 remains339/1160 (short821); the minimum additional workload is1136 judgments, about4.19 person-hours, with a modest initial projection of1219 judgments/about4.49 hours. B-short option A costs approximately106 judgments/0.374 hours; option B requires no new review but carries a substantial language mismatch. Option B is recommended only as an explicitly descriptive diagnostic; the human has not selected an option.

R0 remains active, with R1/R2 not entered. No new review, final assignment, JSONL, training, replay or numerical memory-budget selection occurred. The older ordinal-stratum replenishment/re-calibration recommendation is now historical, not a current instruction.

See [amendment](h2_mini_taxonomy_amendment.md), [simplified design](h2_mini_simplified_design.md), [BH audit](h2_mini_bh_capacity_audit.md), [targeted B0 plan](h2_mini_b0_replenishment_plan.md), and [B-short options](b_short_resolution_options.md).
"""
    for name in ["h2_mini_design.md","replay_readiness_review.md"]:block(ROOT/"docs"/name,summary)
    risk_rows=[
      ["R082","Protocol amendment could be misrepresented as predeclared or a retrospective rubric-v1 PASS","Name the failed stratum reliability, retain all original evidence, date the new proposal before H2-mini training; human adoption pending."],
      ["R083","BH source-defined hardness could be read as measured detector hardness or ordinal progression","Human-confirmed benignness only under the approved policy; no calibrated hardness or easy/medium/hard claim."],
      ["R084","BH is all English whereas current OASST B0 is multilingual and source/format differs","Matched malicious/training controls do not identify a pure benign-difficulty causal effect; report composition and confounds."],
      ["R085","B-short Option B could silently substitute for a language-matched control","42/78 is only35% English; require human approval, separate language reporting and descriptive interpretation."],
      ["R086","Workload projections could be mistaken for sufficient accepted counts","821 new B0 accepted records remain necessary;1136 judgments is an all-pass lower bound,1219 a planning projection, with explicit top-up review."],
      ["R087","Training double reviews could be counted again as evaluation reserves","The BH witness uses691 training and308 disjoint reserves; the B0 plan explicitly accounts for training-audit credit."],
    ]
    block(ROOT/"docs/assumptions_and_risks.md","## Proposed BH amendment risks\n\n"+table(["ID","Risk","Proposed control/status"],risk_rows))
    qrows=[
      ["Q050","Adopt B0/BH in place of mandatory ordinal B1/B2/B3?","Proposal is defensible; human approval required. No new large hard-benign review recommended."],
      ["Q051","Approve targeted B0 review preparation?","Initial872 candidates/1219 judgments, approximately4.49 person-hours; exact acceptance still must be verified."],
      ["Q052","Which B-short objective and option is approved?","B for descriptive-only lowest-cost control; A if approximate language matching remains essential. No option selected."],
      ["Q053","Approve final BH partitions and train/validation/test IDs?","Not yet; only anonymous metadata-count feasibility exists. Full control capacity and re-audit must pass first."],
      ["Q054","Can the operational research question drop ordinal difficulty?","Proposed question: "+question],
      ["Q044","Choose numerical replay memory budget now?","No; unchanged future no-replay evidence, frozen entry criteria and separate approval required."],
    ]
    block(ROOT/"docs/open_questions.md","## Current proposed next decisions\n\n"+table(["ID","Question","Status/recommendation"],qrows)+"\nThe previous Q047/Q048 ordinal-resolution/replenishment recommendation is superseded as a proposal by the user's simplification-audit request. It was not executed. Historical labels remain unchanged.\n\nExact recommended next prompt (includes the conditional Option B recommendation):\n\n"+next_prompt)
    decision={"date":stamp,"id":"D033","decision":"Evaluate a proposed B0/BH taxonomy amendment based on failed production-level ordinal-stratum reliability",
      "authorization":"User approved simplification audit and proposal only; final taxonomy, new review and final assignments remain unapproved",
      "motivation":"Benign decisions are sufficiently reliable under the approved course-scale policy, but B1/B2/B3 difficulty is not operationally reliable across reviewers",
      "findings":{"BH_before_grouping":999,"BH_after_grouping":999,"training_lane":691,"all_double_reviewed":376,
        "validation_test_reserves":308,"three_BH_partitions_feasible":True,"B0_capacity":339,"B0_shortage":821},
      "consequences":"Retain all historical fields and rubric-v1 failure. Propose source-defined nonordinal BH partitions; do not launch B1/B2/B3 replenishment, final assignment, JSONL, training or replay.",
      "Bshort_recommendation":"Option B only for an explicitly unmatched descriptive diagnostic, pending human choice; Option A remains preferable if language matching is essential",
      "replay_readiness":"R0","status":"proposal only; human adoption and targeted-review approval pending"}
    changed=["scripts/audit_h2_mini_bh_simplification.py","scripts/report_h2_mini_taxonomy_amendment.py",
      "tests/test_h2_mini_bh_simplification.py","results/h2_mini_bh_capacity.json","results/h2_mini_bh_partition_feasibility.json"]
    changed += ["docs/"+name for name in documents]
    changed += ["docs/"+name for name in ["h2_mini_design.md","replay_readiness_review.md","decision_log.md","change_log.md","assumptions_and_risks.md","open_questions.md"]]
    changed += ["history/decisions.jsonl","history/changes.jsonl"]
    change={"date":stamp,"id":"C020","files_changed":changed,"reason":"Authorized benign-taxonomy simplification audit only",
      "effect":"Recompute benign-policy BH eligibility; repeat group/contamination checks; construct anonymous count feasibility; document targeted B0 and two B-short review options",
      "tests_run":["4 synthetic tests passed: rejected decisions never rescued, stratum disagreement ignored only for BH, no review-capacity double counting, deterministic anonymous witness without record IDs",
        "999 eligible BH freshly tokenized and checked against source hashes; max45 tokens",
        "0 BH group or external contamination exclusions against339 NotInject,150 proposed B-short,14640 AF1-AF4 and339 existing B0",
        "3 anonymous partitions satisfy200 train/30 validation/60 test with20 audited training records each; no IDs selected",
        "90 protected historical files retained identical SHA256 and modification timestamps"],
      "scientific_behavior_changed":"Proposed nonordinal benchmark interpretation only; not adopted or executed",
      "limitations":["Missing official OR parent lineage; lexical semantic grouping approximate","B0 remains short821 accepted records","B-short language choice awaits human approval",
        "No final BH or split assignment, new review, materialization, training or replay; numerical memory budget unspecified"]}
    for path,item in [(ROOT/"history/decisions.jsonl",decision),(ROOT/"history/changes.jsonl",change)]:
        original=path.read_bytes();rows=[json.loads(line) for line in original.decode("utf-8-sig").splitlines() if line.strip()]
        existing=next((r for r in rows if r.get("id")==item["id"]),None)
        if existing is not None:
            item.clear();item.update(existing)
        else:
            assert original.endswith(b"\n")
            with path.open("ab") as stream:stream.write((json.dumps(item,ensure_ascii=False)+"\n").encode("utf-8"))
            assert path.read_bytes().startswith(original),"History prefix changed"
    block(ROOT/"docs/decision_log.md","## D033 - Proposed nonordinal BH taxonomy\n\n"+"~~~json\n"+json.dumps(decision,ensure_ascii=False,indent=2)+"\n~~~")
    block(ROOT/"docs/change_log.md","## C020 - BH simplification audit and proposals\n\n"+"~~~json\n"+json.dumps(change,ensure_ascii=False,indent=2)+"\n~~~")
    for name,state in c["historical_evidence_before_and_after"].items():
        assert file_sha(ROOT/name)==state["sha256"] and (ROOT/name).stat().st_mtime_ns==state["mtime_ns"],name
    print(json.dumps({"status":"PROPOSAL_DOCUMENTS_COMPLETE","BH":n["total_unique"],"BH_partition_feasible":f["BH_three_partition_feasible"],
      "B0_shortage":b0["unique_shortage"],"documents_created":list(documents),"decision":"D033","change":"C020",
      "historical_evidence_unchanged":True,"new_review_or_training":False},indent=2))

if __name__=="__main__":main()

