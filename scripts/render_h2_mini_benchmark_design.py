"""Render the approved H2-mini benchmark-readiness analysis and design record."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
HISTORY = ROOT / "history"
STAMP = "2026-10-05"
MARKER = "H2_MINI_BENCHMARK_DESIGN_20261005"


def write(path: Path, text: str):
    path.write_text(text.strip() + "\n", encoding="utf-8")


def prepend_block(path: Path, block: str):
    text = path.read_text(encoding="utf-8")
    begin = f"<!-- {MARKER}:BEGIN -->"
    end = f"<!-- {MARKER}:END -->"
    if begin in text:
        before, rest = text.split(begin, 1)
        _, after = rest.split(end, 1)
        text = before.rstrip() + "\n\n" + block.strip() + "\n\n" + after.lstrip()
    else:
        lines = text.splitlines()
        insert = 1 if lines and lines[0].startswith("# ") else 0
        lines[insert:insert] = ["", block.strip(), ""]
        text = "\n".join(lines)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def append_jsonl_once(path: Path, record: dict):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("id") != record["id"]]
    rows.append(record)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


CEILING = r'''# H2-mini ceiling analysis

## Scope and frozen evidence

This is an analysis-only follow-up to the accepted H2-mini no-replay baseline. The six runs, 24 checkpoints, raw predictions, metrics, manifests, and R0 decision are frozen. No model was loaded, no inference or training was run, and no threshold or record selection changed. The analysis reads the existing 18,216 prediction rows and already audited local datasets.

The fixed decision boundary remains `p_attack = 0.5` with argmax ties assigned to benign. The descriptive near-threshold band is **[0.40, 0.60], inclusive**. It is symmetric around 0.5 and was not tuned.

Machine-readable evidence:

- `results/h2_mini_score_saturation_analysis.json`
- `results/h2_mini_family_separability_analysis.json`
- `results/h2_mini_checkpoint_lineage_comparison.json`

## Ceiling diagnosis

The diagnosis is **C: both malicious-side and benign-side margin saturation**.

| Evidence | Result | Meaning |
|---|---:|---|
| First-acquisition malicious observations | 1,440 | 4 families × 2 arms × 3 seeds × 60 test attacks |
| Correct at first acquisition | 100% | classification ceiling |
| `p_attack >= 0.95` | 100% | all attacks have a large positive margin |
| `p_attack >= 0.99` | 99.722% | nearly all attacks are at the extreme end |
| Malicious near [0.40, 0.60] | 0% | no boundary cases |
| All available malicious p01 / median | 0.992188 / 1.000000 | strong malicious margin saturation |
| All available benign correct | 99.665% | near-ceiling benign classification |
| Benign `p_attack <= 0.10` | 99.603% | strong negative margin |
| Benign `p_attack <= 0.01` | 98.864% | extreme negative margin for almost all observations |
| Benign near [0.40, 0.60] | 0.0616% | almost no boundary mass |
| All available benign p99 / median | 0.011528 / 0.000054 | central benign distribution is far from attacks |

Rare high-confidence benign errors, mainly in B-short trajectories, make the total support overlap: some benign scores reach 1.0. That does not contradict the quantile separation. The central 98–99% of benign observations and essentially all malicious observations occupy opposite extremes.

Correctness and margin saturation are separate findings. Correctness is at ceiling, and the probability distributions show that this is not merely a threshold artifact. For AF1–AF4, acquisition-to-final correctness changes by 0 percentage points in every arm and seed. Mean paired malicious-score movement ranges from 0 to +0.007756; none moves toward the boundary enough to change a label. The stream therefore contains little current-task versus old-task conflict.

This is a **validated negative benchmark-readiness result showing ceiling saturation and insufficient retention pressure**. It is not a failed experiment.

## AF1–AF4 data separability

Each H2-mini family has 290 malicious records across train/validation/test, 290 unique payloads, 290 unique base documents, and balanced head/middle/tail positions (97/96/97). Median payload length is 9–10 Unicode word tokens.

| Family | Construction | Full template IDs | Payloads | Base documents |
|---|---|---:|---:|---:|
| AF1 | Naive | 1 | 290 | 290 |
| AF2 | Ignore | 10 | 290 | 290 |
| AF3 | Escape-Separation | 38 | 290 | 290 |
| AF4 | Fake Completion | 1 | 290 | 290 |

Identity controls are strong: every AF1–AF4 pair has zero exact payload, full-template-ID, base-document-ID, and normalized-full-text intersections. Surface vocabularies still overlap: pairwise corpus token-vocabulary Jaccard is 0.277–0.310 and bigram Jaccard is 0.119–0.129. At the individual-payload level, cross-family TF-IDF cosine is low (means 0.0173–0.0183; p95 0.0636–0.0666), because payload instructions are diverse. Frequent phrases such as “the following” and “the given” recur across families.

The identity disjointness did not force distinct decision boundaries. All families share the same SQuAD-derived document format, Alpaca-derived instruction payload lineage, appended/inserted instruction semantics, binary label, and position scheme. A broad “instruction-like span inside a document” boundary can solve them simultaneously. This is supported by the score ceiling, but it remains an inference: lexical statistics do not reveal the model's causal features.

## Benign source and format comparison

| Pool | Pinned source | n | Language | DeBERTa token median (p95) | Format | Selected cue findings |
|---|---|---:|---|---:|---|---|
| B0 | OASST1 `fdf72ae…` (Apache-2.0) | 1,160 | 18 recorded labels; en 399, es 475, ru 82 | 23 (75) | 98.5% ≤64 words; short ordinary prompts | instruction-verb cue in 13.5% |
| BH | OR-Bench `e36d8b8…` (CC BY 4.0) | 870 | English only | 25 (32) | 100% ≤64 words; reviewed hard benign | prompt/system cue 13.3%; security cue 8.0% |
| B-short | OASST1 `fdf72ae…` (Apache-2.0) | 120 | 42 English, 78 non-English | 23 (51) | short ordinary external control | no security cue under the fixed lexicon |
| NotInject | `847ae76…` (dataset-card MIT) | 339 | official rows lack a language field | 21 (36) | short benign trigger-bearing prompts | instruction cue 39.5%; prompt/system 12.1% |

All six normalized exact-text intersections between B0, BH, B-short, and NotInject are zero. BH and NotInject have the largest token-vocabulary Jaccard, 0.194; all other pairs are 0.069–0.097. These statistics show source, language, format, and cue-composition differences. They do not causally isolate “semantic hard benign.” The H2m-C0 versus H2m-F comparison changes OASST1 multilingual ordinary prompts to English OR-Bench reviewed prompts, so source, language, and semantic hardness remain bundled.

## Why NotInject changed from about 96% FPR to about 0%

The earlier H2-D1 and H2-mini evaluations use the same 339-record NotInject revision and the same fixed decision rule, but they evaluate different checkpoint lineages.

| Property | Earlier H2-D1 checkpoint family | H2-mini checkpoint family |
|---|---|---|
| Per-stage train rows | 3,000 (1,500/class) | 400 (200/class) |
| Benign train context | Chen/SQuAD-derived long clean documents | Stage 1 reviewed OASST1 B0; later C0=B0 or F=OR-Bench BH |
| Malicious context | full-scale Chen-derived AF1–AF4 | down-selected Chen-derived AF1–AF4 |
| Stream start | pinned base at start of earlier full stream | fresh pinned base for every arm/seed |
| Primary Stage-1 NotInject FPR | 96.067% ± 1.802 pp | 0.098% ± 0.152 pp across the six paired Stage-1 observations |
| Final NotInject FPR | primary 84.464% ± 8.123; secondary 95.772% ± 4.506 | 0% in all six final checkpoints |

Confidence is **high** that the contrast is a checkpoint-lineage and training-distribution effect, because the evaluation bank is fixed while the training regimes differ. Confidence is **moderate** that short ordinary-benign B0 exposure is a major contributor: the collapse is already present after shared Stage 1, before H2m-F sees BH, so hard-benign pressure is not necessary. Confidence is **low** for assigning a single cause among B0 source/format, the 7.5× smaller stage, and fresh initialization, because those factors changed together. A matched intervention is needed for attribution.

## AF5 status

AF5 Combined is already present locally in `data/processed/chen_attack_family_shift.jsonl`. It contains 3,660 total rows: train 3,000, validation 300, test 360, each split balanced by label. Its 1,830 malicious rows have 1,830 unique payloads, 190 full templates, 1,830 unique base documents, and exactly 610 head, 610 middle, and 610 tail injections.

AF5 combines Ignore, Escape-Separation, and Fake Completion components in every malicious record. Cross-stage exact text, base-document identity, payload identity, full-template identity, and normalized-template identity are zero under the frozen audit, but component reuse is deliberate. **AF5 is not family-disjoint.** Its correct description is **compositional/combined attack shift**.

The source, deterministic labels, local materialization, hashes, and grouping evidence can be reused without new content labeling. A future course-scale subset still needs a prediction-independent hash-only assignment, leakage/count audit, and human approval before materialization.
'''


DESIGNS = r'''# Stronger stream design options

## Fixed protocol for every option

The detector remains `microsoft/deberta-v3-base` revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`. Labels remain 0=benign and 1=indirect prompt injection; `max_length=1280`; seeds are 17, 42, and 2026; one epoch, batch size 4, learning rate `1e-5`, constant schedule, optimizer reset per stage, model weights carried forward, and current-stage-only training remain fixed. The threshold and evaluation protocol remain fixed.

Every stage uses exactly 200 malicious + 200 benign train records, 30 + 30 validation records, and 60 + 60 test records. Per arm this is 2,320 rows and 290 unique records per class per stage. Increasing volume is not the intervention.

The unchanged acquisition gate is attack TPR ≥85%, benign TNR ≥85%, and paired macro-F1 ≥85% in at least 2/3 seeds. Retention is measured from first acquisition. The existing R0/R1/R2 criteria remain unchanged.

## Candidate A — compositional attack shift

Candidate A uses a clean component hierarchy and two matched benign arms:

| Stage | Malicious family | A-C0 benign | A-F benign |
|---|---|---|---|
| 1 | AF2 Ignore | B0-1 | same B0-1 |
| 2 | AF3 Escape-Separation | B0-2 | BH-1 |
| 3 | AF4 Fake Completion | B0-3 | BH-2 |
| 4 | AF5 Combined | B0-4 | BH-3 |

AF2, AF3, and AF4 expose all three AF5 components before their combination. AF1 is omitted because it is not an AF5 component and its ease is already established. AF5 is called a compositional/combined shift and never a family-disjoint family.

The arms share malicious IDs, positions, splits, stage sizes, labels, seeds, initialization, optimization, and evaluation. The one intended arm difference is the post-Stage-1 benign curriculum: ordinary OASST1 B0 versus reviewed OR-Bench BH. Source and language remain residual parts of that intervention.

Scientific value: AF5 can test whether composition changes confidence/acquisition after all components were seen. Limitation: component reuse can create positive transfer and may further reduce, rather than increase, forgetting. Repeating the H2-mini B0/BH contrast also has a substantial chance of preserving the ceiling.

Workload: reuse the existing 290-record AF2–AF4 assignments, all B0/BH assignments, external banks, and review evidence. Select 290 AF5 records by a predeclared split/position/template-aware hash rule that never reads predictions. No new content labels are needed. Hash-assignment and audit review is estimated at 0.5–1.0 person-hour.

Compute: 2 arms × 3 seeds = 6 runs and 24 stage-end checkpoints. H2-mini used 1,258.0 stage-runtime seconds and 1,556.0 cumulative run-wall seconds for this shape. Project 26–39 cumulative GPU-minutes including a 1.5× uncertainty allowance. A float32 184.4M-parameter model is about 0.738 GB; 24 stage-end model snapshots imply roughly 17.7 GB plus small metadata, as a projection rather than a storage audit.

## Candidate B — matched format/context plus compositional attack shift

Candidate B is the recommended stronger pilot. All three arms share the same AF2→AF3→AF4→AF5 malicious assignment and a common Stage-1 Chen-clean anchor.

| Stage | Malicious | B-DOC stable document control | B-SHORT ordinary-format control | B-HARD evolving security arm |
|---|---|---|---|---|
| 1 | AF2 Ignore | paired Chen-clean documents | same paired Chen-clean documents | same paired Chen-clean documents |
| 2 | AF3 Escape-Separation | paired Chen-clean documents | B0-2 | same B0-2 |
| 3 | AF4 Fake Completion | paired Chen-clean documents | B0-3 | BH-1 |
| 4 | AF5 Combined | paired Chen-clean documents | B0-4 | BH-2 |

The paired Chen-clean controls use the exact base-document IDs and split of the selected malicious record with the injection removed. This changes no source document within a pair and is already supported by the accepted Chen-derived construction; it remains a deterministic reconstruction with the existing rights caveats.

The design gives two predeclared matched contrasts:

1. **B-DOC vs B-SHORT:** identical malicious stream and Stage 1; the main pressure difference is transition from stable document-context benign examples to ordinary short prompts.
2. **B-SHORT vs B-HARD:** identical malicious stream and Stages 1–2; the main pressure difference at Stages 3–4 is ordinary short benign versus reviewed hard-benign training.

The second contrast does not purely isolate semantics because OASST1 is multilingual and OR-Bench is English-only. It controls format more closely than the document comparison, while source and language remain declared confounds. The three-arm design is preferred over a single two-arm source swap because it separates the large document-to-short transition from the later hard-benign package.

Workload: reuse AF2–AF4, B0, BH, B-short, NotInject, all existing human judgments, and all grouping/provenance manifests. AF5 needs one new hash-only subset assignment; Chen-clean counterparts need an identity mapping and audit. No new source download or content labeling is required. Estimated human work is 1–2 person-hours for manifest/provenance review, within the stated preference. Engineering/audit work is about one day.

Compute: 3 arms × 3 seeds = 9 runs and 36 stage-end checkpoints. Linear projection from H2-mini is 38.9 cumulative run-wall minutes; allow 39–58 GPU-minutes because document-control batches may be longer. The model-snapshot projection is about 26.6 GB plus small metadata. End-to-end execution, validation, aggregation, and report rendering should fit comfortably within one working day once materialization is approved.

Why Candidate B is preferable: the lineage evidence independently identifies benign context/format exposure as a plausible driver of the earlier NotInject contrast; the malicious order follows the AF5 component hierarchy; and the two matched contrasts preserve interpretability. It has a better chance than Candidate A of generating decision-boundary interference without weakening the detector or mining errors.

## Optional Candidate C — reverse hierarchy control

A predeclared lower-cost order control could use AF5→AF4→AF3→AF2 with a single stable benign arm. Its justification would be construction semantics: composition before components, not observed H2-mini outcomes. It would require 3 runs, 12 checkpoints, roughly 13–20 GPU-minutes, and about 8.9 GB of model snapshots.

**Recommendation: NO-GO for Candidate C now.** Composition-first training can make later single components easier through positive transfer, and the project already has a completed three-seed reverse-order AF1–AF4 control. It adds limited value relative to the two-week deadline.

## Reuse and provenance contract

| Asset | Reuse | Source/revision/license | Rule |
|---|---|---|---|
| AF1–AF4 | AF2–AF4 reused; AF1 retained as prior evidence | Chen `1e9e0f0…`; repository MIT; underlying SQuAD CC BY-SA 4.0 and Alpaca payloads CC BY-NC 4.0 | local research only; deterministic reconstruction caveat |
| AF5 | yes, after new hash-only subset approval | same Chen lineage | call compositional; maintain split/position/template grouping |
| B0 | yes | OASST1 `fdf72ae…`, Apache-2.0 | reuse reviewed IDs and native/tree/semantic/template groups |
| BH | yes | OR-Bench `e36d8b8…`, CC BY 4.0 | reuse reviewed IDs; retain unavailable official rewrite-parent caveat |
| B-short | evaluation only | OASST1 `fdf72ae…`, Apache-2.0 | never train, tune, select, or rank methods on it |
| NotInject | evaluation only | `847ae76…`, dataset-card MIT | never train, tune, select, or rank methods on it |
| Chen clean documents | Candidate B control | Chen/SQuAD lineage above | pair by exact base ID; injection absent; keep local |

Every proposal must freeze source/native ID, revision, raw and normalized hashes, base/parent/group identity, label, stage, split, position, payload/template/component IDs, and source license. No exact, normalized/model-view, base-document, parent, semantic/template group, or conversation tree may cross train/validation/test or stages where disjointness is required. AF5 component overlap is allowed only as the declared intervention; identity-level overlap remains forbidden. Existing H2-mini overlap is recorded, and reused records must be named rather than presented as independent replication.

Future assignments must be based on source metadata and deterministic hashing before any new model run. Existing predictions may not choose IDs, order, difficulty, thresholds, or hyperparameters.

## Go/no-go and future replay scope

Candidate B may advance only after a human approves its hash-only assignment and audit, then separately approves materialization, then separately approves no-replay training. After all nine no-replay runs, apply the frozen acquisition and R0/R1/R2 gates.

- **GO to replay proposal:** only if acquisition passes and the already frozen R1 or R2 criteria are met after matched-control analysis.
- **NO-GO:** if acquisition fails, effects are source/optimization artifacts, or status remains R0. Stop experimental escalation and finish the negative-readiness report.

If later eligible, the minimal method set remains no replay, Reservoir ER, DER++, and security-aware replay, with optional loss-aware/BRS/LARS only if time remains. No method is implemented here, and the common numerical memory budget remains unselected.

## Exact recommended next prompt

> Approve Candidate B benchmark preparation and audit only. Keep `microsoft/deberta-v3-base` revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3` fixed. Prepare a prediction-independent hash-only proposal for the AF2→AF3→AF4→AF5 compositional malicious stream at 200/30/60 records per class for train/validation/test and three matched benign arms: B-DOC with paired Chen-clean documents at all stages; B-SHORT with the shared Chen-clean Stage 1 then B0-2/B0-3/B0-4; and B-HARD with the shared Chen-clean Stage 1, shared B0-2 Stage 2, then BH-1/BH-2. Reuse the pinned audited sources, existing human judgments, B-short and NotInject as evaluation-only banks, and all current grouping controls. Audit exact counts, IDs, hashes, source revisions/licenses, pair mappings, split/stage leakage, AF5 component overlap, and prior-benchmark reuse. Do not materialize JSONL, train, rerun H2-mini, implement replay or selective adaptation, select a numerical memory budget, or change R0/R1/R2. Stop for human review with the exact manifest and validation results.
'''


OPTIONS = r'''# Project completion options

## Decision context

H2-mini is a sound negative result: all acquisition gates pass, but both classes are score-saturated and no R1 retention threshold is reached. About two weeks remain. The choice is whether to finish around that result or spend one tightly gated week on one independently predeclared stronger pilot.

| Dimension | Path 1 — stop after H2-mini | Path 2 — one Candidate B pilot |
|---|---|---|
| Scientific value | Clear evidence that a carefully audited AF1–AF4/B0/BH stream is too easy for replay evaluation | Tests a concrete, independently motivated explanation: compositional attack progression plus benign context/format pressure |
| Novelty | Strong benchmark-readiness and negative-result discipline | Adds a matched three-arm intervention and may produce a valid replay entry case |
| Main risk | Report may be viewed as stopping before a challenging stream | Pilot may remain saturated; source/language confounds may limit attribution |
| Expected active time | 4–6 working days for analysis, figures, writing, and reproducibility pass | 1–2 days audit/materialization gates, <1 GPU-hour projected, Day-5 decision, then report or conditional replay week |
| Likely report strength | Methodologically careful negative result with complete audit trail | Stronger if either it remains R0 for a well-explained reason or reaches a prospective replay gate |
| Instructor alignment | Preserves fixed detector and refuses unjustified replay | Preserves fixed detector and tests replay only if prospective evidence supports it |
| Comfortable two-week finish probability | about 95% planning estimate | about 80–85% if approvals occur by Day 2 and the Day-5 stop is enforced |

## Recommendation

Choose **Path 2 with Candidate B**, subject to the staged approvals and hard Day-5 decision. The projected GPU cost is small relative to the time remaining, the design reuses audited data and judgments, and it addresses a concrete weakness revealed by the checkpoint-lineage comparison. Do not run Candidate A and B both; that would dilute the schedule. Candidate A remains the fallback if the Chen-clean pairing audit fails.

Switch to Path 1 immediately if the hash-only design is not approved by the end of Day 2, exact pairing/grouping cannot pass without new large annotation, or the no-replay pilot remains R0 on Day 5. Under Path 1, the current negative readiness result is the endpoint, not an incomplete replay study.

Replay has one explicit gate: **GO only after human review confirms R1/R2 under the unchanged criteria; otherwise NO-GO and stop experimental escalation.** The numerical replay memory budget is not selected in either path at this stage.
'''


PLAN = r'''# Two-week experiment plan

This schedule assumes Candidate B is selected. Runtime projections use the observed H2-mini totals: 1,258.0 seconds across 24 stages and 1,556.0 cumulative run-wall seconds across six runs. Candidate B has nine runs and 36 checkpoints, with a 39–58 GPU-minute planning range.

| Day | Work | Exit condition |
|---|---|---|
| 1 | Human reviews this diagnosis and Candidate B design | approve hash-only preparation, or choose Path 1 |
| 2 | Build hash-only AF5/Chen-pair/B0/BH assignments; run count, provenance, license, grouping, leakage, and prior-overlap audits | human reviews exact manifest; no JSONL yet |
| 3 | After separate approval, materialize locally and validate hashes/counts; run technical preflight | human approves no-replay execution |
| 3–4 | Run 3 arms × 3 seeds, 4 stages; validate 36 checkpoints and all predictions; aggregate acquisition, retention, class-direction and external-bank metrics | complete no-replay evidence package |
| 5 | Human applies unchanged acquisition and R0/R1/R2 gates | hard GO/NO-GO |
| 6–9 if R1/R2 | Human first approves one common numerical budget and the minimal no-replay/Reservoir ER/DER++/security-aware protocol; implement and run only that comparison | validated method-isolated result |
| 6–9 if R0 | stop experimental escalation; finalize negative-readiness narrative, figures, and reproducibility bundle | no replay branch |
| 10 | Only if justified, run a predeclared control/ablation; otherwise reserve for error checking | no opportunistic method expansion |
| 11–12 | Analysis, uncertainty, class-direction plots, lineage and limitations | frozen figures/tables |
| 13 | Full artifact and report consistency pass | all hashes/references/checks pass |
| 14 | Final write-up and contingency | submission-ready project |

The schedule has three separate approvals before compute: hash-only design, local materialization, and no-replay execution. Replay remains a fourth approval after Day 5. If any gate slips by more than one working day, default to Path 1.

The replay memory budget remains deliberately unselected. If R1/R2 is reached, a later proposal may derive one common cap from per-stage train size, historical-stage count, retained-example percentage, RTX 2060 feasibility, and identical accounting across methods. It cannot be finalized from the current R0 evidence.
'''


READINESS_BLOCK = r'''<!-- H2_MINI_BENCHMARK_DESIGN_20261005:BEGIN -->
## Benchmark-readiness analysis after H2-mini — 2026-10-05

H2-mini remains **R0 active; R1/R2 not entered**. Existing predictions show both malicious and benign score-margin saturation: first-acquisition attacks are 100% correct, all are at least 0.95, 99.722% are at least 0.99, and no malicious example is in [0.40,0.60]; 98.864% of benign trajectory observations are at or below 0.01. AF1–AF4 correctness never changes from acquisition to final.

Candidate B is recommended for human design review because it adds a matched document→short→hard-benign context curriculum while holding an AF2→AF3→AF4→AF5 compositional malicious stream fixed. This recommendation does not construct or authorize that benchmark. The prospective acquisition and R0/R1/R2 rules are unchanged. Replay, selective adaptation, and numerical memory-budget selection remain blocked.
<!-- H2_MINI_BENCHMARK_DESIGN_20261005:END -->'''


COMPARISON_BLOCK = r'''<!-- H2_MINI_BENCHMARK_DESIGN_20261005:BEGIN -->
## Replay comparison remains conditional — 2026-10-05

The completed H2-mini analysis supplies no replay entry: status stays R0. A stronger no-replay Candidate B pilot is recommended for separate audit and approval, but no benchmark has been materialized and no method has been implemented.

Only if a future human-reviewed pilot satisfies the unchanged R1/R2 rules may a later protocol select one common numerical memory budget and compare no replay, Reservoir ER, DER++, and security-aware replay. Loss-aware/BRS/LARS-style replay is optional only after the minimal set and only if time remains. Detector, stream, seeds, optimization, evaluation, and eventual memory accounting must be identical across methods.
<!-- H2_MINI_BENCHMARK_DESIGN_20261005:END -->'''


DECISION_BLOCK = r'''<!-- H2_MINI_BENCHMARK_DESIGN_20261005:BEGIN -->
## D046 — Diagnose H2-mini ceiling and recommend one stronger design (2026-10-05)

Accept H2-mini as a validated negative benchmark-readiness result. Existing predictions show both-class margin saturation and minimal current-versus-old task conflict. Keep R0 active and all replay work blocked. Recommend Path 2 with the three-arm Candidate B AF2→AF3→AF4→AF5 design for a later, separately approved hash-only audit because it provides matched document, ordinary-short, and hard-benign contrasts. Candidate A remains a lower-cost fallback; Candidate C is no-go. No benchmark was constructed or materialized, no model was loaded or trained, and no numerical memory budget was chosen.
<!-- H2_MINI_BENCHMARK_DESIGN_20261005:END -->'''


CHANGE_BLOCK = r'''<!-- H2_MINI_BENCHMARK_DESIGN_20261005:BEGIN -->
## C034 — Add H2-mini ceiling and stronger-stream design analysis (2026-10-05)

Added an analysis-only script and three machine-readable reports for all checkpoint score distributions, AF1–AF5/benign model-independent separability, and H2-D1 versus H2-mini lineage. Added four design/completion documents and synchronized readiness, comparison, decision, risk, question, and history records. Validation reruns the deterministic analysis, parses all outputs, checks all 24 checkpoint summaries and 10 family pairs, and confirms the five frozen input hashes are unchanged. No frozen result/manifests, raw benchmark, checkpoint, or scientific metric changed.
<!-- H2_MINI_BENCHMARK_DESIGN_20261005:END -->'''


RISK_BLOCK = r'''<!-- H2_MINI_BENCHMARK_DESIGN_20261005:BEGIN -->
## Risks for the proposed stronger pilot — 2026-10-05

| ID | Risk | Severity | Mitigation/status |
|---|---|---|---|
| R076 | AF5 component reuse may produce positive transfer and keep the stream saturated | high | call it compositional, expose AF2/AF3/AF4 first, retain hard Day-5 stop |
| R077 | Candidate B's OASST1-versus-OR-Bench contrast still bundles source, language, and semantic hardness | high | use B-DOC/B-SHORT/B-HARD matched arms; report package effect, not pure semantics |
| R078 | Chen clean controls and injected examples can expose a broad document/instruction boundary | high | diagnose margins; do not weaken detector or mine errors; stop if still R0 |
| R079 | Reusing prior IDs is not independent replication | medium | disclose exact prior overlap and call the run a controlled follow-up |
| R080 | New AF5 down-selection could become test-guided | high | freeze a metadata/hash rule before any new run; prohibit all existing prediction inputs |
| R081 | Three arms project about 26.6 GB of stage-end model files | medium | plan capacity manually; no automatic size audit or cleanup; keep required stage endpoints |
| R082 | Two-week schedule could be consumed by approvals rather than compute | high | separate gates but default to Path 1 if design approval is not complete by Day 2 |

Underlying Chen/SQuAD/Alpaca rights and reconstruction caveats, unavailable official OR-Bench rewrite-parent lineage, B-short/NotInject mismatch, three-seed uncertainty, and fixed 60-per-class tests remain active.
<!-- H2_MINI_BENCHMARK_DESIGN_20261005:END -->'''


QUESTION_BLOCK = r'''<!-- H2_MINI_BENCHMARK_DESIGN_20261005:BEGIN -->
## Open decisions after ceiling analysis — 2026-10-05

| ID | Question | Current recommendation | Status |
|---|---|---|---|
| Q050 | Approve Path 2 and Candidate B hash-only preparation/audit? | Yes, if approved by end of Day 2; otherwise Path 1 | open human decision |
| Q051 | If Chen-clean pair mapping fails exact grouping/leakage checks, use Candidate A? | Yes as fallback; do not repair by loosening controls | conditional |
| Q052 | May Candidate B be materialized? | No; decide only after exact hash-only manifest review | blocked |
| Q053 | May Candidate B no-replay training run? | No; separate approval after materialization/preflight | blocked |
| Q054 | May replay or selective adaptation begin? | No; only after future human-reviewed R1/R2 entry | blocked at R0 |
| Q055 | What numerical replay memory budget should be used? | Remains unselected until replay is eligible | deferred |

The exact next prompt is recorded in `docs/stronger_stream_design_options.md`.
<!-- H2_MINI_BENCHMARK_DESIGN_20261005:END -->'''


def main():
    write(DOCS / "h2_mini_ceiling_analysis.md", CEILING)
    write(DOCS / "stronger_stream_design_options.md", DESIGNS)
    write(DOCS / "project_completion_options.md", OPTIONS)
    write(DOCS / "two_week_experiment_plan.md", PLAN)

    prepend_block(DOCS / "replay_readiness_review.md", READINESS_BLOCK)
    prepend_block(DOCS / "replay_entry_criteria.md", READINESS_BLOCK)
    prepend_block(DOCS / "replay_comparison_protocol.md", COMPARISON_BLOCK)
    prepend_block(DOCS / "decision_log.md", DECISION_BLOCK)
    prepend_block(DOCS / "change_log.md", CHANGE_BLOCK)
    prepend_block(DOCS / "assumptions_and_risks.md", RISK_BLOCK)
    prepend_block(DOCS / "open_questions.md", QUESTION_BLOCK)

    now = datetime.now(timezone.utc).isoformat()
    append_jsonl_once(HISTORY / "decisions.jsonl", {
        "date": now,
        "id": "D046",
        "decision": "Keep R0 after H2-mini ceiling analysis and recommend Candidate B for a separate hash-only design audit",
        "diagnosis": "both malicious-side and benign-side margin saturation; insufficient retention pressure",
        "recommended_path": "Path 2, one predeclared three-arm Candidate B pilot with a hard Day-5 R0/R1/R2 decision",
        "candidate_B": {"malicious_order": ["AF2", "AF3", "AF4", "AF5"], "arms": ["B-DOC", "B-SHORT", "B-HARD"], "runs_if_later_approved": 9, "checkpoints_if_later_approved": 36},
        "readiness": {"R0": "active", "R1": "not entered", "R2": "not entered"},
        "training": False,
        "materialization": False,
        "replay": False,
        "selective_adaptation": False,
        "numerical_memory_budget": None,
    })
    append_jsonl_once(HISTORY / "changes.jsonl", {
        "date": now,
        "id": "C034",
        "decision_id": "D046",
        "reason": "Diagnose accepted H2-mini ceiling saturation and design one stronger interpretable pilot without constructing or training it",
        "files_changed": [
            "scripts/analyze_h2_mini_benchmark_readiness.py",
            "results/h2_mini_score_saturation_analysis.json",
            "results/h2_mini_family_separability_analysis.json",
            "results/h2_mini_checkpoint_lineage_comparison.json",
            "docs/h2_mini_ceiling_analysis.md",
            "docs/stronger_stream_design_options.md",
            "docs/project_completion_options.md",
            "docs/two_week_experiment_plan.md",
            "docs/replay_entry_criteria.md",
            "docs/replay_readiness_review.md",
            "docs/replay_comparison_protocol.md",
            "docs/decision_log.md",
            "docs/change_log.md",
            "docs/assumptions_and_risks.md",
            "docs/open_questions.md",
            "history/decisions.jsonl",
            "history/changes.jsonl",
        ],
        "validation": [
            "24 checkpoint summaries and 10 family-pair analyses generated",
            "Five frozen H2-mini/materialization input hashes recorded and unchanged",
            "No detector load, inference, training, materialization, replay, or memory-budget selection",
        ],
        "scientific_behavior_changed": False,
    })
    print(json.dumps({"status": "PASS", "documents_written": 11, "history_ids": ["D046", "C034"]}, indent=2))


if __name__ == "__main__":
    main()
