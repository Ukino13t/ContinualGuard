from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(r"E:\ContinualGuard")
DOCS = ROOT / "docs"
RESULTS = ROOT / "results"
MANIFESTS = ROOT / "data" / "manifests" / "stronger_pilot"
MARKER = "STRONGER_PILOT_CANDIDATE_B_20261005"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def upsert(path: Path, body: str) -> None:
    begin = f"<!-- {MARKER}:BEGIN -->"
    end = f"<!-- {MARKER}:END -->"
    block = f"{begin}\n{body.rstrip()}\n{end}"
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if begin in old and end in old:
        prefix, remainder = old.split(begin, 1)
        _, suffix = remainder.split(end, 1)
        new = prefix.rstrip() + "\n\n" + block + suffix
    else:
        new = old.rstrip() + ("\n\n" if old.strip() else "") + block + "\n"
    write(path, new)


capacity = load_json(RESULTS / "stronger_pilot_capacity_audit.json")
validation = load_json(RESULTS / "stronger_pilot_assignment_validation.json")
assert capacity["status"] == "PASS"
assert validation["status"] == "READY_FOR_HUMAN_MATERIALIZATION_APPROVAL"
assert validation["all_checks_pass"] is True

proposal_hashes = validation["proposal_hashes"]
for rel, expected in proposal_hashes.items():
    assert sha256(ROOT / rel) == expected, rel

protocol = r"""# Candidate B stronger security-bidirectional pilot protocol

## Status and scope

Candidate B is selected. This document fixes the benchmark design before raw-text materialization or model execution. The current state is **READY FOR HUMAN MATERIALIZATION APPROVAL**. No stronger-pilot JSONL has been materialized, no model has been trained, no replay or selective adaptation has been implemented, and the replay memory budget remains unselected.

The detector remains `microsoft/deberta-v3-base` at revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`, with a fixed binary head, label 0 = benign, label 1 = indirect prompt injection, and `max_length=1280`. The benchmark must not be tuned to produce forgetting.

## Exact three-arm benchmark

| Stage | Shared malicious half | B-DOC benign half | B-SHORT benign half | B-HARD benign half |
|---|---|---|---|---|
| 1 | AF2 Ignore | D0-1 | exact same D0-1 | exact same D0-1 |
| 2 | AF3 Escape-Separation | D0-2 | B0-2 | exact same B0-2 |
| 3 | AF4 Fake Completion | D0-3 | B0-3 | BH-1 |
| 4 | AF5 Combined | D0-4 | B0-4 | BH-2 |

Each stage has 200 malicious and 200 benign training records, 30 per class for validation, and 60 per class for test: 580 records per stage, 2,320 memberships per arm, and 6,960 memberships across the three arms. Intentional cross-arm anchors are counted once when reporting 3,770 unique selected records.

For each malicious stage, the head/middle/tail targets are 67/66/67 for training, 10/10/10 for validation, and 20/20/20 for test. The 1,160 malicious assignments are selected once and copied exactly across arms, including IDs, labels, family, stage, split, position, and hashes.

## Scientific interpretation

AF5 contains Ignore, Escape-Separation, and Fake Completion components. Stage 4 is therefore a **compositional/combined attack shift**, not a fourth independent or family-disjoint attack family. The sequence tests stability after simple components are later combined.

The predeclared contrasts are:

1. **B-SHORT minus B-DOC:** effect associated with changing from Chen-derived document context to OASST1 ordinary short prompts.
2. **B-HARD minus B-SHORT:** incremental effect associated with introducing reviewed OR-Bench hard-benign pressure after a common Stage 2.
3. **Within-arm retention:** malicious attack-TPR retention from first acquisition and benign FPR increase from first acquisition, with macro-F1, accuracy, TNR, peak forgetting, and BWT as secondary measures.

These comparisons are matched more closely than Candidate A because all three arms start identically, B-SHORT and B-HARD remain identical through Stage 2, and B-DOC provides a stable document-context control. Source, language, domain, and format still differ; B-HARD minus B-SHORT does not identify a pure semantic-hardness effect. BH is a source-defined hard-benign group, not an ordinal difficulty label.

## Acquisition and replay-readiness gates

Each stage must reach attack TPR >=85%, benign TNR >=85%, and paired macro-F1 >=85% in at least two of three seeds. The existing R0/R1/R2 criteria are frozen. R1 continues to require a harmful direction in at least two seeds, mean degradation of at least 2 percentage points on a predeclared primary metric, and persistent or localized matched evidence. R2 retains harmful direction in all three seeds, at least 5 percentage points mean degradation, replication or matched evidence, and no acquisition, threshold, or label confound.

NotInject (339 records) and B-short (120 records) remain frozen evaluation-only banks. They may not affect training, validation, thresholds, selection, balancing, or method choice.
"""

source_audit = f"""# Candidate B source and capacity audit

## Audit outcome

The source, capacity, provenance, license, grouping, and token-length audit passed. Existing reviewed or audited evidence is sufficient; no new content review is required. Human work is limited to an estimated 0.5–1.0 hours for manifest sign-off.

| Pool | Available | Required | Selected | Result |
|---|---:|---:|---:|---|
| D0 clean AF2–AF5 records | 7,320 total; 1,160 eligible exact clean companions | 1,160 | 1,160 | PASS |
| B0 finalized reviewed pool | 1,160 | 870 | 870 | PASS |
| BH finalized reviewed pool | 870 | 580 | 580 | PASS |

All 1,160 selected D0 rows satisfy label 0, no attack family, no injection ID or position, `text == base_document`, and the pinned Chen revision. Each is a distinct clean record paired to the malicious row's base document; raw clean text is not equal to injected malicious text. The shared base document within each stage is intentional matched pairing. No D0 base document crosses a stage or split.

## Sources and rights

| Source | Pinned revision | License evidence | Local evidence and SHA-256 | Role and caveat |
|---|---|---|---|---|
| Chen-derived attack-family reconstruction | `1e9e0f00179adc154e9ca991b9cd3206c99f9588` | repository MIT; SQuAD CC BY-SA 4.0; Alpaca-derived payloads CC BY-NC 4.0 | `data/processed/chen_attack_family_shift.jsonl`, `4b4efaf7640f31f5f12af2f2b3a40b91bbb71789c91c059d21ce4af1e6a9e43a` | malicious AF2–AF5 and clean D0; deterministic reconstruction, not an exact paper-time reproduction |
| OASST1 | `fdf72ae0827c1cda404aff25b6603abec9e3399b` | Apache-2.0 | local gzip `286a6e9a5a413b3272ae9c0b5a20d327983dea1c24342ae28cb244a6da65185c` | reviewed B0 and B-short; multilingual source distribution |
| OR-Bench | `e36d8b80e81837c8a8f264bbb2a49f1b32c7e272` | CC BY 4.0 | local CSV `22e956020cdfdd43c328f02441f003c933ee4e7083d02d9d24e414e4b42ae2e6` | reviewed BH; official rewrite-parent map remains unavailable, so project grouping is conservative |
| NotInject | `847ae76cf8fea5ed325429e569ae8cfef022d2e0` | MIT in dataset card | records manifest `c3b3cdedcf00f84b24c72b27bbf7d5ff24fb4a19c6a0faf6d2c103e1767a9fde`; pinned raw parquet hashes are recorded in the machine audit | 339-record evaluation-only external bank |

## AF5 audit

AF5 contains exactly 3,660 records: train 3,000, validation 300, and test 360, equally divided between benign and malicious. Its malicious half has 1,830 records, 1,830 payload IDs, 1,830 base documents, 190 complete templates, and 610 examples at each of head, middle, and tail.

The proposed 290-record AF5 subset has train/validation/test position counts of 67/66/67, 10/10/10, and 20/20/20. Complete-template ownership is split-exclusive for the new AF5 assignment; selected AF5 complete-template cross-split overlap is zero. Against selected AF2–AF4, exact text, base-document ID, payload ID, complete-template ID, and normalized-template hash overlaps are all zero. All 290 selected AF5 malicious records intentionally contain the three AF2/AF3/AF4 components. AF2–AF4 retain their frozen construction-wrapper policy, including repeated family patterns; exact examples, base documents, and payload identities remain isolated.

## Leakage and reuse

All audited cross-stage or cross-split collision counts are zero for applicable native IDs, document/conversation parents, base-document IDs, raw/normalized/model-view hashes, semantic-parent groups, template groups, and payload identities. There is no selected raw or normalized hash overlap with NotInject or B-short.

The stronger pilot contains 3,770 unique selected records. Of these, 2,320 were already present in H2-mini: 870 AF2–AF4 malicious records, 870 B0 records, and 580 BH records. The remaining 1,450 are new to H2-mini: 290 AF5 malicious records and 1,160 D0 records. This is an explicitly reused, predeclared experiment rather than an independent held-out replication. No prior prediction influenced selection.

## Token lengths

Pinned-tokenizer statistics for the 3,770 unique records are min 6, median 119, p95 261, p99 354, and max 491. Counts above 1,280 are zero. Per-arm maxima are 491 and per-arm counts above 1,280 are zero. The audit tokenized with truncation disabled.

## Residual confounds

B-DOC is English SQuAD-style document context; B0 is multilingual OASST1 short-form material; BH is English OR-Bench. Source, language, domain, and format are bundled with the benign interventions. OR-Bench lacks an official parent mapping in the local public artifact. Chen data retains reconstruction and upstream-license caveats. AF5 can benefit from prior exposure to its components. These limits must accompany any later causal language.
"""

assignment_plan = f"""# Candidate B deterministic assignment and materialization plan

## Frozen assignment

- Policy: `candidate-b-hash-assignment-v1-20261005`
- Salt: `stronger-pilot-candidate-b-v1-20261005`
- Selection inputs: source metadata, reviewed eligibility, grouping fields, and deterministic hashes only
- Excluded inputs: detector predictions, margins, errors, losses, NotInject scores, and salt search
- Status: **READY FOR HUMAN MATERIALIZATION APPROVAL**

| Proposal | SHA-256 |
|---|---|
| `stronger_shared_malicious_assignment.json` | `{proposal_hashes['data/manifests/stronger_pilot/stronger_shared_malicious_assignment.json']}` |
| `stronger_b_doc_assignment_proposal.json` | `{proposal_hashes['data/manifests/stronger_pilot/stronger_b_doc_assignment_proposal.json']}` |
| `stronger_b_short_assignment_proposal.json` | `{proposal_hashes['data/manifests/stronger_pilot/stronger_b_short_assignment_proposal.json']}` |
| `stronger_b_hard_assignment_proposal.json` | `{proposal_hashes['data/manifests/stronger_pilot/stronger_b_hard_assignment_proposal.json']}` |

The shared malicious proposal has 1,160 records. Each arm has exactly 2,320 memberships. Total arm memberships are 6,960 and unique selected records are 3,770. The proposal files contain metadata and hashes, not raw text.

## Selection and matching rules

AF2–AF4 reuse the frozen H2-mini assignments. AF5 uses one deterministic assignment: scarce validation template ownership first, then test, then train; within each split, position deficits are filled with a fixed hash tie-break. Complete AF5 templates are split-exclusive. D0 maps each selected malicious record to its audited clean companion. B0 and BH reuse frozen reviewed pools and balance available language, length-bin, category, and descriptive metadata without consulting model behavior.

The three malicious signature sets are identical at 1,160 records each. All arms share the same 290-record D0-1 Stage-1 anchor. B-SHORT and B-HARD share the same 290-record B0-2 Stage-2 anchor. Cross-arm benign identity overlaps are exactly the declared anchors: B-DOC/B-SHORT 290, B-DOC/B-HARD 290, and B-SHORT/B-HARD 580. No other benign identity is shared.

## Future local materialization procedure

Materialization requires a separate human approval. After approval, a dedicated script will:

1. verify all four proposal hashes above and every pinned source/authority-manifest hash;
2. resolve each frozen ID from its pinned local source without changing membership or order;
3. write to a temporary quarantine directory under `data/processed/stronger_pilot/`;
4. produce 36 arm/stage/split JSONL files plus a materialization manifest; external NotInject and B-short remain evaluation-only references to their existing frozen local banks;
5. validate exact counts, labels, stages, source revisions, raw/normalized/model-view hashes, group isolation, matched anchors, malicious positions, and zero token lengths above 1,280;
6. atomically publish the validated directory only after every check passes.

No materialization has occurred in this phase. Materialization approval will not authorize training. A second human review of the materialized hashes and validation must precede the nine no-replay runs.

## Exact recommended next prompt

> Approve Candidate B local materialization only from the frozen metadata-only proposals. Verify proposal SHA-256 values `211f986ef0d2c479d4198c65f6b4ab59789a77aa70d860af2ab0f02dd2f6d6ad`, `b2aae59c201c7302bb85b9da79bc5b841fe7de94490fc6660b30d2aa9ba05f6b`, `e4ecda4fa312976de0692287444da138b011cac93f9e833a4de6cd4e7625ade0`, and `43a10c9de0144ec2f8f6fd5e5bf33eff7390a9f573f4217ce24957acf192c841`; resolve records only from the pinned local sources; write `data/processed/stronger_pilot/` through a temporary quarantine directory and atomic publication; produce the 36 arm/stage/split JSONL files plus a materialization manifest; keep NotInject and B-short as evaluation-only references; and run the full count, label, hash, provenance, grouping, matched-anchor, position, leakage, and no-truncation validation. Do not train, implement replay or selective adaptation, choose a replay memory budget, or change R0/R1/R2. Stop for human review of the materialized benchmark before no-replay execution.
"""

no_replay = r"""# Candidate B future no-replay protocol

## Execution state

This is a predeclared protocol only. No training has started. Replay and selective adaptation remain blocked, and the numerical replay memory budget remains unselected.

## Fixed execution factors

- Detector: `microsoft/deberta-v3-base`
- Revision: `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`
- Binary labels: 0 benign, 1 indirect prompt injection
- Maximum length: 1,280
- Seeds: 17, 42, 2026
- Training: one epoch per stage, batch size 4, learning rate `1e-5`, constant scheduler
- State: model weights carry across stages; optimizer resets at every stage
- Data access: current-stage training data only; no historical sample, replay, selective adaptation, or external-bank training

There are three arms times three seeds, for nine sequential runs and 36 stage-end checkpoints. Each run initializes once from the pinned base model and proceeds through AF2, AF3, AF4, and AF5. At each stage end it evaluates every acquired internal stage plus the frozen NotInject and B-short banks.

## Required artifacts and metrics

Save stage-end checkpoints, raw predictions, confusion counts, per-stage losses, runtime, peak GPU memory, and stage-by-stage matrices for accuracy, macro-F1, attack TPR, benign FPR/TNR, retention, BWT, peak forgetting, and position-specific head/middle/tail attack TPR. Record acquisition separately from retention. The primary retention metrics are attack TPR change from first acquisition and benign FPR increase from first acquisition.

Report per seed and mean ± sample SD. Quantify all changes in percentage points and state whether harmful direction is consistent across seeds. Do not label a decrease catastrophic forgetting without passing the frozen evidence gates. Keep acquisition limits, retention, order-dependent transfer, and optimization instability distinct.

## Compute estimate and stopping rule

H2-mini used 1,556 seconds of cumulative run wall time for six comparable runs. Linear scaling gives 38.9 minutes for nine runs; the planning range is 39–58 GPU-minutes because the document-control batches are longer. The expected 36 float32 model snapshots are about 26.6 GB before optimizer or metadata overhead; this is a planning estimate, not a storage audit. A technical GPU-memory and disk-write preflight remains required after materialization.

On Day 5, apply the unchanged gates. If the result remains R0, stop benchmark escalation, do not implement replay, and finish the project around the negative result. If R1 or R2 is reached, replay remains blocked until a human separately approves a common numerical memory budget and a method-isolated comparison.
"""

write(DOCS / "stronger_pilot_candidate_b_protocol.md", protocol)
write(DOCS / "stronger_pilot_source_audit.md", source_audit)
write(DOCS / "stronger_pilot_assignment_plan.md", assignment_plan)
write(DOCS / "stronger_pilot_no_replay_protocol.md", no_replay)

upsert(DOCS / "stronger_stream_design_options.md", r"""## Candidate B selected and audited

Candidate B is the accepted stronger pilot. The exact three-arm structure, source/capacity audit, prediction-independent hash-only assignments, matched-control verification, leakage audit, and materialization plan are complete. All validation checks pass and the state is **READY FOR HUMAN MATERIALIZATION APPROVAL**.

The design contains 3,770 unique records and 6,960 arm memberships. It reuses 2,320 H2-mini records and adds 1,160 audited Chen-clean D0 records plus 290 AF5 records. AF5 remains a compositional/combined shift. R0 remains active; no raw stronger-pilot JSONL, training, replay, selective adaptation, or numerical memory budget exists.

See `docs/stronger_pilot_candidate_b_protocol.md`, `docs/stronger_pilot_source_audit.md`, and `docs/stronger_pilot_assignment_plan.md`.
""")

upsert(DOCS / "two_week_experiment_plan.md", r"""## Candidate B post-audit schedule

| Day | Work | Gate |
|---|---|---|
| 1 | Human reviews the frozen hash-only proposals and source audit | approve local materialization only |
| 2 | Resolve frozen IDs, materialize locally, and validate all files atomically | separate human approval for no-replay execution |
| 3–4 | Run 3 arms × 3 seeds and validate 36 stage checkpoints | complete no-replay evidence package |
| 5 | Apply unchanged acquisition and R0/R1/R2 criteria | hard GO/NO-GO |

If Candidate B remains R0, stop benchmark escalation and do not implement replay. If it reaches R1 or R2, replay still needs a separate human-approved common memory budget and method protocol. Current planning estimate is 39–58 GPU-minutes; no experiment has run in this phase.
""")

upsert(DOCS / "replay_readiness_review.md", r"""## Candidate B hash-only readiness update

The accepted stronger-pilot design and assignment audit do not change the scientific classification. **R0 remains active; R1 and R2 are not entered.** Replay and selective adaptation remain blocked, and no numerical replay memory budget has been selected.

Candidate B is ready only for human approval of local materialization. Its frozen three-arm controls are designed to separate document-to-short format/source pressure from the incremental hard-benign package while using an identical AF2→AF3→AF4→AF5 malicious stream. Data capacity, matched anchors, provenance, grouping, leakage, and token-length checks pass. These are benchmark-readiness results, not retention evidence.
""")

upsert(DOCS / "replay_comparison_protocol.md", r"""## Stronger-pilot prerequisite

Candidate B has passed hash-only assignment validation but has not been materialized or trained. It cannot enter a replay comparison unless a separately approved no-replay execution passes the unchanged acquisition gates and reaches R1 or R2 under the frozen retention criteria.

If replay later becomes eligible, the detector revision, initialization, Candidate B data, stage order, seeds, one-epoch schedule, learning rate, optimizer-reset policy, evaluation protocol, and one common numerical memory budget must be fixed across methods. The primary changing variable will be replay strategy. The budget remains unselected and no replay code is authorized now.
""")

decision_block = r"""## D047 — Select Candidate B and freeze validated hash-only assignments (2026-10-05)

Candidate B is the accepted stronger security-bidirectional pilot. Freeze the three-arm AF2→AF3→AF4→AF5 specification, deterministic salt and policy, four metadata-only assignment proposals, and validation report. All capacity, matched-control, leakage, provenance, and token-length checks pass. The state is **READY FOR HUMAN MATERIALIZATION APPROVAL**.

This decision does not authorize raw JSONL materialization or model execution. R0 remains active; R1/R2 are not entered. Replay, selective adaptation, and memory-budget selection remain blocked.
"""
upsert(DOCS / "decision_log.md", decision_block)

change_block = r"""## C035 — Candidate B audit and frozen assignment proposal (2026-10-05)

Added the Candidate B preparation script, four metadata-only assignment proposals, two validation/audit results, four protocol/audit documents, and synchronized planning, readiness, risk, question, decision, and change records. Validation covers exact counts, AF5 composition and template isolation, D0 label logic, B0/BH reuse, matched anchors, source hashes/licenses, group leakage, H2-mini reuse, external-bank separation, and pinned-tokenizer lengths.

No stronger-pilot raw JSONL, checkpoints, predictions, training, replay, selective adaptation, or numerical memory budget was created.
"""
upsert(DOCS / "change_log.md", change_block)

risks_block = r"""## Candidate B assumptions and residual risks

| ID | Assumption or risk | Treatment |
|---|---|---|
| R083 | AF5 recombines components already exposed in AF2–AF4 and may induce positive transfer rather than retention pressure. | Call it compositional/combined; do not claim family-disjoint novelty. |
| R084 | D0 shares the malicious base document by design and may make B-DOC an easier matched control. | Preserve exact pairing and interpret B-SHORT minus B-DOC as a bundled format/source transition. |
| R085 | D0, B0, and BH differ in source, language, domain, and format. | Do not claim pure causal semantic-hardness isolation; report slice behavior. |
| R086 | OR-Bench public data lacks an official rewrite-parent map. | Retain conservative project groups and the lineage caveat. |
| R087 | 2,320 of 3,770 unique records appeared in H2-mini. | Record exact reuse and do not call this an independent replication. |
| R088 | Frozen AF2–AF4 construction patterns can repeat across splits. | Require identity/base/payload isolation; AF5 complete templates are additionally split-exclusive. |
| R089 | Runtime and checkpoint-space values are projections. | Run a technical preflight only after approved materialization; do not treat estimates as an audit. |
"""
upsert(DOCS / "assumptions_and_risks.md", risks_block)

questions_block = r"""## Candidate B open questions

| ID | Question | Current answer/status |
|---|---|---|
| Q056 | May the frozen Candidate B proposals be materialized into local JSONL? | Open human approval; assignments are validated and ready. |
| Q057 | May the nine Candidate B no-replay runs begin? | Blocked until materialization is separately approved, completed, validated, and reviewed. |
| Q058 | May any replay or selective-adaptation method be implemented? | No; blocked unless future no-replay evidence reaches R1/R2 and a human approves the protocol. |
| Q059 | What numerical replay memory budget should be used? | Unselected; decide only after R1/R2 eligibility and human approval. |
"""
upsert(DOCS / "open_questions.md", questions_block)

decision = {
    "date": capacity["created_at"],
    "id": "D047",
    "decision": "Select Candidate B and freeze validated prediction-independent hash-only assignments for human materialization approval",
    "detector": capacity["detector"],
    "malicious_order": ["AF2 Ignore", "AF3 Escape-Separation", "AF4 Fake Completion", "AF5 Combined"],
    "AF5_interpretation": "compositional/combined; not family-disjoint",
    "arms": {
        "B-DOC": ["D0-1", "D0-2", "D0-3", "D0-4"],
        "B-SHORT": ["D0-1", "B0-2", "B0-3", "B0-4"],
        "B-HARD": ["D0-1", "B0-2", "BH-1", "BH-2"],
    },
    "counts": validation["counts"],
    "proposal_hashes": proposal_hashes,
    "validation": validation["status"],
    "future_execution": validation["future_execution"],
    "readiness": validation["readiness"],
    "materialized": False,
    "training": False,
    "replay": False,
    "selective_adaptation": False,
    "numerical_memory_budget": None,
}

change = {
    "date": capacity["created_at"],
    "id": "C035",
    "decision_id": "D047",
    "reason": "Audit and freeze the human-selected Candidate B benchmark before any raw-text materialization or training",
    "files_changed": [
        "scripts/prepare_stronger_pilot_candidate_b.py",
        "data/manifests/stronger_pilot/stronger_shared_malicious_assignment.json",
        "data/manifests/stronger_pilot/stronger_b_doc_assignment_proposal.json",
        "data/manifests/stronger_pilot/stronger_b_short_assignment_proposal.json",
        "data/manifests/stronger_pilot/stronger_b_hard_assignment_proposal.json",
        "data/manifests/stronger_pilot/stronger_assignment_validation.json",
        "results/stronger_pilot_capacity_audit.json",
        "results/stronger_pilot_assignment_validation.json",
        "docs/stronger_pilot_candidate_b_protocol.md",
        "docs/stronger_pilot_source_audit.md",
        "docs/stronger_pilot_assignment_plan.md",
        "docs/stronger_pilot_no_replay_protocol.md",
        "docs/stronger_stream_design_options.md",
        "docs/two_week_experiment_plan.md",
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
        "source and authority hashes verified",
        "exact counts and position targets pass",
        "matched malicious and benign anchors pass",
        "all applicable cross-stage/split leakage counts are zero",
        "3770 unique records tokenized without truncation; max 491",
        "metadata-only proposals contain no raw text",
    ],
    "scope": "Audit, hash-only assignment, documentation, and future materialization plan only",
    "materialized": False,
    "training": False,
    "replay": False,
    "selective_adaptation": False,
    "numerical_memory_budget": None,
}


def append_jsonl_unique(path: Path, record: dict) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    records = [json.loads(line) for line in lines if line.strip()]
    ids = [r.get("id") for r in records]
    if record["id"] in ids:
        records[ids.index(record["id"])] = record
    else:
        records.append(record)
    write(path, "\n".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in records))


append_jsonl_unique(ROOT / "history" / "decisions.jsonl", decision)
append_jsonl_unique(ROOT / "history" / "changes.jsonl", change)

print(json.dumps({
    "status": "PASS",
    "created_docs": [
        "docs/stronger_pilot_candidate_b_protocol.md",
        "docs/stronger_pilot_source_audit.md",
        "docs/stronger_pilot_assignment_plan.md",
        "docs/stronger_pilot_no_replay_protocol.md",
    ],
    "decision": "D047",
    "change": "C035",
}, indent=2))
