"""Write order-control reporting, harder-benchmark design audit, and history entries."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
HISTORY = ROOT / "history"
DATA_SHA = "4b4efaf7640f31f5f12af2f2b3a40b91bbb71789c91c059d21ce4af1e6a9e43a"
SEEDS = (17, 42, 2026)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def fmt(value):
    return f"{value * 100:.3f}"


def stat(value):
    return f"{value['mean'] * 100:.3f} ± {value['sample_sd'] * 100:.3f}"


def values(value):
    return ", ".join(fmt(item) for item in value["values"])


def upsert_section(path, marker, content):
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    begin = f"<!-- {marker}:BEGIN -->"
    end = f"<!-- {marker}:END -->"
    block = f"{begin}\n{content.rstrip()}\n{end}"
    if begin in text and end in text:
        before, rest = text.split(begin, 1)
        _, after = rest.split(end, 1)
        text = before.rstrip() + "\n\n" + block + after
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def append_jsonl_unique(path, key, record):
    rows = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = [row for row in rows if row.get(key) != record[key]]
    rows.append(record)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def combined_matrix(item, order):
    lines = ["| Trained through | AF1 | AF4 | AF3 | AF2 |", "|---|---:|---:|---:|---:|"]
    for trained in order:
        cells = []
        for family in order:
            if family not in item["matrix"]["accuracy"][trained]:
                cells.append("—")
                continue
            cell = [item["matrix"][metric][trained][family] for metric in ("accuracy", "macro_f1", "attack_recall", "benign_fpr")]
            cells.append("/".join(fmt(value) for value in cell))
        lines.append(f"| {trained} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def mean_matrix(aggregate, metric):
    order = aggregate["stage_order"]
    lines = ["| Trained through | AF1 | AF4 | AF3 | AF2 |", "|---|---:|---:|---:|---:|"]
    for trained in order:
        cells = []
        for family in order:
            value = aggregate["mean_sample_sd_matrix"][metric][trained].get(family)
            cells.append(stat(value) if value else "—")
        lines.append(f"| {trained} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def order_report(secondary, primary):
    order = secondary["stage_order"]
    lines = [
        "# Attack-family order control",
        "",
        "Status: **complete; human review required before any replay or harder-benchmark execution.**",
        "",
        "This is the predeclared no-replay order control `AF1 -> AF4 -> AF3 -> AF2`. It uses the same audited Chen-derived deterministic reconstruction, fixed DeBERTa-v3-base revision, `max_length=1280`, batch size 4, learning rate 1e-5, one epoch per stage, optimizer reset, carried model weights, and seeds 17/42/2026. Each stage trained only its current 3,000 records. AF5, historical samples, replay, and selective adaptation were absent.",
        "",
        "All values below are percentages. A combined matrix cell is `accuracy / macro-F1 / attack TPR / benign FPR`.",
    ]
    for item in secondary["per_seed"]:
        lines += ["", f"## Seed {item['seed']}", "", combined_matrix(item, order)]
    lines += ["", "## Acquisition diagnostics", "", "| Seed | Stage | Accuracy | Macro-F1 | Attack TPR | Benign FPR | Train loss | Validation loss | Runtime (s) | Peak GPU (GiB) |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in secondary["per_seed"]:
        for row in item["stage_diagnostics"]:
            lines.append(
                f"| {item['seed']} | {row['stage']} | {fmt(row['accuracy'])} | {fmt(row['macro_f1'])} | {fmt(row['attack_tpr'])} | {fmt(row['benign_fpr'])} | {row['training_loss']:.6g} | {row['validation_loss']:.6g} | {row['runtime_seconds']:.1f} | {row['peak_gpu_memory_bytes'] / 2**30:.3f} |"
            )
    lines += ["", "Across seeds, attack-TPR acquisition was AF1 `100.000 ± 0.000`, AF4 `99.815 ± 0.321`, AF3 `99.630 ± 0.642`, and AF2 `100.000 ± 0.000`. AF3 therefore shows a small order-dependent acquisition limitation; a later family below perfect acquisition is not counted as forgetting.", "", "## Mean ± sample SD matrices", ""]
    for metric, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("attack_recall", "Attack TPR"), ("benign_fpr", "Benign FPR")):
        lines += [f"### {label}", "", mean_matrix(secondary, metric), ""]
    lines += ["## Final retention and continual-learning summaries", "", "| Seed | Attack-TPR BWT | Macro-F1 BWT | Accuracy BWT | Mean benign-FPR change | Runtime (s) | Peak GPU (GiB) |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for item in secondary["per_seed"]:
        lines.append(f"| {item['seed']} | {fmt(item['final_bwt']['attack_recall'])} | {fmt(item['final_bwt']['macro_f1'])} | {fmt(item['final_bwt']['accuracy'])} | {fmt(item['final_benign_fpr_increase'])} | {item['runtime_seconds']:.1f} | {item['peak_gpu_memory_bytes']/2**30:.3f} |")
    lines += [
        f"| Mean ± SD | {stat(secondary['final_bwt_mean_sample_sd']['attack_recall'])} | {stat(secondary['final_bwt_mean_sample_sd']['macro_f1'])} | {stat(secondary['final_bwt_mean_sample_sd']['accuracy'])} | {stat(secondary['final_benign_fpr_increase_mean_sample_sd'])} | {secondary['runtime_mean_sample_sd']['mean']:.1f} ± {secondary['runtime_mean_sample_sd']['sample_sd']:.1f} | {secondary['peak_gpu_memory_mean_sample_sd']['mean']/2**30:.3f} ± {secondary['peak_gpu_memory_mean_sample_sd']['sample_sd']/2**30:.3f} |",
        "",
        "Final family changes show no common negative direction across all seeds. AF1 attack TPR changes were `0.000, 0.000, -1.111` pp; AF4 were `0.000, 0.000, +0.556` pp; AF3 were `+0.556, 0.000, -0.556` pp. Mean final AF1 attack-TPR change was `-0.370 ± 0.642` pp and mean final AF1 macro-F1 change was `-0.185 ± 0.321` pp.",
        "",
        "True per-seed AF1 peak attack-TPR forgetting was `1.111, 1.111, 4.444` pp (mean `2.222 ± 1.925`). The active metric code and this aggregate define peak forgetting as a nonnegative maximum drop from the best score observed up to each endpoint.",
        "",
        "## Transition and position diagnostics",
        "",
        "AF4 immediately after AF1 reduced AF1 attack TPR by `-1.111, -1.111, -4.444` pp across seeds (mean `-2.222 ± 1.925`), with no benign-FPR increase. AF3 then restored AF1 by `+1.111, +1.111, +4.444` pp. AF2 at the final transition changed AF1 by `0.000, 0.000, -1.111` pp. Thus the first drop is consistent but mostly transient.",
        "",
        "| Endpoint/evaluated family | Head TPR mean ± SD | Middle TPR mean ± SD | Tail TPR mean ± SD |", "|---|---:|---:|---:|",
    ]
    for trained in order:
        for family in order[: order.index(trained) + 1]:
            pos = secondary["position_attack_tpr_mean_sample_sd"]
            lines.append(f"| {trained}/{family} | {stat(pos['head'][trained][family])} | {stat(pos['middle'][trained][family])} | {stat(pos['tail'][trained][family])} |")
    lines += [
        "",
        "The strongest position effect is AF1-middle immediately after AF4: `96.667, 98.333, 90.000%` (mean `95.000 ± 4.410%`). Head was `100.000, 98.333, 96.667%`, while tail stayed `100%` in every seed. The earlier primary seed-42 AF1-middle fall to `86.667%` after AF2 was not reproduced: secondary seed 42 reached `98.333%` after AF4 and recovered to `100%` after AF3.",
        "",
        "## Primary versus secondary order",
        "",
        "| Summary | Primary AF1-AF2-AF3-AF4 | Secondary AF1-AF4-AF3-AF2 |", "|---|---:|---:|",
        f"| Final attack-TPR BWT | {stat(primary['final_bwt_mean_sample_sd']['attack_recall'])} | {stat(secondary['final_bwt_mean_sample_sd']['attack_recall'])} |",
        f"| Final macro-F1 BWT | {stat(primary['final_bwt_mean_sample_sd']['macro_f1'])} | {stat(secondary['final_bwt_mean_sample_sd']['macro_f1'])} |",
        f"| Final accuracy BWT | {stat(primary['final_bwt_mean_sample_sd']['accuracy'])} | {stat(secondary['final_bwt_mean_sample_sd']['accuracy'])} |",
        f"| Mean benign-FPR change | {stat(primary['final_benign_fpr_increase_mean_sample_sd'])} | {stat(secondary['final_benign_fpr_increase_mean_sample_sd'])} |",
        f"| AF1 peak attack-TPR forgetting | {stat(primary['peak_forgetting_mean_sample_sd']['AF1']['attack_recall'])} | {stat(secondary['peak_forgetting_mean_sample_sd']['AF1']['attack_recall'])} |",
        "",
        "AF4-before-AF2 creates more consistent immediate pressure on AF1 than AF2-before-AF4: all secondary seeds drop after AF4, while the primary first transition drops only seed 42 (`0.000, -6.111, 0.000` pp). It does not create stronger final forgetting: secondary final attack-TPR BWT is closer to zero and has mixed signs. The same family therefore has order-dependent trajectories, but neither order produces a stable final retention problem.",
        "",
        "Benign-boundary drift is tiny and changes sign: primary mean final FPR change is `+0.062 ± 0.107` pp; secondary is `-0.062 ± 0.107` pp. No secondary transition increased benign FPR on an earlier family; seed 2026's AF3 acquisition FPR of `0.556%` returned to zero after AF2. Malicious recall and benign FPR therefore move differently and must remain separate.",
        "",
        "## Interpretation",
        "",
        "Changing order reveals reproducible short-lived AF1 interference from AF4 and mild acquisition variation for AF3/AF4, but it does not create a stronger reproducible final retention problem. The result remains dominated by ceiling performance, transient transfer, position effects, and seed variation. It is not evidence of catastrophic forgetting and does not justify replay as the dominant intervention.",
        "",
        "Machine-readable aggregate: `results/attack_family_order_control_aggregate.json`. Artifact and checkpoint-load validation: `results/attack_family_order_control_artifact_validation.json` and `results/attack_family_order_control_checkpoint_validation.json`.",
    ]
    return "\n".join(lines) + "\n"


def main():
    secondary = load(RESULTS / "attack_family_order_control_aggregate.json")
    primary = load(RESULTS / "attack_family_baseline_aggregate.json")
    (DOCS / "attack_family_order_control.md").write_text(order_report(secondary, primary), encoding="utf-8")

    design_dir = ROOT / "data" / "manifests" / "harder_benchmark_design"
    h1 = {
        "candidate": "H1",
        "name": "Controlled Combined-Attack Diagnostic",
        "status": "design_and_existing_data_audit_complete; no evaluation_or_training_authorized",
        "source_artifact": "data/processed/chen_attack_family_shift.jsonl",
        "source_sha256": DATA_SHA,
        "AF5_counts": {"train": {"total": 3000, "benign": 1500, "attack": 1500}, "val": {"total": 300, "benign": 150, "attack": 150}, "test": {"total": 360, "benign": 180, "attack": 180}},
        "positions_per_split_attack": {"train": {"head": 500, "middle": 500, "tail": 500}, "val": {"head": 50, "middle": 50, "tail": 50}, "test": {"head": 60, "middle": 60, "tail": 60}},
        "component_overlap": ["AF2_ignore", "AF3_escape_separation", "AF4_fake_completion"],
        "full_record_overlap": {"cross_stage_exact_text": 0, "cross_stage_base_document": 0, "cross_stage_payload": 0, "cross_stage_full_template": 0},
        "scientific_question": "Does a detector trained sequentially on individual components generalize compositionally to their combination, and, only in an optional later stage, does acquiring that controlled-overlap combination disturb earlier components?",
        "recommended_protocol": "H1-HO: keep AF5 test frozen and evaluate approved AF1-AF4 endpoints; do not train AF5",
        "optional_protocol_requiring_later_approval": "H1-S5: add AF5 as a fifth current-stage-only acquisition step and reevaluate AF1-AF5",
        "claim_limits": ["AF5 is not family-disjoint", "H1-HO measures compositional generalization, not forgetting of AF5", "H1-S5 mixes component reuse with sequential retention pressure", "synthetic lexical shortcuts and one document source remain"],
        "training_executed": False,
        "evaluation_executed": False,
        "replay": False,
    }
    h2 = {
        "candidate": "H2",
        "name": "Security-Bidirectional / Hard-Benign Benchmark",
        "status": "design_audit_only; zero records constructed",
        "design": {
            "stream_principle": "Alternate one-factor transitions so malicious-family and benign-difficulty changes are not introduced in the same transition.",
            "proposed_stages": [
                {"stage": "H2-1", "malicious": "AF1", "benign": "ordinary document benign"},
                {"stage": "H2-2", "malicious": "AF2", "benign": "ordinary document benign", "changed_axis": "malicious family"},
                {"stage": "H2-3", "malicious": "AF2", "benign": "trigger-rich hard benign", "changed_axis": "benign difficulty"},
                {"stage": "H2-4", "malicious": "AF3", "benign": "trigger-rich hard benign", "changed_axis": "malicious family"},
                {"stage": "H2-5", "malicious": "AF3", "benign": "context/domain-shifted hard benign", "changed_axis": "benign context/domain"},
                {"stage": "H2-6", "malicious": "AF4", "benign": "context/domain-shifted hard benign", "changed_axis": "malicious family"},
            ],
            "frozen_evaluation_banks": ["ordinary benign", "trigger-rich hard benign", "context/domain hard benign", "AF1", "AF2", "AF3", "AF4"],
            "required_metrics": ["attack TPR and FNR by IPI family", "benign FPR by benign stratum", "accuracy", "macro-F1", "head/middle/tail TPR", "acquisition", "retention", "BWT", "peak forgetting"],
        },
        "sources": {
            "NotInject": {"role": "frozen trigger-rich benign diagnostic first", "paper_reported_records": 339, "labels": "all benign", "subsets": ["one trigger", "two triggers", "three triggers"], "topics": ["Common Queries", "Technique Queries", "Virtual Creation", "Multilingual Queries"], "official_dataset_card_license": "MIT as observed 2026-09-13", "local_artifact": False, "required_before_use": ["pin revision and file hashes", "verify license file and attribution", "deduplicate against all train/val/test text and model views", "human label review", "freeze split before training"]},
            "CAPTURE": {"role": "design inspiration; possible benign SAFE-GEN/context diagnostic only after artifact and rights audit", "paper_reported_benign": {"train": 339, "test_validation": 171}, "paper_reported_malicious": {"train": 1274, "test_validation": 641}, "threat_scope": "direct, single-turn context-aware prompt injection", "local_artifact": False, "authoritative_data_release_found": False, "explicit_dataset_license_found": False, "restriction": "CAPTURE malicious examples must never be labeled indirect prompt injection; keep as a separate direct-PI diagnostic unless the classifier task is explicitly broadened"},
            "current_Chen_reconstruction": {"role": "IPI malicious strata and ordinary benign anchor candidates", "source_sha256": DATA_SHA, "redistribution": "local research only; do not commit raw/processed text"},
        },
        "construction_gates": ["human source approval", "authoritative artifact availability", "file-level license/provenance record", "fixed split and target counts", "identity/near-duplicate audit", "length and injection-span audit", "task-label compatibility review"],
        "actual_records_constructed": 0,
        "training_executed": False,
        "replay": False,
    }
    audit = {
        "date": "2026-09-13",
        "status": "design_audit_complete; no data construction, evaluation, or training",
        "candidates": ["H1", "H2"],
        "existing_AF5_verified": True,
        "new_raw_content_downloaded": False,
        "new_processed_content_created": False,
        "replay_implemented": False,
        "recommended_next_benchmark": "H2 after NotInject local acquisition/provenance/overlap audit; retain H1-HO as a lower-cost compositional diagnostic",
        "blocking_items": ["H2 source and split approval", "NotInject pinned-artifact audit", "CAPTURE artifact/license unavailable", "task-scope decision for any direct-PI diagnostic"],
    }
    write_json(design_dir / "h1_controlled_combined.json", h1)
    write_json(design_dir / "h2_security_bidirectional.json", h2)
    write_json(ROOT / "data" / "audits" / "harder_benchmark_design" / "design_audit.json", audit)

    harder_doc = """# Harder benchmark design audit

Status: **design and data audit only; no new records were constructed, evaluated, or trained.**

## H1: controlled Combined diagnostic

The existing AF5 Combined partition has 3,000 train, 300 validation, and 360 test records, balanced by label and attack position. It shares no exact text, base-document identity, payload identity, or full-template hash with AF1-AF4, but it deliberately composes Ignore, Escape-Separation, and Fake Completion. It is therefore controlled component overlap and cannot support a family-disjoint claim.

The recommended H1-HO protocol is to keep the 360-record AF5 test set frozen and, after human approval, evaluate the existing AF1-AF4 endpoints without AF5 training. This asks whether individually learned components support compositional generalization. An optional H1-S5 protocol could later train AF5 as a fifth current-stage-only step and measure acquisition plus retention, but its component reuse must remain explicit. Neither protocol is replay.

## H2: security-bidirectional / hard-benign stream

The proposed six-stage stream alternates one changed axis at a time: AF1/ordinary benign; AF2/ordinary benign; AF2/trigger-rich benign; AF3/trigger-rich benign; AF3/context-domain hard benign; AF4/context-domain hard benign. Frozen evaluation banks retain every malicious family and each benign stratum. This makes false-negative pressure from malicious evolution and false-positive pressure from benign difficulty separately measurable.

NotInject is the preferred first hard-benign source. Its official dataset card reports 339 all-benign samples, three trigger-count subsets, four topic groups, and an MIT license. No NotInject content was downloaded in this phase. Before use, pin a revision, hash every file, preserve attribution, audit overlap/model views, review labels, and freeze a test-only diagnostic split.

CAPTURE supports context-aware and over-defense design, reporting MALICIOUS-GEN and SAFE-GEN data. Its malicious task is direct, single-turn prompt injection, so those records cannot be relabeled as indirect prompt injection. No authoritative released dataset or explicit dataset license was located in the supplied paper/supplementary material during this audit. CAPTURE remains blocked as data; its malicious portion may only become a separate direct-PI diagnostic after task-scope approval.

## Recommendation

Prefer H2 after a pinned NotInject artifact and full provenance/overlap audit because both completed family orders remain near ceiling. Keep H1-HO as a lower-cost compositional diagnostic. Do not choose AF5 alone as evidence that replay is needed, and do not introduce CAPTURE-generated substitutes because that would add generator and task confounds.

Machine-readable designs are `data/manifests/harder_benchmark_design/`; the audit decision is `data/audits/harder_benchmark_design/design_audit.json`.
"""
    (DOCS / "harder_benchmark_design.md").write_text(harder_doc, encoding="utf-8")

    replay_doc = """# Replay readiness review

Decision: **R0 — replay is still not experimentally justified.**

The primary order produced final attack-TPR BWT `-0.185 ± 0.185` percentage points, macro-F1 BWT `-0.123 ± 0.053` points, and benign-FPR change `+0.062 ± 0.107` points. The secondary order produced attack-TPR BWT `-0.062 ± 0.283` points, macro-F1 BWT `0.000 ± 0.093` points, and benign-FPR change `-0.062 ± 0.107` points.

AF4 after AF1 caused a reproducible immediate AF1 attack-TPR drop across seeds (`-1.111, -1.111, -4.444` points), concentrated in middle position, but AF3 restored it in every seed. Secondary final AF1 changes were `0.000, 0.000, -1.111` points. Other final family changes have zero, positive, and negative signs. The primary seed-42 middle-position failure was not reproduced at comparable severity. Both orders remain at or close to ceiling, and benign-FPR drift is tiny and order-dependent.

These data demonstrate transient order-dependent interference, position sensitivity, small acquisition variation, and optimization/seed variation. They do not demonstrate a stable final retention problem or show that forgetting is the dominant limitation. R1 would be defensible only if a future harder no-replay benchmark first shows stable degradation and replay is then framed as a sensitivity ablation. R2 is unsupported.

Replay, selective adaptation, and all named replay algorithms remain unimplemented and blocked pending human review. The next evidence should come from a harder no-replay benchmark with an audited hard-benign diagnostic, not from replay on the saturated stream.
"""
    (DOCS / "replay_readiness_review.md").write_text(replay_doc, encoding="utf-8")

    exp_section = """## 2026-09-12 — Three-seed secondary attack-family order control

The no-replay `AF1 -> AF4 -> AF3 -> AF2` control completed from `E:\\ContinualGuard` for seeds 17, 42, and 2026. Artifact lineage and all 12 stage-end checkpoints passed independent validation. Final attack-TPR BWT was `-0.062 ± 0.283` percentage points; macro-F1 BWT was `0.000 ± 0.093`; benign-FPR change was `-0.062 ± 0.107`. AF4 caused a consistent but mostly recovered immediate AF1 TPR decline. H1/H2 were designed and audited without data construction, evaluation, or training. Replay readiness remains R0.
"""
    upsert_section(DOCS / "experiment_history.md", "ORDER_CONTROL_20260912", exp_section)

    decision_section = """## D020

- **date:** 2026-09-13
- **decision:** Classify replay readiness as R0 after the three-seed secondary order control.
- **motivation:** Final retention remains near ceiling and sign-inconsistent: secondary attack-TPR BWT is -0.062 ± 0.283 pp, while the consistent AF4-to-AF1 interference is transient and recovers after AF3.
- **consequences:** No replay or selective-adaptation method may be implemented or run; obtain human review of a harder no-replay benchmark first.
- **status:** accepted analytical stop; human review required for the next experiment.

## D021

- **date:** 2026-09-13
- **decision:** Keep AF5 as a controlled-overlap H1 diagnostic and prefer an H2 hard-benign/security-bidirectional benchmark after a pinned NotInject audit.
- **motivation:** Both family orders saturate. AF5 tests compositional generalization but is not family-disjoint; NotInject directly measures over-defense; CAPTURE malicious data are direct/single-turn and cannot be relabeled IPI.
- **consequences:** No harder-benchmark data are constructed or trained. CAPTURE remains blocked until authoritative artifacts, licensing, and task scope are resolved.
- **status:** design audit complete; source and execution decisions remain human-gated.
"""
    upsert_section(DOCS / "decision_log.md", "DECISIONS_D020_D021", decision_section)

    change_section = """## C012

- **date:** 2026-09-13
- **files_changed:** `configs/attack_family_order_control.yaml`, `src/continualguard/{family_baseline,metrics}.py`, order-control run scripts/verifiers/summarizer, three result runs and aggregate, harder-benchmark design manifests, reports, and history files
- **reason:** Execute and independently verify the approved secondary no-replay order; audit H1/H2 designs; make a quantitative replay-readiness decision.
- **effect:** Adds complete secondary matrices and checkpoints, paired primary/secondary analysis, nonnegative peak-forgetting semantics, H1/H2 design-only records, and an R0 stop.
- **tests_run:** artifact/lineage validation passed; 12/12 order-control checkpoints loaded; final pytest/compileall/pip-check recorded in migration completion record.
- **scientific_behavior_changed:** true for the secondary experiment; no replay behavior added.
- **limitations:** near-ceiling benchmark, small seed-dependent acquisition effects, transient position-specific interference, H2 source data not acquired, CAPTURE artifact/license unresolved.
"""
    upsert_section(DOCS / "change_log.md", "CHANGE_C012", change_section)

    risks = """| R039 | Secondary-order AF4 causes a consistent but mostly transient AF1 recall drop, which an endpoint-only summary can hide | high | Preserve every transition and position matrix; separate peak interference from final retention | confirmed and quantified |
| R040 | A negative or positive endpoint delta can be mislabeled as peak forgetting | high | Define peak forgetting as a nonnegative maximum historical drop; use the independently recomputed aggregate | mitigated in active metric code and order aggregate |
| R041 | NotInject's trigger-rich benign prompts may change language/topic as well as lexical difficulty | high | Stratify trigger count/topic/language and use it first as a frozen diagnostic | open; no data acquired |
| R042 | CAPTURE artifacts and dataset rights are not currently auditable, and its malicious task is direct rather than indirect injection | high | Do not ingest or relabel; require authoritative artifacts, license review, and separate task scope | blocks CAPTURE use |
| R043 | External-drive availability and I/O can interrupt future runs | medium | Treat E:\\ContinualGuard as canonical, verify mount/path before runs, and preserve atomic stage checkpoints | accepted operational limitation |"""
    upsert_section(DOCS / "assumptions_and_risks.md", "RISKS_R039_R043", risks)

    questions = """| Q013 | Should AF5 Combined be used next? | Tests composition but reuses AF2-AF4 components | Approve only H1-HO as a frozen controlled-overlap diagnostic first | Human review of H1 design and frozen AF5 test | open human decision |
| Q014 | Which hard-benign source should be approved? | H2 needs false-positive pressure | Pin and audit NotInject first; keep CAPTURE blocked | Revision/hash/license/overlap/label audit | open human decision |
| Q016 | Is current retention pressure sufficient for replay? | Replay must address a stable limitation | No; classify R0 | Primary and secondary three-seed aggregates | resolved 2026-09-13; R0 |
| Q017 | Did the secondary order run? | Separates order-dependent transfer | Yes; all three seeds completed and validated | Complete matrices and 12 checkpoint loads | resolved 2026-09-13 |
| Q019 | May H1-HO evaluate frozen AF5 after current endpoints? | Low-cost compositional diagnostic without training | Approve only after reviewing controlled-overlap claim limits | H1 manifest and AF5 audit | open human decision |
| Q020 | May NotInject be acquired locally for H2 audit? | Required before a hard-benign baseline | Approve pinned local acquisition and audit, without training | Official revision, file hashes, license and overlap review | open human decision |"""
    upsert_section(DOCS / "open_questions.md", "QUESTIONS_ORDER_AND_HARDER", questions)

    migration_addendum = """## Post-migration scientific operation

The canonical E: environment completed three new CUDA training trajectories from `E:\\ContinualGuard`, generated all expected results, and passed full artifact plus 12-checkpoint validation. New metadata points the pinned model path to E:. This supplies an operational validation beyond the preflight.

Source removal remains the sole migration operation pending because the current Codex workspace is rooted at `D:\\project`. The source must not be force-deleted while this process holds it. Reopen the workspace at `E:\\ContinualGuard`, verify the completion record, then remove `D:\\project` in that new session.
"""
    upsert_section(DOCS / "project_migration.md", "POST_MIGRATION_OPERATION", migration_addendum)

    replay_decision = {
        "date": "2026-09-13T00:00:00+08:00", "id": "D020",
        "decision": "Classify replay readiness as R0 after the three-seed secondary order control",
        "motivation": "Secondary final attack-TPR BWT is -0.062 +/- 0.283 percentage points; consistent AF4-to-AF1 interference is transient and final changes are not sign-consistent",
        "alternatives_considered": "R1 replay sensitivity immediately; R2 full replay comparison; catastrophic-forgetting claim",
        "source_evidence": "results/attack_family_order_control_aggregate.json and passed artifact/checkpoint validations",
        "consequences": "Replay and selective adaptation remain blocked; review a harder no-replay benchmark first",
        "status": "accepted analytical stop; human review required",
    }
    design_decision = {
        "date": "2026-09-13T00:05:00+08:00", "id": "D021",
        "decision": "Keep AF5 as a controlled-overlap H1 diagnostic and prefer H2 after a pinned NotInject data audit",
        "motivation": "Both family orders are near ceiling; NotInject measures hard-benign over-defense; CAPTURE malicious examples are direct/single-turn and its data/license are unavailable for audit",
        "alternatives_considered": "Call AF5 family-disjoint; construct CAPTURE substitutes; start replay on the saturated stream",
        "source_evidence": "Existing AF5 audit, PIGuard/NotInject paper and official dataset card, CAPTURE paper and supplementary artifact review",
        "consequences": "No H1/H2 evaluation or training; source acquisition and execution require human review",
        "status": "design audit complete",
    }
    append_jsonl_unique(HISTORY / "decisions.jsonl", "id", replay_decision)
    append_jsonl_unique(HISTORY / "decisions.jsonl", "id", design_decision)
    change = {
        "date": "2026-09-13T00:10:00+08:00", "id": "C012",
        "files_changed": ["configs/attack_family_order_control.yaml", "src/continualguard/family_baseline.py", "src/continualguard/metrics.py", "scripts/order-control reporting and validation", "results/attack-family-order-control-*", "results/attack_family_order_control_*.json", "data/manifests/harder_benchmark_design/", "data/audits/harder_benchmark_design/", "docs/", "history/"],
        "reason": "Complete secondary-order no-replay control, harder-benchmark design audit, and replay-readiness review",
        "effect": "Complete paired order evidence, independently validated artifacts/checkpoints, H1/H2 design-only records, and R0 stop",
        "tests_run": ["order-control artifact/lineage verifier passed", "all 12 order-control model and resume-state checkpoints loaded", "final suite recorded in migration completion record"],
        "scientific_behavior_changed": True,
        "limitations": ["near-ceiling benchmark", "transient seed/position effects", "no H2 source acquired", "no replay implemented or run"],
    }
    append_jsonl_unique(HISTORY / "changes.jsonl", "id", change)
    experiment = {
        "run_id": "analysis-attack-family-order-control-20260913",
        "date": "2026-09-13T00:15:00+08:00",
        "experiment_regime": "secondary_order_aggregate_and_harder_benchmark_design_audit",
        "project_root": "E:\\ContinualGuard",
        "seeds": list(SEEDS),
        "stage_order": secondary["stage_order"],
        "source_run_ids": [item["run_id"] for item in secondary["per_seed"]],
        "attack_tpr_bwt_mean": secondary["final_bwt_mean_sample_sd"]["attack_recall"]["mean"],
        "attack_tpr_bwt_sample_sd": secondary["final_bwt_mean_sample_sd"]["attack_recall"]["sample_sd"],
        "macro_f1_bwt_mean": secondary["final_bwt_mean_sample_sd"]["macro_f1"]["mean"],
        "benign_fpr_change_mean": secondary["final_benign_fpr_increase_mean_sample_sd"]["mean"],
        "artifact_validation": "passed",
        "checkpoint_validation": "12/12 passed",
        "H1_H2": "design audit only; no evaluation or training",
        "replay_readiness": "R0",
        "replay": False,
        "status": "complete_human_review_required",
    }
    append_jsonl_unique(HISTORY / "experiments.jsonl", "run_id", experiment)


if __name__ == "__main__":
    main()
