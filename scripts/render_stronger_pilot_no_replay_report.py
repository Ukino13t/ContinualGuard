"""Render the completed Candidate B result and update decision/history documents."""
from __future__ import annotations

import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
DOCS = ROOT / "docs"
RESULT = ROOT / "results/stronger_pilot_no_replay"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


AGG = load(ROOT / "results/stronger_pilot_no_replay_aggregate.json")
PAIR = load(ROOT / "results/stronger_pilot_no_replay_pairwise.json")
READY = load(ROOT / "results/stronger_pilot_no_replay_readiness.json")
VALID = load(ROOT / "results/stronger_pilot_no_replay_artifact_validation.json")
PREFLIGHT = load(RESULT / "preflight/preflight.json")
INTEGRITY = load(RESULT / "preflight/artifact_integrity_preflight.json")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(value):
    return f"{100 * value:.3f}%"


def pp(value):
    return f"{100 * value:+.3f} pp"


def mean_sd(values):
    values = list(values)
    return f"{100 * statistics.mean(values):.3f} ± {100 * statistics.stdev(values):.3f} pp"


def block_update(path: Path, marker: str, body: str):
    begin = f"<!-- {marker}:BEGIN -->"
    end = f"<!-- {marker}:END -->"
    text = path.read_text(encoding="utf-8") if path.exists() else f"# {path.stem.replace('_', ' ').title()}\n"
    block = f"{begin}\n{body.strip()}\n{end}"
    if begin in text and end in text:
        before, rest = text.split(begin, 1)
        _, after = rest.split(end, 1)
        text = before.rstrip() + "\n\n" + block + after
    else:
        first, sep, remainder = text.partition("\n")
        text = first + "\n\n" + block + ("\n\n" + remainder.lstrip() if sep else "\n")
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def history_upsert(path: Path, record: dict):
    rows = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("id") != record["id"]]
    rows.append(record)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


acq_lines = ["| Arm | Stage | Seed 17 TPR/TNR/FPR/F1 | Seed 42 | Seed 2026 | Gate |", "|---|---|---:|---:|---:|---|"]
for arm in ("B-DOC", "B-SHORT", "B-HARD"):
    for stage in ("AF2", "AF3", "AF4", "AF5"):
        rows = [x for x in AGG["acquisition"] if x["arm"] == arm and x["stage"] == stage]
        values = []
        for seed in (17, 42, 2026):
            x = next(row for row in rows if row["seed"] == seed)
            values.append("/".join(f"{100*x[k]:.2f}" for k in ("attack_tpr", "benign_tnr", "benign_fpr", "paired_macro_f1")))
        acq_lines.append(f"| {arm} | {stage} | {values[0]} | {values[1]} | {values[2]} | PASS 3/3 |")

ret_lines = ["| Arm | Seed | attack-TPR BWT | macro-F1 BWT | benign-FPR increase | attack peak forgetting (mean) |", "|---|---:|---:|---:|---:|---:|"]
for arm in ("B-DOC", "B-SHORT", "B-HARD"):
    rows = [x for x in AGG["retention_by_run"] if x["arm"] == arm]
    for x in rows:
        ret_lines.append(f"| {arm} | {x['seed']} | {pp(x['attack_tpr_bwt'])} | {pp(x['paired_macro_f1_bwt'])} | {pp(x['benign_fpr_increase'])} | {100*x['attack_peak_forgetting_mean']:.3f} pp |")
    ret_lines.append(
        f"| **{arm}** | **mean ± SD** | **{mean_sd(x['attack_tpr_bwt'] for x in rows)}** | **{mean_sd(x['paired_macro_f1_bwt'] for x in rows)}** | **{mean_sd(x['benign_fpr_increase'] for x in rows)}** | **{mean_sd(x['attack_peak_forgetting_mean'] for x in rows)}** |"
    )

af5_lines = ["| Arm | Earlier family | attack-TPR change after AF5, mean ± SD | Seed changes |", "|---|---|---:|---|"]
for arm in ("B-DOC", "B-SHORT", "B-HARD"):
    for family in ("AF2", "AF3", "AF4"):
        rows = [x for x in AGG["AF5_compositional_transfer"] if x["arm"] == arm and x["evaluated_family"] == family]
        vals = [x["attack_tpr_change_after_AF5"] for x in rows]
        af5_lines.append(f"| {arm} | {family} | {mean_sd(vals)} | {' / '.join(pp(v) for v in vals)} |")

position_lines = ["| Arm | Acquisition stage | head TPR | middle TPR | tail TPR |", "|---|---|---:|---:|---:|"]
for arm in ("B-DOC", "B-SHORT", "B-HARD"):
    for stage in ("AF2", "AF3", "AF4", "AF5"):
        rows = [x for x in AGG["attack_trajectory"] if x["arm"] == arm and x["checkpoint_stage"] == stage and x["evaluated_family"] == stage]
        cells = []
        for position in ("head", "middle", "tail"):
            vals = [x[f"attack_tpr_{position}"] for x in rows]
            cells.append(f"{100*statistics.mean(vals):.3f} ± {100*statistics.stdev(vals):.3f}%")
        position_lines.append(f"| {arm} | {stage} | {cells[0]} | {cells[1]} | {cells[2]} |")

ext_lines = ["| Arm | Bank/slice | AF2 mean ± SD | AF5 mean ± SD | change mean ± SD |", "|---|---|---:|---:|---:|"]
for arm in ("B-DOC", "B-SHORT", "B-HARD"):
    for label, metric in (("NotInject", "notinject_fpr"), ("B-short", "bshort_fpr"), ("B-short English", "bshort_english_fpr"), ("B-short multilingual", "bshort_multilingual_fpr")):
        starts = [x[metric] for x in AGG["external_trajectory"] if x["arm"] == arm and x["checkpoint_stage"] == "AF2"]
        finals = [x[metric] for x in AGG["external_trajectory"] if x["arm"] == arm and x["checkpoint_stage"] == "AF5"]
        ext_lines.append(f"| {arm} | {label} | {mean_sd(starts)} | {mean_sd(finals)} | {mean_sd(y-x for x,y in zip(starts,finals))} |")

pair_ret = {x["comparison"]: x for x in PAIR["retention_aggregate"]}
pair_lines = ["| Matched comparison | attack-TPR BWT difference | macro-F1 BWT difference | benign-FPR-increase difference |", "|---|---:|---:|---:|"]
for comparison in ("B-SHORT_minus_B-DOC", "B-HARD_minus_B-SHORT"):
    x = pair_ret[comparison]
    pair_lines.append(
        f"| {comparison.replace('_minus_', ' − ')} | {100*x['attack_tpr_bwt_difference']['mean']:.3f} ± {100*x['attack_tpr_bwt_difference']['sample_sd']:.3f} pp | {100*x['paired_macro_f1_bwt_difference']['mean']:.3f} ± {100*x['paired_macro_f1_bwt_difference']['sample_sd']:.3f} pp | {100*x['benign_fpr_increase_difference']['mean']:.3f} ± {100*x['benign_fpr_increase_difference']['sample_sd']:.3f} pp |"
    )

loss_candidates = READY["R1_numeric_candidates"]
candidate_lines = ["| Tier | Arm/task/metric | Mean ± SD | Seed values | Harmful seeds | Matched replication |", "|---|---|---:|---|---:|---|"]
for x in loss_candidates:
    tier = "R2" if x["R2_numeric_gate"] and x["R2_replication_gate"] else "R1"
    candidate_lines.append(
        f"| {tier} | {x['arm']} / {x['task']} / {x['metric']} | {100*x['mean_degradation']:.3f} ± {100*x['sample_sd']:.3f} pp | {' / '.join(f'{100*v:.3f}' for v in x['values'])} | {x['harmful_seed_count']}/3 | {', '.join(x['R2_replicated_arms']) or 'not required / none'} |"
    )

run_runtime = {(arm, seed): next(x for x in AGG["retention_by_run"] if x["arm"] == arm and x["seed"] == seed) for arm in ("B-DOC","B-SHORT","B-HARD") for seed in (17,42,2026)}
runtime_lines = ["| Arm | Seed | Runtime (s) |", "|---|---:|---:|"]
for arm_key, arm_name in (("doc","B-DOC"),("short","B-SHORT"),("hard","B-HARD")):
    for seed in (17,42,2026):
        meta = load(RESULT / arm_key / f"seed{seed}/run_metadata.json")
        runtime_lines.append(f"| {arm_name} | {seed} | {meta['runtime_seconds']:.3f} |")

r2 = READY["R2_replication_qualified_candidates"][0]
next_prompt = (
    "Accept the completed Candidate B no-replay result and its frozen R2 classification. Keep all replay implementation and execution blocked. "
    "Proceed with replay-protocol and memory-budget design only: propose one shared example-count budget, stored fields and privacy/data-retention policy, "
    "and a method-isolated comparison of no replay, Reservoir ER, DER++, and security-aware replay while fixing the DeBERTa revision, Candidate B assignments, "
    "stage order, seeds, one-epoch budget, learning rate, optimizer resets, threshold, and evaluation protocol. Include compute estimates, buffer accounting, stopping rules, "
    "and the exact tests needed to preserve current-stage acquisition. Do not implement or run any replay method, selective adaptation, or new benchmark; stop for human approval of the protocol and numerical memory budget."
)

report = f"""# Candidate B no-replay baseline results

## Decision

The fixed nine-run Candidate B baseline is **COMPLETE_VALIDATED** and reaches **R2** under the prospective gates. This is eligibility to propose a full replay comparison for human review. It does not authorize replay implementation or execution. No replay, selective adaptation, new benchmark, or numerical memory budget was created.

The decisive R2 result is B-SHORT AF3 attack-TPR loss from first acquisition to final: **{100*r2['mean_degradation']:.3f} ± {100*r2['sample_sd']:.3f} percentage points**, harmful in all seeds (**{' / '.join(f'{100*v:.3f}' for v in r2['values'])} pp**). The matched B-DOC AF3 result reproduces the harmful direction in all seeds at **3.333 ± 1.667 pp**. Magnitude is highly seed-sensitive, so the result is strong descriptive retention evidence rather than a significance claim.

## A–G. Integrity and execution

- **A:** read-only materialization integrity preflight **PASS**.
- **B:** the prior transient capacity message caused **NO** artifact issue.
- **C:** technical GPU preflight **PASS** on `{PREFLIGHT['gpu']}`; CUDA `{PREFLIGHT['cuda_version']}`; no OOM; longest tested token lengths `{PREFLIGHT['selected_token_lengths']}`; save/load and evaluation passed.
- **D/E:** **9/9 runs** and **36/36 stage-end checkpoints** complete.
- **F:** failed runs **0**; resumed runs **0**.
- **G:** cumulative stage runtime **{AGG['total_stage_runtime_seconds']:.3f} s ({AGG['total_stage_runtime_seconds']/60:.2f} min)**; summed run wall time **{AGG['total_run_wall_seconds']:.3f} s ({AGG['total_run_wall_seconds']/60:.2f} min)**; preflight **{AGG['preflight_runtime_seconds']:.3f} s**; peak allocated GPU memory **{AGG['peak_gpu_memory_bytes']/2**30:.3f} GiB**.

{chr(10).join(runtime_lines)}

Artifact validation rehashed **{VALID['checkpoint_files_hashed']} checkpoint files**, verified **{VALID['prediction_files']} prediction files / {VALID['prediction_rows']:,} rows**, exact lineage, current-stage-only training, no optimizer-state carryover, same-seed initialization, three-arm Stage-1 equality, B-SHORT/B-HARD Stage-2 equality, and absence of unpublished stage attempts. The materialization manifest remains `{INTEGRITY['manifest_sha256']}` and validation hash `{INTEGRITY['validation_sha256']}`.

## H–K. Acquisition

Values are attack TPR / benign TNR / benign FPR / paired macro-F1 in percent. Every arm-stage gate passes in **3/3 seeds**, above the required 2/3.

{chr(10).join(acq_lines)}

AF5 compositional acquisition is 100% attack TPR, 100% benign TNR, and 100% macro-F1 in every arm and seed. Poor acquisition does not explain the later retention result.

## L–P. Retention, BWT, peak forgetting, and seed consistency

{chr(10).join(ret_lines)}

B-DOC shows small, consistent malicious retention loss and no benign-FPR increase. B-SHORT creates strong malicious interference, especially AF3, with large magnitude variation. B-HARD largely protects malicious retention but produces a different failure mode: AF2 benign-FPR final increase is **62.222 ± 54.297 pp**, harmful in seeds 42 and 2026 (`0 / 100 / 86.667 pp`). Therefore malicious TPR and benign FPR move differently across arms; the stronger stream does not reduce to one scalar forgetting effect.

Frozen readiness candidates:

{chr(10).join(candidate_lines)}

## Q–T. Compositional AF5 transfer

AF5 is compositional/combined and is not family-disjoint.

{chr(10).join(af5_lines)}

AF5 itself is learned at ceiling. The largest degradation after AF5 is B-SHORT AF3 (`-30.556 ± 37.131 pp`), harmful in all seeds but strongly variable. B-DOC AF3 declines modestly in all seeds; B-HARD AF3 is nearly stable. AF4 stays stable in B-SHORT and B-HARD and declines only slightly in B-DOC.

## U–X. Matched arm comparisons

Differences are treatment minus reference, paired by seed. Negative attack/macro BWT means worse retention; positive benign-FPR-increase difference means worse benign retention.

{chr(10).join(pair_lines)}

The document-to-short transition is associated with measurable interference: B-SHORT minus B-DOC attack-TPR BWT is **-11.296 ± 12.427 pp**. The hard-benign arm shifts the failure mode: B-HARD minus B-SHORT attack-TPR BWT improves **+12.407 ± 13.337 pp**, while benign-FPR retention worsens **+11.852 ± 15.539 pp**. These are matched associations, not pure causal semantic isolation; source, language, length, and format remain intertwined.

## Y–AA. External diagnostics

{chr(10).join(ext_lines)}

B-DOC retains severe external over-defense at AF5: NotInject FPR **69.715 ± 15.933 pp** and B-short FPR **37.222 ± 7.561 pp**. B-SHORT and B-HARD both end at 0 FPR on NotInject and both B-short language slices in every seed. NotInject and B-short were evaluation-only and did not select data, thresholds, epochs, methods, or readiness rules. B-short remains language-unmatched; no pure trigger-effect claim is made.

## AB–AC. Score margins and position diagnostics

Current-stage acquisition remains hard-label and margin saturated, but retained earlier tasks do not. The B-SHORT final AF3 malicious mean `p_attack` is approximately `0.800 / 0.935 / 0.232` for seeds 17/42/2026, with TPR losses `20.000 / 6.667 / 78.333 pp`. B-HARD final AF2 benign mean `p_attack` is approximately `0.003 / 1.000 / 0.874`, matching the seed-specific FPR split. These are large polarized margin shifts with hard-label consequences, not only subtle movement near 0.5. All raw distributions are in `tables/score_margin.csv`.

Acquisition TPR by injection position:

{chr(10).join(position_lines)}

There is no severe acquisition-only position failure; the only acquisition deficit is the AF2/AF4 middle slice in the affected seeds. Full checkpoint trajectories remain in `tables/attack_tpr_trajectory.csv`.

## AD–AH. Replay readiness

- **AD:** final classification **R2**; R0 exited, R1 satisfied, R2 active.
- **AE:** every acquisition and artifact gate passes; B-SHORT AF3 attack-TPR loss is ≥5 pp on average, harmful in all three seeds, and reproduced at the R1 evidence level in the matched B-DOC arm. No threshold, label, leakage, test-guided selection, or acquisition failure was found.
- **AF:** seven R1 numeric candidates are recorded above and in the readiness JSON.
- **AG:** two R2 numeric candidates exist in B-SHORT AF3 (attack TPR and macro-F1); attack TPR passes the matched-replication gate and is the R2-qualified candidate.
- **AH:** **GO for a separate human review of a full replay-comparison proposal; NO-GO for implementation or execution in this task.**

## AI–AN. Project decision

- **AI:** not applicable because the final result is R2; benchmark escalation nevertheless ends because Candidate B was the final escalation attempt.
- **AJ:** a replay comparison may now be **proposed** for human review. It may not yet be implemented or run.
- **AK:** detector remained fixed at `microsoft/deberta-v3-base`, revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`, `max_length=1280`.
- **AL:** numerical replay memory budget remains **unselected**.
- **AM:** remaining confounds are three-seed descriptive uncertainty, large seed-dependent magnitude, document/short/hard source-language-format differences, language-unmatched B-short, AF5 component overlap, and polarized score saturation. BH-1/BH-2 are source-defined partitions, not ordinal difficulty levels.
- **AN exact recommended next prompt:**

> {next_prompt}

## Reproducibility and outputs

- Python `{load(RESULT / 'doc/seed17/run_metadata.json')['environment']['python']}`
- PyTorch `{load(RESULT / 'doc/seed17/run_metadata.json')['environment']['packages']['torch']}`
- Transformers `{load(RESULT / 'doc/seed17/run_metadata.json')['environment']['packages']['transformers']}`
- CUDA `{PREFLIGHT['cuda_version']}`; GPU `{PREFLIGHT['gpu']}`
- Seeds `17, 42, 2026`; order `AF2 → AF3 → AF4 → AF5`
- Focused validation: **22 passed, 0 failed, 1 existing warning**
- Full project validation: **124 passed, 0 failed, 1 existing warning**
- Machine-readable results: `results/stronger_pilot_no_replay_aggregate.json`, `results/stronger_pilot_no_replay_pairwise.json`, `results/stronger_pilot_no_replay_readiness.json`, and `results/stronger_pilot_no_replay_artifact_validation.json`
- Table-ready CSVs: `results/stronger_pilot_no_replay/tables/`
"""

(DOCS / "stronger_pilot_no_replay_results.md").write_text(report, encoding="utf-8")

status_block = f"""## Completed Candidate B no-replay baseline — 2026-10-05

The approved 9-run/36-checkpoint baseline is complete and independently validated. All 12 acquisition gates pass in 3/3 seeds. Final readiness is **R2**: B-SHORT AF3 attack-TPR loss is `35.000 ± 38.115 pp`, harmful in all seeds (`20.000 / 6.667 / 78.333 pp`), with matched B-DOC AF3 replication at `3.333 ± 1.667 pp` in the same harmful direction for all seeds.

R2 permits a full replay-comparison **proposal** for human review. Replay implementation/execution, selective adaptation, any new benchmark, and numerical memory-budget selection remain blocked. See [complete results](stronger_pilot_no_replay_results.md)."""

block_update(DOCS / "stronger_pilot_no_replay_protocol.md", "CANDIDATE_B_NO_REPLAY_RESULT_20261005", status_block)
block_update(DOCS / "replay_readiness_review.md", "CANDIDATE_B_NO_REPLAY_RESULT_20261005", status_block)

comparison_block = """## Candidate B R2 planning gate — 2026-10-05

Candidate B reaches R2 and makes a full replay comparison eligible for **proposal**, subject to a separate human decision. No replay method is implemented or authorized. The proposal must freeze the current detector revision, benchmark bytes, stage order, seeds, one-epoch budget, learning rate, optimizer-reset policy, threshold, and evaluation protocol. It must select one shared example-count memory budget and specify stored fields, privacy/data-retention rules, buffer accounting, compute, stopping rules, and acquisition-preservation tests.

The likely minimal proposal remains no replay, Reservoir ER, DER++, and security-aware replay. This list is planning guidance only. LARS/BRS is optional only if separately approved and time remains. MIR, GSS, GEM, UPGD, and selective adaptation remain outside the next minimal proposal. Implementation and execution require another human approval after the protocol and numerical budget are reviewed."""
block_update(DOCS / "replay_comparison_protocol.md", "CANDIDATE_B_NO_REPLAY_RESULT_20261005", comparison_block)

plan_block = """## Day-5 Candidate B result and next gate — 2026-10-05

The final benchmark-escalation attempt is complete: 9/9 runs, 36/36 checkpoints, every acquisition gate passing, and final R2 readiness. Benchmark escalation now ends. The next possible work is a replay-protocol and common-memory-budget design package for human review; no implementation or execution is authorized. If the human accepts R2, prepare that proposal and stop again before code or training."""
block_update(DOCS / "two_week_experiment_plan.md", "CANDIDATE_B_NO_REPLAY_RESULT_20261005", plan_block)

decision_block = f"""## D049 — Candidate B reaches R2; replay proposal eligible, execution blocked (2026-10-05)

Accept the complete validated Candidate B no-replay evidence. All 9 runs and 36 checkpoints pass; all arm-stage acquisition gates pass in 3/3 seeds. B-SHORT AF3 attack-TPR loss is `35.000 ± 38.115 pp`, harmful in all seeds, and matched B-DOC AF3 reproduces the direction at `3.333 ± 1.667 pp`. Classify readiness as **R2**.

End benchmark escalation. A full replay comparison may be proposed for human review, but replay/selective-adaptation implementation and execution remain blocked. The common numerical memory budget remains unselected. Exact next prompt: `{next_prompt}`"""
block_update(DOCS / "decision_log.md", "CANDIDATE_B_NO_REPLAY_RESULT_20261005", decision_block)

change_block = """## C037 — Execute and validate Candidate B no-replay baseline (2026-10-05)

Added the frozen Candidate B config, no-replay runner, result summarizer, report renderer, and focused tests. Executed 9 GPU runs and saved 36 checkpoints, 162 prediction files, 27,324 prediction rows, confusion counts, matrices, loss/runtime/memory metadata, score margins, pairwise tables, and R0/R1/R2 evidence. Independent validation passed 252 checkpoint-file hashes and both deterministic anchor controls. Focused tests: 22 passed. Full suite: 124 passed. One existing Python 3.14 `torch.jit` FutureWarning remains.

Scientific state changed from R0 to R2. No replay, selective adaptation, benchmark escalation, or numerical memory-budget selection occurred."""
block_update(DOCS / "change_log.md", "CANDIDATE_B_NO_REPLAY_RESULT_20261005", change_block)

risk_block = """## Candidate B result assumptions and risks — 2026-10-05

- R2 is descriptive at n=3. Harmful direction is consistent for the decisive B-SHORT AF3 attack-TPR metric, but magnitude is highly seed-dependent (`6.667–78.333 pp`).
- The matched arms preserve malicious identities and declared anchors, yet benign source, language, length, and format differ. Arm contrasts are associated effects, not pure causal semantic isolation.
- AF5 is compositional and reuses AF2/AF3/AF4 components; it is not family-disjoint. BH-1/BH-2 are source-defined partitions, not increasing difficulty levels.
- B-HARD changes the failure mode: malicious retention improves relative to B-SHORT while AF2 benign FPR worsens in two seeds. Replay design must protect both security recall and benign acceptance.
- NotInject and B-short are frozen external diagnostics only. B-short is language-unmatched. Neither bank may choose training data, thresholds, epochs, memory samples, or methods.
- Scores are often polarized near 0/1. The observed failures include hard-label flips and are not merely small threshold drift, but future replay evaluation must retain margin diagnostics.
- Python 3.14 emits the existing `torch.jit.script` FutureWarning; it did not affect the completed CUDA path."""
block_update(DOCS / "assumptions_and_risks.md", "CANDIDATE_B_NO_REPLAY_RESULT_20261005", risk_block)

questions_block = """## Candidate B post-result questions

| ID | Question | Current status |
|---|---|---|
| Q056 | May Candidate B be locally materialized? | Resolved: approved and validated. |
| Q057 | May the nine Candidate B no-replay runs begin? | Resolved: approved; 9/9 complete and validated. |
| Q058 | May replay or selective adaptation be implemented? | Replay: open separate human approval after protocol/budget review. Selective adaptation remains blocked. |
| Q059 | What one numerical replay memory budget should all methods use? | Open; no value selected. |
| Q060 | Does the human accept the R2 classification and authorize replay-protocol design? | Open human decision. |
| Q061 | Which stored fields and privacy/data-retention policy apply to replay memory? | Open; must be fixed before implementation. |
| Q062 | Should the minimal proposal include exactly no replay, Reservoir ER, DER++, and security-aware replay? | Open; recommended planning set, not authorized execution. |
| Q063 | What compute/stopping rule will govern a future comparison? | Open; must be predeclared in the proposal. |"""
block_update(DOCS / "open_questions.md", "CANDIDATE_B_NO_REPLAY_RESULT_20261005", questions_block)

timestamp = datetime.now(timezone.utc).isoformat()
code_hashes = {
    "configs/stronger_pilot_no_replay.yaml": sha256(ROOT / "configs/stronger_pilot_no_replay.yaml"),
    "src/continualguard/stronger_pilot_no_replay.py": sha256(ROOT / "src/continualguard/stronger_pilot_no_replay.py"),
    "src/continualguard/stronger_pilot_summary.py": sha256(ROOT / "src/continualguard/stronger_pilot_summary.py"),
    "scripts/run_stronger_pilot_no_replay.py": sha256(ROOT / "scripts/run_stronger_pilot_no_replay.py"),
    "scripts/summarize_stronger_pilot_no_replay.py": sha256(ROOT / "scripts/summarize_stronger_pilot_no_replay.py"),
}
decision = {
    "date": timestamp, "id": "D049", "decision": "Classify the complete validated Candidate B no-replay baseline as R2 and end benchmark escalation",
    "execution": {"runs": 9, "stage_end_checkpoints": 36, "failed": 0, "resumed": 0, "prediction_files": 162, "prediction_rows": 27324},
    "acquisition": "all 12 arm-stage gates pass in 3/3 seeds",
    "R2_evidence": {"arm": "B-SHORT", "task": "AF3", "metric": "attack_tpr_loss", "mean_pp": 35.0, "sample_sd_pp": 38.11532208676424, "per_seed_pp": [20.0, 6.666666666666665, 78.33333333333333], "harmful_seeds": 3, "matched_replication": "B-DOC AF3 attack-TPR loss 3.333 +/- 1.667 pp, harmful 3/3 seeds"},
    "readiness": {"R0": "exited", "R1": "satisfied", "R2": "active"},
    "replay_scientifically_justified": True, "full_replay_comparison_eligible_for_proposal": True,
    "replay_implemented": False, "selective_adaptation": False, "numerical_memory_budget": None,
    "benchmark_escalation": "ended", "next_prompt": next_prompt,
}
change = {
    "date": timestamp, "id": "C037", "decision_id": "D049", "reason": "Execute, validate, aggregate, and document the approved Candidate B three-arm no-replay baseline",
    "files_changed": ["configs/stronger_pilot_no_replay.yaml", "src/continualguard/stronger_pilot_no_replay.py", "src/continualguard/stronger_pilot_summary.py", "scripts/run_stronger_pilot_no_replay.py", "scripts/summarize_stronger_pilot_no_replay.py", "scripts/render_stronger_pilot_no_replay_report.py", "tests/test_stronger_pilot_no_replay.py", "tests/test_stronger_pilot_no_replay_results.py", "results/stronger_pilot_no_replay/**", "results/stronger_pilot_no_replay_aggregate.json", "results/stronger_pilot_no_replay_pairwise.json", "results/stronger_pilot_no_replay_readiness.json", "results/stronger_pilot_no_replay_artifact_validation.json", "docs/stronger_pilot_no_replay_results.md", "docs/stronger_pilot_no_replay_protocol.md", "docs/replay_readiness_review.md", "docs/replay_comparison_protocol.md", "docs/two_week_experiment_plan.md", "docs/decision_log.md", "docs/change_log.md", "docs/assumptions_and_risks.md", "docs/open_questions.md", "history/decisions.jsonl", "history/changes.jsonl"],
    "validation": ["read-only materialization integrity PASS", "technical CUDA preflight PASS", "9/9 runs and 36/36 checkpoints complete", "252 checkpoint file hashes PASS", "162 prediction files and 27,324 rows PASS", "Stage-1 three-arm and Stage-2 B-SHORT/B-HARD deterministic equivalence PASS", "focused tests 22 passed", "full suite 124 passed, 0 failed, 1 existing warning"],
    "code_config_hashes": code_hashes,
    "scope": "No-replay baseline and readiness classification only; replay/selective adaptation not implemented; numerical memory budget unselected",
}
history_upsert(ROOT / "history/decisions.jsonl", decision)
history_upsert(ROOT / "history/changes.jsonl", change)

print(json.dumps({"status": "PASS", "readiness": READY["classification"], "decision_id": "D049", "change_id": "C037", "next_prompt": next_prompt}, ensure_ascii=False, indent=2))
