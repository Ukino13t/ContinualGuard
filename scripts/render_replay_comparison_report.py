"""Render the validated Candidate B replay comparison into a review document."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def pct(value):
    return f"{100 * value:.3f}"


def stat(value):
    return f"{pct(value['mean'])} ± {pct(value['sample_sd'])} pp"


def method_label(key):
    return {"no_replay": "No Replay", "reservoir": "Reservoir ER", "derpp": "DER++", "security_aware": "Security-aware"}[key]


def main():
    agg = load("results/replay_comparison_aggregate.json")
    art = load("results/replay_artifact_validation.json")
    fair = load("results/replay_fairness_validation.json")
    memory = load("results/replay_memory_composition.json")["rows"]
    runagg = {(x["method"], x["arm"]): x for x in agg["run_summary_aggregate"]}
    retagg = {(x["method"], x["arm"], x["task"]): x for x in agg["retention_aggregate"]}
    lines = [
        "# Candidate B primary replay comparison results", "",
        "## Validated scope", "",
        f"The fixed comparison completed {art['new_runs']}/27 new runs and {art['new_checkpoints']}/108 new stage-end checkpoints with {art['failed_runs']} failed runs and {art['resumed_runs']} resumed runs. Artifact and fairness validation are **{art['status']}** and **{fair['status']}**. M0 reuses the previously validated nine-run no-replay baseline.", "",
        "All methods use the pinned DeBERTa-v3-base revision, Candidate B bytes and splits, AF2→AF3→AF4→AF5 order, seeds 17/42/2026, one epoch, current batch 4, AdamW at `1e-5`, optimizer resets, carried model weights, fixed threshold, and the same evaluation sets. Each replay method has `M=200` historical TRAIN examples and a four-example replay batch for every nonempty-memory current batch.", "",
        "DER++ uses `alpha=beta=0.5`. The paper reports this symmetric setting as stable; the project divides the summed objective by two to align overall loss scale. See [Buzzega et al.](https://papers.neurips.cc/paper_files/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf) and the authors' [Mammoth repository](https://github.com/aimagelab/mammoth). No coefficient was tuned on Candidate B or external diagnostics.", "",
        "## Retention summary", "",
        "Values are mean ± sample SD across three seeds in percentage points. Attack-TPR and macro-F1 BWT are signed higher-is-better changes; benign-FPR increase is signed harm when positive.", "",
        "| Method | Arm | Attack-TPR BWT | Macro-F1 BWT | Benign-FPR increase | Attack peak forgetting |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for method in ("no_replay", "reservoir", "derpp", "security_aware"):
        for arm in ("B-DOC", "B-SHORT", "B-HARD"):
            row = runagg[(method, arm)]
            lines.append(f"| {method_label(method)} | {arm} | {stat(row['attack_tpr_bwt'])} | {stat(row['macro_f1_bwt'])} | {stat(row['benign_fpr_increase'])} | {stat(row['attack_peak_forgetting_mean'])} |")
    lines += ["", "## Decisive B-SHORT AF3 trajectory", "", "| Method | Seed | Acquisition TPR | Final TPR | Final loss | Peak loss |", "|---|---:|---:|---:|---:|---:|"]
    short = sorted(agg["key_b_short_af3"], key=lambda x: (("no_replay", "reservoir", "derpp", "security_aware").index(x["method"]), x["seed"]))
    for row in short:
        lines.append(f"| {method_label(row['method'])} | {row['seed']} | {pct(row['attack_tpr_acquisition'])} | {pct(row['attack_tpr_final'])} | {pct(row['attack_tpr_loss'])} pp | {pct(row['attack_peak_forgetting'])} pp |")
    lines += ["", "Aggregate final B-SHORT AF3 losses:", ""]
    for method in ("no_replay", "reservoir", "derpp", "security_aware"):
        lines.append(f"- {method_label(method)}: {stat(retagg[(method, 'B-SHORT', 'AF3')]['attack_tpr_loss'])}.")
    lines += ["", "## Acquisition gates", ""]
    for method in ("no_replay", "reservoir", "derpp", "security_aware"):
        gates = [x for x in agg["acquisition_gates"] if x["method"] == method]
        passed = sum(x["pass"] for x in gates)
        failures = [f"{x['arm']}/{x['stage']} ({x['passing_seeds']}/3)" for x in gates if not x["pass"]]
        lines.append(f"- **{method_label(method)}:** {passed}/12 arm-stage gates pass; failures: {', '.join(failures) if failures else 'none'}.")
    lines += ["", "## B-HARD benign retention", "", "Mean signed benign-FPR increase across the three earlier tasks:", ""]
    for method in ("no_replay", "reservoir", "derpp", "security_aware"):
        lines.append(f"- {method_label(method)}: {stat(runagg[(method, 'B-HARD')]['benign_fpr_increase'])}.")
    lines += ["", "Partition-level B0-2, BH-1, and BH-2 trajectories for every seed and checkpoint are in `aggregate.table_ready.b_hard_benign_fpr_trajectory`.", "", "## External diagnostics at AF5", "", "| Method | Arm | NotInject FPR | B-short FPR | B-short English FPR | B-short multilingual FPR |", "|---|---|---:|---:|---:|---:|"]
    external = {(x["method"], x["arm"], x["checkpoint_stage"]): x for x in agg["external_aggregate"]}
    for method in ("no_replay", "reservoir", "derpp", "security_aware"):
        for arm in ("B-DOC", "B-SHORT", "B-HARD"):
            row = external[(method, arm, "AF5")]
            lines.append(f"| {method_label(method)} | {arm} | {stat(row['notinject_fpr'])} | {stat(row['bshort_fpr'])} | {stat(row['bshort_english_fpr'])} | {stat(row['bshort_multilingual_fpr'])} |")
    lines += ["", "These banks remained diagnostic-only and did not affect memory, coefficients, thresholds, or stopping.", "", "## Memory behavior", ""]
    for method in ("reservoir", "derpp", "security_aware"):
        finals = [x for x in memory if x["method"] == method and x["stage"] == "AF5"]
        benign = [x["label_counts"].get("0", 0) for x in finals]
        malicious = [x["label_counts"].get("1", 0) for x in finals]
        logits = [x["auxiliary_logit_bytes_float32"] for x in finals]
        lines.append(f"- **{method_label(method)}:** final buffers 200/200 in all nine runs; benign range {min(benign)}–{max(benign)}, malicious range {min(malicious)}–{max(malicious)}, auxiliary logit bytes range {min(logits)}–{max(logits)}.")
    lines += ["", "Security-aware buffers target and achieve 100 benign plus 100 malicious examples, use deterministic per-stage quotas, and rank within quota by frozen pre-update risk with coverage and hash tie-breaking. Full risk quantiles, high-risk counts, stage quotas, families, groups, positions, and sources are in `results/replay_memory_composition.json`.", "", "## Runtime and memory", "", "| Method | Arm | Runtime | Peak allocated GPU memory | Replay forward passes |", "|---|---|---:|---:|---:|"]
    for method in ("no_replay", "reservoir", "derpp", "security_aware"):
        for arm in ("B-DOC", "B-SHORT", "B-HARD"):
            row = runagg[(method, arm)]
            runtime = row["runtime_seconds"]
            peak = row["peak_gpu_memory_bytes"]
            passes = row["replay_forward_passes"]
            lines.append(f"| {method_label(method)} | {arm} | {runtime['mean']/60:.2f} ± {runtime['sample_sd']/60:.2f} min | {peak['mean']/2**30:.3f} ± {peak['sample_sd']/2**30:.3f} GiB | {passes['mean']:.0f} |")
    lines += ["", "## Validation and limits", "", f"Validation hashed {art['checkpoint_files_hashed']} checkpoint files and checked {art['prediction_files']} raw-prediction files containing {art['prediction_rows']} rows. Initialization matched M0, Candidate B hashes matched, every memory identity resolved to an arm-local TRAIN record with matching text hash, all stage lineages passed, and no future-stage, validation, test, NotInject, or B-short record entered memory. No unpublished attempt directory remained.", "", "There are only three seeds. Candidate B is a constructed, reviewed pilot; AF5 is compositional rather than family-disjoint; source/language/format factors differ among arms after declared anchors. Differences support a method comparison within this protocol and should not be overstated as broad deployment evidence.", "", "## Machine-readable outputs", "", "- `results/replay_comparison_aggregate.json`", "- `results/replay_comparison_pairwise.json`", "- `results/replay_memory_composition.json`", "- `results/replay_fairness_validation.json`", "- `results/replay_artifact_validation.json`", ""]
    (ROOT / "docs/replay_comparison_results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
