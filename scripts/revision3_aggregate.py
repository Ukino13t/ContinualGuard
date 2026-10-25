"""Aggregate only fully validated A0/A1 M=200 trajectories and frozen M1 context."""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
ZROOT = Path(r"Z:\ContinualGuard_Revision3")
OUT = ROOT / "results/revision3/analysis"
REPORT = ROOT / "reports/revision3/02_class_balance_ablation.md"
SELECTOR_SUMMARY = ROOT / "results/revision3/selector_audit/summary.json"
sys.path.insert(0, str(ROOT / "scripts"))
from aggregate_route_a import cell, summarize_run  # noqa: E402

STAGES = ("AF2", "AF3", "AF4", "AF5")
ARMS = ("doc", "short", "hard")
SEEDS = (17, 42, 2026)
METHODS = {
    "A0_matched": ZROOT / "results/revision3/matched_ablation_amp256/M200/a0_matched",
    "A1_matched": ROOT / "results/revision2/matched_ablation_amp256/M200/a1_matched",
    "M1_Reservoir_original": ROOT / "results/replay_comparison_candidate_b/reservoir",
}
ENDPOINTS = ("tpr_bwt_pp", "macro_f1_bwt_pp", "benign_fpr_increase_pp",
             "attack_peak_forgetting_pp", "acquisition_tpr_pct", "acquisition_macro_f1_pct",
             "worst_benign_group_fpr_drift_pp", "worst_attack_position_tpr_loss_pp",
             "final_attack_tpr_pct", "final_benign_fpr_pct", "final_macro_f1_pct",
             "runtime_seconds", "peak_gpu_memory_bytes")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    validation = load(OUT / "validation_M200.json")
    assert validation["status"] == "PASS" and validation["full_checkpoint_hashes"]
    assert validation["runs_checked"] == 9
    assert {(r["arm"], r["seed"]) for r in validation["runs"]} == {
        (arm, seed) for arm in ARMS for seed in SEEDS}
    per_seed, curves, buffers, acquisitions = [], [], [], []
    for method, base in METHODS.items():
        for arm in ARMS:
            for seed in SEEDS:
                stages = [load(base / arm / f"seed{seed}/stage{i}/metrics.json") for i in range(1, 5)]
                row = {"method": method, "arm": arm, "seed": seed, **summarize_run(stages)}
                for metric, name in (("tpr", "attack_tpr"), ("fpr", "benign_fpr"),
                                     ("macro_f1", "macro_f1")):
                    row[f"final_{name}_pct"] = statistics.mean(
                        cell(stages[-1], task, metric) * 100 for task in STAGES)
                per_seed.append(row)
                for trained_index, stage_metrics in enumerate(stages):
                    trained = STAGES[trained_index]
                    acquisitions.append({"method": method, "arm": arm, "seed": seed,
                                         "trained_stage": trained,
                                         "attack_tpr_pct": cell(stage_metrics, trained, "tpr") * 100,
                                         "benign_fpr_pct": cell(stage_metrics, trained, "fpr") * 100,
                                         "macro_f1_pct": cell(stage_metrics, trained, "macro_f1") * 100,
                                         "training_loss": stage_metrics.get("training_loss"),
                                         "validation_loss": stage_metrics.get("validation_loss"),
                                         "runtime_seconds": stage_metrics["runtime_seconds"],
                                         "peak_gpu_memory_bytes": stage_metrics["peak_gpu_memory_bytes"]})
                    for task in STAGES[:trained_index + 1]:
                        evaluation = stage_metrics["evaluations"][task]
                        curves.append({"method": method, "arm": arm, "seed": seed,
                                       "trained_stage": trained, "eval_stage": task,
                                       "attack_tpr_pct": cell(stage_metrics, task, "tpr") * 100,
                                       "benign_fpr_pct": cell(stage_metrics, task, "fpr") * 100,
                                       "macro_f1_pct": cell(stage_metrics, task, "macro_f1") * 100,
                                       "accuracy_pct": cell(stage_metrics, task, "accuracy") * 100,
                                       "attack_tpr_by_position": json.dumps(
                                           evaluation.get("attack_tpr_by_position", {}), sort_keys=True),
                                       "benign_fpr_by_group": json.dumps(
                                           evaluation.get("benign_fpr_by_group", {}), sort_keys=True)})
                    composition = stage_metrics.get("buffer_after_training", {})
                    buffers.append({"method": method, "arm": arm, "seed": seed,
                                    "stage": trained,
                                    "size": composition.get("size", composition.get("buffer_size")),
                                    "by_label": json.dumps(composition.get("by_label", composition.get("label_counts")), sort_keys=True),
                                    "by_stage": json.dumps(composition.get("by_stage", composition.get("stage_distribution")), sort_keys=True)})
    lookup = {(r["method"], r["arm"], r["seed"]): r for r in per_seed}
    paired = []
    for arm in ARMS:
        for seed in SEEDS:
            a0 = lookup[("A0_matched", arm, seed)]
            a1 = lookup[("A1_matched", arm, seed)]
            paired.append({"arm": arm, "seed": seed,
                           **{f"a1_minus_a0_{key}": a1[key] - a0[key]
                              for key in ENDPOINTS}})
    summary = []
    for method in METHODS:
        for arm in ARMS:
            group = [r for r in per_seed if r["method"] == method and r["arm"] == arm]
            result = {"method": method, "arm": arm, "n_seeds": 3}
            for key in ENDPOINTS:
                values = [r[key] for r in group]
                result[f"{key}_mean"] = statistics.mean(values)
                result[f"{key}_sample_sd"] = statistics.stdev(values)
            summary.append(result)
    paired_summary = []
    for arm in ARMS:
        group = [r for r in paired if r["arm"] == arm]
        result = {"arm": arm, "n_seeds": 3}
        for key in ENDPOINTS:
            values = [r[f"a1_minus_a0_{key}"] for r in group]
            result[f"a1_minus_a0_{key}_mean"] = statistics.mean(values)
            result[f"a1_minus_a0_{key}_sample_sd"] = statistics.stdev(values)
            result[f"a1_minus_a0_{key}_positive_seeds"] = sum(x > 1e-9 for x in values)
            result[f"a1_minus_a0_{key}_negative_seeds"] = sum(x < -1e-9 for x in values)
        paired_summary.append(result)
    for name, records in (("per_seed.csv", per_seed), ("aggregate.csv", summary),
                          ("paired_per_seed.csv", paired), ("paired_aggregate.csv", paired_summary),
                          ("stage_curves.csv", curves), ("memory_composition.csv", buffers),
                          ("stage_acquisition.csv", acquisitions)):
        write_csv(OUT / name, records)
    selector = load(SELECTOR_SUMMARY)
    lines = ["# Matched A0/A1 class-balancing ablation", "",
             "Status: complete, full-hash validated; Candidate B only. Original M1 Reservoir is contextual and uses a different selection/numerical protocol.", "",
             f"The selector audit reproduced all frozen A1 buffers and found quota intervention in {selector['cells_with_actual_selection_change']}/36 arm--seed--stage cells. Actual later eligible pools can diverge following an earlier selection difference. Candidate B stage-test TPR/FPR resolution is 1.667 pp; each 20-item attack-position slice resolves 5 pp.", "",
             "## Per-seed endpoints", "",
             "| Arm | Method | Seed | TPR BWT (pp) | macro-F1 BWT (pp) | FPR increase (pp) | Final TPR (%) | Final FPR (%) | Acquisition TPR (%) | Peak forgetting (pp) | Worst position loss (pp) | Worst benign-group drift (pp) | Runtime (min) | Peak GPU (GiB) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for method in METHODS:
            for seed in SEEDS:
                r = lookup[(method, arm, seed)]
                lines.append("| {arm} | {method} | {seed} | {tpr:+.2f} | {f1:+.2f} | {fpr:+.2f} | {final_tpr:.2f} | {final_fpr:.2f} | {acq:.2f} | {peak:.2f} | {pos:.2f} | {group:.2f} | {runtime:.2f} | {gpu:.3f} |".format(
                    arm=arm, method=method, seed=seed, tpr=r["tpr_bwt_pp"],
                    f1=r["macro_f1_bwt_pp"], fpr=r["benign_fpr_increase_pp"],
                    final_tpr=r["final_attack_tpr_pct"], final_fpr=r["final_benign_fpr_pct"],
                    acq=r["acquisition_tpr_pct"], peak=r["attack_peak_forgetting_pp"],
                    pos=r["worst_attack_position_tpr_loss_pp"],
                    group=r["worst_benign_group_fpr_drift_pp"],
                    runtime=r["runtime_seconds"] / 60,
                    gpu=r["peak_gpu_memory_bytes"] / (1024 ** 3)))
    lines += ["", "## Paired A1 minus A0 differences", "",
              "Positive TPR/macro-F1 BWT is beneficial; negative FPR increase is beneficial. These are descriptive three-seed contrasts, not significance tests.", "",
              "| Arm | Seed | TPR BWT difference (pp) | macro-F1 BWT difference (pp) | FPR-increase difference (pp) |",
              "|---|---:|---:|---:|---:|"]
    for p in paired:
        lines.append(f"| {p['arm']} | {p['seed']} | {p['a1_minus_a0_tpr_bwt_pp']:+.2f} | {p['a1_minus_a0_macro_f1_bwt_pp']:+.2f} | {p['a1_minus_a0_benign_fpr_increase_pp']:+.2f} |")
    lines += ["", "| Arm | TPR BWT difference mean ± sample SD (pp) | macro-F1 BWT difference | FPR-increase difference |",
              "|---|---:|---:|---:|"]
    for p in paired_summary:
        lines.append("| {arm} | {t:+.2f} ± {ts:.2f} | {f:+.2f} ± {fs:.2f} | {b:+.2f} ± {bs:.2f} |".format(
            arm=p["arm"], t=p["a1_minus_a0_tpr_bwt_pp_mean"],
            ts=p["a1_minus_a0_tpr_bwt_pp_sample_sd"],
            f=p["a1_minus_a0_macro_f1_bwt_pp_mean"],
            fs=p["a1_minus_a0_macro_f1_bwt_pp_sample_sd"],
            b=p["a1_minus_a0_benign_fpr_increase_pp_mean"],
            bs=p["a1_minus_a0_benign_fpr_increase_pp_sample_sd"]))
    lines += ["", "Every stage/evaluation cell, acquisition loss, position and benign-group slice, and memory composition is in the sibling CSVs. Selected IDs, buffer Jaccard, rank medians, and quota-affected counts are in `results/revision3/selector_audit/`. The original online Reservoir is retained separately and must not be interpreted as A0-matched.", ""]
    lines += ["## Interpretation", "",
              "Class quotas changed selected identities in 32/36 audited updates, so this is an active selector intervention. Its mean selected-ID Jaccard is 0.9683; eligible pools can diverge after the first intervention. Attack-retention differences are small and change direction across B-DOC and B-HARD seeds. B-SHORT benign-FPR increase is lower for A1 in two seeds and tied in the third, but this is only a three-seed, 60-benign-item-per-test descriptive result.", "",
              "Neither these data nor the stage-quota A1/A2 results establish a universally superior structured memory policy. The original M1 online Reservoir remains competitive and is a contextual baseline because its selection and numerical policy differ. Per-stage attack and benign trajectories are in `stage_curves.csv` and the vector plots `figures/revision3/matched_retention_and_fpr.pdf`; final memory composition and selector overlap are in `memory_composition.csv`, `selector_cells.csv`, and `figures/revision3/matched_memory_and_overlap.pdf`.", ""]
    for arm in ARMS:
        final_a0 = [int(json.loads(row["by_label"])["0"])
                    for row in buffers if row["method"] == "A0_matched" and
                    row["arm"] == arm and row["stage"] == "AF5"]
        assert len(final_a0) == 3
        lines.append(f"Final A0 benign occupancy in {arm}, seeds 17/42/2026: "
                     + "/".join(map(str, final_a0)) + " of 200 (A1 fixes 100/200).")
    lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "PASS", "per_seed_rows": len(per_seed),
                      "paired_rows": len(paired), "curves": len(curves)}))


if __name__ == "__main__":
    main()
