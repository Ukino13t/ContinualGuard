"""Aggregate frozen and Route A Candidate B runs without changing source results."""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
STAGES = ("AF2", "AF3", "AF4", "AF5")
SEEDS = (17, 42, 2026)
ARMS = ("doc", "short", "hard")
METHODS = {
    "a0_reservoir": ROOT / "results/replay_comparison_candidate_b/reservoir",
    "a1_class": ROOT / "results/extended/ablation/a1_class",
    "a2_class_stage": ROOT / "results/extended/ablation/a2_class_stage",
    "a3_coverage": ROOT / "results/extended/ablation/a3_coverage",
    "a4_ce": ROOT / "results/replay_comparison_candidate_b/loss_aware",
    "a5_security": ROOT / "results/replay_comparison_candidate_b/security_aware",
    "m0_no_replay": ROOT / "results/stronger_pilot_no_replay",
    "compute_matched": ROOT / "results/extended/offline_controls/compute_matched_current_only",
}
JOINT = ROOT / "results/extended/offline_controls/joint_union_1epoch"
OUT = ROOT / "results/extended/analysis"


def stage_metrics(base: Path, arm: str, seed: int):
    path = base / arm / f"seed{seed}"
    result = []
    for i in range(1, 5):
        p = path / f"stage{i}" / "metrics.json"
        if not p.exists():
            return None
        result.append(json.loads(p.read_text(encoding="utf-8")))
    return result


def cell(metric: dict, task: str, kind: str) -> float:
    item = metric["evaluations"][task]
    return {"tpr": item["malicious"]["attack_tpr"],
            "fpr": item["benign"]["benign_fpr"],
            "macro_f1": item["paired"]["macro_f1"],
            "accuracy": item["paired"]["accuracy"]}[kind]


def summarize_run(metrics: list[dict]) -> dict:
    final = metrics[-1]
    earlier = STAGES[:-1]
    result = {}
    for key in ("tpr", "macro_f1", "accuracy"):
        delta = [(cell(final, task, key) - cell(metrics[i], task, key)) * 100
                 for i, task in enumerate(earlier)]
        result[f"{key}_bwt_pp"] = statistics.mean(delta)
    result["benign_fpr_increase_pp"] = statistics.mean(
        (cell(final, task, "fpr") - cell(metrics[i], task, "fpr")) * 100
        for i, task in enumerate(earlier))
    result["attack_peak_forgetting_pp"] = statistics.mean(
        max((max(cell(metrics[k], task, "tpr") for k in range(i, j + 1))
             - cell(metrics[j], task, "tpr")) * 100 for j in range(i, 4))
        for i, task in enumerate(earlier))
    result["acquisition_tpr_pct"] = statistics.mean(
        cell(metrics[i], task, "tpr") * 100 for i, task in enumerate(STAGES))
    result["acquisition_macro_f1_pct"] = statistics.mean(
        cell(metrics[i], task, "macro_f1") * 100 for i, task in enumerate(STAGES))
    result["acquisition_fpr_pct"] = statistics.mean(
        cell(metrics[i], task, "fpr") * 100 for i, task in enumerate(STAGES))
    result["runtime_seconds"] = sum(x["runtime_seconds"] for x in metrics)
    result["peak_gpu_memory_bytes"] = max(x["peak_gpu_memory_bytes"] for x in metrics)
    group_drifts = []
    attack_position_losses = []
    for i, task in enumerate(earlier):
        before = metrics[i]["evaluations"][task].get("benign_fpr_by_group", {})
        after = final["evaluations"][task].get("benign_fpr_by_group", {})
        for group in before.keys() & after.keys():
            group_drifts.append((after[group] - before[group]) * 100)
        before_pos = metrics[i]["evaluations"][task].get("attack_tpr_by_position", {})
        after_pos = final["evaluations"][task].get("attack_tpr_by_position", {})
        for position in before_pos.keys() & after_pos.keys():
            attack_position_losses.append((before_pos[position] - after_pos[position]) * 100)
    result["worst_benign_group_fpr_drift_pp"] = max([0., *group_drifts])
    result["worst_attack_position_tpr_loss_pp"] = max([0., *attack_position_losses])
    return result


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per_run, curves, memory, available = [], [], [], {}
    for method, base in METHODS.items():
        available[method] = []
        for arm in ARMS:
            for seed in SEEDS:
                metrics = stage_metrics(base, arm, seed)
                if metrics is None:
                    continue
                available[method].append(f"{arm}/seed{seed}")
                per_run.append({"method": method, "arm": arm, "seed": seed, **summarize_run(metrics)})
                for train_i, metric in enumerate(metrics):
                    for eval_task in STAGES[:train_i + 1]:
                        curves.append({"method": method, "arm": arm, "seed": seed,
                            "trained_stage": STAGES[train_i], "eval_stage": eval_task,
                            "attack_tpr_pct": cell(metric, eval_task, "tpr") * 100,
                            "benign_fpr_pct": cell(metric, eval_task, "fpr") * 100,
                            "macro_f1_pct": cell(metric, eval_task, "macro_f1") * 100})
                    buffer = metric.get("buffer_after_training")
                    if buffer:
                        memory.append({"method": method, "arm": arm, "seed": seed,
                            "stage": STAGES[train_i], "size": buffer.get("size", buffer.get("buffer_size")),
                            "coverage_keys": buffer.get("coverage_keys", buffer.get("unique_coverage_keys")),
                            "by_label": json.dumps(buffer.get("by_label", buffer.get("label_counts"))),
                            "by_stage": json.dumps(buffer.get("by_stage", buffer.get("stage_distribution")))})
    aggregates = []
    for method in METHODS:
        for arm in ARMS:
            selected = [x for x in per_run if x["method"] == method and x["arm"] == arm]
            if not selected:
                continue
            row = {"method": method, "arm": arm, "n_seeds": len(selected)}
            for key in selected[0]:
                if key in ("method", "arm", "seed"):
                    continue
                values = [x[key] for x in selected]
                row[f"{key}_mean"] = statistics.mean(values)
                row[f"{key}_sample_sd"] = statistics.stdev(values) if len(values) > 1 else ""
            aggregates.append(row)
    write_csv(OUT / "per_run.csv", per_run)
    write_csv(OUT / "aggregate.csv", aggregates)
    write_csv(OUT / "per_stage_curves.csv", curves)
    write_csv(OUT / "memory_composition.csv", memory)
    by_key = {(r["method"], r["arm"], r["seed"]): r for r in per_run}
    comparisons = (("a0_reservoir", "a1_class"), ("a1_class", "a2_class_stage"),
                   ("a2_class_stage", "a3_coverage"), ("a3_coverage", "a4_ce"),
                   ("a4_ce", "a5_security"), ("m0_no_replay", "a0_reservoir"),
                   ("m0_no_replay", "compute_matched"))
    paired = []
    paired_metrics = ("tpr_bwt_pp", "macro_f1_bwt_pp", "benign_fpr_increase_pp",
                      "attack_peak_forgetting_pp", "acquisition_tpr_pct",
                      "acquisition_macro_f1_pct", "worst_benign_group_fpr_drift_pp",
                      "worst_attack_position_tpr_loss_pp",
                      "runtime_seconds")
    for before, after in comparisons:
        for arm in ARMS:
            for seed in SEEDS:
                a = by_key.get((before, arm, seed))
                b = by_key.get((after, arm, seed))
                if a is None or b is None:
                    continue
                paired.append({"before": before, "after": after, "arm": arm, "seed": seed,
                               **{f"delta_{k}": b[k] - a[k] for k in paired_metrics}})
    paired_aggregate = []
    for before, after in comparisons:
        for arm in ARMS:
            subset = [x for x in paired if x["before"] == before and x["after"] == after and x["arm"] == arm]
            if not subset:
                continue
            row = {"before": before, "after": after, "arm": arm, "n_seeds": len(subset)}
            for key in paired_metrics:
                vals = [x[f"delta_{key}"] for x in subset]
                row[f"delta_{key}_mean"] = statistics.mean(vals)
                row[f"delta_{key}_sample_sd"] = statistics.stdev(vals) if len(vals) > 1 else ""
                row[f"delta_{key}_positive_seeds"] = sum(x > 0 for x in vals)
                row[f"delta_{key}_negative_seeds"] = sum(x < 0 for x in vals)
            paired_aggregate.append(row)
    write_csv(OUT / "paired_per_seed.csv", paired)
    write_csv(OUT / "paired_aggregate.csv", paired_aggregate)
    joint = []
    for arm in ARMS:
        for seed in SEEDS:
            p = JOINT / arm / f"seed{seed}" / "joint/metrics.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            for task in STAGES:
                v = d["evaluations"][task]
                joint.append({"arm": arm, "seed": seed, "task": task,
                    "attack_tpr_pct": v["malicious"]["attack_tpr"] * 100,
                    "benign_fpr_pct": v["benign"]["benign_fpr"] * 100,
                    "macro_f1_pct": v["paired"]["macro_f1"] * 100,
                    "accuracy_pct": v["paired"]["accuracy"] * 100,
                    "runtime_seconds": d["runtime_seconds"],
                    "peak_gpu_memory_bytes": d["peak_gpu_memory_bytes"]})
    write_csv(OUT / "joint_reference.csv", joint)
    (OUT / "availability.json").write_text(json.dumps(available, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(per_run), "joint_task_rows": len(joint), "availability": available}))


if __name__ == "__main__":
    main()
