"""Summarize only the full-hash-validated two-arm matched capacity grid."""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
ZROOT = Path(r"Z:\ContinualGuard_Revision3")
OUT = ROOT / "results/revision3/analysis"
sys.path.insert(0, str(ROOT / "scripts"))
from aggregate_route_a import cell, summarize_run  # noqa: E402
from scripts.revision3_capacity_validate import source  # noqa: E402

STAGES = ("AF2", "AF3", "AF4", "AF5")
ENDPOINTS = ("tpr_bwt_pp", "macro_f1_bwt_pp", "benign_fpr_increase_pp",
             "attack_peak_forgetting_pp", "acquisition_tpr_pct",
             "acquisition_macro_f1_pct", "final_attack_tpr_pct",
             "final_benign_fpr_pct", "final_macro_f1_pct",
             "worst_benign_group_fpr_drift_pp", "worst_attack_position_tpr_loss_pp",
             "final_buffer_occupancy", "final_benign_memory_count", "runtime_seconds",
             "peak_gpu_memory_bytes")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    validation = load(OUT / "capacity_validation.json")
    assert validation["status"] == "PASS" and validation["runs_checked"] == 36
    assert validation["full_checkpoint_hashes"] and len(validation["matched_pairs"]) == 18
    per_seed, stage_curves, memory_overlap = [], [], []
    for arm in ("short", "hard"):
        for capacity in (50, 200, 400):
            for seed in (17, 42, 2026):
                for method in ("a0_matched", "a1_matched"):
                    run = source(method, arm, seed, capacity)
                    stages = [load(run / f"stage{i}/metrics.json") for i in range(1, 5)]
                    row = {"arm": arm, "capacity": capacity, "seed": seed,
                           "method": method, **summarize_run(stages)}
                    for metric, name in (("tpr", "attack_tpr"), ("fpr", "benign_fpr"),
                                         ("macro_f1", "macro_f1")):
                        row[f"final_{name}_pct"] = statistics.mean(
                            cell(stages[-1], task, metric) * 100 for task in STAGES)
                    memory = load(run / "stage4/memory_composition.json")
                    row["final_buffer_occupancy"] = memory["size"]
                    row["final_benign_memory_count"] = memory["by_label"]["0"]
                    per_seed.append(row)
                    for i, stage in enumerate(stages):
                        for task in STAGES[:i + 1]:
                            stage_curves.append({"arm": arm, "capacity": capacity,
                                                 "seed": seed, "method": method,
                                                 "trained_stage": STAGES[i], "eval_stage": task,
                                                 "attack_tpr_pct": cell(stage, task, "tpr") * 100,
                                                 "benign_fpr_pct": cell(stage, task, "fpr") * 100,
                                                 "macro_f1_pct": cell(stage, task, "macro_f1") * 100})
    lookup = {(r["arm"], r["capacity"], r["seed"], r["method"]): r for r in per_seed}
    paired = []
    for arm in ("short", "hard"):
        for capacity in (50, 200, 400):
            for seed in (17, 42, 2026):
                a0 = lookup[(arm, capacity, seed, "a0_matched")]
                a1 = lookup[(arm, capacity, seed, "a1_matched")]
                paired.append({"arm": arm, "capacity": capacity, "seed": seed,
                               **{f"a1_minus_a0_{metric}": a1[metric] - a0[metric]
                                  for metric in ENDPOINTS}})
                for i, stage in enumerate(STAGES, 1):
                    a0_ids = {x["example_id"] for x in jsonl(
                        source("a0_matched", arm, seed, capacity) / f"stage{i}/buffer.jsonl")}
                    a1_ids = {x["example_id"] for x in jsonl(
                        source("a1_matched", arm, seed, capacity) / f"stage{i}/buffer.jsonl")}
                    assert len(a0_ids) == len(a1_ids) == capacity
                    memory_overlap.append({"arm": arm, "capacity": capacity,
                                           "seed": seed, "stage": stage,
                                           "selected_id_jaccard": len(a0_ids & a1_ids) / len(a0_ids | a1_ids),
                                           "selected_id_intersection": len(a0_ids & a1_ids)})
    summary, paired_summary = [], []
    for arm in ("short", "hard"):
        for capacity in (50, 200, 400):
            for method in ("a0_matched", "a1_matched"):
                group = [r for r in per_seed if r["arm"] == arm and
                         r["capacity"] == capacity and r["method"] == method]
                row = {"arm": arm, "capacity": capacity, "method": method, "n_seeds": 3}
                for metric in ENDPOINTS:
                    values = [r[metric] for r in group]
                    row[f"{metric}_mean"] = statistics.mean(values)
                    row[f"{metric}_sample_sd"] = statistics.stdev(values)
                summary.append(row)
            group = [r for r in paired if r["arm"] == arm and r["capacity"] == capacity]
            row = {"arm": arm, "capacity": capacity, "n_seeds": 3}
            for metric in ENDPOINTS:
                values = [r[f"a1_minus_a0_{metric}"] for r in group]
                row[f"a1_minus_a0_{metric}_mean"] = statistics.mean(values)
                row[f"a1_minus_a0_{metric}_sample_sd"] = statistics.stdev(values)
            paired_summary.append(row)
    for name, rows in (("capacity_per_seed.csv", per_seed),
                       ("capacity_aggregate.csv", summary),
                       ("capacity_paired_per_seed.csv", paired),
                       ("capacity_paired_aggregate.csv", paired_summary),
                       ("capacity_stage_curves.csv", stage_curves),
                       ("capacity_memory_overlap.csv", memory_overlap)):
        write_csv(OUT / name, rows)
    print(json.dumps({"status": "PASS", "runs": len(per_seed),
                      "paired": len(paired), "stage_curve_rows": len(stage_curves),
                      "memory_overlap_rows": len(memory_overlap)}))


if __name__ == "__main__":
    main()
