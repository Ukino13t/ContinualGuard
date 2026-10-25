"""Aggregate versioned Revision 2 runs and frozen references, without rewriting either."""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from aggregate_route_a import STAGES, SEEDS, ARMS, ROOT, cell, summarize_run

OUT = ROOT / "results/revision2/analysis"
MATCHED = ROOT / "results/revision2/matched_ablation_amp256"
CAPACITY_RESERVOIR = ROOT / "results/revision2/memory_sensitivity_amp256"
NEW_COMPUTE = ROOT / "results/revision2/offline_controls/compute_matched_current_only"
OLD_COMPUTE = ROOT / "results/extended/offline_controls/compute_matched_current_only"
BASES = {
    "M0_NoReplay": ROOT / "results/stronger_pilot_no_replay",
    "A0_Reservoir": ROOT / "results/replay_comparison_candidate_b/reservoir",
    "A4_CE": ROOT / "results/replay_comparison_candidate_b/loss_aware",
    "A5_Risk": ROOT / "results/replay_comparison_candidate_b/security_aware",
    "A1_matched": MATCHED / "M200/a1_matched",
    "A2_matched": MATCHED / "M200/a2_matched",
}
KEYS = ("tpr_bwt_pp", "macro_f1_bwt_pp", "benign_fpr_increase_pp",
        "attack_peak_forgetting_pp", "acquisition_tpr_pct",
        "acquisition_macro_f1_pct", "worst_benign_group_fpr_drift_pp",
        "worst_attack_position_tpr_loss_pp", "final_attack_tpr_pct",
        "final_benign_fpr_pct", "final_macro_f1_pct", "final_accuracy_pct",
        "runtime_seconds", "peak_gpu_memory_bytes")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def read_stages(base: Path, arm: str, seed: int):
    folder = base / arm / f"seed{seed}"
    files = [folder / f"stage{i}/metrics.json" for i in range(1, 5)]
    if not all(path.is_file() for path in files):
        return None
    return [json.loads(path.read_text(encoding="utf-8")) for path in files]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_run, curves, buffers, diagnostics = [], [], [], []
    methods = dict(BASES)
    for cap in (50, 200, 400):
        if cap != 200:
            for variant in ("a1_matched",):
                methods[f"{variant}_M{cap}"] = MATCHED / f"M{cap}/{variant}"
        methods[f"reservoir_capacity_M{cap}"] = CAPACITY_RESERVOIR / f"M{cap}/reservoir_capacity"
    methods["compute_current_only"] = NEW_COMPUTE
    for method, base in methods.items():
        for arm in ARMS:
            for seed in SEEDS:
                chosen = OLD_COMPUTE if method == "compute_current_only" and seed == 42 else base
                stages = read_stages(chosen, arm, seed)
                if stages is None:
                    continue
                row = {"method": method, "arm": arm, "seed": seed, **summarize_run(stages)}
                row.update({f"final_{name}_pct": statistics.mean(
                    cell(stages[-1], task, metric) * 100 for task in STAGES)
                    for name, metric in (("attack_tpr", "tpr"), ("benign_fpr", "fpr"),
                                         ("macro_f1", "macro_f1"), ("accuracy", "accuracy"))})
                row["memory_capacity"] = (int(method.rsplit("M", 1)[-1])
                                          if method.startswith(("a1_matched_M", "reservoir_capacity_M")) else 200)
                per_run.append(row)
                for i, stage in enumerate(stages):
                    acquired = STAGES[i]
                    training = stage.get("training", {})
                    diagnostics.append({
                        "method": method, "arm": arm, "seed": seed,
                        "trained_stage": acquired,
                        "acquisition_attack_tpr_pct": cell(stage, acquired, "tpr") * 100,
                        "acquisition_benign_fpr_pct": cell(stage, acquired, "fpr") * 100,
                        "acquisition_macro_f1_pct": cell(stage, acquired, "macro_f1") * 100,
                        "acquisition_accuracy_pct": cell(stage, acquired, "accuracy") * 100,
                        "training_loss": stage.get("training_loss"),
                        "validation_loss": stage.get("validation_loss"),
                        "optimizer_steps": training.get("optimizer_steps"),
                        "current_examples_exposed": training.get("current_examples", training.get("current_examples_exposed")),
                        "historical_examples_exposed": training.get("replay_examples_exposed", training.get("historical_examples_exposed")),
                        "runtime_seconds": stage.get("runtime_seconds"),
                        "peak_gpu_memory_bytes": stage.get("peak_gpu_memory_bytes"),
                    })
                    for task in STAGES[:i + 1]:
                        curves.append({"method": method, "arm": arm, "seed": seed,
                                       "trained_stage": STAGES[i], "tested_stage": task,
                                       "attack_tpr_pct": cell(stage, task, "tpr") * 100,
                                       "benign_fpr_pct": cell(stage, task, "fpr") * 100,
                                       "macro_f1_pct": cell(stage, task, "macro_f1") * 100,
                                       "accuracy_pct": cell(stage, task, "accuracy") * 100})
                    if "buffer_after_training" in stage:
                        buffers.append({"method": method, "arm": arm, "seed": seed,
                                        "stage": STAGES[i],
                                        "buffer": json.dumps(stage["buffer_after_training"], sort_keys=True)})
    write_csv(OUT / "per_seed.csv", per_run)
    write_csv(OUT / "stage_curves.csv", curves)
    write_csv(OUT / "stage_diagnostics.csv", diagnostics)
    write_csv(OUT / "buffer_composition.csv", buffers)
    aggregated = []
    for (method, arm), rows in sorted(defaultdict(list, {
        key: [r for r in per_run if (r["method"], r["arm"]) == key]
        for key in {(r["method"], r["arm"]) for r in per_run}
    }).items()):
        out = {"method": method, "arm": arm, "n_seeds": len(rows)}
        for key in KEYS:
            values = [r[key] for r in rows]
            out[f"{key}_mean"] = statistics.mean(values)
            out[f"{key}_sample_sd"] = statistics.stdev(values) if len(values) > 1 else ""
        aggregated.append(out)
    write_csv(OUT / "aggregate.csv", aggregated)
    pairs = []
    contrasts = (("A1_matched", "A2_matched"), ("A0_Reservoir", "A1_matched"),
                 ("M0_NoReplay", "compute_current_only"),
                 ("compute_current_only", "A0_Reservoir"))
    lookup = {(r["method"], r["arm"], r["seed"]): r for r in per_run}
    for before, after in contrasts:
        for arm in ARMS:
            for seed in SEEDS:
                a = lookup.get((before, arm, seed)); b = lookup.get((after, arm, seed))
                if a is None or b is None:
                    continue
                pairs.append({"before": before, "after": after, "arm": arm, "seed": seed,
                              **{f"delta_{key}": b[key] - a[key] for key in KEYS}})
    write_csv(OUT / "paired_per_seed.csv", pairs)
    pair_summary = []
    for before, after in contrasts:
        for arm in ARMS:
            subset = [r for r in pairs if r["before"] == before and r["after"] == after and r["arm"] == arm]
            if not subset:
                continue
            row = {"before": before, "after": after, "arm": arm, "n_seeds": len(subset)}
            for key in KEYS:
                values = [r[f"delta_{key}"] for r in subset]
                row[f"delta_{key}_mean"] = statistics.mean(values)
                row[f"delta_{key}_sample_sd"] = statistics.stdev(values) if len(values) > 1 else ""
                row[f"delta_{key}_positive_seeds"] = sum(v > 0 for v in values)
                row[f"delta_{key}_negative_seeds"] = sum(v < 0 for v in values)
            pair_summary.append(row)
    write_csv(OUT / "paired_aggregate.csv", pair_summary)
    availability = {method: len([r for r in per_run if r["method"] == method]) for method in methods}
    (OUT / "availability.json").write_text(json.dumps(availability, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(availability))


if __name__ == "__main__":
    main()
