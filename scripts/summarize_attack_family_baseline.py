"""Aggregate the completed three-seed AF1-AF4 no-replay baseline."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("AF1", "AF2", "AF3", "AF4")
METRICS = ("accuracy", "macro_f1", "attack_recall", "benign_fpr")
POSITIONS = ("head", "middle", "tail")
SEEDS = (17, 42, 2026)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def latest(seed):
    found = []
    for directory in (ROOT / "results").glob(f"attack-family-sequential-seed{seed}-*"):
        metadata = load(directory / "metadata.json")
        if metadata.get("status") == "complete":
            found.append((metadata["finished_at"], directory, metadata))
    if not found:
        raise RuntimeError(f"No complete attack-family run for seed {seed}")
    return max(found, key=lambda item: item[0])[1:]


def stats(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "values": values}


def aggregate_nested(per_seed, getter):
    return stats([getter(item) for item in per_seed])


def main():
    per_seed = []
    for seed in SEEDS:
        directory, metadata = latest(seed)
        cells = {(row["trained_stage"], row["eval_stage"]): row for row in jsonl(directory / "metrics.jsonl") if row["phase"] == "post"}
        summary = load(directory / "summary.json")
        acquisitions = {stage: cells[(stage, stage)] for stage in STAGES}
        final = {stage: cells[("AF4", stage)] for stage in STAGES}
        final_changes = {
            stage: {metric: final[stage][metric] - acquisitions[stage][metric] for metric in METRICS}
            for stage in STAGES[:-1]
        }
        peak = {}
        for eval_index, eval_stage in enumerate(STAGES):
            peak[eval_stage] = {}
            for metric in ("accuracy", "macro_f1", "attack_recall"):
                observed = [cells[(trained, eval_stage)][metric] for trained in STAGES[eval_index:]]
                peak[eval_stage][metric] = max(observed[:-1] or observed) - observed[-1] if len(observed) > 1 else 0.0
                peak[eval_stage][metric] = max(
                    max(observed[:index + 1]) - observed[index] for index in range(len(observed))
                )
        transitions = {}
        for trained_index in range(1, len(STAGES)):
            before_stage, after_stage = STAGES[trained_index - 1], STAGES[trained_index]
            for eval_stage in STAGES[:trained_index]:
                key = f"{eval_stage}:{before_stage}->{after_stage}"
                transitions[key] = {
                    metric: cells[(after_stage, eval_stage)][metric] - cells[(before_stage, eval_stage)][metric]
                    for metric in METRICS
                }
        per_seed.append({
            "seed": seed, "run_id": directory.name, "runtime_seconds": metadata["runtime_seconds"],
            "peak_gpu_memory_bytes": metadata["peak_gpu_memory_bytes"],
            "matrix": {metric: {trained: {stage: cells[(trained, stage)][metric] for stage in STAGES[:STAGES.index(trained)+1]} for trained in STAGES} for metric in METRICS},
            "acquisition": {stage: {metric: acquisitions[stage][metric] for metric in METRICS} for stage in STAGES},
            "final_retention": {stage: {metric: final[stage][metric] for metric in METRICS} for stage in STAGES},
            "final_change": final_changes, "peak_forgetting": peak, "transitions": transitions,
            "final_bwt": {
                metric: statistics.mean([final[stage][metric] - acquisitions[stage][metric] for stage in STAGES[:-1]])
                for metric in ("accuracy", "macro_f1", "attack_recall")
            },
            "final_benign_fpr_increase": statistics.mean([final[stage]["benign_fpr"] - acquisitions[stage]["benign_fpr"] for stage in STAGES[:-1]]),
            "position_attack_tpr": summary["position_attack_tpr"],
            "stage_diagnostics": summary["stage_acquisition_diagnostics"],
        })

    matrix = {}
    for metric in METRICS:
        matrix[metric] = {}
        for trained_index, trained in enumerate(STAGES):
            matrix[metric][trained] = {}
            for eval_stage in STAGES[:trained_index + 1]:
                matrix[metric][trained][eval_stage] = aggregate_nested(per_seed, lambda item, m=metric, t=trained, e=eval_stage: item["matrix"][m][t][e])
    acquisitions = {stage: {metric: aggregate_nested(per_seed, lambda item, s=stage, m=metric: item["acquisition"][s][m]) for metric in METRICS} for stage in STAGES}
    retention = {stage: {metric: aggregate_nested(per_seed, lambda item, s=stage, m=metric: item["final_change"][s][m]) for metric in METRICS} for stage in STAGES[:-1]}
    transition_keys = list(per_seed[0]["transitions"])
    transitions = {key: {metric: aggregate_nested(per_seed, lambda item, k=key, m=metric: item["transitions"][k][m]) for metric in METRICS} for key in transition_keys}
    positions = {position: {trained: {stage: aggregate_nested(per_seed, lambda item, p=position, t=trained, e=stage: item["position_attack_tpr"][p][t][e]) for stage in STAGES[:STAGES.index(trained)+1]} for trained in STAGES} for position in POSITIONS}
    diagnostics = {stage: {field: aggregate_nested(per_seed, lambda item, s=stage, f=field: next(row[f] for row in item["stage_diagnostics"] if row["stage"] == s)) for field in ("attack_tpr", "benign_fpr", "macro_f1", "accuracy", "training_loss", "validation_loss", "runtime_seconds", "peak_gpu_memory_bytes")} for stage in STAGES}
    earlier = load(ROOT / "results/real_baseline_aggregate.json")
    doc_changes = [row["s1_attack_recall_change"] for row in earlier["sequential_per_seed"]]
    report = {
        "status": "complete", "stage_order": list(STAGES), "seeds": list(SEEDS),
        "dataset_sha256": "4b4efaf7640f31f5f12af2f2b3a40b91bbb71789c91c059d21ce4af1e6a9e43a",
        "per_seed": per_seed, "mean_sample_sd_matrix": matrix, "acquisition_mean_sample_sd": acquisitions,
        "final_retention_change_mean_sample_sd": retention,
        "final_bwt_mean_sample_sd": {metric: aggregate_nested(per_seed, lambda item, m=metric: item["final_bwt"][m]) for metric in ("accuracy", "macro_f1", "attack_recall")},
        "final_benign_fpr_increase_mean_sample_sd": aggregate_nested(per_seed, lambda item: item["final_benign_fpr_increase"]),
        "peak_forgetting_mean_sample_sd": {stage: {metric: aggregate_nested(per_seed, lambda item, s=stage, m=metric: item["peak_forgetting"][s][m]) for metric in ("accuracy", "macro_f1", "attack_recall")} for stage in STAGES},
        "transition_change_mean_sample_sd": transitions,
        "position_attack_tpr_mean_sample_sd": positions,
        "stage_diagnostics_mean_sample_sd": diagnostics,
        "runtime_mean_sample_sd": aggregate_nested(per_seed, lambda item: item["runtime_seconds"]),
        "peak_gpu_memory_mean_sample_sd": aggregate_nested(per_seed, lambda item: item["peak_gpu_memory_bytes"]),
        "earlier_document_pilot": {"s1_attack_tpr_change": stats(doc_changes)},
        "scope": {"AF5_included": False, "variant_b_constructed": False, "replay": False, "secondary_order_run": False},
    }
    output = ROOT / "results/attack_family_baseline_aggregate.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
