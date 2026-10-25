"""Aggregate the completed no-replay baseline sanity controls."""
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from continualguard.artifacts import read_jsonl, write_json

SEEDS = (17, 42, 2026)
METRICS = ("accuracy", "macro_f1", "attack_recall", "benign_fpr")


def latest(control, seed):
    candidates = []
    for directory in (ROOT / "results").glob(f"control-*-seed{seed}-*"):
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("status") == "complete" and metadata.get("control_type") == control:
            candidates.append((metadata["finished_at"], directory, metadata))
    if not candidates:
        raise RuntimeError(f"Missing complete {control} seed {seed}")
    return max(candidates, key=lambda item: item[0])[1:]


def stats(values):
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values) if len(values) > 1 else None, "n": len(values), "min": min(values), "max": max(values)}


def epoch_timings(directory):
    events = [event for event in read_jsonl(directory / "stage_events.jsonl") if event["event"] == "epoch_complete"]
    elapsed = [event["elapsed_seconds"] for event in events]
    total = 0.0
    output = {}
    for index, value in enumerate(elapsed, 1):
        total += value
        output[index] = {"epoch_runtime_seconds": value, "cumulative_runtime_seconds": total, "peak_gpu_memory_bytes": events[index - 1]["peak_gpu_memory_bytes"], "peak_gpu_reserved_bytes": events[index - 1]["peak_gpu_reserved_bytes"]}
    return output


def epoch_control(control, seeds):
    rows = []
    for seed in seeds:
        directory, metadata = latest(control, seed)
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        timings = epoch_timings(directory)
        for item in summary["epoch_results"]:
            row = {"seed": seed, "epoch": item["epoch"], "run_id": directory.name, **timings[item["epoch"]], "per_stage": item["per_stage"]}
            for key in ("s1_retention", "s1_bwt", "s1_benign_fpr_change"):
                if key in item:
                    row[key] = item[key]
            rows.append(row)
    aggregate = {}
    for epoch in (1, 2, 3):
        subset = [row for row in rows if row["epoch"] == epoch]
        values = {"cumulative_runtime_seconds": stats([row["cumulative_runtime_seconds"] for row in subset])}
        stages = sorted({stage for row in subset for stage in row["per_stage"]})
        for stage in stages:
            for metric in METRICS:
                values[f"{stage.lower()}_{metric}"] = stats([row["per_stage"][stage][metric] for row in subset])
        if control == "sequential_s2_epochs":
            for metric in ("accuracy", "macro_f1", "attack_recall"):
                values[f"s1_{metric}_bwt"] = stats([row["s1_bwt"][metric] for row in subset])
            values["s1_benign_fpr_change"] = stats([row["s1_benign_fpr_change"] for row in subset])
        aggregate[str(epoch)] = values
    return rows, aggregate


def reverse():
    directory, metadata = latest("reverse_order", 42)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    first = summary["stage_summaries"][0]
    second = summary["stage_summaries"][1]
    return {
        "seed": 42,
        "run_id": directory.name,
        "runtime_seconds": metadata["runtime_seconds"],
        "peak_gpu_memory_bytes": metadata["peak_gpu_memory_bytes"],
        "peak_gpu_reserved_bytes": metadata["peak_gpu_reserved_bytes"],
        "s2_acquisition": {metric: first[metric]["acquisition"] for metric in METRICS},
        "s2_after_s1": {metric: second[metric]["previous_stage_mean"] for metric in METRICS},
        "s2_bwt": {metric: second[metric]["bwt"] for metric in ("accuracy", "macro_f1", "attack_recall")},
        "s2_benign_fpr_change": second["benign_fpr"]["increase_from_acquisition"],
        "s1_acquisition": {metric: second[metric]["acquisition"] for metric in METRICS},
    }


def main():
    s2_rows, s2_aggregate = epoch_control("s2_only", SEEDS)
    sequential_rows, sequential_aggregate = epoch_control("sequential_s2_epochs", SEEDS)
    joint_rows, _ = epoch_control("joint_epochs", (42,))
    reverse_result = reverse()
    all_run_ids = sorted({row["run_id"] for row in s2_rows + sequential_rows + joint_rows} | {reverse_result["run_id"]})
    metadata = [json.loads((ROOT / "results" / run_id / "metadata.json").read_text(encoding="utf-8")) for run_id in all_run_ids]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Chen-derived deterministic reconstruction; SQuAD/TriviaQA document-domain pilot only; no replay",
        "fixed_contract": {"model": "microsoft/deberta-v3-base", "max_length": 1280, "batch_size": 4, "learning_rate": 1e-5, "seeds": list(SEEDS)},
        "s2_only_per_seed_epoch": s2_rows,
        "s2_only_mean_std_by_epoch": s2_aggregate,
        "sequential_per_seed_epoch": sequential_rows,
        "sequential_mean_std_by_epoch": sequential_aggregate,
        "reverse_order_seed42": reverse_result,
        "joint_seed42_by_epoch": joint_rows,
        "total_instrumented_runtime_seconds": sum(item["runtime_seconds"] for item in metadata),
        "interpretation_gate": "Human review required before reverse/joint seed expansion, attack-family stream execution, or replay implementation.",
    }
    write_json(ROOT / "results/baseline_sanity_aggregate.json", report)
    print(ROOT / "results/baseline_sanity_aggregate.json")


if __name__ == "__main__":
    main()
