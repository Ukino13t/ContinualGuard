"""Aggregate the approved three-seed real baselines without replay."""
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


def completed_run(regime, seed):
    found = []
    for path in (ROOT / "results").glob(f"real-*-seed{seed}-*/metadata.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("status") == "complete" and metadata.get("experiment_regime") == regime:
            found.append((metadata["finished_at"], path.parent, metadata))
    if not found:
        raise RuntimeError(f"Missing complete {regime} run for seed {seed}")
    return max(found, key=lambda item: item[0])[1:]


def sequential(seed):
    run_dir, metadata = completed_run("sequential", seed)
    cells = read_jsonl(run_dir / "metrics.jsonl")
    lookup = {(cell["phase"], cell["trained_stage"], cell["eval_stage"]): cell for cell in cells}
    before = lookup[("post", "S1", "S1")]
    after = lookup[("post", "S2", "S1")]
    s2 = lookup[("post", "S2", "S2")]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    final = next(item for item in summary["stage_summaries"] if item["stage"] == "S2")
    result = {"seed": seed, "run_id": run_dir.name}
    for metric in METRICS:
        result[f"s1_{metric}_before_s2"] = before[metric]
        result[f"s1_{metric}_after_s2"] = after[metric]
        result[f"s1_{metric}_change"] = after[metric] - before[metric]
        result[f"s2_{metric}_acquisition"] = s2[metric]
    attack_before = before["attack_recall"]
    result["s1_attack_recall_relative_change"] = (after["attack_recall"] - attack_before) / attack_before if attack_before else None
    for metric in ("accuracy", "macro_f1", "attack_recall"):
        result[f"{metric}_retention"] = final[metric]["previous_stage_mean"]
        result[f"{metric}_bwt"] = final[metric]["bwt"]
    result["benign_fpr_increase"] = final["benign_fpr"]["increase_from_acquisition"]
    events = [event for event in read_jsonl(run_dir / "stage_events.jsonl") if event.get("event") == "stage_complete"]
    result["runtime_seconds"] = metadata["runtime_seconds"]
    result["peak_gpu_memory_bytes"] = max(event["peak_gpu_memory_bytes"] for event in events)
    result["peak_gpu_reserved_bytes"] = metadata["peak_gpu_reserved_bytes"]
    result["stage_runtime_seconds"] = {event["stage"]: event["elapsed_seconds"] for event in events}
    for event in events:
        result[f"{event['stage'].lower()}_runtime_seconds"] = event["elapsed_seconds"]
    return result


def joint(seed):
    run_dir, metadata = completed_run("joint_offline_reference", seed)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    result = {"seed": seed, "run_id": run_dir.name, "runtime_seconds": metadata["runtime_seconds"], "peak_gpu_memory_bytes": metadata["peak_gpu_memory_bytes"], "peak_gpu_reserved_bytes": metadata["peak_gpu_reserved_bytes"]}
    for stage in ("S1", "S2"):
        for metric in METRICS:
            result[f"{stage.lower()}_{metric}"] = summary["per_stage"][stage][metric]
    return result


def aggregate(rows):
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float)) and key != "seed"})
    output = {}
    for key in keys:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        output[key] = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values) if len(values) > 1 else None, "n": len(values)}
    return output


def main():
    sequential_rows = [sequential(seed) for seed in SEEDS]
    joint_rows = [joint(seed) for seed in SEEDS]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Chen-derived deterministic reconstruction; SQuAD-to-TriviaQA document/domain-distribution pilot only",
        "seeds": list(SEEDS),
        "max_length": 1280,
        "model": "microsoft/deberta-v3-base",
        "sequential_per_seed": sequential_rows,
        "sequential_mean_std": aggregate(sequential_rows),
        "joint_per_seed": joint_rows,
        "joint_mean_std": aggregate(joint_rows),
        "total_instrumented_runtime_seconds": sum(row["runtime_seconds"] for row in sequential_rows + joint_rows),
        "interpretation": "No replay result is included. Human review is required before replay.",
    }
    write_json(ROOT / "results/real_baseline_aggregate.json", report)
    print(ROOT / "results/real_baseline_aggregate.json")


if __name__ == "__main__":
    main()
