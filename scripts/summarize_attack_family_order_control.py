"""Aggregate the three-seed AF1->AF4->AF3->AF2 no-replay order control."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORDER = ("AF1", "AF4", "AF3", "AF2")
METRICS = ("accuracy", "macro_f1", "attack_recall", "benign_fpr")
HIGHER = ("accuracy", "macro_f1", "attack_recall")
POSITIONS = ("head", "middle", "tail")
SEEDS = (17, 42, 2026)
DATA_SHA = "4b4efaf7640f31f5f12af2f2b3a40b91bbb71789c91c059d21ce4af1e6a9e43a"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def latest(seed: int):
    found = []
    for directory in (ROOT / "results").glob(f"attack-family-order-control-seed{seed}-*"):
        metadata = load(directory / "metadata.json")
        if metadata.get("status") == "complete":
            found.append((metadata["finished_at"], directory, metadata))
    if not found:
        raise RuntimeError(f"No complete order-control run for seed {seed}")
    return max(found, key=lambda item: item[0])[1:]


def stats(values):
    values = list(values)
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def aggregate(per_seed, getter):
    return stats(getter(item) for item in per_seed)


def seed_report(seed: int):
    directory, metadata = latest(seed)
    cells = {
        (row["trained_stage"], row["eval_stage"]): row
        for row in jsonl(directory / "metrics.jsonl")
        if row["phase"] == "post"
    }
    summary = load(directory / "summary.json")
    acquisition = {stage: cells[(stage, stage)] for stage in ORDER}
    last = ORDER[-1]
    final = {stage: cells[(last, stage)] for stage in ORDER}
    final_change = {
        stage: {metric: final[stage][metric] - acquisition[stage][metric] for metric in METRICS}
        for stage in ORDER[:-1]
    }
    peak = {}
    for acquired_at, family in enumerate(ORDER):
        peak[family] = {}
        observed_stages = ORDER[acquired_at:]
        for metric in HIGHER:
            values = [cells[(trained, family)][metric] for trained in observed_stages]
            peak[family][metric] = max(
                max(values[: index + 1]) - values[index] for index in range(len(values))
            )
    transitions = {}
    for index in range(1, len(ORDER)):
        before, after = ORDER[index - 1], ORDER[index]
        for family in ORDER[:index]:
            transitions[f"{family}:{before}->{after}"] = {
                metric: cells[(after, family)][metric] - cells[(before, family)][metric]
                for metric in METRICS
            }
    return {
        "seed": seed,
        "run_id": directory.name,
        "runtime_seconds": metadata["runtime_seconds"],
        "peak_gpu_memory_bytes": metadata["peak_gpu_memory_bytes"],
        "matrix": {
            metric: {
                trained: {
                    family: cells[(trained, family)][metric]
                    for family in ORDER[: ORDER.index(trained) + 1]
                }
                for trained in ORDER
            }
            for metric in METRICS
        },
        "acquisition": {
            stage: {metric: acquisition[stage][metric] for metric in METRICS}
            for stage in ORDER
        },
        "final_retention": {
            stage: {metric: final[stage][metric] for metric in METRICS} for stage in ORDER
        },
        "final_change": final_change,
        "peak_forgetting": peak,
        "transitions": transitions,
        "final_bwt": {
            metric: statistics.mean(final[stage][metric] - acquisition[stage][metric] for stage in ORDER[:-1])
            for metric in HIGHER
        },
        "final_benign_fpr_increase": statistics.mean(
            final[stage]["benign_fpr"] - acquisition[stage]["benign_fpr"] for stage in ORDER[:-1]
        ),
        "position_attack_tpr": summary["position_attack_tpr"],
        "stage_diagnostics": summary["stage_acquisition_diagnostics"],
    }


def main():
    per_seed = [seed_report(seed) for seed in SEEDS]
    matrix = {
        metric: {
            trained: {
                family: aggregate(per_seed, lambda item, m=metric, t=trained, f=family: item["matrix"][m][t][f])
                for family in ORDER[: ORDER.index(trained) + 1]
            }
            for trained in ORDER
        }
        for metric in METRICS
    }
    acquisitions = {
        stage: {
            metric: aggregate(per_seed, lambda item, s=stage, m=metric: item["acquisition"][s][m])
            for metric in METRICS
        }
        for stage in ORDER
    }
    retention = {
        stage: {
            metric: aggregate(per_seed, lambda item, s=stage, m=metric: item["final_change"][s][m])
            for metric in METRICS
        }
        for stage in ORDER[:-1]
    }
    transition_keys = list(per_seed[0]["transitions"])
    transitions = {
        key: {
            metric: aggregate(per_seed, lambda item, k=key, m=metric: item["transitions"][k][m])
            for metric in METRICS
        }
        for key in transition_keys
    }
    positions = {
        position: {
            trained: {
                family: aggregate(
                    per_seed,
                    lambda item, p=position, t=trained, f=family: item["position_attack_tpr"][p][t][f],
                )
                for family in ORDER[: ORDER.index(trained) + 1]
            }
            for trained in ORDER
        }
        for position in POSITIONS
    }
    diagnostics = {
        stage: {
            field: aggregate(
                per_seed,
                lambda item, s=stage, f=field: next(row[f] for row in item["stage_diagnostics"] if row["stage"] == s),
            )
            for field in (
                "attack_tpr", "benign_fpr", "macro_f1", "accuracy", "training_loss",
                "validation_loss", "runtime_seconds", "peak_gpu_memory_bytes",
            )
        }
        for stage in ORDER
    }
    primary = load(ROOT / "results/attack_family_baseline_aggregate.json")
    primary_by_seed = {item["seed"]: item for item in primary["per_seed"]}
    paired = {}
    for seed_item in per_seed:
        seed = seed_item["seed"]
        first = primary_by_seed[seed]
        paired[str(seed)] = {
            "final_bwt_secondary_minus_primary": {
                metric: seed_item["final_bwt"][metric] - first["final_bwt"][metric] for metric in HIGHER
            },
            "final_fpr_increase_secondary_minus_primary": (
                seed_item["final_benign_fpr_increase"] - first["final_benign_fpr_increase"]
            ),
            "AF1_pressure_first_transition": {
                "primary_AF1_to_AF2": first["transitions"]["AF1:AF1->AF2"],
                "secondary_AF1_to_AF4": seed_item["transitions"]["AF1:AF1->AF4"],
            },
            "AF1_middle_after_first_transition": {
                "primary_after_AF2": first["position_attack_tpr"]["middle"]["AF2"]["AF1"],
                "secondary_after_AF4": seed_item["position_attack_tpr"]["middle"]["AF4"]["AF1"],
            },
        }
    report = {
        "status": "complete",
        "project_root": str(ROOT.resolve()),
        "stage_order": list(ORDER),
        "seeds": list(SEEDS),
        "dataset_sha256": DATA_SHA,
        "per_seed": per_seed,
        "mean_sample_sd_matrix": matrix,
        "acquisition_mean_sample_sd": acquisitions,
        "final_retention_change_mean_sample_sd": retention,
        "final_bwt_mean_sample_sd": {
            metric: aggregate(per_seed, lambda item, m=metric: item["final_bwt"][m]) for metric in HIGHER
        },
        "final_benign_fpr_increase_mean_sample_sd": aggregate(
            per_seed, lambda item: item["final_benign_fpr_increase"]
        ),
        "peak_forgetting_mean_sample_sd": {
            stage: {
                metric: aggregate(per_seed, lambda item, s=stage, m=metric: item["peak_forgetting"][s][m])
                for metric in HIGHER
            }
            for stage in ORDER
        },
        "transition_change_mean_sample_sd": transitions,
        "position_attack_tpr_mean_sample_sd": positions,
        "stage_diagnostics_mean_sample_sd": diagnostics,
        "runtime_mean_sample_sd": aggregate(per_seed, lambda item: item["runtime_seconds"]),
        "peak_gpu_memory_mean_sample_sd": aggregate(per_seed, lambda item: item["peak_gpu_memory_bytes"]),
        "primary_order_reference": {
            "stage_order": primary["stage_order"],
            "final_bwt_mean_sample_sd": primary["final_bwt_mean_sample_sd"],
            "final_benign_fpr_increase_mean_sample_sd": primary["final_benign_fpr_increase_mean_sample_sd"],
            "paired_seed_comparison": paired,
        },
        "scope": {
            "AF5_included": False,
            "variant_b_constructed": False,
            "replay": False,
            "order_control": True,
        },
    }
    output = ROOT / "results/attack_family_order_control_aggregate.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
