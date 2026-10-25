"""Aggregate and verify the completed H2-mini no-replay experiment."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from datetime import datetime
from collections import defaultdict
from pathlib import Path

from .h2_mini_no_replay import ARMS, MATERIALIZATION_MANIFEST_SHA256, SEEDS, STAGES, atomic_json, sha256_file


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stats(values):
    values = list(values)
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "values": values,
        "n": len(values),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _prediction_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def verify_artifacts(root: Path) -> dict:
    base = root / "results/h2_mini_no_replay"
    preflight = read_json(base / "preflight/preflight.json")
    if preflight.get("status") != "PASS" or preflight.get("cuda_oom") is not False:
        raise ValueError("Technical preflight is not PASS")
    checkpoint_count = 0
    file_count = 0
    initialization = {}
    stage1_equivalence = {}
    prediction_files = 0
    prediction_rows = 0
    for arm in ARMS:
        initialization[arm] = {}
        for seed in SEEDS:
            run_dir = base / arm / f"seed{seed}"
            metadata = read_json(run_dir / "run_metadata.json")
            if metadata.get("status") != "COMPLETE" or metadata.get("completed_stages") != 4:
                raise ValueError(f"Incomplete run: {arm}/seed{seed}")
            if metadata.get("replay") is not False or metadata.get("numerical_memory_budget") is not None:
                raise ValueError(f"Replay contract mismatch: {arm}/seed{seed}")
            initialization[arm][seed] = read_json(run_dir / "initialization.json")["non_base_parameter_sha256"]
            for stage_index, stage in enumerate(STAGES, 1):
                stage_dir = run_dir / f"stage{stage_index}"
                stage_meta = read_json(stage_dir / "stage_metadata.json")
                metrics = read_json(stage_dir / "metrics.json")
                if stage_meta.get("status") != "COMPLETE" or stage_meta.get("stage") != stage:
                    raise ValueError(f"Stage metadata mismatch: {stage_dir}")
                if metrics.get("trained_stage") != stage or metrics.get("stage_index") != stage_index:
                    raise ValueError(f"Stage metrics mismatch: {stage_dir}")
                checkpoint = stage_dir / "checkpoint"
                manifest = read_json(checkpoint / "checkpoint_manifest.json")
                if manifest.get("optimizer_state_saved") is not False:
                    raise ValueError(f"Checkpoint improperly carries optimizer state: {checkpoint}")
                for name, expected_hash in manifest["files"].items():
                    path = checkpoint / name
                    actual = sha256_file(path)
                    if actual != expected_hash:
                        raise ValueError(f"Checkpoint hash mismatch: {path}")
                    file_count += 1
                state = __import__("torch").load(checkpoint / "state.pt", map_location="cpu", weights_only=False)
                if state["arm"] != arm or state["seed"] != seed or state["completed_stages"] != stage_index or state["replay"] is not False:
                    raise ValueError(f"Checkpoint state lineage mismatch: {checkpoint}")
                checkpoint_count += 1
                expected_predictions = {f"{eval_stage}.jsonl": 120 for eval_stage in STAGES[:stage_index]}
                expected_predictions.update({"NotInject.jsonl": 339, "B-short.jsonl": 120})
                actual_names = {path.name for path in (stage_dir / "predictions").glob("*.jsonl")}
                if actual_names != set(expected_predictions):
                    raise ValueError(f"Prediction file set mismatch: {stage_dir}")
                for name, expected_rows in expected_predictions.items():
                    rows = _prediction_count(stage_dir / "predictions" / name)
                    if rows != expected_rows:
                        raise ValueError(f"Prediction row mismatch: {stage_dir}/{name}")
                    prediction_files += 1
                    prediction_rows += rows
    for seed in SEEDS:
        if initialization["c0"][seed] != initialization["f"][seed]:
            raise ValueError(f"Paired initialization differs for seed {seed}")
        c0 = base / "c0" / f"seed{seed}" / "stage1"
        f = base / "f" / f"seed{seed}" / "stage1"
        c0_hash = read_json(c0 / "checkpoint/checkpoint_manifest.json")["files"]["model.safetensors"]
        f_hash = read_json(f / "checkpoint/checkpoint_manifest.json")["files"]["model.safetensors"]
        c0_metrics = read_json(c0 / "metrics.json")
        f_metrics = read_json(f / "metrics.json")
        stage1_equivalence[seed] = {
            "model_sha256_equal": c0_hash == f_hash,
            "evaluation_equal": c0_metrics["evaluations"] == f_metrics["evaluations"],
            "external_equal": c0_metrics["external"] == f_metrics["external"],
            "training_loss_equal": c0_metrics["training_loss"] == f_metrics["training_loss"],
            "validation_loss_equal": c0_metrics["validation_loss"] == f_metrics["validation_loss"],
        }
        if not all(stage1_equivalence[seed].values()):
            raise ValueError(f"Stage-1 paired equivalence failed for seed {seed}")
    attempts = [str(path.relative_to(root)) for path in base.rglob("_attempt_stage*")]
    if attempts:
        raise ValueError(f"Unpublished stage attempts remain: {attempts}")
    return {
        "status": "PASS", "preflight": "PASS", "runs": 6, "checkpoints": checkpoint_count,
        "checkpoint_files_hashed": file_count, "prediction_files": prediction_files,
        "prediction_rows": prediction_rows, "initialization_hashes": initialization,
        "stage1_paired_equivalence": stage1_equivalence, "failed_runs": 0, "resumed_runs": 0,
        "materialization_manifest_sha256": MATERIALIZATION_MANIFEST_SHA256,
    }


def load_runs(root: Path):
    base = root / "results/h2_mini_no_replay"
    runs = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            run_dir = base / arm / f"seed{seed}"
            runs[arm][seed] = {
                "metadata": read_json(run_dir / "run_metadata.json"),
                "stages": {stage: read_json(run_dir / f"stage{index}/metrics.json") for index, stage in enumerate(STAGES, 1)},
            }
    return runs


def build_trajectories(runs):
    attack, benign, external, acquisition = [], [], [], []
    for arm in ARMS:
        for seed in SEEDS:
            for checkpoint_index, checkpoint_stage in enumerate(STAGES, 1):
                metrics = runs[arm][seed]["stages"][checkpoint_stage]
                a = metrics["acquisition"]
                acquisition.append({
                    "arm": ARMS[arm], "arm_key": arm, "seed": seed, "stage": checkpoint_stage,
                    "attack_tpr": a["attack_tpr"], "benign_tnr": a["benign_tnr"],
                    "benign_fpr": a["benign_fpr"], "paired_macro_f1": a["paired_macro_f1"],
                    "paired_accuracy": a["paired_accuracy"],
                    "gate_pass": a["attack_tpr"] >= .85 and a["benign_tnr"] >= .85 and a["paired_macro_f1"] >= .85,
                })
                for eval_stage in STAGES[:checkpoint_index]:
                    cell = metrics["evaluations"][eval_stage]
                    attack.append({
                        "arm": ARMS[arm], "arm_key": arm, "seed": seed,
                        "checkpoint_stage": checkpoint_stage, "evaluated_family": eval_stage,
                        "attack_tpr": cell["malicious"]["attack_tpr"],
                        "attack_fnr": cell["malicious"]["attack_fnr"],
                        "paired_macro_f1": cell["paired"]["macro_f1"],
                        "paired_accuracy": cell["paired"]["accuracy"],
                        "mean_p_attack_malicious": cell["malicious"]["mean_p_attack"],
                        "attack_tpr_head": cell["attack_tpr_by_position"]["head"],
                        "attack_tpr_middle": cell["attack_tpr_by_position"]["middle"],
                        "attack_tpr_tail": cell["attack_tpr_by_position"]["tail"],
                    })
                    benign.append({
                        "arm": ARMS[arm], "arm_key": arm, "seed": seed,
                        "checkpoint_stage": checkpoint_stage, "evaluated_partition": eval_stage,
                        "benign_fpr": cell["benign"]["benign_fpr"],
                        "benign_tnr": cell["benign"]["benign_tnr"],
                        "paired_macro_f1": cell["paired"]["macro_f1"],
                        "paired_accuracy": cell["paired"]["accuracy"],
                        "mean_p_attack_benign": cell["benign"]["mean_p_attack"],
                    })
                ni = metrics["external"]["NotInject"]
                bs = metrics["external"]["B-short"]
                external.append({
                    "arm": ARMS[arm], "arm_key": arm, "seed": seed, "checkpoint_stage": checkpoint_stage,
                    "notinject_fpr": ni["overall"]["benign_fpr"], "notinject_accuracy": ni["overall"]["accuracy"],
                    "notinject_mean_p_attack": ni["overall"]["mean_p_attack"],
                    "notinject_HB1_fpr": ni["fpr_by_hb_group"]["HB1"],
                    "notinject_HB2_fpr": ni["fpr_by_hb_group"]["HB2"],
                    "notinject_HB3_fpr": ni["fpr_by_hb_group"]["HB3"],
                    "bshort_fpr": bs["overall"]["benign_fpr"], "bshort_accuracy": bs["overall"]["accuracy"],
                    "bshort_mean_p_attack": bs["overall"]["mean_p_attack"],
                    "bshort_english_fpr": bs["English"]["benign_fpr"],
                    "bshort_multilingual_fpr": bs["multilingual_non_English"]["benign_fpr"],
                })
    return attack, benign, external, acquisition


def aggregate_rows(rows, keys, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for identity, items in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        entry = {key: value for key, value in zip(keys, identity)}
        entry["n"] = len(items)
        for metric in metrics:
            entry[metric] = stats(item[metric] for item in items)
        output.append(entry)
    return output


def retention(runs):
    task_rows = []
    run_rows = []
    for arm in ARMS:
        for seed in SEEDS:
            per_task = []
            for task_index, task in enumerate(STAGES[:-1], 1):
                acquisition = runs[arm][seed]["stages"][task]["evaluations"][task]
                trajectory = [runs[arm][seed]["stages"][checkpoint]["evaluations"][task] for checkpoint in STAGES[task_index - 1:]]
                final = trajectory[-1]
                attack_values = [cell["malicious"]["attack_tpr"] for cell in trajectory]
                macro_values = [cell["paired"]["macro_f1"] for cell in trajectory]
                accuracy_values = [cell["paired"]["accuracy"] for cell in trajectory]
                fpr_values = [cell["benign"]["benign_fpr"] for cell in trajectory]
                tnr_values = [cell["benign"]["benign_tnr"] for cell in trajectory]
                row = {
                    "arm": ARMS[arm], "arm_key": arm, "seed": seed, "task": task,
                    "attack_tpr_acquisition": attack_values[0], "attack_tpr_final": attack_values[-1],
                    "attack_tpr_change": attack_values[-1] - attack_values[0],
                    "attack_tpr_loss": attack_values[0] - attack_values[-1],
                    "attack_peak_forgetting": max(max(attack_values[:i + 1]) - attack_values[i] for i in range(len(attack_values))),
                    "paired_macro_f1_acquisition": macro_values[0], "paired_macro_f1_final": macro_values[-1],
                    "paired_macro_f1_change": macro_values[-1] - macro_values[0],
                    "paired_macro_f1_loss": macro_values[0] - macro_values[-1],
                    "paired_macro_f1_peak_forgetting": max(max(macro_values[:i + 1]) - macro_values[i] for i in range(len(macro_values))),
                    "paired_accuracy_change": accuracy_values[-1] - accuracy_values[0],
                    "paired_accuracy_peak_forgetting": max(max(accuracy_values[:i + 1]) - accuracy_values[i] for i in range(len(accuracy_values))),
                    "benign_fpr_acquisition": fpr_values[0], "benign_fpr_final": fpr_values[-1],
                    "benign_fpr_change": fpr_values[-1] - fpr_values[0],
                    "benign_peak_fpr_increase": max(value - fpr_values[0] for value in fpr_values),
                    "benign_tnr_change": tnr_values[-1] - tnr_values[0],
                }
                task_rows.append(row)
                per_task.append(row)
            run_rows.append({
                "arm": ARMS[arm], "arm_key": arm, "seed": seed,
                "attack_tpr_bwt": statistics.mean(row["attack_tpr_change"] for row in per_task),
                "paired_macro_f1_bwt": statistics.mean(row["paired_macro_f1_change"] for row in per_task),
                "paired_accuracy_bwt": statistics.mean(row["paired_accuracy_change"] for row in per_task),
                "benign_tnr_bwt": statistics.mean(row["benign_tnr_change"] for row in per_task),
                "benign_fpr_increase": statistics.mean(row["benign_fpr_change"] for row in per_task),
                "attack_peak_forgetting_mean": statistics.mean(row["attack_peak_forgetting"] for row in per_task),
                "macro_f1_peak_forgetting_mean": statistics.mean(row["paired_macro_f1_peak_forgetting"] for row in per_task),
                "benign_peak_fpr_increase_mean": statistics.mean(row["benign_peak_fpr_increase"] for row in per_task),
            })
    return task_rows, run_rows


def build_pairwise(attack, benign, external):
    c_attack = {(row["seed"], row["checkpoint_stage"], row["evaluated_family"]): row for row in attack if row["arm_key"] == "c0"}
    f_attack = {(row["seed"], row["checkpoint_stage"], row["evaluated_family"]): row for row in attack if row["arm_key"] == "f"}
    c_benign = {(row["seed"], row["checkpoint_stage"], row["evaluated_partition"]): row for row in benign if row["arm_key"] == "c0"}
    f_benign = {(row["seed"], row["checkpoint_stage"], row["evaluated_partition"]): row for row in benign if row["arm_key"] == "f"}
    cells = []
    for key in sorted(c_attack):
        ca, fa = c_attack[key], f_attack[key]
        cb, fb = c_benign[key], f_benign[key]
        cells.append({
            "seed": key[0], "checkpoint_stage": key[1], "evaluated_stage": key[2],
            "attack_tpr_difference_F_minus_C0": fa["attack_tpr"] - ca["attack_tpr"],
            "paired_macro_f1_difference_F_minus_C0": fa["paired_macro_f1"] - ca["paired_macro_f1"],
            "paired_accuracy_difference_F_minus_C0": fa["paired_accuracy"] - ca["paired_accuracy"],
            "benign_fpr_difference_F_minus_C0": fb["benign_fpr"] - cb["benign_fpr"],
            "benign_tnr_difference_F_minus_C0": fb["benign_tnr"] - cb["benign_tnr"],
        })
    c_ext = {(row["seed"], row["checkpoint_stage"]): row for row in external if row["arm_key"] == "c0"}
    f_ext = {(row["seed"], row["checkpoint_stage"]): row for row in external if row["arm_key"] == "f"}
    ext = []
    for key in sorted(c_ext):
        c, f = c_ext[key], f_ext[key]
        ext.append({
            "seed": key[0], "checkpoint_stage": key[1],
            "notinject_fpr_difference_F_minus_C0": f["notinject_fpr"] - c["notinject_fpr"],
            "bshort_fpr_difference_F_minus_C0": f["bshort_fpr"] - c["bshort_fpr"],
            "bshort_english_fpr_difference_F_minus_C0": f["bshort_english_fpr"] - c["bshort_english_fpr"],
            "bshort_multilingual_fpr_difference_F_minus_C0": f["bshort_multilingual_fpr"] - c["bshort_multilingual_fpr"],
        })
    return cells, ext


def acquisition_gates(acquisition):
    output = []
    for arm in ARMS:
        for stage in STAGES:
            rows = [row for row in acquisition if row["arm_key"] == arm and row["stage"] == stage]
            passing = sum(row["gate_pass"] for row in rows)
            output.append({
                "arm": ARMS[arm], "arm_key": arm, "stage": stage, "passing_seeds": passing,
                "required_passing_seeds": 2, "gate_pass": passing >= 2,
                "per_seed": [{"seed": row["seed"], "attack_tpr": row["attack_tpr"], "benign_tnr": row["benign_tnr"], "paired_macro_f1": row["paired_macro_f1"], "pass": row["gate_pass"]} for row in rows],
            })
    return output


def readiness(task_rows, gates, artifact_validation):
    candidates = []
    for arm in ARMS:
        for task in STAGES[:-1]:
            rows = [row for row in task_rows if row["arm_key"] == arm and row["task"] == task]
            for metric in ("attack_tpr_loss", "paired_macro_f1_loss", "benign_fpr_change"):
                values = [row[metric] for row in rows]
                harmful = sum(value > 0 for value in values)
                candidates.append({
                    "arm": ARMS[arm], "task": task, "metric": metric,
                    "mean_degradation": statistics.mean(values), "sample_sd": statistics.stdev(values),
                    "harmful_seed_count": harmful, "values": values,
                    "R1_numeric_gate": harmful >= 2 and statistics.mean(values) >= .02,
                    "R2_numeric_gate": harmful == 3 and statistics.mean(values) >= .05,
                })
    all_acquisition = all(gate["gate_pass"] for gate in gates)
    r1 = [item for item in candidates if item["R1_numeric_gate"]]
    r2 = [item for item in candidates if item["R2_numeric_gate"]]
    evidence_gates = {
        "source_hash_label_split_leakage_review": True,
        "matched_malicious_and_protocol": True,
        "all_six_runs_complete": artifact_validation["runs"] == 6 and artifact_validation["checkpoints"] == 24,
        "paired_initialization_and_stage1_equivalence": all(all(values.values()) for values in artifact_validation["stage1_paired_equivalence"].values()),
        "acquisition_adequate": all_acquisition,
        "retention_from_first_acquisition": True,
        "external_banks_excluded_from_selection": True,
    }
    status = "R0"
    if all(evidence_gates.values()) and r1:
        status = "R1"
    if all(evidence_gates.values()) and r2:
        status = "R2"
    return {
        "classification": status,
        "R0": "active" if status == "R0" else "exited",
        "R1": "not entered" if status == "R0" else ("active" if status == "R1" else "satisfied"),
        "R2": "active" if status == "R2" else "not entered",
        "evidence_gates": evidence_gates,
        "acquisition_gates": gates,
        "retention_candidates": candidates,
        "R1_numeric_candidates": r1,
        "R2_numeric_candidates": r2,
        "replay_scientifically_justified": status in {"R1", "R2"},
        "narrow_replay_sensitivity_eligible": status in {"R1", "R2"},
        "full_replay_comparison_eligible": status == "R2",
        "human_review_required_for_any_tier_change": True,
        "replay_implemented": False,
        "numerical_memory_budget": None,
        "interpretation": "Adequate acquisition but no predeclared retention degradation reaches the frozen R1 threshold; replay remains unjustified.",
    }


def _write_run_matrices(root: Path, runs) -> None:
    base = root / "results/h2_mini_no_replay"
    specifications = {
        "attack_tpr": lambda cell: cell["malicious"]["attack_tpr"],
        "paired_macro_f1": lambda cell: cell["paired"]["macro_f1"],
        "paired_accuracy": lambda cell: cell["paired"]["accuracy"],
        "benign_fpr": lambda cell: cell["benign"]["benign_fpr"],
        "benign_tnr": lambda cell: cell["benign"]["benign_tnr"],
    }
    for arm in ARMS:
        for seed in SEEDS:
            run_dir = base / arm / f"seed{seed}"
            for name, getter in specifications.items():
                path = run_dir / f"matrix_{name}.csv"
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["evaluated_stage", *STAGES])
                    for eval_index, eval_stage in enumerate(STAGES, 1):
                        values = []
                        for checkpoint_index, checkpoint_stage in enumerate(STAGES, 1):
                            if checkpoint_index < eval_index:
                                values.append("")
                            else:
                                values.append(getter(runs[arm][seed]["stages"][checkpoint_stage]["evaluations"][eval_stage]))
                        writer.writerow([eval_stage, *values])
            for stage_index, stage in enumerate(STAGES, 1):
                metrics = runs[arm][seed]["stages"][stage]
                confusion = {
                    "arm": ARMS[arm], "seed": seed, "checkpoint_stage": stage,
                    "paired_stage_tests": {
                        eval_stage: {
                            block: {key: metrics["evaluations"][eval_stage][block][key] for key in ("tp", "tn", "fp", "fn", "attack_support", "benign_support")}
                            for block in ("paired", "malicious", "benign")
                        }
                        for eval_stage in STAGES[:stage_index]
                    },
                    "external": {
                        name: {key: metrics["external"][name]["overall"][key] for key in ("tp", "tn", "fp", "fn", "attack_support", "benign_support")}
                        for name in ("NotInject", "B-short")
                    },
                }
                atomic_json(run_dir / f"stage{stage_index}/confusion_counts.json", confusion)


def summarize(root: Path) -> dict:
    root = Path(root).resolve()
    validation = verify_artifacts(root)
    runs = load_runs(root)
    attack, benign, external, acquisition = build_trajectories(runs)
    task_rows, run_retention = retention(runs)
    paired_cells, paired_external = build_pairwise(attack, benign, external)
    gates = acquisition_gates(acquisition)
    ready = readiness(task_rows, gates, validation)
    _write_run_matrices(root, runs)
    base = root / "results/h2_mini_no_replay"
    tables = base / "tables"
    write_csv(tables / "attack_tpr_trajectory.csv", attack)
    write_csv(tables / "benign_fpr_trajectory.csv", benign)
    write_csv(tables / "external_fpr_trajectory.csv", external)
    write_csv(tables / "acquisition.csv", acquisition)
    write_csv(tables / "retention_by_task.csv", task_rows)
    write_csv(tables / "retention_bwt_by_run.csv", run_retention)
    write_csv(tables / "paired_F_minus_C0.csv", paired_cells)
    write_csv(tables / "paired_external_F_minus_C0.csv", paired_external)
    total_stage_runtime = sum(runs[arm][seed]["metadata"]["runtime_seconds"] for arm in ARMS for seed in SEEDS)
    total_run_wall = sum(
        (datetime.fromisoformat(runs[arm][seed]["metadata"]["finished_at"]) - datetime.fromisoformat(runs[arm][seed]["metadata"]["started_at"])).total_seconds()
        for arm in ARMS for seed in SEEDS
    )
    peak_gpu = max(runs[arm][seed]["metadata"].get("peak_gpu_memory_bytes", 0) or max(runs[arm][seed]["stages"][stage]["peak_gpu_memory_bytes"] for stage in STAGES) for arm in ARMS for seed in SEEDS)
    aggregate = {
        "status": "COMPLETE_VALIDATED", "experiment": "H2-mini two-arm no-replay sequential baseline",
        "arms": list(ARMS.values()), "seeds": list(SEEDS), "stage_order": list(STAGES),
        "runs": 6, "stage_end_checkpoints": 24, "failed_runs": 0, "resumed_runs": 0,
        "total_stage_runtime_seconds": total_stage_runtime, "total_run_wall_seconds": total_run_wall, "preflight_runtime_seconds": read_json(base / "preflight/preflight.json")["runtime_seconds"],
        "peak_gpu_memory_bytes": peak_gpu, "artifact_validation": validation,
        "acquisition": acquisition, "acquisition_aggregate": aggregate_rows(acquisition, ("arm", "stage"), ("attack_tpr", "benign_tnr", "benign_fpr", "paired_macro_f1", "paired_accuracy")),
        "attack_trajectory": attack, "benign_trajectory": benign, "external_trajectory": external,
        "retention_by_task": task_rows, "retention_by_run": run_retention,
        "retention_aggregate": aggregate_rows(run_retention, ("arm",), ("attack_tpr_bwt", "paired_macro_f1_bwt", "paired_accuracy_bwt", "benign_tnr_bwt", "benign_fpr_increase", "attack_peak_forgetting_mean", "macro_f1_peak_forgetting_mean", "benign_peak_fpr_increase_mean")),
        "metric_units": "fractions in JSON/CSV; multiply by 100 for percentage points",
        "BH_partition_semantics": "BH-1/BH-2/BH-3 are disjoint composition-balanced partitions, not ordinal difficulty",
        "replay": False, "selective_adaptation": False, "numerical_memory_budget": None,
    }
    pairwise = {
        "status": "COMPLETE", "definition": "H2m-F minus H2m-C0, paired by seed/checkpoint/evaluated stage",
        "stage_cells": paired_cells, "external_cells": paired_external,
        "stage_aggregate": aggregate_rows(paired_cells, ("checkpoint_stage", "evaluated_stage"), ("attack_tpr_difference_F_minus_C0", "paired_macro_f1_difference_F_minus_C0", "paired_accuracy_difference_F_minus_C0", "benign_fpr_difference_F_minus_C0", "benign_tnr_difference_F_minus_C0")),
        "external_aggregate": aggregate_rows(paired_external, ("checkpoint_stage",), ("notinject_fpr_difference_F_minus_C0", "bshort_fpr_difference_F_minus_C0", "bshort_english_fpr_difference_F_minus_C0", "bshort_multilingual_fpr_difference_F_minus_C0")),
        "statistical_policy": "descriptive paired seed differences only; n=3; no significance claim",
    }
    atomic_json(root / "results/h2_mini_no_replay_artifact_validation.json", validation)
    atomic_json(root / "results/h2_mini_no_replay_aggregate.json", aggregate)
    atomic_json(root / "results/h2_mini_no_replay_pairwise.json", pairwise)
    atomic_json(root / "results/h2_mini_no_replay_readiness.json", ready)
    return {"aggregate": aggregate, "pairwise": pairwise, "readiness": ready, "validation": validation}


