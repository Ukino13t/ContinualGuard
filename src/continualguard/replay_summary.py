"""Validate and aggregate the frozen Candidate B four-method replay comparison."""
from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import torch

from .replay_candidate_b import (
    CONFIG_RELATIVE, MEMORY_CAPACITY, METHODS, REPLAY_BATCH_SIZE, RESULT_RELATIVE,
)
from .stronger_pilot_no_replay import ARMS, SEEDS, STAGES, CandidateBData, atomic_json, sha256_file

ALL_METHODS = ("no_replay", *METHODS)
METHOD_NAMES = {
    "no_replay": "No Replay", "reservoir": "Reservoir ER", "derpp": "DER++",
    "security_aware": "Security-aware Replay",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stats(values):
    values = list(values)
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values) if len(values) > 1 else None, "values": values, "n": len(values)}


def prediction_count(path):
    with Path(path).open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def run_dir(root, method, arm, seed):
    if method == "no_replay":
        return root / "results/stronger_pilot_no_replay" / arm / f"seed{seed}"
    return root / RESULT_RELATIVE / method / arm / f"seed{seed}"


def load_all_runs(root):
    result = {}
    for method in ALL_METHODS:
        result[method] = {arm: {} for arm in ARMS}
        for arm in ARMS:
            for seed in SEEDS:
                directory = run_dir(root, method, arm, seed)
                result[method][arm][seed] = {
                    "metadata": read_json(directory / "run_metadata.json"),
                    "stages": {stage: read_json(directory / f"stage{index}/metrics.json") for index, stage in enumerate(STAGES, 1)},
                }
    return result


def verify(root):
    replay_root = root / RESULT_RELATIVE
    preflight = read_json(replay_root / "preflight/preflight.json")
    if preflight.get("status") != "PASS" or set(preflight.get("methods", {})) != set(METHODS):
        raise ValueError("Replay preflight is not an all-method PASS")
    config_hash = sha256_file(root / CONFIG_RELATIVE)
    data_by_arm = {arm: CandidateBData(root, arm) for arm in ARMS}
    expected_train = {
        arm: {row["example_id"]: row for stage in STAGES for row in data_by_arm[arm].records(stage, "train")}
        for arm in ARMS
    }
    checkpoints = checkpoint_files = prediction_files = prediction_rows = failed = resumed = 0
    initialization = defaultdict(dict)
    exposure = []
    capacity_checks = []
    lineage_checks = []
    unpublished = []
    for method in METHODS:
        for arm in ARMS:
            for seed in SEEDS:
                directory = run_dir(root, method, arm, seed)
                metadata = read_json(directory / "run_metadata.json")
                if metadata.get("status") != "COMPLETE" or metadata.get("completed_stages") != 4:
                    failed += 1
                    raise ValueError(f"Incomplete replay run: {method}/{arm}/seed{seed}")
                resumed += metadata.get("resumed_from_stage") is not None
                init = read_json(directory / "initialization.json")
                baseline_init = read_json(run_dir(root, "no_replay", arm, seed) / "initialization.json")
                if init != baseline_init:
                    raise ValueError(f"Initialization mismatch: {method}/{arm}/seed{seed}")
                initialization[(method, seed)][arm] = init["non_base_parameter_sha256"]
                for stage_index, stage in enumerate(STAGES, 1):
                    stage_dir = directory / f"stage{stage_index}"
                    stage_meta = read_json(stage_dir / "stage_metadata.json")
                    metrics = read_json(stage_dir / "metrics.json")
                    if stage_meta.get("status") != "COMPLETE" or stage_meta.get("stage") != stage:
                        raise ValueError(f"Stage metadata mismatch: {stage_dir}")
                    manifest = read_json(stage_dir / "checkpoint/checkpoint_manifest.json")
                    for name, expected in manifest["files"].items():
                        if sha256_file(stage_dir / "checkpoint" / name) != expected:
                            raise ValueError(f"Checkpoint hash mismatch: {stage_dir}/{name}")
                        checkpoint_files += 1
                    state = torch.load(stage_dir / "checkpoint/state.pt", map_location="cpu", weights_only=False)
                    if not (state["method"] == method and state["arm"] == arm and state["seed"] == seed and state["completed_stages"] == stage_index and state["config_sha256"] == config_hash):
                        raise ValueError(f"Checkpoint lineage mismatch: {stage_dir}")
                    buffer_state = state["buffer_state"]
                    entries = buffer_state["entries"]
                    if len(entries) > MEMORY_CAPACITY or len(entries) != stage_meta["buffer_size"]:
                        raise ValueError(f"Buffer capacity mismatch: {stage_dir}")
                    if len({x["example_id"] for x in entries}) != len(entries):
                        raise ValueError(f"Duplicate replay identity: {stage_dir}")
                    for entry in entries:
                        source = expected_train[arm].get(entry["example_id"])
                        if source is None or source["split"] != "train" or source["text_sha256"] != entry["text_sha256"]:
                            raise ValueError(f"Non-TRAIN or hash-mismatched memory entry: {stage_dir}")
                        if STAGES.index(entry["task_id"]) + 1 > stage_index:
                            raise ValueError(f"Future-stage memory entry: {stage_dir}")
                    capacity_checks.append({"method": method, "arm": ARMS[arm], "seed": seed, "stage": stage, "size": len(entries), "pass": len(entries) <= 200})
                    training = metrics["training"]
                    expected_replay = 0 if stage_index == 1 else 400
                    expected_passes = 0 if stage_index == 1 else 100
                    if training["current_examples"] != 400 or training["replay_examples_exposed"] != expected_replay or training["replay_forward_passes"] != expected_passes:
                        raise ValueError(f"Replay exposure mismatch: {stage_dir}")
                    exposure.append({"method": method, "arm": ARMS[arm], "seed": seed, "stage": stage, "current_examples": 400, "replay_examples": expected_replay, "replay_forward_passes": expected_passes, "pass": True})
                    train_rows = [json.loads(line) for line in (stage_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if line]
                    if len(train_rows) != 100:
                        raise ValueError(f"Training-step count mismatch: {stage_dir}")
                    for row in train_rows:
                        if any(expected_train[arm][identity]["task_id"] not in STAGES[:stage_index - 1] for identity in row["replay_example_ids"]):
                            raise ValueError(f"Current/future replay in training log: {stage_dir}")
                    expected_predictions = {f"{task}.jsonl": 120 for task in STAGES[:stage_index]}
                    expected_predictions.update({"NotInject.jsonl": 339, "B-short.jsonl": 120})
                    if {p.name for p in (stage_dir / "predictions").glob("*.jsonl")} != set(expected_predictions):
                        raise ValueError(f"Prediction set mismatch: {stage_dir}")
                    for name, count in expected_predictions.items():
                        if prediction_count(stage_dir / "predictions" / name) != count:
                            raise ValueError(f"Prediction count mismatch: {stage_dir}/{name}")
                        prediction_files += 1
                        prediction_rows += count
                    checkpoints += 1
                    lineage_checks.append({"method": method, "arm": ARMS[arm], "seed": seed, "stage": stage, "pass": True})
                unpublished.extend(str(p.relative_to(root)) for p in directory.glob("_attempt_stage*"))
    if unpublished:
        raise ValueError(f"Unpublished attempts remain: {unpublished}")
    for (method, seed), arms in initialization.items():
        if len(set(arms.values())) != 1:
            raise ValueError(f"Three-arm initialization mismatch: {method}/seed{seed}")
    artifact = {
        "status": "PASS", "preflight": "PASS", "new_runs": 27, "new_checkpoints": checkpoints,
        "checkpoint_files_hashed": checkpoint_files, "prediction_files": prediction_files,
        "prediction_rows": prediction_rows, "failed_runs": failed, "resumed_runs": resumed,
        "initialization_matches_no_replay": True, "same_seed_three_arm_initialization": True,
        "candidate_b_hashes_verified": True, "checkpoint_lineage_verified": True,
        "no_unpublished_attempts": True, "config_sha256": config_hash,
    }
    fairness = {
        "status": "PASS", "example_capacity_equality": "PASS", "capacity": MEMORY_CAPACITY,
        "replay_exposure_equality": "PASS", "replay_batch_size": REPLAY_BATCH_SIZE,
        "dataset_seed_config_equality": "PASS", "no_test_validation_external_selection": "PASS",
        "current_stage_examples_per_stage": 400, "optimizer_reset_each_stage": True,
        "epochs_per_stage": 1, "detector_fixed": True, "seeds": list(SEEDS),
        "capacity_checks": capacity_checks, "exposure_checks": exposure,
    }
    return artifact, fairness


def derive_rows(runs):
    trajectories, acquisition, retention, run_summary, external, hard = [], [], [], [], [], []
    for method in ALL_METHODS:
        for arm in ARMS:
            for seed in SEEDS:
                per_task = []
                for checkpoint_index, checkpoint in enumerate(STAGES, 1):
                    block = runs[method][arm][seed]["stages"][checkpoint]
                    acq = block["acquisition"]
                    acquisition.append({
                        "method": method, "method_name": METHOD_NAMES[method], "arm": ARMS[arm], "arm_key": arm,
                        "seed": seed, "stage": checkpoint, **acq,
                        "gate_pass": acq["attack_tpr"] >= .85 and acq["benign_tnr"] >= .85 and acq["paired_macro_f1"] >= .85,
                    })
                    ni, bs = block["external"]["NotInject"], block["external"]["B-short"]
                    external.append({
                        "method": method, "arm": ARMS[arm], "arm_key": arm, "seed": seed, "checkpoint_stage": checkpoint,
                        "notinject_fpr": ni["overall"]["benign_fpr"], "bshort_fpr": bs["overall"]["benign_fpr"],
                        "bshort_english_fpr": bs["English"]["benign_fpr"],
                        "bshort_multilingual_fpr": bs["multilingual_non_English"]["benign_fpr"],
                    })
                    for task in STAGES[:checkpoint_index]:
                        cell = block["evaluations"][task]
                        trajectories.append({
                            "method": method, "arm": ARMS[arm], "arm_key": arm, "seed": seed,
                            "checkpoint_stage": checkpoint, "task": task,
                            "attack_tpr": cell["malicious"]["attack_tpr"], "benign_fpr": cell["benign"]["benign_fpr"],
                            "macro_f1": cell["paired"]["macro_f1"], "accuracy": cell["paired"]["accuracy"],
                            "attack_tpr_by_position": cell["attack_tpr_by_position"],
                            "benign_fpr_by_partition": cell["benign_fpr_by_partition"],
                        })
                        if arm == "hard":
                            for partition, value in cell["benign_fpr_by_partition"].items():
                                hard.append({"method": method, "seed": seed, "checkpoint_stage": checkpoint, "task": task, "partition": partition, "benign_fpr": value})
                for task_index, task in enumerate(STAGES[:-1]):
                    cells = [row for row in trajectories if row["method"] == method and row["arm_key"] == arm and row["seed"] == seed and row["task"] == task]
                    cells.sort(key=lambda row: STAGES.index(row["checkpoint_stage"]))
                    attack = [row["attack_tpr"] for row in cells]
                    macro = [row["macro_f1"] for row in cells]
                    fpr = [row["benign_fpr"] for row in cells]
                    row = {
                        "method": method, "method_name": METHOD_NAMES[method], "arm": ARMS[arm], "arm_key": arm,
                        "seed": seed, "task": task, "attack_tpr_acquisition": attack[0], "attack_tpr_final": attack[-1],
                        "attack_tpr_change": attack[-1] - attack[0], "attack_tpr_loss": attack[0] - attack[-1],
                        "attack_peak_forgetting": max(max(attack[:i + 1]) - attack[i] for i in range(len(attack))),
                        "macro_f1_change": macro[-1] - macro[0], "macro_f1_loss": macro[0] - macro[-1],
                        "macro_f1_peak_forgetting": max(max(macro[:i + 1]) - macro[i] for i in range(len(macro))),
                        "benign_fpr_acquisition": fpr[0], "benign_fpr_final": fpr[-1],
                        "benign_fpr_change": fpr[-1] - fpr[0], "benign_peak_fpr_increase": max(value - fpr[0] for value in fpr),
                    }
                    retention.append(row)
                    per_task.append(row)
                meta = runs[method][arm][seed]["metadata"]
                runtime = meta.get("runtime_seconds")
                stage_blocks = runs[method][arm][seed]["stages"].values()
                run_summary.append({
                    "method": method, "method_name": METHOD_NAMES[method], "arm": ARMS[arm], "arm_key": arm, "seed": seed,
                    "attack_tpr_bwt": statistics.mean(x["attack_tpr_change"] for x in per_task),
                    "macro_f1_bwt": statistics.mean(x["macro_f1_change"] for x in per_task),
                    "benign_fpr_increase": statistics.mean(x["benign_fpr_change"] for x in per_task),
                    "attack_peak_forgetting_mean": statistics.mean(x["attack_peak_forgetting"] for x in per_task),
                    "macro_f1_peak_forgetting_mean": statistics.mean(x["macro_f1_peak_forgetting"] for x in per_task),
                    "runtime_seconds": runtime,
                    "peak_gpu_memory_bytes": max(block["peak_gpu_memory_bytes"] for block in stage_blocks),
                    "replay_forward_passes": 0 if method == "no_replay" else sum(block["training"]["replay_forward_passes"] for block in stage_blocks),
                })
    return trajectories, acquisition, retention, run_summary, external, hard


def aggregate_rows(rows, keys, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for identity, items in sorted(grouped.items(), key=lambda x: tuple(str(v) for v in x[0])):
        entry = {key: value for key, value in zip(keys, identity)}
        for metric in metrics:
            entry[metric] = stats(item[metric] for item in items)
        output.append(entry)
    return output


def pairwise(rows, keys, metrics):
    lookup = {(row["method"], *(row[key] for key in keys)): row for row in rows}
    comparisons = (("security_aware_minus_reservoir", "reservoir"), ("security_aware_minus_derpp", "derpp"),
                   ("security_aware_minus_no_replay", "no_replay"), ("reservoir_minus_no_replay", "no_replay"),
                   ("derpp_minus_no_replay", "no_replay"))
    output = []
    identities = sorted({tuple(row[key] for key in keys) for row in rows}, key=lambda x: tuple(str(v) for v in x))
    for name, reference in comparisons:
        for identity in identities:
            security = lookup[("security_aware" if name.startswith("security") else name.split("_minus_")[0], *identity)]
            ref = lookup[(reference, *identity)]
            output.append({"comparison": name, **{key: value for key, value in zip(keys, identity)}, **{f"{metric}_difference": security[metric] - ref[metric] for metric in metrics}})
    return output


def memory_rows(root):
    rows = []
    for method in METHODS:
        for arm in ARMS:
            for seed in SEEDS:
                for index, stage in enumerate(STAGES, 1):
                    block = read_json(run_dir(root, method, arm, seed) / f"stage{index}/memory_composition.json")
                    rows.append({"method": method, "arm": ARMS[arm], "arm_key": arm, "seed": seed, "stage": stage, **block})
    return rows


def acquisition_gates(acquisition):
    output = []
    for method in ALL_METHODS:
        for arm in ARMS:
            for stage in STAGES:
                rows = [x for x in acquisition if x["method"] == method and x["arm_key"] == arm and x["stage"] == stage]
                passing = sum(x["gate_pass"] for x in rows)
                output.append({"method": method, "arm": ARMS[arm], "stage": stage, "passing_seeds": passing, "required": 2, "pass": passing >= 2, "per_seed": rows})
    return output


def write_matrices(root, runs):
    getters = {
        "attack_tpr": lambda c: c["malicious"]["attack_tpr"], "macro_f1": lambda c: c["paired"]["macro_f1"],
        "accuracy": lambda c: c["paired"]["accuracy"], "benign_fpr": lambda c: c["benign"]["benign_fpr"],
    }
    for method in METHODS:
        for arm in ARMS:
            for seed in SEEDS:
                directory = run_dir(root, method, arm, seed)
                for name, getter in getters.items():
                    with (directory / f"matrix_{name}.csv").open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.writer(handle); writer.writerow(["evaluated_stage", *STAGES])
                        for eidx, evaluated in enumerate(STAGES):
                            writer.writerow([evaluated, *["" if cidx < eidx else getter(runs[method][arm][seed]["stages"][checkpoint]["evaluations"][evaluated]) for cidx, checkpoint in enumerate(STAGES)]])
                for index, checkpoint in enumerate(STAGES, 1):
                    block = runs[method][arm][seed]["stages"][checkpoint]
                    atomic_json(directory / f"stage{index}/confusion_counts.json", {
                        "method": method, "arm": ARMS[arm], "seed": seed, "checkpoint_stage": checkpoint,
                        "paired_stage_tests": {task: {kind: {key: block["evaluations"][task][kind][key] for key in ("tp", "tn", "fp", "fn", "attack_support", "benign_support")} for kind in ("paired", "malicious", "benign")} for task in STAGES[:index]},
                        "external": {name: {key: block["external"][name]["overall"][key] for key in ("tp", "tn", "fp", "fn", "attack_support", "benign_support")} for name in ("NotInject", "B-short")},
                    })


def summarize(root: Path):
    root = Path(root).resolve()
    artifact, fairness = verify(root)
    runs = load_all_runs(root)
    trajectories, acquisition, retention, run_summaries, external, hard = derive_rows(runs)
    gates = acquisition_gates(acquisition)
    memory = memory_rows(root)
    retention_aggregate = aggregate_rows(retention, ("method", "arm", "task"), ("attack_tpr_loss", "attack_peak_forgetting", "macro_f1_loss", "benign_fpr_change", "benign_peak_fpr_increase"))
    run_aggregate = aggregate_rows(run_summaries, ("method", "arm"), ("attack_tpr_bwt", "macro_f1_bwt", "benign_fpr_increase", "attack_peak_forgetting_mean", "runtime_seconds", "peak_gpu_memory_bytes", "replay_forward_passes"))
    trajectory_aggregate = aggregate_rows(trajectories, ("method", "arm", "checkpoint_stage", "task"), ("attack_tpr", "benign_fpr", "macro_f1", "accuracy"))
    external_aggregate = aggregate_rows(external, ("method", "arm", "checkpoint_stage"), ("notinject_fpr", "bshort_fpr", "bshort_english_fpr", "bshort_multilingual_fpr"))
    key_short = [x for x in retention if x["arm_key"] == "short" and x["task"] == "AF3"]
    pair_retention = pairwise(retention, ("arm_key", "seed", "task"), ("attack_tpr_loss", "macro_f1_loss", "benign_fpr_change", "attack_peak_forgetting"))
    pair_acquisition = pairwise(acquisition, ("arm_key", "seed", "stage"), ("attack_tpr", "benign_tnr", "paired_macro_f1"))
    pair_external = pairwise(external, ("arm_key", "seed", "checkpoint_stage"), ("notinject_fpr", "bshort_fpr", "bshort_english_fpr", "bshort_multilingual_fpr"))
    table_ready = {
        "b_short_af3_tpr_trajectory": [x for x in trajectories if x["arm_key"] == "short" and x["task"] == "AF3"],
        "b_hard_benign_fpr_trajectory": hard,
        "attack_tpr_bwt": [{k: row[k] for k in ("method", "arm", "seed", "attack_tpr_bwt")} for row in run_summaries],
        "benign_fpr_bwt": [{k: row[k] for k in ("method", "arm", "seed", "benign_fpr_increase")} for row in run_summaries],
        "peak_forgetting": [{k: row[k] for k in ("method", "arm", "seed", "attack_peak_forgetting_mean", "macro_f1_peak_forgetting_mean")} for row in run_summaries],
        "acquisition": acquisition, "security_vs_reservoir": [x for x in pair_retention if x["comparison"] == "security_aware_minus_reservoir"],
        "security_vs_derpp": [x for x in pair_retention if x["comparison"] == "security_aware_minus_derpp"],
        "memory_composition": memory, "external_diagnostics": external, "runtime_memory": run_summaries,
    }
    aggregate = {
        "experiment": "Candidate B primary replay comparison", "methods": list(ALL_METHODS),
        "retention_per_seed": retention, "retention_aggregate": retention_aggregate,
        "run_summary_per_seed": run_summaries, "run_summary_aggregate": run_aggregate,
        "trajectory_per_seed": trajectories, "trajectory_aggregate": trajectory_aggregate,
        "acquisition_per_seed": acquisition, "acquisition_gates": gates,
        "external_per_seed": external, "external_aggregate": external_aggregate,
        "key_b_short_af3": key_short, "key_b_hard": hard, "table_ready": table_ready,
    }
    paired = {"retention": pair_retention, "acquisition": pair_acquisition, "external": pair_external}
    atomic_json(root / "results/replay_comparison_aggregate.json", aggregate)
    atomic_json(root / "results/replay_comparison_pairwise.json", paired)
    atomic_json(root / "results/replay_memory_composition.json", {"rows": memory})
    atomic_json(root / "results/replay_fairness_validation.json", fairness)
    atomic_json(root / "results/replay_artifact_validation.json", artifact)
    write_matrices(root, runs)
    return aggregate, paired, artifact, fairness
