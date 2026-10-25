"""Frozen NotInject checkpoint evaluation. This module contains no training path."""
from __future__ import annotations

import ast
import csv
import gc
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .model import batch_inputs

DATASET_ID = "leolee99/NotInject"
REVISION = "847ae76cf8fea5ed325429e569ae8cfef022d2e0"
EXPECTED_SEEDS = (17, 42, 2026)
ORDER_SPECS = {
    "primary": {
        "prefix": "attack-family-sequential-seed",
        "stages": ("AF1", "AF2", "AF3", "AF4"),
    },
    "secondary": {
        "prefix": "attack-family-order-control-seed",
        "stages": ("AF1", "AF4", "AF3", "AF2"),
    },
}
SOURCE_SPLITS = {
    "NotInject_one": (1, "HB1"),
    "NotInject_two": (2, "HB2"),
    "NotInject_three": (3, "HB3"),
}
CATEGORY_NAMES = {
    "Common Queries": "Common",
    "Technique Queries": "Technique",
    "Virtual Creation": "Virtual Creation",
    "Multilingual": "Multilingual",
}
RESULT_STEM = "notinject_checkpoint_evaluation"
THRESHOLD_RULE = "argmax over fixed binary logits; equivalent decision boundary p_attack=0.5"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def load_notinject(root: Path) -> list[dict]:
    raw = root / "data/raw/notinject" / REVISION / "dataset_server"
    manifest_rows = read_jsonl(root / "data/manifests/hard_benign/notinject/records.jsonl")
    manifest = {(row["source_split"], row["source_row_idx"]): row for row in manifest_rows}
    records = []
    for source_split, (trigger_count, hb_group) in SOURCE_SPLITS.items():
        wrapped = []
        for offset in (0, 100):
            payload = json.loads((raw / f"{source_split}_{offset:03d}.json").read_text(encoding="utf-8"))
            if payload["num_rows_total"] != 113 or payload["partial"]:
                raise ValueError(f"Unexpected frozen source response: {source_split}/{offset}")
            wrapped.extend(payload["rows"])
        by_index = {item["row_idx"]: item["row"] for item in wrapped}
        if sorted(by_index) != list(range(113)):
            raise ValueError(f"Frozen split is incomplete: {source_split}")
        for row_index, source in sorted(by_index.items()):
            if set(source) != {"prompt", "word_list", "category"}:
                raise ValueError("Unexpected NotInject fields")
            audited = manifest[(source_split, row_index)]
            prompt_hash = hashlib.sha256(source["prompt"].encode("utf-8")).hexdigest()
            if prompt_hash != audited["prompt_sha256"]:
                raise ValueError(f"Prompt hash mismatch: {source_split}/{row_index}")
            if len(source["word_list"]) != trigger_count or audited["label"] != 0:
                raise ValueError(f"Group/label mismatch: {source_split}/{row_index}")
            records.append({
                "record_id": audited["record_id"],
                "prompt": source["prompt"],
                "true_label": 0,
                "source_split": source_split,
                "source_row_idx": row_index,
                "hb_group": hb_group,
                "trigger_count": trigger_count,
                "category": CATEGORY_NAMES[source["category"]],
            })
    if len(records) != 339:
        raise ValueError(f"Expected 339 records, found {len(records)}")
    if Counter(row["hb_group"] for row in records) != Counter({"HB1": 113, "HB2": 113, "HB3": 113}):
        raise ValueError("Trigger-group count mismatch")
    if Counter(row["category"] for row in records) != Counter({"Common": 126, "Technique": 87, "Virtual Creation": 42, "Multilingual": 84}):
        raise ValueError("Category count mismatch")
    return records


def model_input_records(records: list[dict]) -> list[dict]:
    """Return the only objects passed to the tokenizer/model input builder."""
    return [{"text": row["prompt"]} for row in records]


def labels_and_probabilities(logits: torch.Tensor) -> tuple[list[int], list[float]]:
    labels = logits.argmax(dim=-1).detach().cpu().tolist()
    probabilities = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
    return labels, probabilities


def discover_checkpoints(root: Path) -> list[dict]:
    descriptors = []
    for order_name, specification in ORDER_SPECS.items():
        for seed in EXPECTED_SEEDS:
            matches = []
            for run_dir in (root / "results").glob(f"{specification['prefix']}{seed}-*"):
                metadata_path = run_dir / "metadata.json"
                if metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if metadata.get("status") == "complete" and metadata.get("seed") == seed:
                        matches.append((run_dir, metadata))
            if len(matches) != 1:
                raise ValueError(f"Expected one complete {order_name}/seed{seed} run, found {len(matches)}")
            run_dir, metadata = matches[0]
            if tuple(metadata["stage_order"]) != specification["stages"] or metadata.get("replay") is not False:
                raise ValueError(f"Run contract mismatch: {run_dir.name}")
            if metadata.get("max_length") != 1280 or metadata.get("label_mapping") != {"0": "benign", "1": "indirect_prompt_injection"}:
                raise ValueError(f"Model/label contract mismatch: {run_dir.name}")
            for stage_index, stage in enumerate(specification["stages"], 1):
                checkpoint = run_dir / "checkpoints" / stage
                required = ("config.json", "model.safetensors", "state.pt", "spm.model", "tokenizer_config.json")
                if not checkpoint.is_dir() or any(not (checkpoint / name).is_file() for name in required):
                    raise ValueError(f"Incomplete checkpoint: {checkpoint}")
                descriptors.append({
                    "checkpoint_id": f"{order_name}:seed{seed}:stage{stage_index}:{stage}",
                    "order": order_name,
                    "stage_order": list(specification["stages"]),
                    "seed": seed,
                    "stage": stage,
                    "stage_index": stage_index,
                    "run_id": run_dir.name,
                    "run_dir": run_dir,
                    "checkpoint": checkpoint,
                })
    if len(descriptors) != 24:
        raise ValueError(f"Expected 24 checkpoints, found {len(descriptors)}")
    return descriptors


def checkpoint_fingerprint(descriptors: list[dict]) -> dict:
    """Hash every checkpoint file; hard-linked files are read once per pass."""
    cache = {}
    result = {}
    for descriptor in descriptors:
        files = {}
        for path in sorted(descriptor["checkpoint"].iterdir()):
            if not path.is_file():
                continue
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if identity not in cache:
                cache[identity] = sha256_file(path)
            files[path.name] = {
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": cache[identity],
            }
        result[descriptor["checkpoint_id"]] = files
        print(f"fingerprinted {descriptor['checkpoint_id']}", flush=True)
    return result


def infer_checkpoint(descriptor: dict, records: list[dict], device: str = "cuda", batch_size: int = 32) -> tuple[list[dict], float]:
    checkpoint = descriptor["checkpoint"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, use_fast=False, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    if model.config.id2label != {0: "benign", 1: "indirect_prompt_injection"}:
        raise ValueError(f"Checkpoint label mapping changed: {descriptor['checkpoint_id']}")
    model.to(device)
    model.eval()
    isolated = model_input_records(records)
    output = []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(records), batch_size):
            metadata_batch = records[offset:offset + batch_size]
            input_batch = isolated[offset:offset + batch_size]
            inputs = batch_inputs(input_batch, tokenizer, 1280, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(**inputs).logits
            labels, probabilities = labels_and_probabilities(logits)
            for row, label, probability in zip(metadata_batch, labels, probabilities):
                output.append({
                    "record_id": row["record_id"],
                    "true_label": row["true_label"],
                    "predicted_label": int(label),
                    "p_attack": float(probability),
                    "checkpoint": descriptor["checkpoint_id"],
                    "run_id": descriptor["run_id"],
                    "seed": descriptor["seed"],
                    "stage": descriptor["stage"],
                    "stage_index": descriptor["stage_index"],
                    "order": descriptor["order"],
                    "hb_group": row["hb_group"],
                    "category": row["category"],
                })
    elapsed = time.perf_counter() - started
    if len(output) != len(records):
        raise ValueError("Checkpoint evaluation emitted the wrong prediction count")
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output, elapsed


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def mean_sd(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "values": values,
    }


def summarize_checkpoint(rows: list[dict]) -> dict:
    if len(rows) != 339 or any(row["true_label"] != 0 for row in rows):
        raise ValueError("Each checkpoint must have exactly 339 benign records")
    predicted = [row["predicted_label"] for row in rows]
    probabilities = [row["p_attack"] for row in rows]

    def slice_summary(subset: list[dict]) -> dict:
        false_positives = sum(row["predicted_label"] == 1 for row in subset)
        return {"count": len(subset), "false_positives": false_positives, "fpr": false_positives / len(subset)}

    groups = {group: slice_summary([row for row in rows if row["hb_group"] == group]) for group in ("HB1", "HB2", "HB3")}
    categories = {category: slice_summary([row for row in rows if row["category"] == category]) for category in ("Common", "Technique", "Virtual Creation", "Multilingual")}
    false_positives = sum(predicted)
    return {
        "records": 339,
        "false_positives": false_positives,
        "overall_fpr": false_positives / 339,
        "benign_accuracy": 1 - false_positives / 339,
        "group_fpr": groups,
        "category_fpr": categories,
        "confidence": {
            "mean_p_attack": statistics.mean(probabilities),
            "median_p_attack": statistics.median(probabilities),
            "p90_p_attack": percentile(probabilities, 0.90),
            "p95_p_attack": percentile(probabilities, 0.95),
            "p99_p_attack": percentile(probabilities, 0.99),
        },
    }


def load_attack_metric_lookup(descriptors: list[dict]) -> dict:
    lookup = {}
    for descriptor in descriptors:
        summary = json.loads((descriptor["run_dir"] / "summary.json").read_text(encoding="utf-8"))
        stage_summary = next(row for row in summary["stage_summaries"] if row["stage"] == descriptor["stage"])
        lookup[descriptor["checkpoint_id"]] = {
            "current_stage_attack_tpr": stage_summary["attack_recall"]["acquisition"],
            "retained_attack_family_tpr": stage_summary["attack_recall"]["seen_mean"],
            "previous_family_attack_tpr": stage_summary["attack_recall"]["previous_stage_mean"],
            "macro_f1": stage_summary["macro_f1"]["seen_mean"],
            "previous_family_macro_f1": stage_summary["macro_f1"]["previous_stage_mean"],
        }
    return lookup


def aggregate_predictions(predictions: list[dict], descriptors: list[dict], runtimes: dict, fingerprints: dict) -> dict:
    descriptor_by_id = {row["checkpoint_id"]: row for row in descriptors}
    attack_lookup = load_attack_metric_lookup(descriptors)
    by_checkpoint = defaultdict(list)
    for row in predictions:
        by_checkpoint[row["checkpoint"]].append(row)
    if set(by_checkpoint) != set(descriptor_by_id):
        raise ValueError("Prediction/checkpoint set mismatch")
    per_checkpoint = []
    for descriptor in descriptors:
        checkpoint_id = descriptor["checkpoint_id"]
        item = {
            "checkpoint": checkpoint_id,
            "checkpoint_path": descriptor["checkpoint"].relative_to(descriptor["checkpoint"].parents[2]).as_posix(),
            "run_id": descriptor["run_id"],
            "order": descriptor["order"],
            "seed": descriptor["seed"],
            "stage": descriptor["stage"],
            "stage_index": descriptor["stage_index"],
            "evaluation_runtime_seconds": runtimes[checkpoint_id],
            "model_sha256": fingerprints[checkpoint_id]["model.safetensors"]["sha256"],
            **summarize_checkpoint(by_checkpoint[checkpoint_id]),
            **attack_lookup[checkpoint_id],
        }
        per_checkpoint.append(item)

    item_lookup = {(row["order"], row["seed"], row["stage_index"]): row for row in per_checkpoint}
    trajectories = []
    for order in ORDER_SPECS:
        for seed in EXPECTED_SEEDS:
            sequence = [item_lookup[(order, seed, index)] for index in range(1, 5)]
            baseline = sequence[0]
            trajectory = {"order": order, "seed": seed, "stages": []}
            for index, row in enumerate(sequence):
                previous = sequence[index - 1] if index else row
                trajectory["stages"].append({
                    "stage": row["stage"],
                    "stage_index": row["stage_index"],
                    "overall_fpr": row["overall_fpr"],
                    "absolute_change_from_AF1_pp": 100 * (row["overall_fpr"] - baseline["overall_fpr"]),
                    "step_change_pp": 100 * (row["overall_fpr"] - previous["overall_fpr"]),
                    "mean_p_attack": row["confidence"]["mean_p_attack"],
                    "mean_p_attack_change_from_AF1": row["confidence"]["mean_p_attack"] - baseline["confidence"]["mean_p_attack"],
                    "group_change_from_AF1_pp": {group: 100 * (row["group_fpr"][group]["fpr"] - baseline["group_fpr"][group]["fpr"]) for group in ("HB1", "HB2", "HB3")},
                    "category_change_from_AF1_pp": {category: 100 * (row["category_fpr"][category]["fpr"] - baseline["category_fpr"][category]["fpr"]) for category in ("Common", "Technique", "Virtual Creation", "Multilingual")},
                })
            deltas = [stage["absolute_change_from_AF1_pp"] for stage in trajectory["stages"][1:]]
            trajectory["final_change_pp"] = deltas[-1]
            trajectory["peak_increase_pp"] = max([0.0, *deltas])
            trajectory["increase_pattern"] = "persistent" if deltas[-1] > 0 else ("transient" if max([0.0, *deltas]) > 0 else "none")
            trajectories.append(trajectory)

    cross_seed = {}
    for order, specification in ORDER_SPECS.items():
        cross_seed[order] = {}
        for stage_index, stage in enumerate(specification["stages"], 1):
            rows = [item_lookup[(order, seed, stage_index)] for seed in EXPECTED_SEEDS]
            baselines = [item_lookup[(order, seed, 1)] for seed in EXPECTED_SEEDS]
            metric_values = {
                "overall_fpr": [row["overall_fpr"] for row in rows],
                "mean_p_attack": [row["confidence"]["mean_p_attack"] for row in rows],
            }
            metric_values.update({f"{group}_fpr": [row["group_fpr"][group]["fpr"] for row in rows] for group in ("HB1", "HB2", "HB3")})
            metric_values.update({f"category_{category}_fpr": [row["category_fpr"][category]["fpr"] for row in rows] for category in ("Common", "Technique", "Virtual Creation", "Multilingual")})
            entry = {name: mean_sd(values) for name, values in metric_values.items()}
            overall_deltas = [100 * (row["overall_fpr"] - baseline["overall_fpr"]) for row, baseline in zip(rows, baselines)]
            p_deltas = [row["confidence"]["mean_p_attack"] - baseline["confidence"]["mean_p_attack"] for row, baseline in zip(rows, baselines)]
            entry["overall_fpr_change_from_AF1_pp"] = mean_sd(overall_deltas)
            entry["mean_p_attack_change_from_AF1"] = mean_sd(p_deltas)
            entry["harmful_fpr_change_seed_count"] = sum(value > 0 for value in overall_deltas)
            cross_seed[order][stage] = entry

    immediate = []
    final = []
    for seed in EXPECTED_SEEDS:
        p1, p2, p4 = item_lookup[("primary", seed, 1)], item_lookup[("primary", seed, 2)], item_lookup[("primary", seed, 4)]
        s1, s2, s4 = item_lookup[("secondary", seed, 1)], item_lookup[("secondary", seed, 2)], item_lookup[("secondary", seed, 4)]
        immediate.append({
            "seed": seed,
            "AF2_after_AF1_change_pp": 100 * (p2["overall_fpr"] - p1["overall_fpr"]),
            "AF4_after_AF1_change_pp": 100 * (s2["overall_fpr"] - s1["overall_fpr"]),
            "AF4_minus_AF2_immediate_change_pp": 100 * ((s2["overall_fpr"] - s1["overall_fpr"]) - (p2["overall_fpr"] - p1["overall_fpr"])),
        })
        final.append({
            "seed": seed,
            "primary_final_change_pp": 100 * (p4["overall_fpr"] - p1["overall_fpr"]),
            "secondary_final_change_pp": 100 * (s4["overall_fpr"] - s1["overall_fpr"]),
        })
    group_overall = {group: mean_sd([row["group_fpr"][group]["fpr"] for row in per_checkpoint]) for group in ("HB1", "HB2", "HB3")}
    category_overall = {category: mean_sd([row["category_fpr"][category]["fpr"] for row in per_checkpoint]) for category in ("Common", "Technique", "Virtual Creation", "Multilingual")}

    tradeoff = [{
        "checkpoint": row["checkpoint"], "seed": row["seed"], "order": row["order"], "current_stage": row["stage"],
        "retained_attack_family_tpr": row["retained_attack_family_tpr"], "macro_f1": row["macro_f1"],
        "notinject_fpr": row["overall_fpr"], "notinject_mean_p_attack": row["confidence"]["mean_p_attack"],
    } for row in per_checkpoint]
    return {
        "status": "complete_frozen_evaluation_only",
        "dataset": {"id": DATASET_ID, "revision": REVISION, "records": 339, "label": 0, "use": "frozen evaluation only"},
        "checkpoint_count": 24,
        "prediction_count": len(predictions),
        "orders": {name: list(spec["stages"]) for name, spec in ORDER_SPECS.items()},
        "seeds": list(EXPECTED_SEEDS),
        "threshold_rule": THRESHOLD_RULE,
        "model_input": "prompt text only",
        "per_checkpoint": per_checkpoint,
        "trajectories": trajectories,
        "cross_seed_mean_sample_sd": cross_seed,
        "order_comparison": {"immediate_after_AF1": immediate, "final_vs_AF1": final, "all_checkpoint_group_fpr": group_overall, "all_checkpoint_category_fpr": category_overall},
        "security_tradeoff": tradeoff,
        "scope": {"training": False, "optimizer_loaded": False, "threshold_changed": False, "model_selection": False, "replay": False, "selective_adaptation": False, "H2_F_constructed": False},
    }



def write_csv_outputs(root: Path, aggregate: dict) -> None:
    per_checkpoint = aggregate["per_checkpoint"]
    with (root / "results/notinject_checkpoint_fpr_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order", "seed", "stage_1", "stage_2", "stage_3", "stage_4"])
        lookup = {(row["order"], row["seed"], row["stage_index"]): row["overall_fpr"] for row in per_checkpoint}
        for order in ORDER_SPECS:
            for seed in EXPECTED_SEEDS:
                writer.writerow([order, seed, *[lookup[(order, seed, index)] for index in range(1, 5)]])
    with (root / "results/notinject_checkpoint_group_fpr.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["checkpoint", "order", "seed", "stage", "stage_index", "group", "count", "false_positives", "fpr"])
        for row in per_checkpoint:
            for group, value in row["group_fpr"].items():
                writer.writerow([row["checkpoint"], row["order"], row["seed"], row["stage"], row["stage_index"], group, value["count"], value["false_positives"], value["fpr"]])
    with (root / "results/notinject_checkpoint_category_fpr.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["checkpoint", "order", "seed", "stage", "stage_index", "category", "count", "false_positives", "fpr"])
        for row in per_checkpoint:
            for category, value in row["category_fpr"].items():
                writer.writerow([row["checkpoint"], row["order"], row["seed"], row["stage"], row["stage_index"], category, value["count"], value["false_positives"], value["fpr"]])


def verify_saved(root: Path, before: dict | None = None, after: dict | None = None, reproducibility: dict | None = None) -> dict:
    predictions_path = root / f"results/{RESULT_STEM}_predictions.jsonl"
    aggregate_path = root / f"results/{RESULT_STEM}_aggregate.json"
    predictions = read_jsonl(predictions_path)
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    descriptors = discover_checkpoints(root)
    by_checkpoint = defaultdict(list)
    for row in predictions:
        by_checkpoint[row["checkpoint"]].append(row)
    counts_ok = len(predictions) == 24 * 339 and len(by_checkpoint) == 24 and all(len(rows) == 339 for rows in by_checkpoint.values())
    slice_counts_ok = all(
        Counter(row["hb_group"] for row in rows) == Counter({"HB1": 113, "HB2": 113, "HB3": 113})
        and Counter(row["category"] for row in rows) == Counter({"Common": 126, "Technique": 87, "Virtual Creation": 42, "Multilingual": 84})
        for rows in by_checkpoint.values()
    )
    runtimes = {row["checkpoint"]: row["evaluation_runtime_seconds"] for row in aggregate["per_checkpoint"]}
    fingerprint_for_recompute = before or {
        row["checkpoint"]: {"model.safetensors": {"sha256": row["model_sha256"]}}
        for row in aggregate["per_checkpoint"]
    }
    recomputed = aggregate_predictions(predictions, descriptors, runtimes, fingerprint_for_recompute)
    aggregate_matches = (
        recomputed["per_checkpoint"] == aggregate["per_checkpoint"]
        and recomputed["trajectories"] == aggregate["trajectories"]
        and recomputed["cross_seed_mean_sample_sd"] == aggregate["cross_seed_mean_sample_sd"]
        and recomputed["order_comparison"] == aggregate["order_comparison"]
        and recomputed["security_tradeoff"] == aggregate["security_tradeoff"]
    )
    fingerprints_equal = before == after if before is not None and after is not None else None
    optimizer_states_equal = (
        all(before[key]["state.pt"] == after[key]["state.pt"] for key in before)
        if before is not None and after is not None else None
    )
    module_text = Path(__file__).read_text(encoding="utf-8-sig")
    tree = ast.parse(module_text)
    forbidden_calls = {"train", "backward", "step", "zero_grad"}
    no_training_api = not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_calls
        for node in ast.walk(tree)
    )
    checks = {
        "exactly_24_checkpoints": len(by_checkpoint) == 24,
        "exactly_339_records_per_checkpoint": counts_ok,
        "prediction_count_8136": len(predictions) == 8136,
        "group_and_category_counts_match_audit": slice_counts_ok,
        "aggregate_matches_raw_predictions": aggregate_matches,
        "fixed_threshold_rule": aggregate["threshold_rule"] == THRESHOLD_RULE,
        "prompt_only_model_input": aggregate["model_input"] == "prompt text only",
        "no_training_api_in_evaluator": no_training_api,
        "scope_records_no_training_or_replay": aggregate["scope"] == {"training": False, "optimizer_loaded": False, "threshold_changed": False, "model_selection": False, "replay": False, "selective_adaptation": False, "H2_F_constructed": False},
        "checkpoint_contents_unchanged": fingerprints_equal,
        "optimizer_state_unchanged": optimizer_states_equal,
        "repeated_sample_reproducible": reproducibility["passed"] if reproducibility else None,
    }
    required = [value for value in checks.values() if value is not None]
    validation = {
        "status": "passed" if all(required) else "failed",
        "checks": checks,
        "checkpoint_count": len(by_checkpoint),
        "records_per_checkpoint": sorted({len(rows) for rows in by_checkpoint.values()}),
        "prediction_count": len(predictions),
        "reproducibility": reproducibility,
        "checkpoint_fingerprints_before": before,
        "checkpoint_fingerprints_after": after,
        "aggregate_sha256": sha256_file(aggregate_path),
        "predictions_sha256": sha256_file(predictions_path),
    }
    if not all(required):
        raise ValueError(f"Artifact validation failed: {checks}")
    return validation


def run(root: Path) -> dict:
    root = root.resolve()
    if root != Path(r"E:\ContinualGuard"):
        raise ValueError(f"H2-D1 must run from E:\\ContinualGuard, got {root}")
    if not torch.cuda.is_available():
        raise RuntimeError("Approved checkpoint evaluation requires the verified CUDA environment")
    descriptors = discover_checkpoints(root)
    records = load_notinject(root)
    print("Hashing 24 frozen checkpoints before evaluation", flush=True)
    before = checkpoint_fingerprint(descriptors)
    predictions = []
    runtimes = {}
    started = time.perf_counter()
    for index, descriptor in enumerate(descriptors, 1):
        rows, elapsed = infer_checkpoint(descriptor, records)
        predictions.extend(rows)
        runtimes[descriptor["checkpoint_id"]] = elapsed
        print(f"evaluated {index}/24 {descriptor['checkpoint_id']} in {elapsed:.2f}s", flush=True)
    predictions_path = root / f"results/{RESULT_STEM}_predictions.jsonl"
    atomic_text(predictions_path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
    aggregate = aggregate_predictions(predictions, descriptors, runtimes, before)
    aggregate["evaluation_runtime_seconds"] = time.perf_counter() - started
    write_json(root / f"results/{RESULT_STEM}_aggregate.json", aggregate)
    write_csv_outputs(root, aggregate)

    sample_records = records[:16]
    repeated, repeat_runtime = infer_checkpoint(descriptors[0], sample_records)
    original = [row for row in predictions if row["checkpoint"] == descriptors[0]["checkpoint_id"]][:16]
    max_difference = max(abs(left["p_attack"] - right["p_attack"]) for left, right in zip(original, repeated))
    reproducibility = {
        "checkpoint": descriptors[0]["checkpoint_id"],
        "sample_records": 16,
        "repeat_runtime_seconds": repeat_runtime,
        "labels_identical": [row["predicted_label"] for row in original] == [row["predicted_label"] for row in repeated],
        "max_abs_p_attack_difference": max_difference,
        "tolerance": 1e-5,
    }
    reproducibility["passed"] = reproducibility["labels_identical"] and max_difference <= reproducibility["tolerance"]
    print("Hashing 24 frozen checkpoints after evaluation", flush=True)
    after = checkpoint_fingerprint(descriptors)
    validation = verify_saved(root, before, after, reproducibility)
    write_json(root / f"results/{RESULT_STEM}_artifact_validation.json", validation)
    print(json.dumps({
        "status": validation["status"],
        "checkpoints": 24,
        "records_per_checkpoint": 339,
        "predictions": len(predictions),
        "runtime_seconds": aggregate["evaluation_runtime_seconds"],
    }, indent=2), flush=True)
    return aggregate




