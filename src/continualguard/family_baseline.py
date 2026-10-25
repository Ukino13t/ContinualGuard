"""Approved four-stage attack-family sequential baseline without replay."""

from __future__ import annotations

import copy
import csv
import json
import statistics
import time
import uuid
from collections import Counter
from pathlib import Path

import torch
import yaml

from .artifacts import append_jsonl, digest, history, now, read_jsonl, write_json
from .data import validate_record
from .evaluate import evaluate
from .metrics import HIGHER, METRICS, summarize
from .model import batch_inputs, make_model, make_tokenizer
from .reproducibility import environment, isolated_evaluation, save_checkpoint, seed_everything
from .train import REAL_MODEL_SHA256, REAL_REVISION, file_sha256, make_optimizer, train_stage


DATA_SHA256 = "4b4efaf7640f31f5f12af2f2b3a40b91bbb71789c91c059d21ce4af1e6a9e43a"
STAGES = ("AF1", "AF2", "AF3", "AF4")
SECONDARY_ORDER = ("AF1", "AF4", "AF3", "AF2")
APPROVED_ORDERS = (STAGES, SECONDARY_ORDER)
FAMILIES = {"AF1": "naive", "AF2": "ignore", "AF3": "escape_separation", "AF4": "fake_completion"}
SEEDS = (17, 42, 2026)
POSITIONS = ("head", "middle", "tail")
EXPECTED = {"train": (3000, 1500, 1500, 500), "val": (300, 150, 150, 50), "test": (360, 180, 180, 60)}


class ApprovedFamilyData:
    """Validated view of the immutable audited AF1-AF4 subset."""

    def __init__(self, all_rows):
        if len(all_rows) != 18_300:
            raise ValueError("Audited source must contain exactly 18,300 records")
        for row in all_rows:
            validate_record(row)
        self.rows = [row for row in all_rows if row["task_id"] in STAGES]
        if len(self.rows) != 14_640 or any(row["task_id"] == "AF5" for row in self.rows):
            raise ValueError("Primary AF1-AF4 view must contain exactly 14,640 records and exclude AF5")
        counts = []
        for stage in STAGES:
            for split, expected in EXPECTED.items():
                part = self.records(stage, split)
                labels = Counter(row["label"] for row in part)
                positions = Counter(row["injection_position"] for row in part if row["label"] == 1)
                observed = (len(part), labels[0], labels[1], positions["head"])
                if observed != expected or any(positions[position] != expected[3] for position in POSITIONS):
                    raise ValueError(f"Audited count/position mismatch for {stage}/{split}: {observed}, {positions}")
                if {row["attack_family"] for row in part if row["label"] == 1} != {FAMILIES[stage]}:
                    raise ValueError(f"Family mismatch for {stage}/{split}")
                counts.append({"stage": stage, "split": split, "total": len(part), "benign": labels[0], "injection": labels[1], "positions": dict(positions)})
        self.audit = {"status": "passed", "records": len(self.rows), "stage_order": list(STAGES), "AF5_included": False, "counts": counts}

    def records(self, stage, split):
        return [row for row in self.rows if row["task_id"] == stage and row["split"] == split]


def validate_contract(config, root: Path, expected_order=STAGES) -> ApprovedFamilyData:
    order = tuple(expected_order)
    if order not in APPROVED_ORDERS:
        raise ValueError(f"Unapproved attack-family order: {order}")
    if not (
        config["run_type"] == "reportable research result"
        and config["stage_order"] == list(order)
        and config["seed"] in SEEDS
        and config["model"]["kind"] == "pretrained_deberta"
        and config["model"]["checkpoint"] == "microsoft/deberta-v3-base"
        and config["model"]["revision"] == REAL_REVISION
        and config["model"]["max_length"] == 1280
        and config["training"]["epochs"] == 1
        and config["training"]["batch_size"] == 4
        and config["training"]["gradient_accumulation_steps"] == 1
        and config["training"]["gradient_checkpointing"] is True
        and config["training"]["learning_rate"] == 1e-5
        and config["training"]["optimizer_transition"] == "reset"
        and config["training"]["pre_adaptation"] is False
        and config["device"] == "cuda"
        and config["precision"] == "float16"
    ):
        raise ValueError("Run does not match the human-approved AF1-AF4 baseline contract")
    source = (root / config["data_path"]).resolve()
    if not source.is_relative_to(root.resolve()) or file_sha256(source) != DATA_SHA256:
        raise ValueError("Attack-family dataset path or SHA-256 differs from the audited artifact")
    construction = json.loads((root / "data/manifests/attack_family_shift/construction.json").read_text(encoding="utf-8"))
    audit_index = json.loads((root / "data/manifests/attack_family_shift/audit_index.json").read_text(encoding="utf-8"))
    overlap = json.loads((root / "data/audits/attack_family_shift/overlap_audit.json").read_text(encoding="utf-8"))
    lengths = json.loads((root / "data/audits/attack_family_shift/length_audit.json").read_text(encoding="utf-8"))
    if (
        construction["processed_sha256"] != DATA_SHA256
        or construction["primary_stage_order"] != list(STAGES)
        or construction["variant_b"]["status"] != "not_constructed"
        or construction["blocking_findings"]
        or audit_index["primary_records"] != 14_640
        or audit_index.get("blocking_findings", [])
        or overlap["blocking_findings"]
        or overlap["audits"]["template_component_semantics"]["component_overlap_in_primary"]
        or lengths.get("exclusions")
    ):
        raise ValueError("Existing attack-family audit no longer passes the approved gates")
    rows = read_jsonl(source)
    return ApprovedFamilyData(rows)


def validation_loss(model, tokenizer, records, config):
    total_loss = 0.0
    total = 0
    batch_size = config["training"]["batch_size"]
    with isolated_evaluation(model):
        for offset in range(0, len(records), batch_size):
            batch = records[offset : offset + batch_size]
            inputs = batch_inputs(batch, tokenizer, config["model"]["max_length"], config["device"])
            labels = torch.tensor([row["label"] for row in batch], dtype=torch.long, device=config["device"])
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                loss = model(**inputs, labels=labels).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite validation loss")
            total_loss += loss.item() * len(batch)
            total += len(batch)
    return total_loss / total


def training_loss(run_dir: Path, stage: str):
    rows = [row for row in read_jsonl(run_dir / "train_log.jsonl") if row["stage"] == stage]
    count = sum(row["examples_in_step"] for row in rows)
    return sum(row["loss"] * row["examples_in_step"] for row in rows) / count


def position_tpr(cell, position):
    return cell["slices"]["injection_position"][position]["attack_recall"]


def write_outputs(run_dir: Path, cells, diagnostics, stage_order=STAGES, experiment_regime="attack_family_sequential_no_replay"):
    order = tuple(stage_order)
    lookup = {(cell["trained_stage"], cell["eval_stage"]): cell for cell in cells if cell["phase"] == "post"}
    names = {"accuracy": "accuracy", "macro_f1": "macro_f1", "attack_recall": "attack_tpr", "benign_fpr": "benign_fpr"}
    for metric, name in names.items():
        with (run_dir / f"matrix_{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["trained_stage", *order])
            for trained in order:
                writer.writerow([trained, *[lookup[(trained, stage)][metric] if (trained, stage) in lookup else "" for stage in order]])
    # Preserve the established internal metric filename as an alias.
    (run_dir / "matrix_attack_recall.csv").write_bytes((run_dir / "matrix_attack_tpr.csv").read_bytes())
    position_payload = {}
    for position in POSITIONS:
        position_payload[position] = {}
        with (run_dir / f"matrix_attack_tpr_{position}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["trained_stage", *order])
            for trained in order:
                row = {}
                for stage in order:
                    if (trained, stage) in lookup:
                        row[stage] = position_tpr(lookup[(trained, stage)], position)
                position_payload[position][trained] = row
                writer.writerow([trained, *[row.get(stage, "") for stage in order]])
    stage_summary = summarize(cells, order)
    summary = {
        "stage_order": list(order),
        "families": FAMILIES,
        "stage_summaries": stage_summary,
        "stage_acquisition_diagnostics": diagnostics,
        "position_attack_tpr": position_payload,
        "metric_convention": "Signed BWT and peak forgetting apply only to higher-is-better metrics; benign-FPR increase is separate",
        "experiment_regime": experiment_regime,
        "evidence_type": "Chen-derived deterministic reconstruction; four-stage component-disjoint attack-family stream",
        "replay": "not implemented or run",
        "AF5": "excluded from primary baseline",
        "variant_b": "not constructed",
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def run_family_baseline(config, root, config_file="programmatic", experiment_regime="attack_family_sequential_no_replay", run_prefix="attack-family-sequential"):
    root = Path(root)
    order = tuple(config["stage_order"])
    data = validate_contract(config, root, order)
    if not torch.cuda.is_available():
        raise ValueError("Approved attack-family baseline requires CUDA")
    weight_path = (root / config["model"]["local_artifact"] / "pytorch_model.bin").resolve()
    weight_hash = file_sha256(weight_path)
    if weight_hash != REAL_MODEL_SHA256:
        raise ValueError("Pinned base-model weight hash mismatch")
    seed = config["seed"]
    run_id = f"{run_prefix}-seed{seed}-" + time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:10]
    run_dir = root / "results" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "predictions").mkdir()
    resolved = copy.deepcopy(config)
    resolved["replay"] = False
    resolved["historical_training_samples"] = 0
    config_hash = digest(json.dumps(resolved, sort_keys=True).encode())
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
    metadata = {
        "run_id": run_id, "date": now(), "config_file": str(config_file), "config_hash": config_hash,
        "project_root": str(root.resolve()),
        "seed": seed, "stage_order": list(order), "stage_families": FAMILIES,
        "experiment_regime": experiment_regime, "run_type": config["run_type"],
        "model_checkpoint": config["model"]["checkpoint"], "model_revision": config["model"]["revision"],
        "max_length": 1280, "label_mapping": {"0": "benign", "1": "indirect_prompt_injection"},
        "optimizer_transition": "reset", "model_weights_carried": True, "current_stage_only": True,
        "replay": False, "AF5_included": False, "variant_b_constructed": False,
        "effective_batch_size": 4, "learning_rate": 1e-5, "device": "cuda", "precision": "float16",
        "data_manifest_hash": DATA_SHA256, "model_weights_sha256": weight_hash,
        "warnings": [
            "Chen-derived deterministic reconstruction, not an exact paper reproduction",
            "Synthetic family wrappers may permit family-specific lexical shortcuts",
            "No semantic-paraphrase guarantee; AF5 Combined and Variant B are excluded",
            "No replay or selective adaptation is implemented or run",
        ],
        "status": "running",
    }
    started = time.perf_counter()
    completed = 0
    cells = []
    diagnostics = []
    summary = None
    try:
        seed_everything(seed, config["deterministic"], config["threads"])
        env = environment(root)
        metadata.update(environment=env, hardware_device=env["gpu"], git_commit=env["git_commit"])
        metadata["warnings"].extend(env["compatibility_warnings"])
        write_json(run_dir / "data_audit.json", data.audit)
        write_json(run_dir / "length_audit.json", {"approved_source": "data/audits/attack_family_shift/length_audit.json", "max_length": 1280, "exclusions": [], "all_primary_records_retained": True})
        write_json(run_dir / "exclusions.json", [])
        write_json(run_dir / "data_manifest.json", {
            "source_path": config["data_path"], "source_sha256": DATA_SHA256,
            "audit_manifests": ["data/manifests/attack_family_shift/construction.json", "data/manifests/attack_family_shift/audit_index.json"],
            "records": [{key: row[key] for key in ("example_id", "text_sha256", "task_id", "split", "source")} for row in data.rows],
        })
        tokenizer = make_tokenizer(config["model"], root)
        model = make_model(config["model"], root)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.to("cuda")
        metadata["model_architecture"] = model.config.to_dict()
        metadata["architecture_hash"] = digest(model.config.to_json_string().encode())
        metadata["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        write_json(run_dir / "metadata.json", metadata)
        global_step = 0
        for index, stage in enumerate(order):
            stage_started = time.perf_counter()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            optimizer, scheduler = make_optimizer(model, config)
            append_jsonl(run_dir / "stage_events.jsonl", {
                "stage": stage, "event": "stage_start", "optimizer_transition": "reset",
                "optimizer_state_entries_before_training": len(optimizer.state), "model_weights_carried": index > 0,
                "current_stage_training_records": len(data.records(stage, "train")), "historical_training_records": 0,
                "allowed_task_ids": [stage], "global_step_before": global_step,
            })
            global_step = train_stage(
                model, tokenizer, data.records(stage, "train"), config, optimizer, scheduler,
                run_dir, stage, global_step, allowed_task_ids=(stage,),
            )
            for seen in order[: index + 1]:
                cells.append(evaluate(model, tokenizer, data.records(seen, "test"), config, run_dir, stage, seen, "post"))
            val_loss = validation_loss(model, tokenizer, data.records(stage, "val"), config)
            acquisition = next(cell for cell in cells if cell["trained_stage"] == stage and cell["eval_stage"] == stage)
            diagnostic = {
                "stage": stage, "family": FAMILIES[stage], "attack_tpr": acquisition["attack_recall"],
                "benign_fpr": acquisition["benign_fpr"], "macro_f1": acquisition["macro_f1"],
                "accuracy": acquisition["accuracy"], "training_loss": training_loss(run_dir, stage),
                "validation_loss": val_loss, "runtime_seconds": None, "peak_gpu_memory_bytes": None,
                "peak_gpu_reserved_bytes": None,
            }
            diagnostic["runtime_seconds"] = time.perf_counter() - stage_started
            diagnostic["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
            diagnostic["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
            checkpoint = run_dir / "checkpoints" / stage
            save_checkpoint(checkpoint, model, tokenizer, optimizer, scheduler, {
                "experiment_regime": experiment_regime, "completed_stages": index + 1,
                "global_step": global_step, "config_hash": config_hash, "data_hash": DATA_SHA256,
                "environment": env, "cells": cells, "diagnostics": diagnostics + [diagnostic],
                "replay": False, "historical_training_samples": 0,
            })
            diagnostics.append(diagnostic)
            append_jsonl(run_dir / "stage_events.jsonl", {
                "stage": stage, "event": "stage_complete", "elapsed_seconds": diagnostic["runtime_seconds"],
                "peak_gpu_memory_bytes": diagnostic["peak_gpu_memory_bytes"],
                "peak_gpu_reserved_bytes": diagnostic["peak_gpu_reserved_bytes"], "global_step_after": global_step,
            })
            summary = write_outputs(run_dir, cells, diagnostics, order, experiment_regime)
            completed = index + 1
        metadata["status"] = "complete"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["completed_stages"] = completed
        metadata["finished_at"] = now()
        metadata["runtime_seconds"] = time.perf_counter() - started
        metadata["peak_gpu_memory_bytes"] = max((item["peak_gpu_memory_bytes"] for item in diagnostics), default=0)
        metadata["peak_gpu_reserved_bytes"] = max((item["peak_gpu_reserved_bytes"] for item in diagnostics), default=0)
        metadata["summary_metrics"] = summary
        write_json(run_dir / "metadata.json", metadata)
        write_json(root / "history/runs" / f"{run_id}.json", metadata)
        history(root, "experiments", metadata)
    return run_dir


def mean_sd(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "values": values}
