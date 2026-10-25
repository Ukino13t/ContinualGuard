"""Audited acquisition, epoch-budget, order, and joint optimization controls."""
import copy
import csv
import json
import time
import uuid
from pathlib import Path

import torch
import yaml

from .artifacts import append_jsonl, digest, history, now, read_jsonl, write_json
from .evaluate import evaluate, write_summary
from .model import make_model, make_tokenizer
from .reproducibility import environment, load_checkpoint, restore_rng, save_checkpoint, seed_everything
from .train import (
    ApprovedRealData,
    REAL_DATA_SHA256,
    REAL_MODEL_SHA256,
    REAL_REVISION,
    file_sha256,
    make_optimizer,
    train_stage,
)


SEEDS = (17, 42, 2026)
METRICS = ("accuracy", "macro_f1", "attack_recall", "benign_fpr")
CONTROL_TYPES = {
    "s2_only": "S2-only acquisition control",
    "sequential_s2_epochs": "Sequential S1-to-S2 epoch ablation",
    "reverse_order": "Reverse-order S2-to-S1 control",
    "joint_epochs": "Joint optimization epoch check",
}


def _approved_config(config, control):
    allowed_seed = config["seed"] in SEEDS if control != "reverse_order" else config["seed"] == 42
    if not (
        control in CONTROL_TYPES
        and config["run_type"] == "pilot experiment"
        and config["model"]["kind"] == "pretrained_deberta"
        and config["model"]["checkpoint"] == "microsoft/deberta-v3-base"
        and config["model"]["revision"] == REAL_REVISION
        and config["model"]["max_length"] == 1280
        and config["stage_order"] == ["S1", "S2"]
        and allowed_seed
        and config["training"]["epochs"] == 1
        and config["training"]["batch_size"] == 4
        and config["training"]["gradient_accumulation_steps"] == 1
        and config["training"]["gradient_checkpointing"] is True
        and config["training"]["learning_rate"] == 1e-5
        and config["training"]["optimizer_transition"] == "reset"
        and config["device"] == "cuda"
        and config["precision"] == "float16"
    ):
        raise ValueError("Control does not match the human-approved baseline-sanity contract")


def _load_approved_data(config, root, run_dir, metadata):
    source = (root / config["data_path"]).resolve()
    if not source.is_relative_to(root.resolve()):
        raise ValueError("Dataset path must stay in project root")
    raw = source.read_bytes()
    metadata["data_manifest_hash"] = digest(raw)
    if metadata["data_manifest_hash"] != REAL_DATA_SHA256:
        raise ValueError("Real dataset differs from the human-approved audited artifact")
    construction = json.loads((root / "data/audits/real_data/construction_manifest.json").read_text(encoding="utf-8"))
    overlap = json.loads((root / "data/audits/real_data/overlap_audit.json").read_text(encoding="utf-8"))
    if construction["processed_sha256"] != REAL_DATA_SHA256 or construction["blocking_overlap_findings"] or overlap.get("errors"):
        raise ValueError("Approved real-data audit gate no longer passes")
    rows = read_jsonl(source)
    data = ApprovedRealData(rows, config["stage_order"])
    write_json(run_dir / "data_audit.json", data.audit)
    write_json(
        run_dir / "data_manifest.json",
        {
            "source_path": config["data_path"],
            "sha256": metadata["data_manifest_hash"],
            "records": [
                {key: row[key] for key in ("example_id", "text_sha256", "task_id", "split", "source")}
                for row in rows
            ],
        },
    )
    approved_length = json.loads((root / "data/audits/real_data/length_audit.json").read_text(encoding="utf-8"))
    audit = {
        "approved_source": "data/audits/real_data/length_audit.json",
        "max_length": 1280,
        "coverage": [item for item in approved_length["coverage"] if item["max_length"] == 1280],
        "exclusions": [item for item in approved_length["exclusions"] if item["max_length"] == 1280],
    }
    write_json(run_dir / "length_audit.json", audit)
    write_json(run_dir / "exclusions.json", audit["exclusions"])
    if audit["exclusions"]:
        raise ValueError("Approved 1280-token audit unexpectedly has exclusions")
    weight_path = (root / config["model"]["local_artifact"] / "pytorch_model.bin").resolve()
    metadata["model_weights_sha256"] = file_sha256(weight_path)
    if metadata["model_weights_sha256"] != REAL_MODEL_SHA256 or not torch.cuda.is_available():
        raise ValueError("Pinned model or CUDA gate failed")
    return data


def _fresh_model(config, root):
    tokenizer = make_tokenizer(config["model"], root)
    model = make_model(config["model"], root)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(config["device"])
    return model, tokenizer


def _write_epoch_matrices(run_dir, cells, rows, eval_stages, acquisition=None):
    run_dir = Path(run_dir)
    for metric in METRICS:
        with (run_dir / f"matrix_{metric}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["trained_stage", *eval_stages])
            if acquisition is not None:
                writer.writerow(["S1", acquisition[metric], ""])
            for row_id in rows:
                writer.writerow(
                    [
                        row_id,
                        *[
                            next(
                                (
                                    cell[metric]
                                    for cell in cells
                                    if cell["phase"] == "post"
                                    and cell["trained_stage"] == row_id
                                    and cell["eval_stage"] == stage
                                ),
                                "",
                            )
                            for stage in eval_stages
                        ],
                    ]
                )


def _epoch_summary(run_dir, control, cells, row_ids, eval_stages, acquisition=None):
    _write_epoch_matrices(run_dir, cells, row_ids, eval_stages, acquisition)
    lookup = {(cell["trained_stage"], cell["eval_stage"]): cell for cell in cells if cell["phase"] == "post"}
    epochs = []
    for epoch, row_id in enumerate(row_ids, 1):
        result = {"epoch": epoch, "row_id": row_id, "per_stage": {}}
        for stage in eval_stages:
            if (row_id, stage) in lookup:
                result["per_stage"][stage] = {key: lookup[(row_id, stage)][key] for key in (*METRICS, "tp", "tn", "fp", "fn", "attack_support", "benign_support")}
        if acquisition is not None and "S1" in result["per_stage"]:
            retained = result["per_stage"]["S1"]
            result["s1_retention"] = {metric: retained[metric] for metric in ("accuracy", "macro_f1", "attack_recall")}
            result["s1_bwt"] = {metric: retained[metric] - acquisition[metric] for metric in ("accuracy", "macro_f1", "attack_recall")}
            result["s1_benign_fpr_change"] = retained["benign_fpr"] - acquisition["benign_fpr"]
        epochs.append(result)
    summary = {
        "control": control,
        "evidence_type": "Chen-derived deterministic reconstruction; SQuAD/TriviaQA document-domain pilot only",
        "epoch_results": epochs,
        "s1_acquisition_reference": acquisition,
        "metric_convention": "Higher-is-better retention/BWT are separate from benign-FPR change",
        "replay": "not implemented or run",
    }
    write_json(Path(run_dir) / "summary.json", summary)
    return summary


def _latest_s1_checkpoint(root, seed):
    aggregate = json.loads((root / "results/real_baseline_aggregate.json").read_text(encoding="utf-8"))
    run_id = next(row["run_id"] for row in aggregate["sequential_per_seed"] if row["seed"] == seed)
    path = root / "results" / run_id / "checkpoints" / "S1"
    if not (path / "state.pt").is_file():
        raise ValueError(f"Missing trusted S1 checkpoint for seed {seed}")
    return path


def run_control(config, root, control, config_file="programmatic"):
    """Run one approved control trajectory and evaluate fixed tests at each budget."""
    root = Path(root)
    _approved_config(config, control)
    seed = config["seed"]
    prefixes = {
        "s2_only": "control-s2-only",
        "sequential_s2_epochs": "control-sequential-s2-epochs",
        "reverse_order": "control-reverse-order",
        "joint_epochs": "control-joint-epochs",
    }
    run_id = f"{prefixes[control]}-seed{seed}-" + time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:10]
    run_dir = root / "results" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "predictions").mkdir()
    resolved = copy.deepcopy(config)
    resolved["control"] = {
        "type": control,
        "label": CONTROL_TYPES[control],
        "epoch_budgets": [1, 2, 3] if control != "reverse_order" else [1, 1],
        "replay": False,
    }
    config_hash = digest(json.dumps(resolved, sort_keys=True).encode())
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
    warnings = [
        "Chen-derived deterministic reconstruction; document/domain-distribution control only",
        "Not an exact reproduction, unseen-attack study, attack-family-evolution study, or replay experiment",
        "Raw TriviaQA and Alpaca content is local and git-ignored",
    ]
    if control == "reverse_order":
        warnings.append("Compute-limited reverse-order pilot uses seed 42 only and must be reviewed before expansion")
    if control == "joint_epochs":
        warnings.append("Joint optimization check uses seed 42 only and must be reviewed before seed expansion")
    metadata = {
        "run_id": run_id,
        "date": now(),
        "config_file": str(config_file),
        "config_hash": config_hash,
        "seed": seed,
        "control_type": control,
        "control_label": CONTROL_TYPES[control],
        "experiment_regime": "baseline_sanity_control",
        "model_checkpoint": config["model"]["checkpoint"],
        "model_revision": config["model"]["revision"],
        "max_length": 1280,
        "effective_batch_size": 4,
        "learning_rate": 1e-5,
        "device": "cuda",
        "precision": "float16",
        "data_manifest_hash": None,
        "model_weights_sha256": None,
        "parent_checkpoint": None,
        "warnings": warnings,
        "status": "running",
    }
    started = time.perf_counter()
    summary = None
    completed = 0
    try:
        seed_everything(seed, config["deterministic"], config["threads"])
        env = environment(root)
        metadata.update(environment=env, hardware_device=env["gpu"] or env["processor"] or "CPU", git_commit=env["git_commit"])
        metadata["warnings"].extend(env["compatibility_warnings"])
        data = _load_approved_data(config, root, run_dir, metadata)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        epoch_config = copy.deepcopy(config)
        epoch_config["training"]["epochs"] = 1
        cells = []
        global_step = 0

        if control == "sequential_s2_epochs":
            parent = _latest_s1_checkpoint(root, seed)
            model, state = load_checkpoint(parent)
            if state["data_hash"] != REAL_DATA_SHA256 or state["completed_stages"] != 1:
                raise ValueError("Trusted S1 checkpoint lineage/data mismatch")
            tokenizer = make_tokenizer(config["model"], root)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.to(config["device"])
            optimizer, scheduler = make_optimizer(model, config)
            restore_rng(state["rng"])
            metadata["parent_checkpoint"] = str(parent.relative_to(root))
            acquisition = next(
                cell for cell in state["cells"]
                if cell["phase"] == "post" and cell["trained_stage"] == "S1" and cell["eval_stage"] == "S1"
            )
            acquisition = {key: acquisition[key] for key in (*METRICS, "tp", "tn", "fp", "fn", "attack_support", "benign_support")}
            metadata["model_architecture"] = model.config.to_dict()
            metadata["architecture_hash"] = digest(model.config.to_json_string().encode())
            metadata["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
            write_json(run_dir / "metadata.json", metadata)
            row_ids = []
            for epoch in range(1, 4):
                row_id = f"S2-E{epoch}"
                row_ids.append(row_id)
                epoch_started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                append_jsonl(run_dir / "stage_events.jsonl", {"stage": row_id, "event": "epoch_start", "optimizer_state_entries_before_training": len(optimizer.state), "optimizer_reset_at_s2_boundary": epoch == 1, "global_step_before": global_step})
                global_step = train_stage(model, tokenizer, data.records("S2", "train"), epoch_config, optimizer, scheduler, run_dir, row_id, global_step, allowed_task_ids=("S2",))
                cells.append(evaluate(model, tokenizer, data.records("S1", "test"), config, run_dir, row_id, "S1", "post"))
                cells.append(evaluate(model, tokenizer, data.records("S2", "test"), config, run_dir, row_id, "S2", "post"))
                save_checkpoint(run_dir / "checkpoints" / row_id, model, tokenizer, optimizer, scheduler, {"control": control, "completed_epochs": epoch, "global_step": global_step, "config_hash": config_hash, "data_hash": REAL_DATA_SHA256, "environment": env, "cells": cells, "parent_checkpoint": metadata["parent_checkpoint"]})
                append_jsonl(run_dir / "stage_events.jsonl", {"stage": row_id, "event": "epoch_complete", "elapsed_seconds": time.perf_counter() - epoch_started, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(), "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(), "global_step_after": global_step})
                completed = epoch
                summary = _epoch_summary(run_dir, control, cells, row_ids, ("S1", "S2"), acquisition)

        elif control in {"s2_only", "joint_epochs"}:
            model, tokenizer = _fresh_model(config, root)
            optimizer, scheduler = make_optimizer(model, config)
            metadata["model_architecture"] = model.config.to_dict()
            metadata["architecture_hash"] = digest(model.config.to_json_string().encode())
            metadata["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
            write_json(run_dir / "metadata.json", metadata)
            training_rows = data.records("S2", "train") if control == "s2_only" else data.records("S1", "train") + data.records("S2", "train")
            allowed = ("S2",) if control == "s2_only" else ("S1", "S2")
            eval_stages = ("S2",) if control == "s2_only" else ("S1", "S2")
            row_ids = []
            prefix = "S2ONLY" if control == "s2_only" else "JOINT"
            for epoch in range(1, 4):
                row_id = f"{prefix}-E{epoch}"
                row_ids.append(row_id)
                epoch_started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                append_jsonl(run_dir / "stage_events.jsonl", {"stage": row_id, "event": "epoch_start", "optimizer_state_entries_before_training": len(optimizer.state), "global_step_before": global_step})
                global_step = train_stage(model, tokenizer, training_rows, epoch_config, optimizer, scheduler, run_dir, row_id, global_step, allowed_task_ids=allowed)
                for stage in eval_stages:
                    cells.append(evaluate(model, tokenizer, data.records(stage, "test"), config, run_dir, row_id, stage, "post"))
                save_checkpoint(run_dir / "checkpoints" / row_id, model, tokenizer, optimizer, scheduler, {"control": control, "completed_epochs": epoch, "global_step": global_step, "config_hash": config_hash, "data_hash": REAL_DATA_SHA256, "environment": env, "cells": cells})
                append_jsonl(run_dir / "stage_events.jsonl", {"stage": row_id, "event": "epoch_complete", "elapsed_seconds": time.perf_counter() - epoch_started, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(), "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(), "global_step_after": global_step})
                completed = epoch
                summary = _epoch_summary(run_dir, control, cells, row_ids, eval_stages)

        else:
            model, tokenizer = _fresh_model(config, root)
            metadata["model_architecture"] = model.config.to_dict()
            metadata["architecture_hash"] = digest(model.config.to_json_string().encode())
            metadata["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
            write_json(run_dir / "metadata.json", metadata)
            for index, stage in enumerate(("S2", "S1")):
                optimizer, scheduler = make_optimizer(model, config)
                stage_started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                append_jsonl(run_dir / "stage_events.jsonl", {"stage": stage, "event": "stage_start", "optimizer_state_entries_before_training": len(optimizer.state), "optimizer_transition": "reset", "global_step_before": global_step})
                global_step = train_stage(model, tokenizer, data.records(stage, "train"), epoch_config, optimizer, scheduler, run_dir, stage, global_step)
                for seen in ("S2", "S1")[:index + 1]:
                    cells.append(evaluate(model, tokenizer, data.records(seen, "test"), config, run_dir, stage, seen, "post"))
                save_checkpoint(run_dir / "checkpoints" / stage, model, tokenizer, optimizer, scheduler, {"control": control, "completed_stages": index + 1, "global_step": global_step, "config_hash": config_hash, "data_hash": REAL_DATA_SHA256, "environment": env, "cells": cells})
                append_jsonl(run_dir / "stage_events.jsonl", {"stage": stage, "event": "stage_complete", "elapsed_seconds": time.perf_counter() - stage_started, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(), "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(), "global_step_after": global_step})
                completed = index + 1
                summary = write_summary(run_dir, cells, ["S2", "S1"], "Chen-derived deterministic reconstruction; reverse-order document/domain control")

        metadata["status"] = "complete"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["completed_units"] = completed
        metadata["finished_at"] = now()
        metadata["runtime_seconds"] = time.perf_counter() - started
        if torch.cuda.is_available():
            complete_events = []
            if (run_dir / "stage_events.jsonl").is_file():
                complete_events = [event for event in read_jsonl(run_dir / "stage_events.jsonl") if "peak_gpu_memory_bytes" in event]
            metadata["peak_gpu_memory_bytes"] = max((event["peak_gpu_memory_bytes"] for event in complete_events), default=torch.cuda.max_memory_allocated())
            metadata["peak_gpu_reserved_bytes"] = max((event["peak_gpu_reserved_bytes"] for event in complete_events), default=torch.cuda.max_memory_reserved())
        metadata["summary_metrics"] = summary
        write_json(run_dir / "metadata.json", metadata)
        write_json(root / "history" / "runs" / f"{run_id}.json", metadata)
        history(root, "experiments", metadata)
    return run_dir
