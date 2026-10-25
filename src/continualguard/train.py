"""Current-stage-only baseline with auditable execution and boundary resume."""
import json
import hashlib
import shutil
import time
import uuid
from pathlib import Path
import torch
import yaml
from .artifacts import append_jsonl, digest, history, now, read_jsonl, write_json
from .data import StageData, length_audit, validate_record
from .evaluate import evaluate, write_joint_summary, write_summary
from .model import batch_inputs, make_model, make_tokenizer
from .reproducibility import environment, load_checkpoint, restore_rng, save_checkpoint, seed_everything


REAL_DATA_SHA256 = "4d5a3e4f98a9df2211d8c2658db68c6ce7bf6d5284ec482a746c10ad9b394fb0"
REAL_MODEL_SHA256 = "691d48a2800b926a19e3051def466fc2cca4f59a15e42ce4a0cf7f1b380b5e33"
REAL_REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


class ApprovedRealData:
    """Lightweight loader for the immutable, separately audited real-data artifact."""
    def __init__(self, rows, stage_order):
        for row in rows:
            validate_record(row)
        self.rows, self.stage_order = rows, list(stage_order)
        counts = {(stage, split): 0 for stage in stage_order for split in ("train", "val", "test")}
        labels = {(stage, split): set() for stage in stage_order for split in ("train", "val", "test")}
        for row in rows:
            key = (row["task_id"], row["split"])
            if key not in counts:
                raise ValueError(f"Unexpected real-data stage/split: {key}")
            counts[key] += 1
            labels[key].add(row["label"])
        if any(value == 0 for value in counts.values()) or any(value != {0, 1} for value in labels.values()):
            raise ValueError("Every approved real-data stage/split must be nonempty and contain both labels")
        self.audit = {"errors": [], "warnings": ["Loaded by approved artifact SHA-256; full overlap and length audits are preserved under data/audits/real_data"], "counts": {f"{key[0]}/{key[1]}": value for key, value in counts.items()}}

    def records(self, stage, split):
        return [row for row in self.rows if row["task_id"] == stage and row["split"] == split]


def make_optimizer(model, config):
    t = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], betas=tuple(t["betas"]), eps=t["epsilon"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return optimizer, scheduler


def train_stage(model, tokenizer, records, config, optimizer, scheduler, run_dir, stage, global_step, allowed_task_ids=None):
    allowed_task_ids = set(allowed_task_ids or (stage,))
    if any(r["split"] != "train" or r["task_id"] not in allowed_task_ids for r in records):
        raise ValueError("Trainer received historical/future/non-training records")
    model.train()
    batch_size = config["training"]["batch_size"]
    accumulation = config["training"]["gradient_accumulation_steps"]
    use_amp = config["precision"] == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=1024.0)
    for epoch in range(config["training"]["epochs"]):
        # Explicitly seeded global torch RNG is checkpointed at boundaries.
        order = torch.randperm(len(records)).tolist()
        optimizer.zero_grad(set_to_none=True)
        accumulated_ids, accumulated_loss = [], 0.0
        for offset in range(0, len(order), batch_size):
            batch = [records[i] for i in order[offset:offset+batch_size]]
            inputs = batch_inputs(batch, tokenizer, config["model"]["max_length"], config["device"])
            labels = torch.tensor([r["label"] for r in batch], dtype=torch.long, device=config["device"])
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(**inputs, labels=labels).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            scaler.scale(loss / accumulation).backward()
            accumulated_loss += loss.item() * len(batch)
            accumulated_ids.extend(r["example_id"] for r in batch)
            microbatch_index = offset // batch_size
            final_microbatch = offset + batch_size >= len(order)
            if (microbatch_index + 1) % accumulation == 0 or final_microbatch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1
                append_jsonl(Path(run_dir) / "train_log.jsonl", {"stage": stage, "epoch": epoch, "global_step": global_step, "loss": accumulated_loss / len(accumulated_ids), "lr": optimizer.param_groups[0]["lr"], "example_ids": accumulated_ids, "examples_in_step": len(accumulated_ids), "microbatch_size": batch_size, "gradient_accumulation_steps": accumulation})
                optimizer.zero_grad(set_to_none=True)
                accumulated_ids, accumulated_loss = [], 0.0
    return global_step


def run(config, root, config_file="programmatic", resume=None, stop_after=None):
    root = Path(root)
    is_smoke = config["run_type"] == "software smoke test" and config["model"]["kind"] == "tiny_deberta"
    is_real = (
        config["run_type"] == "pilot experiment"
        and config["model"]["kind"] == "pretrained_deberta"
        and config["model"]["checkpoint"] == "microsoft/deberta-v3-base"
        and config["model"]["revision"] == REAL_REVISION
        and config["model"]["max_length"] == 1280
        and config["stage_order"] == ["S1", "S2"]
        and config["seed"] in {17, 42, 2026}
        and config["training"]["optimizer_transition"] == "reset"
        and config["training"]["epochs"] == 1
        and config["training"]["batch_size"] == 4
        and config["training"]["gradient_accumulation_steps"] == 1
        and config["training"]["gradient_checkpointing"] is True
        and config["device"] == "cuda"
        and config["precision"] == "float16"
    )
    if not (is_smoke or is_real):
        raise ValueError("Run does not match the approved smoke or real sequential-baseline contract")
    prefix = "smoke" if is_smoke else f"real-sequential-seed{config['seed']}"
    run_id = prefix + "-" + time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:10]
    run_dir = root / "results" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "predictions").mkdir()
    timestamp = now()
    config_hash = digest(json.dumps(config, sort_keys=True).encode())
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    warnings = ["Synthetic software validation only; not scientific evidence"] if is_smoke else ["Chen-derived deterministic reconstruction; document/domain-distribution pilot only; not exact reproduction, unseen-attack evidence, or attack-family evolution", "Raw TriviaQA and Alpaca content is local and git-ignored", "Replay is excluded pending human review of this sequential baseline"]
    metadata = {"run_id": run_id, "date": timestamp, "config_file": str(config_file), "config_hash": config_hash, "seed": config["seed"], "stage_order": config["stage_order"], "device": config["device"], "precision": config["precision"], "run_type": config["run_type"], "experiment_regime": "sequential", "result_artifact_path": f"results/{run_id}", "model_checkpoint": config["model"]["checkpoint"], "model_revision": config["model"]["revision"], "optimizer_transition": config["training"]["optimizer_transition"], "effective_batch_size": config["training"]["batch_size"] * config["training"]["gradient_accumulation_steps"], "data_manifest_hash": None, "model_weights_sha256": None, "exclusions": [], "warnings": warnings, "parent_checkpoint": str(resume) if resume else None, "status": "running"}
    cells, completed, step, summary = [], 0, 0, None
    run_started = time.perf_counter()
    try:
        seed_everything(config["seed"], config["deterministic"], config["threads"])
        env = environment(root)
        metadata.update(environment=env, git_commit=env["git_commit"], hardware_device=env["gpu"] or env["processor"] or "CPU")
        metadata["warnings"].extend(env["compatibility_warnings"])
        source = (root / config["data_path"]).resolve()
        if not source.is_relative_to(root.resolve()):
            raise ValueError("Dataset path must stay in project root")
        raw = source.read_bytes()
        metadata["data_manifest_hash"] = digest(raw)
        rows = read_jsonl(source)
        if is_smoke:
            if any(r.get("source", {}).get("dataset") != "synthetic_software_fixture" for r in rows):
                raise ValueError("Smoke run cannot consume research data")
            data = StageData(rows, config["stage_order"])
        else:
            if digest(raw) != REAL_DATA_SHA256:
                raise ValueError("Real dataset differs from the human-approved audited artifact")
            construction = json.loads((root / "data/audits/real_data/construction_manifest.json").read_text(encoding="utf-8"))
            overlap = json.loads((root / "data/audits/real_data/overlap_audit.json").read_text(encoding="utf-8"))
            if construction["processed_sha256"] != REAL_DATA_SHA256 or construction["blocking_overlap_findings"] or overlap.get("errors"):
                raise ValueError("Approved real-data audit gate no longer passes")
            data = ApprovedRealData(rows, config["stage_order"])
        metadata["warnings"].extend(data.audit["warnings"])
        write_json(run_dir / "data_audit.json", data.audit)
        write_json(run_dir / "data_manifest.json", {"source_path": config["data_path"], "sha256": digest(raw), "records": [{k: r[k] for k in ("example_id", "text_sha256", "task_id", "split", "source")} for r in rows]})
        tokenizer = make_tokenizer(config["model"], root)
        if is_smoke:
            audit = length_audit(rows, tokenizer, (config["model"]["max_length"],))
        else:
            approved_length = json.loads((root / "data/audits/real_data/length_audit.json").read_text(encoding="utf-8"))
            exclusions = [item for item in approved_length["exclusions"] if item["max_length"] == 1280]
            coverage = [item for item in approved_length["coverage"] if item["max_length"] == 1280]
            audit = {"approved_source": "data/audits/real_data/length_audit.json", "max_length": 1280, "coverage": coverage, "exclusions": exclusions}
        write_json(run_dir / "length_audit.json", audit)
        write_json(run_dir / "exclusions.json", audit["exclusions"])
        metadata["exclusions"] = audit["exclusions"]
        if audit["exclusions"]:
            raise ValueError("Coverage loss: stop and review exclusions; no silent filtering/truncation")
        if is_real:
            weight_path = (root / config["model"]["local_artifact"] / "pytorch_model.bin").resolve()
            metadata["model_weights_sha256"] = file_sha256(weight_path)
            if metadata["model_weights_sha256"] != REAL_MODEL_SHA256:
                raise ValueError("Pinned model weights hash mismatch")
            if not torch.cuda.is_available():
                raise ValueError("Approved real baseline requires CUDA on this host")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model = make_model(config["model"], root)
        if config["training"]["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.to(config["device"])
        optimizer, scheduler = make_optimizer(model, config)
        if resume:
            model, state = load_checkpoint(resume)
            if state["config_hash"] != config_hash or state["data_hash"] != digest(raw):
                raise ValueError("Resume configuration/data hash mismatch")
            if state["environment"] != env:
                raise ValueError("Resume environment mismatch")
            model.to(config["device"])
            optimizer, scheduler = make_optimizer(model, config)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            completed, step, cells = state["completed_stages"], state["global_step"], state["cells"]
            parent = Path(resume).parent.parent
            for path in (parent / "predictions").glob("*.jsonl"):
                # Only copy predictions belonging to the checkpoint's completed prefix.
                allowed = {f"{c['phase']}_{c['trained_stage']}_{c['eval_stage']}.jsonl" for c in cells}
                if path.name in allowed:
                    shutil.copyfile(path, run_dir / "predictions" / path.name)
            for cell in cells:
                append_jsonl(run_dir / "metrics.jsonl", cell)
            for entry in read_jsonl(parent / "train_log.jsonl"):
                if entry["global_step"] <= step:
                    append_jsonl(run_dir / "train_log.jsonl", entry)
            restore_rng(state["rng"])
        metadata["model_architecture"] = model.config.to_dict()
        metadata["architecture_hash"] = digest(model.config.to_json_string().encode())
        metadata["parameter_count"] = sum(p.numel() for p in model.parameters())
        write_json(run_dir / "metadata.json", metadata)
        for index, stage in enumerate(config["stage_order"]):
            if index < completed:
                continue
            stage_started = time.perf_counter()
            if is_real:
                torch.cuda.reset_peak_memory_stats()
            if config["training"]["pre_adaptation"]:
                cells.append(evaluate(model, tokenizer, data.records(stage, "test"), config, run_dir, stage, stage, "pre"))
            if config["training"]["optimizer_transition"] == "reset":
                optimizer, scheduler = make_optimizer(model, config)
            append_jsonl(run_dir / "stage_events.jsonl", {"stage": stage, "event": "stage_start", "optimizer_transition": config["training"]["optimizer_transition"], "optimizer_state_entries_before_training": len(optimizer.state), "global_step_before": step})
            step = train_stage(model, tokenizer, data.records(stage, "train"), config, optimizer, scheduler, run_dir, stage, step)
            # Evaluations are read-only; the checkpoint saved afterward is the exact stage-end model.
            payload = {"completed_stages": index+1, "global_step": step, "config_hash": config_hash, "data_hash": digest(raw), "environment": env, "cells": cells}
            checkpoint = run_dir / "checkpoints" / stage
            for seen in config["stage_order"][:index+1]:
                cells.append(evaluate(model, tokenizer, data.records(seen, "test"), config, run_dir, stage, seen, "post"))
            cells.append(evaluate(model, tokenizer, data.records(stage, "val"), config, run_dir, stage, stage, "val"))
            save_checkpoint(checkpoint, model, tokenizer, optimizer, scheduler, {**payload, "cells": cells})
            append_jsonl(run_dir / "stage_events.jsonl", {"stage": stage, "event": "stage_complete", "elapsed_seconds": time.perf_counter() - stage_started, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if is_real else None, "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if is_real else None, "global_step_after": step})
            evidence = "Chen-derived deterministic reconstruction; SQuAD-to-TriviaQA document/domain-distribution pilot" if is_real else "software smoke test; no research evidence"
            summary = write_summary(run_dir, cells, config["stage_order"], evidence)
            completed = index + 1
            if stop_after is not None and completed >= stop_after:
                break
        metadata["status"] = "complete" if completed == len(config["stage_order"]) else "paused_at_stage_boundary"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["completed_stages"] = completed
        metadata["finished_at"] = now()
        metadata["runtime_seconds"] = time.perf_counter() - run_started
        if is_real and torch.cuda.is_available():
            metadata["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
            metadata["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
        metadata["summary_metrics"] = summary
        write_json(run_dir / "metadata.json", metadata)
        write_json(root / "history" / "runs" / f"{run_id}.json", metadata)
        history(root, "experiments", metadata)
    return run_dir


def run_joint(config, root, config_file="programmatic"):
    """Train the feasible pooled S1+S2 offline reference for one approved seed."""
    root = Path(root)
    if not (
        config["run_type"] == "pilot experiment"
        and config["model"]["kind"] == "pretrained_deberta"
        and config["model"]["checkpoint"] == "microsoft/deberta-v3-base"
        and config["model"]["revision"] == REAL_REVISION
        and config["model"]["max_length"] == 1280
        and config["stage_order"] == ["S1", "S2"]
        and config["seed"] in {17, 42, 2026}
        and config["device"] == "cuda"
        and config["precision"] == "float16"
    ):
        raise ValueError("Run does not match the approved real joint-reference contract")
    run_id = f"real-joint-seed{config['seed']}-" + time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:10]
    run_dir = root / "results" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "predictions").mkdir()
    config_hash = digest(json.dumps(config, sort_keys=True).encode())
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    metadata = {"run_id": run_id, "date": now(), "config_file": str(config_file), "config_hash": config_hash, "seed": config["seed"], "stage_order": config["stage_order"], "device": config["device"], "precision": config["precision"], "run_type": config["run_type"], "experiment_regime": "joint_offline_reference", "result_artifact_path": f"results/{run_id}", "model_checkpoint": config["model"]["checkpoint"], "model_revision": config["model"]["revision"], "optimizer_transition": "fresh_joint_optimizer", "effective_batch_size": config["training"]["batch_size"] * config["training"]["gradient_accumulation_steps"], "data_manifest_hash": None, "model_weights_sha256": None, "exclusions": [], "warnings": ["Chen-derived deterministic reconstruction; document/domain-distribution reference only", "Joint reference has no continual-learning retention, BWT, or FPR-increase interpretation", "Raw TriviaQA and Alpaca content is local and git-ignored", "No replay is implemented or run"], "status": "running"}
    summary, completed = None, 0
    started = time.perf_counter()
    try:
        seed_everything(config["seed"], config["deterministic"], config["threads"])
        env = environment(root)
        metadata.update(environment=env, git_commit=env["git_commit"], hardware_device=env["gpu"] or env["processor"] or "CPU")
        metadata["warnings"].extend(env["compatibility_warnings"])
        source = (root / config["data_path"]).resolve()
        raw = source.read_bytes()
        metadata["data_manifest_hash"] = digest(raw)
        if metadata["data_manifest_hash"] != REAL_DATA_SHA256:
            raise ValueError("Real dataset differs from the human-approved audited artifact")
        construction = json.loads((root / "data/audits/real_data/construction_manifest.json").read_text(encoding="utf-8"))
        if construction["processed_sha256"] != REAL_DATA_SHA256 or construction["blocking_overlap_findings"]:
            raise ValueError("Approved real-data audit gate no longer passes")
        rows = read_jsonl(source)
        data = ApprovedRealData(rows, config["stage_order"])
        write_json(run_dir / "data_audit.json", data.audit)
        write_json(run_dir / "data_manifest.json", {"source_path": config["data_path"], "sha256": metadata["data_manifest_hash"], "records": [{k: row[k] for k in ("example_id", "text_sha256", "task_id", "split", "source")} for row in rows]})
        approved_length = json.loads((root / "data/audits/real_data/length_audit.json").read_text(encoding="utf-8"))
        audit = {"approved_source": "data/audits/real_data/length_audit.json", "max_length": 1280, "coverage": [item for item in approved_length["coverage"] if item["max_length"] == 1280], "exclusions": [item for item in approved_length["exclusions"] if item["max_length"] == 1280]}
        write_json(run_dir / "length_audit.json", audit)
        write_json(run_dir / "exclusions.json", audit["exclusions"])
        if audit["exclusions"]:
            raise ValueError("Approved 1280-token audit unexpectedly has exclusions")
        weight_path = (root / config["model"]["local_artifact"] / "pytorch_model.bin").resolve()
        metadata["model_weights_sha256"] = file_sha256(weight_path)
        if metadata["model_weights_sha256"] != REAL_MODEL_SHA256 or not torch.cuda.is_available():
            raise ValueError("Pinned model or CUDA gate failed")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        tokenizer = make_tokenizer(config["model"], root)
        model = make_model(config["model"], root)
        if config["training"]["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.to(config["device"])
        metadata["model_architecture"] = model.config.to_dict()
        metadata["architecture_hash"] = digest(model.config.to_json_string().encode())
        metadata["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        write_json(run_dir / "metadata.json", metadata)
        cells = [evaluate(model, tokenizer, data.records(stage, "test"), config, run_dir, "JOINT", stage, "pre") for stage in config["stage_order"]]
        optimizer, scheduler = make_optimizer(model, config)
        append_jsonl(run_dir / "stage_events.jsonl", {"stage": "JOINT", "event": "stage_start", "optimizer_transition": "fresh_joint_optimizer", "optimizer_state_entries_before_training": len(optimizer.state), "global_step_before": 0})
        training_rows = data.records("S1", "train") + data.records("S2", "train")
        training_started = time.perf_counter()
        step = train_stage(model, tokenizer, training_rows, config, optimizer, scheduler, run_dir, "JOINT", 0, allowed_task_ids=config["stage_order"])
        for stage in config["stage_order"]:
            cells.append(evaluate(model, tokenizer, data.records(stage, "test"), config, run_dir, "JOINT", stage, "post"))
            cells.append(evaluate(model, tokenizer, data.records(stage, "val"), config, run_dir, "JOINT", stage, "val"))
        save_checkpoint(run_dir / "checkpoints" / "JOINT", model, tokenizer, optimizer, scheduler, {"completed_stages": 1, "global_step": step, "config_hash": config_hash, "data_hash": metadata["data_manifest_hash"], "environment": env, "cells": cells})
        append_jsonl(run_dir / "stage_events.jsonl", {"stage": "JOINT", "event": "stage_complete", "elapsed_seconds": time.perf_counter() - training_started, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(), "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(), "global_step_after": step})
        summary = write_joint_summary(run_dir, cells, config["stage_order"])
        completed = 1
        metadata["status"] = "complete"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["completed_stages"] = completed
        metadata["finished_at"] = now()
        metadata["runtime_seconds"] = time.perf_counter() - started
        if torch.cuda.is_available():
            metadata["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
            metadata["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
        metadata["summary_metrics"] = summary
        write_json(run_dir / "metadata.json", metadata)
        write_json(root / "history" / "runs" / f"{run_id}.json", metadata)
        history(root, "experiments", metadata)
    return run_dir
