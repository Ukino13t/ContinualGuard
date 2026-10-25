"""Final Candidate B Generic Loss-aware Replay control."""
from __future__ import annotations

import copy
import gc
import json
import os
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from .artifacts import now
from .model import batch_inputs, make_model, make_tokenizer
from .replay_candidate_b import (
    MEMORY_CAPACITY, REPLAY_BATCH_SIZE, coverage_key, entry_from_row,
    security_risk, stable_hash, train_stage_with_replay,
)
from .reproducibility import environment, isolated_evaluation, restore_rng, seed_everything
from .stronger_pilot_no_replay import (
    ARMS, MATERIALIZATION_MANIFEST_SHA256, MATERIALIZATION_VALIDATION_SHA256,
    SEEDS, STAGES, CandidateBData, atomic_json, evaluate_external,
    evaluate_stage_test, load_boundary_checkpoint, save_boundary_checkpoint,
    sha256_file, state_fingerprint, validation_loss, write_jsonl,
)
from .train import REAL_MODEL_SHA256, REAL_REVISION, make_optimizer

METHOD = "loss_aware"
METHOD_NAME = "Generic Loss-aware Replay"
CONFIG_RELATIVE = "configs/loss_aware_control.yaml"
RESULT_RELATIVE = "results/replay_comparison_candidate_b/loss_aware"
TIE_NAMESPACE = "security-aware"


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def loss_entry(row, *, generic_loss: float, p_attack: float, seed: int) -> dict:
    entry = entry_from_row(
        row, p_attack=p_attack, risk=security_risk(row["label"], p_attack), selection_seed=seed,
    )
    # The tie namespace is deliberately identical to Security-aware Replay.
    entry["tie_break_sha256"] = stable_hash(seed, TIE_NAMESPACE, row["example_id"])
    entry["pre_update_generic_loss"] = float(generic_loss)
    entry["selection_score_name"] = "pre_update_true_label_cross_entropy"
    return entry


def generic_difficulty_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return symmetric, per-example true-label CE in float32."""
    return F.cross_entropy(logits.float(), labels, reduction="none")


def score_candidates(model, tokenizer, records: list[dict], config: dict, seed: int) -> list[dict]:
    if any(row.get("split") != "train" or row.get("task_id") not in STAGES for row in records):
        raise ValueError("Loss-aware scoring accepts Candidate B TRAIN records only")
    output = []
    with isolated_evaluation(model):
        for offset in range(0, len(records), config["training"]["batch_size"]):
            batch = records[offset:offset + config["training"]["batch_size"]]
            inputs = batch_inputs(batch, tokenizer, 1280, "cuda")
            labels = torch.tensor([row["label"] for row in batch], dtype=torch.long, device="cuda")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits = model(**inputs).logits
            logits32 = logits.float()
            losses = generic_difficulty_from_logits(logits, labels).detach().cpu().tolist()
            probabilities = torch.softmax(logits32, dim=-1)[:, 1].detach().cpu().tolist()
            for row, loss, probability in zip(batch, losses, probabilities):
                output.append(loss_entry(row, generic_loss=loss, p_attack=probability, seed=seed))
    if len(output) != len(records):
        raise RuntimeError("Pre-update score count mismatch")
    return output


class LossAwareBuffer:
    def __init__(self, seed: int, capacity: int = MEMORY_CAPACITY):
        if capacity != 200:
            raise ValueError("Loss-aware capacity must be exactly 200")
        self.method, self.seed, self.capacity = METHOD, seed, capacity
        self.sample_rng = random.Random(seed * 2027 + 37)  # identical Security-aware sampling RNG
        self.entries: list[dict] = []
        self.seen_count = 0

    def state_dict(self):
        return {
            "method": self.method, "seed": self.seed, "capacity": self.capacity,
            "entries": copy.deepcopy(self.entries), "seen_count": self.seen_count,
            "sample_rng_state": self.sample_rng.getstate(),
        }

    @classmethod
    def from_state_dict(cls, state):
        if state.get("method") != METHOD:
            raise ValueError("Loss-aware buffer method mismatch")
        obj = cls(state["seed"], state["capacity"])
        obj.entries = copy.deepcopy(state["entries"])
        obj.seen_count = state["seen_count"]
        obj.sample_rng.setstate(state["sample_rng_state"])
        obj.validate()
        return obj

    def validate(self, current_stage_index: int | None = None):
        if len(self.entries) > self.capacity:
            raise ValueError("Loss-aware replay capacity exceeded")
        if len({entry["example_id"] for entry in self.entries}) != len(self.entries):
            raise ValueError("Duplicate identity in loss-aware memory")
        for entry in self.entries:
            if entry.get("task_id") not in STAGES or "pre_update_generic_loss" not in entry:
                raise ValueError("Invalid loss-aware memory entry")
            if not 0 <= entry.get("pre_update_p_attack", -1) <= 1:
                raise ValueError("Invalid stored probability")
            if current_stage_index is not None and entry["stage_index"] >= current_stage_index:
                raise ValueError("Current/future-stage loss-aware replay entry")

    def sample(self, count: int, current_stage_index: int):
        self.validate(current_stage_index)
        size = min(count, len(self.entries))
        return self.sample_rng.sample(self.entries, size) if size else []

    def update(self, candidates: list[dict], contributed_stages: tuple[str, ...]):
        if not contributed_stages or tuple(STAGES[:len(contributed_stages)]) != tuple(contributed_stages):
            raise ValueError("Contributed stages must be a contiguous prefix")
        combined = {entry["example_id"]: copy.deepcopy(entry) for entry in [*self.entries, *candidates]}
        selected, quota_record = [], {}
        for label in (0, 1):
            base, remainder = divmod(self.capacity // 2, len(contributed_stages))
            for index, stage in enumerate(contributed_stages):
                quota = base + (1 if index < remainder else 0)
                pool = [entry for entry in combined.values() if entry["label"] == label and entry["task_id"] == stage]
                pool.sort(key=lambda entry: (-entry["pre_update_generic_loss"], entry["tie_break_sha256"]))
                diverse, deferred, groups = [], [], set()
                for entry in pool:
                    if entry["coverage_key"] not in groups:
                        diverse.append(entry)
                        groups.add(entry["coverage_key"])
                    else:
                        deferred.append(entry)
                chosen = (diverse + deferred)[:quota]
                selected.extend(chosen)
                quota_record[f"label{label}/{stage}"] = {
                    "target": quota, "available": len(pool), "selected": len(chosen),
                }
        self.entries = selected
        self.seen_count += len(candidates)
        self.validate()
        if len(self.entries) != min(self.capacity, len(combined)):
            raise ValueError("Loss-aware structured quota did not fill feasible capacity")
        return quota_record


def memory_composition(buffer: LossAwareBuffer, quota=None):
    entries = buffer.entries
    def counts(key):
        return dict(sorted(Counter(str(entry.get(key)) for entry in entries).items()))
    losses = [entry["pre_update_generic_loss"] for entry in entries]
    risks = [entry["pre_update_security_risk"] for entry in entries]
    return {
        "method": METHOD, "capacity_examples": 200, "buffer_size": len(entries),
        "seen_training_examples": buffer.seen_count, "label_counts": counts("label"),
        "stage_distribution": counts("task_id"), "attack_family_distribution": counts("attack_family"),
        "benign_group_distribution": counts("benign_group"),
        "injection_position_distribution": counts("injection_position"),
        "source_distribution": counts("source_dataset"),
        "coverage_key_distribution": counts("coverage_key"),
        "unique_coverage_keys": len({entry["coverage_key"] for entry in entries}),
        "raw_text_stored": False, "historical_logit_entries": 0, "auxiliary_logit_bytes_float32": 0,
        "generic_loss": {
            "mean": statistics.mean(losses), "median": statistics.median(losses),
            "p90": quantile(losses, .90), "p95": quantile(losses, .95), "p99": quantile(losses, .99),
        } if losses else None,
        "security_risk_diagnostic": {
            "mean": statistics.mean(risks), "median": statistics.median(risks),
            "p90": quantile(risks, .90), "p95": quantile(risks, .95), "p99": quantile(risks, .99),
        } if risks else None,
        "stage_quota_fulfillment": quota or {},
    }


def approved_config(root: Path, seed: int):
    config = yaml.safe_load((Path(root) / CONFIG_RELATIVE).read_text(encoding="utf-8"))
    config["seed"] = seed
    validate_config(config)
    return config


def validate_config(config):
    t, r = config["training"], config["replay"]
    valid = (
        config["run_type"] == "reportable research result"
        and config["experiment"] == "Candidate B final Generic Loss-aware Replay control"
        and config["data_path"] == "data/processed/stronger_pilot_candidate_b"
        and tuple(config["stage_order"]) == STAGES and config["seed"] in SEEDS and config["method"] == METHOD
        and config["model"]["checkpoint"] == "microsoft/deberta-v3-base"
        and config["model"]["revision"] == REAL_REVISION and config["model"]["max_length"] == 1280
        and t == {
            "epochs": 1, "batch_size": 4, "gradient_accumulation_steps": 1,
            "gradient_checkpointing": True, "learning_rate": 1e-5, "betas": [0.9, 0.999],
            "epsilon": 1e-8, "weight_decay": 0.0, "max_grad_norm": 1.0,
            "optimizer_transition": "reset", "scheduler": "constant", "pre_adaptation": False,
        }
        and r["memory_capacity_examples"] == 200 and r["replay_batch_size"] == 4
        and r["classification_loss"] == "(current_ce + replay_ce) / 2"
        and r["selection_score"] == "pre_update_true_label_cross_entropy"
        and r["score_timing"] == "pre_stage_update_on_train_only"
        and r["label_targets"] == {"benign": 100, "malicious": 100}
        and r["stage_quota"] == "deterministic_even_with_earlier_stage_remainder"
        and r["coverage"] == "one_per_coverage_key_before_fill"
        and r["tie_break"] == "sha256(seed|security-aware|example_id)"
        and config["device"] == "cuda" and config["precision"] == "float16"
        and config["deterministic"] is True and config["threads"] == 4
    )
    if not valid:
        raise ValueError("Execution differs from approved Generic Loss-aware protocol")


def record_lookup(data: CandidateBData):
    lookup = {row["example_id"]: row for stage in STAGES for row in data.records(stage, "train")}
    if len(lookup) != 1600:
        raise ValueError("Candidate B training identity contract mismatch")
    return lookup


def initialization_reference(root, arm, seed):
    return json.loads((root / "results/stronger_pilot_no_replay" / arm / f"seed{seed}/initialization.json").read_text(encoding="utf-8"))


def run_preflight(root: Path):
    root = Path(root).resolve()
    if root != Path(r"E:\ContinualGuard").resolve():
        raise ValueError(f"Wrong active project root: {root}")
    output = root / RESULT_RELATIVE / "preflight"
    result_path = output / "preflight.json"
    config_hash = sha256_file(root / CONFIG_RELATIVE)
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") == "PASS" and result.get("config_sha256") == config_hash:
            return output
        raise ValueError("Existing Loss-aware preflight is not a reusable PASS")
    output.mkdir(parents=True, exist_ok=False)
    result = {"started_at": now(), "status": "RUNNING", "config_sha256": config_hash}
    started = time.perf_counter()
    model = tokenizer = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        config = approved_config(root, 42)
        data = CandidateBData(root, "doc")
        lookup = record_lookup(data)
        seed_everything(42, True, 4)
        tokenizer = make_tokenizer(config["model"], root)
        model = make_model(config["model"], root)
        initial = state_fingerprint(model)
        if initial != initialization_reference(root, "doc", 42):
            raise ValueError("Preflight initialization differs from M0")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.to("cuda")
        history = data.records("AF2", "train")[:8]
        current = data.records("AF3", "train")[:4]
        history_scores = score_candidates(model, tokenizer, history, config, 42)
        current_scores = score_candidates(model, tokenizer, current, config, 42)
        buffer = LossAwareBuffer(42)
        buffer.update(history_scores, ("AF2",))
        optimizer, scheduler = make_optimizer(model, config)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        step_started = time.perf_counter()
        global_step, _captured, training = train_stage_with_replay(
            model, tokenizer, current, config, optimizer, scheduler, buffer, lookup,
            output, "AF3", 2, 0,
        )
        quota = buffer.update(current_scores, ("AF2", "AF3"))
        checkpoint_hashes = save_boundary_checkpoint(output / "checkpoint", model, tokenizer, {
            "purpose": "Loss-aware software preflight only", "method": METHOD, "seed": 42,
            "completed_stages": 2, "global_step": global_step, "initialization": initial,
            "buffer_state": buffer.state_dict(),
        })
        del model, tokenizer; model = tokenizer = None; gc.collect(); torch.cuda.empty_cache()
        loaded_model, loaded_tokenizer, state = load_boundary_checkpoint(output / "checkpoint")
        loaded_buffer = LossAwareBuffer.from_state_dict(state["buffer_state"])
        loaded_model.to("cuda")
        test = data.records("AF3", "test")
        balanced = [*[x for x in test if x["label"] == 0][:2], *[x for x in test if x["label"] == 1][:2]]
        metrics = evaluate_stage_test(loaded_model, loaded_tokenizer, balanced, config, "preflight:loss-aware", output / "predictions.jsonl")
        atomic_json(output / "memory_composition.json", memory_composition(loaded_buffer, quota))
        result.update({
            "status": "PASS", "finished_at": now(), "gpu": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda, "loss": training["training_loss"],
            "replay_examples": training["replay_examples_exposed"], "buffer_size": len(loaded_buffer.entries),
            "checkpoint_reload": True, "buffer_roundtrip": len(loaded_buffer.entries) == len(buffer.entries),
            "resume_state_roundtrip": loaded_buffer.state_dict() == buffer.state_dict(),
            "evaluation_records": metrics["paired"]["attack_support"] + metrics["paired"]["benign_support"],
            "step_seconds": time.perf_counter() - step_started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "checkpoint_hashes": checkpoint_hashes, "initialization": initial, "cuda_oom": False,
            "pre_update_loss_scoring": True, "true_label_used": True,
        })
        del loaded_model, loaded_tokenizer
    except Exception as exc:
        result.update(status="FAIL", finished_at=now(), error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["runtime_seconds"] = time.perf_counter() - started
        atomic_json(result_path, result)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return output


def stage_directories(run_dir):
    completed = []
    for index in range(1, 5):
        path = run_dir / f"stage{index}"
        if path.is_dir() and (path / "stage_metadata.json").exists() and (path / "checkpoint/state.pt").exists():
            completed.append(path)
        elif path.exists():
            raise ValueError(f"Incomplete published stage directory: {path}")
    if [int(x.name[5:]) for x in completed] != list(range(1, len(completed) + 1)):
        raise ValueError("Published Loss-aware stages are not contiguous")
    return completed


def run_metadata(root, arm, seed, config, data):
    return {
        "run_id": f"candidate-b-loss-aware-{arm}-seed{seed}", "started_at": now(), "status": "RUNNING",
        "active_root": str(root), "interpreter": os.sys.executable, "method": METHOD, "method_name": METHOD_NAME,
        "arm": ARMS[arm], "arm_key": arm, "seed": seed, "stage_order": list(STAGES), "completed_stages": 0,
        "model_checkpoint": config["model"]["checkpoint"], "model_revision": REAL_REVISION,
        "model_weights_sha256": REAL_MODEL_SHA256, "max_length": 1280,
        "training_policy": copy.deepcopy(config["training"]), "replay_policy": copy.deepcopy(config["replay"]),
        "memory_capacity_examples": 200, "replay_batch_size": 4, "historical_replay_only": True,
        "model_weights_carried": True, "optimizer_reset_each_stage": True, "selective_adaptation": False,
        "materialization_manifest_sha256": MATERIALIZATION_MANIFEST_SHA256,
        "materialization_validation_sha256": MATERIALIZATION_VALIDATION_SHA256,
        "dataset_hashes": data.file_hashes, "environment": environment(root),
    }


def run_control(root: Path, arm: str, seed: int):
    root = Path(root).resolve()
    if root != Path(r"E:\ContinualGuard").resolve():
        raise ValueError(f"Wrong active project root: {root}")
    preflight = root / RESULT_RELATIVE / "preflight/preflight.json"
    if not preflight.exists() or json.loads(preflight.read_text(encoding="utf-8")).get("status") != "PASS":
        raise ValueError("PASS Loss-aware preflight required")
    config = approved_config(root, seed)
    data = CandidateBData(root, arm)
    lookup = record_lookup(data)
    run_dir = root / RESULT_RELATIVE / arm / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = run_dir / "resolved_config.yaml"
    if not resolved.exists():
        resolved.write_text(yaml.safe_dump({**config, "execution_arm": ARMS[arm]}, sort_keys=True), encoding="utf-8")
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else run_metadata(root, arm, seed, config, data)
    if metadata.get("status") == "COMPLETE":
        return run_dir
    completed = stage_directories(run_dir)
    metadata.update(status="RUNNING", resumed_from_stage=len(completed) if completed else None)
    atomic_json(metadata_path, metadata)
    model = tokenizer = None
    stage_started = None
    try:
        seed_everything(seed, True, 4)
        if completed:
            model, tokenizer, state = load_boundary_checkpoint(completed[-1] / "checkpoint")
            if state["method"] != METHOD or state["arm"] != arm or state["seed"] != seed or state["completed_stages"] != len(completed):
                raise ValueError("Loss-aware resume lineage mismatch")
            buffer = LossAwareBuffer.from_state_dict(state["buffer_state"])
            restore_rng(state["rng"])
            global_step, initialization = state["global_step"], state["initialization"]
        else:
            tokenizer = make_tokenizer(config["model"], root)
            model = make_model(config["model"], root)
            initialization = state_fingerprint(model)
            if initialization != initialization_reference(root, arm, seed):
                raise ValueError("Loss-aware initialization differs from frozen comparisons")
            atomic_json(run_dir / "initialization.json", initialization)
            buffer, global_step = LossAwareBuffer(seed), 0
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.to("cuda")
        metadata.update(initialization=initialization, matched_existing_initialization=True,
                        parameter_count=sum(p.numel() for p in model.parameters()))
        atomic_json(metadata_path, metadata)
        for stage_index in range(len(completed) + 1, 5):
            stage = STAGES[stage_index - 1]
            attempt = run_dir / f"_attempt_stage{stage_index}_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{os.getpid()}"
            attempt.mkdir(parents=False, exist_ok=False); (attempt / "predictions").mkdir()
            stage_started = time.perf_counter(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            current_records = data.records(stage, "train")
            scoring_started = time.perf_counter()
            scored = score_candidates(model, tokenizer, current_records, config, seed)
            torch.cuda.synchronize()
            scoring_runtime = time.perf_counter() - scoring_started
            write_jsonl(attempt / "pre_update_loss_scores.jsonl", scored)
            before = memory_composition(buffer)
            optimizer, scheduler = make_optimizer(model, config)
            atomic_json(attempt / "stage_start.json", {
                "stage": stage, "stage_index": stage_index, "started_at": now(),
                "optimizer_state_entries_before_training": len(optimizer.state), "optimizer_reset": True,
                "model_weights_carried": stage_index > 1, "current_stage_training_records": 400,
                "historical_memory_records": len(buffer.entries), "buffer_updated_after_training": True,
                "pre_update_true_label_CE_on_train_only": True,
            })
            global_step, _captured, training = train_stage_with_replay(
                model, tokenizer, current_records, config, optimizer, scheduler, buffer, lookup,
                attempt, stage, stage_index, global_step,
            )
            quota = buffer.update(scored, tuple(STAGES[:stage_index]))
            write_jsonl(attempt / "buffer.jsonl", buffer.entries)
            composition = memory_composition(buffer, quota)
            atomic_json(attempt / "memory_composition.json", composition)
            evaluations = {}
            checkpoint_id = f"loss-aware:{ARMS[arm]}:seed{seed}:stage{stage_index}"
            for eval_index in range(1, stage_index + 1):
                task = STAGES[eval_index - 1]
                evaluations[task] = evaluate_stage_test(
                    model, tokenizer, data.records(task, "test"), config, checkpoint_id,
                    attempt / "predictions" / f"{task}.jsonl",
                )
            external = {
                "NotInject": evaluate_external(model, tokenizer, data.notinject, config, checkpoint_id, "NotInject", attempt / "predictions/NotInject.jsonl"),
                "B-short": evaluate_external(model, tokenizer, data.bshort, config, checkpoint_id, "B-short", attempt / "predictions/B-short.jsonl"),
            }
            current = evaluations[stage]
            val = validation_loss(model, tokenizer, data.records(stage, "val"), config)
            torch.cuda.synchronize(); runtime = time.perf_counter() - stage_started
            metrics = {
                "method": METHOD, "method_name": METHOD_NAME, "arm": ARMS[arm], "seed": seed,
                "trained_stage": stage, "stage_index": stage_index, "evaluations": evaluations, "external": external,
                "acquisition": {
                    "attack_tpr": current["malicious"]["attack_tpr"], "benign_tnr": current["benign"]["benign_tnr"],
                    "benign_fpr": current["benign"]["benign_fpr"], "paired_macro_f1": current["paired"]["macro_f1"],
                    "paired_accuracy": current["paired"]["accuracy"],
                },
                "training": training, "training_loss": training["training_loss"], "validation_loss": val,
                "buffer_scoring_runtime_seconds": scoring_runtime, "buffer_before_training": before,
                "buffer_after_training": composition, "runtime_seconds": runtime,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            atomic_json(attempt / "metrics.json", metrics)
            hashes = save_boundary_checkpoint(attempt / "checkpoint", model, tokenizer, {
                "experiment": "Candidate B final Generic Loss-aware Replay control", "method": METHOD,
                "arm": arm, "arm_name": ARMS[arm], "seed": seed, "completed_stages": stage_index,
                "stage": stage, "global_step": global_step, "initialization": initialization,
                "config_sha256": sha256_file(root / CONFIG_RELATIVE),
                "materialization_manifest_sha256": MATERIALIZATION_MANIFEST_SHA256,
                "dataset_hashes": data.file_hashes, "buffer_state": buffer.state_dict(),
                "memory_capacity_examples": 200, "replay_batch_size": 4,
            })
            atomic_json(attempt / "stage_metadata.json", {
                "status": "COMPLETE", "completed_at": now(), "stage": stage, "stage_index": stage_index,
                "runtime_seconds": runtime, "buffer_scoring_runtime_seconds": scoring_runtime,
                "global_step": global_step, "checkpoint_hashes": hashes, "optimizer_state_saved": False,
                "optimizer_reset_next_stage": True, "buffer_size": len(buffer.entries),
                "replay_examples_exposed": training["replay_examples_exposed"],
            })
            os.replace(attempt, run_dir / f"stage{stage_index}")
            metadata["completed_stages"] = stage_index; metadata["last_completed_stage"] = stage
            metadata["runtime_seconds"] = sum(json.loads((run_dir / f"stage{i}/stage_metadata.json").read_text(encoding="utf-8"))["runtime_seconds"] for i in range(1, stage_index + 1))
            metadata["buffer_scoring_runtime_seconds"] = sum(json.loads((run_dir / f"stage{i}/stage_metadata.json").read_text(encoding="utf-8"))["buffer_scoring_runtime_seconds"] for i in range(1, stage_index + 1))
            atomic_json(metadata_path, metadata)
            print(json.dumps({"event": "stage_complete", "method": METHOD, "arm": arm, "seed": seed, "stage": stage, "runtime_seconds": runtime}), flush=True)
        metadata.update(status="COMPLETE", finished_at=now())
        atomic_json(metadata_path, metadata)
        print(json.dumps({"event": "run_complete", "method": METHOD, "arm": arm, "seed": seed, "runtime_seconds": metadata["runtime_seconds"]}), flush=True)
    except Exception as exc:
        metadata.update(status="FAILED", failed_at=now(), error=f"{type(exc).__name__}: {exc}",
                        failed_stage_elapsed_seconds=time.perf_counter() - stage_started if stage_started else None)
        atomic_json(metadata_path, metadata)
        raise
    finally:
        if model is not None: del model
        if tokenizer is not None: del tokenizer
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return run_dir
