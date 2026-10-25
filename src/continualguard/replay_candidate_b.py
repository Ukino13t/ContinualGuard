"""Fixed Candidate B replay comparison: Reservoir ER, DER++, and project Security-aware Replay."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from .artifacts import append_jsonl, now
from .config import load_config
from .model import batch_inputs, make_model, make_tokenizer
from .reproducibility import environment, isolated_evaluation, restore_rng, seed_everything
from .stronger_pilot_no_replay import (
    ARMS,
    ARM_DIRS,
    MATERIALIZATION_MANIFEST_SHA256,
    MATERIALIZATION_VALIDATION_SHA256,
    SEEDS,
    STAGES,
    CandidateBData,
    approved_config as baseline_approved_config,
    atomic_json,
    evaluate_external,
    evaluate_stage_test,
    load_boundary_checkpoint,
    mean_training_loss,
    save_boundary_checkpoint,
    sha256_file,
    state_fingerprint,
    validation_loss,
    write_jsonl,
)
from .train import REAL_MODEL_SHA256, REAL_REVISION, make_optimizer

METHODS = ("reservoir", "derpp", "security_aware")
METHOD_NAMES = {"reservoir": "Reservoir ER", "derpp": "DER++", "security_aware": "Security-aware Replay"}
METHOD_CODES = {"reservoir": 11, "derpp": 23, "security_aware": 37}
MEMORY_CAPACITY = 200
REPLAY_BATCH_SIZE = 4
DERPP_ALPHA = 0.5
DERPP_BETA = 0.5
CONFIG_RELATIVE = "configs/replay_comparison_candidate_b.yaml"
RESULT_RELATIVE = "results/replay_comparison_candidate_b"


def stable_hash(*values) -> str:
    return hashlib.sha256("|".join(str(value) for value in values).encode("utf-8")).hexdigest()


def quantile(values, q: float):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def security_risk(label: int, p_attack: float) -> float:
    if label not in (0, 1) or not 0 <= p_attack <= 1:
        raise ValueError("Security risk requires a binary label and probability")
    return 1.0 - p_attack if label == 1 else p_attack


def normalized_replay_loss(method: str, current_ce, replay_ce=None, logit_mse=None):
    if replay_ce is None:
        return current_ce
    if method in ("reservoir", "security_aware", "loss_aware"):
        return (current_ce + replay_ce) / 2.0
    if method == "derpp":
        if logit_mse is None:
            raise ValueError("DER++ requires logit MSE")
        return (current_ce + DERPP_ALPHA * logit_mse + DERPP_BETA * replay_ce) / (1.0 + DERPP_ALPHA + DERPP_BETA)
    raise ValueError(method)


def validate_replay_candidate(row: dict, current_stage_index: int | None = None) -> None:
    if row.get("split") != "train":
        raise ValueError("Replay memory accepts TRAIN records only")
    if row.get("task_id") not in STAGES:
        raise ValueError("Replay memory rejects external or unknown records")
    stage_index = STAGES.index(row["task_id"]) + 1
    if current_stage_index is not None and stage_index >= current_stage_index:
        raise ValueError("Replay batch contains current or future-stage examples")


def coverage_key(row: dict) -> str:
    source = row.get("source") or {}
    candidates = (
        row.get("project_template_group"), row.get("project_semantic_parent_group"),
        row.get("duplicate_group_id"), row.get("model_view_group"), row.get("normalized_group"),
        row.get("source_parent_id"), row.get("base_document_id"), row.get("injection_id"),
    )
    group = next((str(value) for value in candidates if value not in (None, "")), row["example_id"])
    return "|".join((str(row["label"]), row["task_id"], str(row.get("attack_family")), str(row.get("benign_group")), str(row.get("injection_position")), str(source.get("dataset")), group))


def entry_from_row(row: dict, *, logits=None, p_attack=None, risk=None, selection_seed=None) -> dict:
    validate_replay_candidate(row)
    source = row.get("source") or {}
    entry = {
        "example_id": row["example_id"], "label": row["label"], "task_id": row["task_id"],
        "stage_index": STAGES.index(row["task_id"]) + 1, "text_sha256": row["text_sha256"],
        "attack_family": row.get("attack_family"), "benign_group": row.get("benign_group"),
        "benign_partition": row.get("benign_partition"), "injection_position": row.get("injection_position"),
        "language": row.get("language"), "domain": row.get("domain"), "source_dataset": source.get("dataset"),
        "source_category": row.get("source_category"), "coverage_key": coverage_key(row),
    }
    if logits is not None:
        entry["historical_logits"] = [float(value) for value in logits]
    if p_attack is not None:
        entry["pre_update_p_attack"] = float(p_attack)
    if risk is not None:
        entry["pre_update_security_risk"] = float(risk)
    if selection_seed is not None:
        entry["tie_break_sha256"] = stable_hash(selection_seed, "security-aware", row["example_id"])
    return entry


class ReplayBuffer:
    def __init__(self, method: str, seed: int, capacity: int = MEMORY_CAPACITY):
        if method not in METHODS or capacity != MEMORY_CAPACITY:
            raise ValueError("Unapproved replay buffer configuration")
        self.method, self.seed, self.capacity = method, seed, capacity
        code = METHOD_CODES[method]
        self.update_rng = random.Random(seed * 1009 + code)
        self.sample_rng = random.Random(seed * 2027 + code)
        self.entries: list[dict] = []
        self.seen_count = 0

    def state_dict(self) -> dict:
        return {
            "method": self.method, "seed": self.seed, "capacity": self.capacity,
            "entries": copy.deepcopy(self.entries), "seen_count": self.seen_count,
            "update_rng_state": self.update_rng.getstate(), "sample_rng_state": self.sample_rng.getstate(),
        }

    @classmethod
    def from_state_dict(cls, state: dict):
        obj = cls(state["method"], state["seed"], state["capacity"])
        obj.entries = copy.deepcopy(state["entries"])
        obj.seen_count = state["seen_count"]
        obj.update_rng.setstate(state["update_rng_state"])
        obj.sample_rng.setstate(state["sample_rng_state"])
        obj.validate()
        return obj

    def validate(self, current_stage_index: int | None = None) -> None:
        if len(self.entries) > self.capacity:
            raise ValueError("Replay capacity exceeded")
        ids = [entry["example_id"] for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate identity in replay memory")
        for entry in self.entries:
            if entry["task_id"] not in STAGES:
                raise ValueError("Non-Candidate-B entry in replay memory")
            if current_stage_index is not None and entry["stage_index"] >= current_stage_index:
                raise ValueError("Current/future entry in replay batch")
            if self.method == "derpp" and len(entry.get("historical_logits", [])) != 2:
                raise ValueError("DER++ entry lacks two historical logits")
            if self.method == "security_aware" and "pre_update_security_risk" not in entry:
                raise ValueError("Security-aware entry lacks frozen risk")

    def sample(self, count: int, current_stage_index: int) -> list[dict]:
        self.validate(current_stage_index)
        size = min(count, len(self.entries))
        return self.sample_rng.sample(self.entries, size) if size else []

    def reservoir_update(self, candidates: list[dict]) -> None:
        if self.method not in ("reservoir", "derpp"):
            raise ValueError("Reservoir update used by wrong method")
        for candidate in candidates:
            self.seen_count += 1
            if len(self.entries) < self.capacity:
                self.entries.append(copy.deepcopy(candidate))
            else:
                index = self.update_rng.randrange(self.seen_count)
                if index < self.capacity:
                    self.entries[index] = copy.deepcopy(candidate)
        self.validate()

    def security_update(self, candidates: list[dict], contributed_stages: tuple[str, ...]) -> dict:
        if self.method != "security_aware":
            raise ValueError("Security update used by wrong method")
        combined = {entry["example_id"]: copy.deepcopy(entry) for entry in [*self.entries, *candidates]}
        selected = []
        quota_record = {}
        for label in (0, 1):
            total = self.capacity // 2
            base, remainder = divmod(total, len(contributed_stages))
            for index, stage in enumerate(contributed_stages):
                quota = base + (1 if index < remainder else 0)
                pool = [entry for entry in combined.values() if entry["label"] == label and entry["task_id"] == stage]
                pool.sort(key=lambda entry: (-entry["pre_update_security_risk"], entry["tie_break_sha256"]))
                diverse, deferred, groups = [], [], set()
                for entry in pool:
                    if entry["coverage_key"] not in groups:
                        diverse.append(entry)
                        groups.add(entry["coverage_key"])
                    else:
                        deferred.append(entry)
                chosen = (diverse + deferred)[:quota]
                selected.extend(chosen)
                quota_record[f"label{label}/{stage}"] = {"target": quota, "available": len(pool), "selected": len(chosen)}
        self.entries = selected
        self.seen_count += len(candidates)
        self.validate()
        return quota_record


def approved_config(root: Path, method: str, seed: int) -> dict:
    config = yaml.safe_load((Path(root) / CONFIG_RELATIVE).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Replay configuration must be a mapping")
    config["seed"] = seed
    config["method"] = method
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    t, r = config["training"], config["replay"]
    valid = (
        config["run_type"] == "reportable research result"
        and config["experiment"] == "Candidate B primary replay comparison"
        and config["data_path"] == "data/processed/stronger_pilot_candidate_b"
        and tuple(config["stage_order"]) == STAGES and config["seed"] in SEEDS and config["method"] in METHODS
        and config["model"]["checkpoint"] == "microsoft/deberta-v3-base"
        and config["model"]["revision"] == REAL_REVISION and config["model"]["max_length"] == 1280
        and t["epochs"] == 1 and t["batch_size"] == 4 and t["gradient_accumulation_steps"] == 1
        and t["gradient_checkpointing"] is True and t["learning_rate"] == 1e-5
        and t["betas"] == [0.9, 0.999] and t["epsilon"] == 1e-8 and t["weight_decay"] == 0.0
        and t["max_grad_norm"] == 1.0 and t["optimizer_transition"] == "reset" and t["scheduler"] == "constant"
        and r["methods"] == list(METHODS) and r["memory_capacity_examples"] == 200 and r["replay_batch_size"] == 4
        and r["derpp_alpha"] == DERPP_ALPHA and r["derpp_beta"] == DERPP_BETA
        and config["device"] == "cuda" and config["precision"] == "float16" and config["deterministic"] is True
    )
    if not valid:
        raise ValueError("Execution differs from approved Candidate B replay protocol")


def record_lookup(data: CandidateBData) -> dict[str, dict]:
    rows = [data.records(stage, "train") for stage in STAGES]
    lookup = {row["example_id"]: row for stage_rows in rows for row in stage_rows}
    if len(lookup) != 1600:
        raise ValueError("Candidate B training identity contract mismatch")
    return lookup


def score_security_candidates(model, tokenizer, records: list[dict], config: dict, seed: int) -> list[dict]:
    if any(row.get("split") != "train" for row in records):
        raise ValueError("Security scoring must use TRAIN records only")
    output = []
    with isolated_evaluation(model):
        for offset in range(0, len(records), config["training"]["batch_size"]):
            batch = records[offset:offset + config["training"]["batch_size"]]
            inputs = batch_inputs(batch, tokenizer, 1280, "cuda")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
            for row, probability in zip(batch, probabilities):
                output.append(entry_from_row(
                    row, p_attack=probability, risk=security_risk(row["label"], probability), selection_seed=seed,
                ))
    return output


def train_stage_with_replay(model, tokenizer, records: list[dict], config: dict, optimizer, scheduler,
                            buffer: ReplayBuffer, lookup: dict[str, dict], run_dir: Path,
                            stage: str, stage_index: int, global_step: int,
                            order_generator: torch.Generator | None = None,
                            grad_scaler_init_scale: float = 1024.0) -> tuple[int, list[dict], dict]:
    if any(row.get("split") != "train" or row.get("task_id") != stage for row in records):
        raise ValueError("Trainer received historical/future/non-training current records")
    buffer.validate(stage_index)
    model.train()
    batch_size = config["training"]["batch_size"]
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=grad_scaler_init_scale)
    order = torch.randperm(len(records), generator=order_generator).tolist()
    captured = []
    replay_examples = replay_forward_passes = 0
    total_current_ce = total_replay_ce = total_mse = total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    for offset in range(0, len(order), batch_size):
        batch = [records[index] for index in order[offset:offset + batch_size]]
        inputs = batch_inputs(batch, tokenizer, 1280, "cuda")
        labels = torch.tensor([row["label"] for row in batch], dtype=torch.long, device="cuda")
        sampled = buffer.sample(REPLAY_BATCH_SIZE, stage_index)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            current_output = model(**inputs)
            current_ce = F.cross_entropy(current_output.logits, labels)
            replay_ce = logit_mse = None
            replay_ids = []
            if sampled:
                replay_rows = [lookup[entry["example_id"]] for entry in sampled]
                if any(row["task_id"] not in STAGES[:stage_index - 1] for row in replay_rows):
                    raise ValueError("Replay minibatch contains current/future-stage records")
                replay_inputs = batch_inputs(replay_rows, tokenizer, 1280, "cuda")
                replay_labels = torch.tensor([entry["label"] for entry in sampled], dtype=torch.long, device="cuda")
                replay_logits = model(**replay_inputs).logits
                replay_ce = F.cross_entropy(replay_logits, replay_labels)
                if config["method"] == "derpp":
                    targets = torch.tensor([entry["historical_logits"] for entry in sampled], dtype=torch.float32, device="cuda")
                    logit_mse = F.mse_loss(replay_logits.float(), targets)
                replay_ids = [entry["example_id"] for entry in sampled]
            loss = normalized_replay_loss(config["method"], current_ce, replay_ce, logit_mse)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite replay training loss")
        current_logits_cpu = current_output.logits.detach().float().cpu().tolist()
        for row, logits in zip(batch, current_logits_cpu):
            captured.append(entry_from_row(row, logits=logits if config["method"] == "derpp" else None))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
        if not torch.isfinite(grad_norm):
            raise RuntimeError("Nonfinite replay gradient norm")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        n = len(batch)
        total_current_ce += float(current_ce.item()) * n
        total_loss += float(loss.item()) * n
        if replay_ce is not None:
            total_replay_ce += float(replay_ce.item()) * n
            replay_examples += len(sampled)
            replay_forward_passes += 1
        if logit_mse is not None:
            total_mse += float(logit_mse.item()) * n
        append_jsonl(Path(run_dir) / "train_log.jsonl", {
            "stage": stage, "global_step": global_step, "loss": float(loss.item()),
            "current_ce": float(current_ce.item()), "replay_ce": float(replay_ce.item()) if replay_ce is not None else None,
            "replay_logit_mse": float(logit_mse.item()) if logit_mse is not None else None,
            "lr": optimizer.param_groups[0]["lr"], "example_ids": [row["example_id"] for row in batch],
            "replay_example_ids": replay_ids, "examples_in_step": n, "replay_examples_in_step": len(sampled),
            "current_task_id": stage, "allowed_replay_task_ids": list(STAGES[:stage_index - 1]),
        })
    if len(captured) != len(records) or len({entry["example_id"] for entry in captured}) != len(records):
        raise ValueError("Current-stage capture contract failed")
    summary = {
        "training_loss": total_loss / len(records), "current_ce": total_current_ce / len(records),
        "replay_ce": total_replay_ce / len(records) if replay_examples else None,
        "replay_logit_mse": total_mse / len(records) if config["method"] == "derpp" and replay_examples else None,
        "current_examples": len(records), "replay_examples_exposed": replay_examples,
        "replay_forward_passes": replay_forward_passes, "optimizer_steps": len(order) // batch_size,
    }
    return global_step, captured, summary


def memory_composition(buffer: ReplayBuffer, quota_fulfillment=None) -> dict:
    entries = buffer.entries
    def counts(key):
        return dict(sorted(Counter(str(entry.get(key)) for entry in entries).items()))
    risks = [entry["pre_update_security_risk"] for entry in entries if "pre_update_security_risk" in entry]
    result = {
        "method": buffer.method, "capacity_examples": buffer.capacity, "buffer_size": len(entries),
        "seen_training_examples": buffer.seen_count,
        "label_counts": counts("label"), "stage_distribution": counts("task_id"),
        "attack_family_distribution": counts("attack_family"), "benign_group_distribution": counts("benign_group"),
        "injection_position_distribution": counts("injection_position"), "source_distribution": counts("source_dataset"),
        "unique_coverage_keys": len({entry["coverage_key"] for entry in entries}),
        "raw_text_stored": False, "historical_logit_entries": sum("historical_logits" in entry for entry in entries),
        "auxiliary_logit_bytes_float32": sum(len(entry.get("historical_logits", [])) * 4 for entry in entries),
    }
    if risks:
        result["security_risk"] = {
            "mean": statistics.mean(risks), "median": statistics.median(risks),
            "p05": quantile(risks, .05), "p10": quantile(risks, .10), "p90": quantile(risks, .90),
            "p95": quantile(risks, .95), "p99": quantile(risks, .99),
            "high_risk_malicious_ge_0_5": sum(entry["label"] == 1 and entry["pre_update_security_risk"] >= .5 for entry in entries),
            "high_risk_benign_ge_0_5": sum(entry["label"] == 0 and entry["pre_update_security_risk"] >= .5 for entry in entries),
        }
        result["stage_quota_fulfillment"] = quota_fulfillment or {}
    return result


def _stage_directories(run_dir: Path) -> list[Path]:
    completed = []
    for index in range(1, 5):
        path = run_dir / f"stage{index}"
        if path.is_dir() and (path / "stage_metadata.json").exists() and (path / "checkpoint/state.pt").exists():
            completed.append(path)
        elif path.exists():
            raise ValueError(f"Incomplete published stage directory: {path}")
    if [int(path.name[5:]) for path in completed] != list(range(1, len(completed) + 1)):
        raise ValueError("Published stages are not a contiguous prefix")
    return completed


def _run_metadata(root: Path, method: str, arm: str, seed: int, config: dict, data: CandidateBData) -> dict:
    return {
        "run_id": f"candidate-b-{method}-{arm}-seed{seed}", "started_at": now(), "status": "RUNNING",
        "active_root": str(root), "interpreter": os.sys.executable, "method": method, "method_name": METHOD_NAMES[method],
        "arm": ARMS[arm], "arm_key": arm, "seed": seed, "stage_order": list(STAGES), "completed_stages": 0,
        "model_checkpoint": config["model"]["checkpoint"], "model_revision": REAL_REVISION,
        "model_weights_sha256": REAL_MODEL_SHA256, "max_length": 1280,
        "training_policy": copy.deepcopy(config["training"]), "replay_policy": copy.deepcopy(config["replay"]),
        "memory_capacity_examples": MEMORY_CAPACITY, "replay_batch_size": REPLAY_BATCH_SIZE,
        "current_stage_only_current_data": True, "historical_replay_only": True, "model_weights_carried": True,
        "optimizer_reset_each_stage": True, "selective_adaptation": False,
        "materialization_manifest_sha256": MATERIALIZATION_MANIFEST_SHA256,
        "materialization_validation_sha256": MATERIALIZATION_VALIDATION_SHA256,
        "dataset_hashes": data.file_hashes, "environment": environment(root),
    }


def _initialization_reference(root: Path, arm: str, seed: int) -> dict:
    path = root / "results/stronger_pilot_no_replay" / arm / f"seed{seed}" / "initialization.json"
    if not path.exists():
        raise ValueError(f"Missing no-replay initialization reference: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _preflight_buffer(method: str, seed: int, model, tokenizer, rows: list[dict], config: dict) -> ReplayBuffer:
    buffer = ReplayBuffer(method, seed)
    if method == "security_aware":
        candidates = score_security_candidates(model, tokenizer, rows, config, seed)
        buffer.security_update(candidates, ("AF2",))
    else:
        candidates = []
        with isolated_evaluation(model):
            for offset in range(0, len(rows), 4):
                batch = rows[offset:offset + 4]
                logits = None
                if method == "derpp":
                    inputs = batch_inputs(batch, tokenizer, 1280, "cuda")
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                        logits = model(**inputs).logits.detach().float().cpu().tolist()
                for index, row in enumerate(batch):
                    candidates.append(entry_from_row(row, logits=logits[index] if logits is not None else None))
        buffer.reservoir_update(candidates)
    return buffer


def run_preflight(root: Path) -> Path:
    root = Path(root).resolve()
    if root != Path(r"E:\ContinualGuard").resolve():
        raise ValueError(f"Wrong active project root: {root}")
    output = root / RESULT_RELATIVE / "preflight"
    result_path = output / "preflight.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous.get("status") == "PASS" and previous.get("config_sha256") == sha256_file(root / CONFIG_RELATIVE):
            return output
        raise ValueError("Existing replay preflight is not a reusable PASS artifact")
    output.mkdir(parents=True, exist_ok=False)
    result = {"started_at": now(), "status": "RUNNING", "methods": {}, "config_sha256": sha256_file(root / CONFIG_RELATIVE)}
    started = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        data = CandidateBData(root, "doc")
        lookup = record_lookup(data)
        history = data.records("AF2", "train")[:8]
        current = data.records("AF3", "train")[:4]
        for method in METHODS:
            method_dir = output / method
            method_dir.mkdir()
            seed_everything(42, True, 4)
            config = approved_config(root, method, 42)
            tokenizer = make_tokenizer(config["model"], root)
            model = make_model(config["model"], root)
            initial = state_fingerprint(model)
            if initial != _initialization_reference(root, "doc", 42):
                raise ValueError(f"{method} preflight initialization differs from M0")
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.to("cuda")
            buffer = _preflight_buffer(method, 42, model, tokenizer, history, config)
            pre_current_scored = score_security_candidates(model, tokenizer, current, config, 42) if method == "security_aware" else None
            optimizer, scheduler = make_optimizer(model, config)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            step_started = time.perf_counter()
            global_step, captured, training = train_stage_with_replay(
                model, tokenizer, current, config, optimizer, scheduler, buffer, lookup,
                method_dir, "AF3", 2, 0,
            )
            if method == "security_aware":
                quota = buffer.security_update(pre_current_scored, ("AF2", "AF3"))
            else:
                buffer.reservoir_update(captured)
                quota = None
            torch.cuda.synchronize()
            checkpoint_hashes = save_boundary_checkpoint(method_dir / "checkpoint", model, tokenizer, {
                "purpose": "replay software preflight only", "method": method, "seed": 42,
                "completed_stages": 2, "global_step": global_step, "initialization": initial,
                "buffer_state": buffer.state_dict(),
            })
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            loaded_model, loaded_tokenizer, state = load_boundary_checkpoint(method_dir / "checkpoint")
            loaded_buffer = ReplayBuffer.from_state_dict(state["buffer_state"])
            loaded_model.to("cuda")
            test_rows = data.records("AF3", "test")
            balanced_test = [*[row for row in test_rows if row["label"] == 0][:2], *[row for row in test_rows if row["label"] == 1][:2]]
            metrics = evaluate_stage_test(
                loaded_model, loaded_tokenizer, balanced_test, config,
                f"preflight:{method}", method_dir / "predictions.jsonl",
            )
            composition = memory_composition(loaded_buffer, quota)
            atomic_json(method_dir / "memory_composition.json", composition)
            result["methods"][method] = {
                "status": "PASS", "loss": training["training_loss"], "current_ce": training["current_ce"],
                "replay_ce": training["replay_ce"], "replay_logit_mse": training["replay_logit_mse"],
                "replay_examples": training["replay_examples_exposed"], "checkpoint_reload": True,
                "buffer_roundtrip": len(loaded_buffer.entries) == len(buffer.entries),
                "buffer_size": len(buffer.entries), "evaluation_records": 4,
                "step_seconds": time.perf_counter() - step_started,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                "checkpoint_hashes": checkpoint_hashes, "initialization": initial,
            }
            del loaded_model, loaded_tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        result.update({
            "status": "PASS", "finished_at": now(), "gpu": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda, "cuda_oom": False, "all_methods": list(METHODS),
            "memory_capacity_examples": MEMORY_CAPACITY, "replay_batch_size": REPLAY_BATCH_SIZE,
        })
    except Exception as exc:
        result.update(status="FAIL", finished_at=now(), error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["runtime_seconds"] = time.perf_counter() - started
        atomic_json(result_path, result)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return output


def run_replay(root: Path, method: str, arm: str, seed: int) -> Path:
    root = Path(root).resolve()
    if root != Path(r"E:\ContinualGuard").resolve():
        raise ValueError(f"Wrong active project root: {root}")
    preflight_path = root / RESULT_RELATIVE / "preflight/preflight.json"
    if not preflight_path.exists() or json.loads(preflight_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise ValueError("PASS all-method replay preflight is required")
    config = approved_config(root, method, seed)
    data = CandidateBData(root, arm)
    lookup = record_lookup(data)
    run_dir = root / RESULT_RELATIVE / method / arm / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = run_dir / "resolved_config.yaml"
    if not resolved.exists():
        resolved.write_text(yaml.safe_dump({**config, "execution_arm": ARMS[arm]}, sort_keys=True), encoding="utf-8")
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else _run_metadata(root, method, arm, seed, config, data)
    if metadata.get("status") == "COMPLETE":
        return run_dir
    completed = _stage_directories(run_dir)
    metadata.update(status="RUNNING", resumed_from_stage=len(completed) if completed else None)
    atomic_json(metadata_path, metadata)
    model = tokenizer = None
    stage_started = None
    try:
        seed_everything(seed, True, 4)
        if completed:
            model, tokenizer, state = load_boundary_checkpoint(completed[-1] / "checkpoint")
            if state["method"] != method or state["arm"] != arm or state["seed"] != seed or state["completed_stages"] != len(completed):
                raise ValueError("Resume checkpoint lineage mismatch")
            buffer = ReplayBuffer.from_state_dict(state["buffer_state"])
            restore_rng(state["rng"])
            global_step, initialization = state["global_step"], state["initialization"]
        else:
            tokenizer = make_tokenizer(config["model"], root)
            model = make_model(config["model"], root)
            initialization = state_fingerprint(model)
            reference = _initialization_reference(root, arm, seed)
            if initialization != reference:
                raise ValueError("Replay initialization differs from matched M0 run")
            atomic_json(run_dir / "initialization.json", initialization)
            buffer, global_step = ReplayBuffer(method, seed), 0
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.to("cuda")
        metadata["initialization"] = initialization
        metadata["matched_no_replay_initialization"] = True
        metadata["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        atomic_json(metadata_path, metadata)
        for stage_index in range(len(completed) + 1, 5):
            stage = STAGES[stage_index - 1]
            attempt = run_dir / f"_attempt_stage{stage_index}_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{os.getpid()}"
            attempt.mkdir(parents=False, exist_ok=False)
            (attempt / "predictions").mkdir()
            stage_started = time.perf_counter()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            current_records = data.records(stage, "train")
            pre_scored = None
            if method == "security_aware":
                pre_scored = score_security_candidates(model, tokenizer, current_records, config, seed)
                write_jsonl(attempt / "pre_update_security_scores.jsonl", pre_scored)
            buffer_before = memory_composition(buffer)
            optimizer, scheduler = make_optimizer(model, config)
            atomic_json(attempt / "stage_start.json", {
                "stage": stage, "stage_index": stage_index, "started_at": now(),
                "optimizer_state_entries_before_training": len(optimizer.state), "optimizer_reset": True,
                "model_weights_carried": stage_index > 1, "current_stage_training_records": 400,
                "historical_memory_records": len(buffer.entries), "buffer_updated_after_training": True,
                "security_scores_pre_update_train_only": method == "security_aware",
            })
            global_step, captured, training = train_stage_with_replay(
                model, tokenizer, current_records, config, optimizer, scheduler, buffer, lookup,
                attempt, stage, stage_index, global_step,
            )
            if method == "security_aware":
                quota = buffer.security_update(pre_scored, tuple(STAGES[:stage_index]))
            else:
                buffer.reservoir_update(captured)
                quota = None
            buffer.validate()
            write_jsonl(attempt / "buffer.jsonl", buffer.entries)
            composition = memory_composition(buffer, quota)
            atomic_json(attempt / "memory_composition.json", composition)
            evaluations = {}
            checkpoint_id = f"{method}:{ARMS[arm]}:seed{seed}:stage{stage_index}"
            for eval_index in range(1, stage_index + 1):
                eval_stage = STAGES[eval_index - 1]
                evaluations[eval_stage] = evaluate_stage_test(
                    model, tokenizer, data.records(eval_stage, "test"), config, checkpoint_id,
                    attempt / "predictions" / f"{eval_stage}.jsonl",
                )
            externals = {
                "NotInject": evaluate_external(model, tokenizer, data.notinject, config, checkpoint_id, "NotInject", attempt / "predictions/NotInject.jsonl"),
                "B-short": evaluate_external(model, tokenizer, data.bshort, config, checkpoint_id, "B-short", attempt / "predictions/B-short.jsonl"),
            }
            current = evaluations[stage]
            val_loss = validation_loss(model, tokenizer, data.records(stage, "val"), config)
            torch.cuda.synchronize()
            runtime = time.perf_counter() - stage_started
            stage_metrics = {
                "method": method, "method_name": METHOD_NAMES[method], "arm": ARMS[arm], "seed": seed,
                "trained_stage": stage, "stage_index": stage_index, "evaluations": evaluations, "external": externals,
                "acquisition": {
                    "attack_tpr": current["malicious"]["attack_tpr"], "benign_tnr": current["benign"]["benign_tnr"],
                    "benign_fpr": current["benign"]["benign_fpr"], "paired_macro_f1": current["paired"]["macro_f1"],
                    "paired_accuracy": current["paired"]["accuracy"],
                },
                "training": training, "training_loss": training["training_loss"], "validation_loss": val_loss,
                "buffer_before_training": buffer_before, "buffer_after_training": composition,
                "runtime_seconds": runtime, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            atomic_json(attempt / "metrics.json", stage_metrics)
            checkpoint_hashes = save_boundary_checkpoint(attempt / "checkpoint", model, tokenizer, {
                "experiment": "Candidate B primary replay comparison", "method": method,
                "arm": arm, "arm_name": ARMS[arm], "seed": seed, "completed_stages": stage_index,
                "stage": stage, "global_step": global_step, "initialization": initialization,
                "config_sha256": sha256_file(root / CONFIG_RELATIVE),
                "materialization_manifest_sha256": MATERIALIZATION_MANIFEST_SHA256,
                "dataset_hashes": data.file_hashes, "buffer_state": buffer.state_dict(),
                "memory_capacity_examples": MEMORY_CAPACITY, "replay_batch_size": REPLAY_BATCH_SIZE,
            })
            atomic_json(attempt / "stage_metadata.json", {
                "status": "COMPLETE", "completed_at": now(), "stage": stage, "stage_index": stage_index,
                "runtime_seconds": runtime, "global_step": global_step, "checkpoint_hashes": checkpoint_hashes,
                "optimizer_state_saved": False, "optimizer_reset_next_stage": True,
                "buffer_size": len(buffer.entries), "replay_examples_exposed": training["replay_examples_exposed"],
            })
            os.replace(attempt, run_dir / f"stage{stage_index}")
            metadata["completed_stages"] = stage_index
            metadata["last_completed_stage"] = stage
            metadata["runtime_seconds"] = sum(json.loads((run_dir / f"stage{i}/stage_metadata.json").read_text(encoding="utf-8"))["runtime_seconds"] for i in range(1, stage_index + 1))
            atomic_json(metadata_path, metadata)
            print(json.dumps({"event": "stage_complete", "method": method, "arm": arm, "seed": seed, "stage": stage, "runtime_seconds": runtime, "replay_examples": training["replay_examples_exposed"]}), flush=True)
        metadata.update(status="COMPLETE", finished_at=now())
        atomic_json(metadata_path, metadata)
        print(json.dumps({"event": "run_complete", "method": method, "arm": arm, "seed": seed, "runtime_seconds": metadata["runtime_seconds"]}), flush=True)
    except Exception as exc:
        metadata.update(status="FAILED", failed_at=now(), error=f"{type(exc).__name__}: {exc}", failed_stage_elapsed_seconds=time.perf_counter() - stage_started if stage_started is not None else None)
        atomic_json(metadata_path, metadata)
        raise
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return run_dir

