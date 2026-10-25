"""Fixed Candidate B three-arm sequential baseline with no replay."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import shutil
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .artifacts import append_jsonl, digest, now, read_jsonl
from .config import load_config
from .metrics import classification_metrics
from .model import batch_inputs, make_model, make_tokenizer
from .notinject_evaluation import load_notinject
from .reproducibility import environment, isolated_evaluation, restore_rng, rng_state, seed_everything
from .train import REAL_MODEL_SHA256, REAL_REVISION, file_sha256, make_optimizer, train_stage

STAGES = ("AF2", "AF3", "AF4", "AF5")
SEEDS = (17, 42, 2026)
ARMS = {"doc": "B-DOC", "short": "B-SHORT", "hard": "B-HARD"}
ARM_DIRS = {"doc": "b_doc", "short": "b_short", "hard": "b_hard"}
EXPECTED = {"train": (400, 200, 200), "val": (60, 30, 30), "test": (120, 60, 60)}
MATERIALIZATION_MANIFEST_SHA256 = "81962e092e1f8502a537d675582549df94f128f5ebf493b1ed0b59540ec3e8f4"
MATERIALIZATION_VALIDATION_SHA256 = "15f2fae9b34595bf586124092eb77b01ffbda44b93f6dd8fc0e78c25581a376a"
TOKENIZER_REVISION = REAL_REVISION
THRESHOLD_RULE = "argmax over fixed binary logits; p_attack=0.5 boundary; ties choose benign/class 0"


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    return file_sha256(path)


def state_fingerprint(model) -> dict:
    """Fingerprint every non-base parameter; base weights use the pinned artifact hash."""
    value = hashlib.sha256()
    names = []
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith("deberta."):
            continue
        names.append(name)
        value.update(name.encode("utf-8"))
        value.update(str(tuple(tensor.shape)).encode("ascii"))
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return {"non_base_parameter_names": names, "non_base_parameter_sha256": value.hexdigest(), "base_model_weights_sha256": REAL_MODEL_SHA256}


def validate_config(config: dict) -> None:
    t = config["training"]
    if not (
        config["run_type"] == "reportable research result"
        and config["data_path"] == "data/processed/stronger_pilot_candidate_b"
        and tuple(config["stage_order"]) == STAGES
        and config["seed"] in SEEDS
        and config["model"]["kind"] == "pretrained_deberta"
        and config["model"]["checkpoint"] == "microsoft/deberta-v3-base"
        and config["model"]["revision"] == REAL_REVISION
        and config["model"]["max_length"] == 1280
        and t["epochs"] == 1
        and t["batch_size"] == 4
        and t["gradient_accumulation_steps"] == 1
        and t["gradient_checkpointing"] is True
        and t["learning_rate"] == 1e-5
        and t["betas"] == [0.9, 0.999]
        and t["epsilon"] == 1e-8
        and t["weight_decay"] == 0.0
        and t["max_grad_norm"] == 1.0
        and t["optimizer_transition"] == "reset"
        and t["scheduler"] == "constant"
        and t["pre_adaptation"] is False
        and config["device"] == "cuda"
        and config["precision"] == "float16"
        and config["deterministic"] is True
        and config["threads"] == 4
    ):
        raise ValueError("Execution differs from the approved Candidate B no-replay protocol")


class CandidateBData:
    def __init__(self, root: Path, arm: str):
        if arm not in ARMS:
            raise ValueError(f"Unknown arm: {arm}")
        self.root = Path(root)
        self.arm = arm
        manifest_path = self.root / "data/processed/stronger_pilot_candidate_b/materialization_manifest.json"
        if sha256_file(manifest_path) != MATERIALIZATION_MANIFEST_SHA256:
            raise ValueError("Materialization manifest hash changed")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validation_path = self.root / self.manifest["validation_artifact"]
        if sha256_file(validation_path) != MATERIALIZATION_VALIDATION_SHA256 or self.manifest["validation_artifact_sha256"] != MATERIALIZATION_VALIDATION_SHA256:
            raise ValueError("Materialization validation hash changed")
        self.rows = {}
        manifest_lookup = {item["relative_path"]: item for item in self.manifest["files"]}
        seen_ids = set()
        for stage_index, stage in enumerate(STAGES, 1):
            for split, (total, benign, malicious) in EXPECTED.items():
                relative = f"data/processed/stronger_pilot_candidate_b/{ARM_DIRS[arm]}/stage{stage_index}/{split}.jsonl"
                path = self.root / relative
                entry = manifest_lookup.get(relative)
                if entry is None or sha256_file(path) != entry["sha256"]:
                    raise ValueError(f"Materialized file hash mismatch: {relative}")
                rows = read_jsonl(path)
                labels = Counter(row["label"] for row in rows)
                if len(rows) != total or labels != Counter({0: benign, 1: malicious}):
                    raise ValueError(f"Count/label mismatch: {relative}")
                for row in rows:
                    if row["task_id"] != stage or row["split"] != split or row["arm"] != ARMS[arm]:
                        raise ValueError(f"Stage/split/arm mismatch: {relative}")
                    identity = row["example_id"]
                    if identity in seen_ids:
                        raise ValueError(f"Duplicate example_id in arm: {identity}")
                    seen_ids.add(identity)
                self.rows[(stage, split)] = rows
        self.file_hashes = {
            key: manifest_lookup[key]["sha256"]
            for key in manifest_lookup
            if key.startswith(f"data/processed/stronger_pilot_candidate_b/{ARM_DIRS[arm]}/")
        }
        refs_path = self.root / "data/processed/stronger_pilot_candidate_b/external/references.json"
        refs_entry = self.manifest["external_reference_metadata"]
        if sha256_file(refs_path) != refs_entry["sha256"]:
            raise ValueError("External reference metadata hash mismatch")
        refs = json.loads(refs_path.read_text(encoding="utf-8"))
        bshort_path = self.root / refs["B-short"]["materialized_path"]
        if sha256_file(bshort_path) != refs["B-short"]["materialized_sha256"]:
            raise ValueError("B-short hash mismatch")
        self.bshort = read_jsonl(bshort_path)
        if len(self.bshort) != 120 or Counter(row["label"] for row in self.bshort) != Counter({0: 120}):
            raise ValueError("B-short contract mismatch")
        frozen = load_notinject(self.root)
        self.notinject = [
            {
                "text": row["prompt"], "label": 0, "example_id": row["record_id"],
                "task_id": "NotInject", "attack_family": "none", "injection_position": "none",
                "domain": "NotInject", "language": "multilingual" if row["category"] == "Multilingual" else "en",
                "hb_group": row["hb_group"], "category": row["category"], "source_split": row["source_split"],
            }
            for row in frozen
        ]
        if len(self.notinject) != 339:
            raise ValueError("NotInject contract mismatch")

    def records(self, stage: str, split: str) -> list[dict]:
        return self.rows[(stage, split)]


def artifact_integrity_preflight(root: Path) -> dict:
    """Validate the frozen materialization before any model or tokenizer is loaded."""
    root = Path(root).resolve()
    datasets = {arm: CandidateBData(root, arm) for arm in ARMS}
    all_ids = set()
    arm_counts = {}
    for arm, data in datasets.items():
        ids = {
            row["example_id"]
            for stage in STAGES
            for split in ("train", "val", "test")
            for row in data.records(stage, split)
        }
        arm_counts[ARMS[arm]] = len(ids)
        all_ids.update(ids)
    malicious_equal = {}
    for stage in STAGES:
        for split in ("train", "val", "test"):
            identities = [
                [row["example_id"] for row in datasets[arm].records(stage, split) if row["label"] == 1]
                for arm in ARMS
            ]
            malicious_equal[f"{stage}/{split}"] = all(value == identities[0] for value in identities[1:])
    stage1_anchor = {}
    stage2_anchor = {}
    for split in ("train", "val", "test"):
        identities = [
            [row["example_id"] for row in datasets[arm].records("AF2", split) if row["label"] == 0]
            for arm in ARMS
        ]
        stage1_anchor[split] = all(value == identities[0] for value in identities[1:])
        short_ids = [row["example_id"] for row in datasets["short"].records("AF3", split) if row["label"] == 0]
        hard_ids = [row["example_id"] for row in datasets["hard"].records("AF3", split) if row["label"] == 0]
        stage2_anchor[split] = short_ids == hard_ids
    manifest_path = root / "data/processed/stronger_pilot_candidate_b/materialization_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_path = root / manifest["validation_artifact"]
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    source_revision_records = (
        manifest.get("source_revision_hashes")
        or manifest.get("source_revisions")
        or validation.get("source_revision_hashes")
        or validation.get("source_revisions")
    )
    checks = {
        "manifest_sha256": sha256_file(manifest_path) == MATERIALIZATION_MANIFEST_SHA256,
        "validation_sha256": sha256_file(validation_path) == MATERIALIZATION_VALIDATION_SHA256,
        "expected_jsonl_files": len(manifest["files"]) == 36,
        "arm_memberships": arm_counts == {name: 2320 for name in ARMS.values()},
        "total_arm_memberships": sum(arm_counts.values()) == 6960,
        "unique_selected_records": len(all_ids) == 3770,
        "shared_malicious_identities": all(malicious_equal.values()),
        "stage1_D0_anchor": all(stage1_anchor.values()),
        "stage2_B0_short_hard_anchor": all(stage2_anchor.values()),
        "validation_status": validation.get("status") == "PASS_READY_FOR_HUMAN_NO_REPLAY_TRAINING_APPROVAL",
        "source_revision_records_present": bool(source_revision_records),
    }
    if not all(checks.values()):
        raise ValueError(f"Candidate B artifact-integrity preflight failed: {checks}")
    return {
        "status": "PASS", "checked_at": now(), "read_only_data_check": True,
        "manifest_sha256": MATERIALIZATION_MANIFEST_SHA256,
        "validation_sha256": MATERIALIZATION_VALIDATION_SHA256,
        "jsonl_files": 36, "arm_counts": arm_counts,
        "arm_memberships": sum(arm_counts.values()), "unique_selected_records": len(all_ids),
        "malicious_identity_checks": malicious_equal,
        "stage1_D0_anchor_checks": stage1_anchor,
        "stage2_B0_short_hard_anchor_checks": stage2_anchor,
        "source_revision_records": source_revision_records,
        "checks": checks,
        "transient_prior_capacity_message_artifact_issue": False,
    }



def predict_records(model, tokenizer, records: list[dict], config: dict, checkpoint_id: str, dataset: str) -> list[dict]:
    output = []
    batch_size = config["training"]["batch_size"]
    with isolated_evaluation(model):
        for offset in range(0, len(records), batch_size):
            batch = records[offset: offset + batch_size]
            inputs = batch_inputs(batch, tokenizer, 1280, "cuda")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
            predictions = logits.argmax(dim=-1).detach().cpu().tolist()
            for row, prediction, probability in zip(batch, predictions, probabilities):
                output.append({
                    "example_id": row["example_id"], "true_label": row["label"],
                    "predicted_label": int(prediction), "p_attack": float(probability),
                    "checkpoint_id": checkpoint_id, "dataset": dataset, "task_id": row.get("task_id"),
                    "attack_family": row.get("attack_family"), "injection_position": row.get("injection_position"),
                    "benign_group": row.get("benign_group"), "benign_partition": row.get("benign_partition"),
                    "language": row.get("language"), "domain": row.get("domain"),
                    "hb_group": row.get("hb_group"), "category": row.get("category"),
                })
    return output


def metric_block(predictions: list[dict]) -> dict:
    labels = [row["true_label"] for row in predictions]
    guesses = [row["predicted_label"] for row in predictions]
    metrics = classification_metrics(labels, guesses)
    metrics["attack_tpr"] = metrics.pop("attack_recall")
    metrics["attack_fnr"] = 1.0 - metrics["attack_tpr"] if metrics["attack_tpr"] is not None else None
    metrics["benign_tnr"] = 1.0 - metrics["benign_fpr"] if metrics["benign_fpr"] is not None else None
    scores = sorted(row["p_attack"] for row in predictions)
    def quantile(q: float) -> float:
        position = (len(scores) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(scores) - 1)
        weight = position - lower
        return scores[lower] * (1.0 - weight) + scores[upper] * weight
    metrics["mean_p_attack"] = statistics.mean(scores)
    metrics["p_attack_summary"] = {
        "mean": metrics["mean_p_attack"], "median": quantile(.5),
        "p05": quantile(.05), "p10": quantile(.10), "p90": quantile(.90),
        "p95": quantile(.95), "p99": quantile(.99),
    }
    metrics["fraction_p_attack_ge_0_99"] = sum(value >= .99 for value in scores) / len(scores)
    metrics["fraction_p_attack_le_0_01"] = sum(value <= .01 for value in scores) / len(scores)
    return metrics


def sliced_metric(predictions: list[dict], field: str, metric: str) -> dict:
    result = {}
    for value in sorted({row.get(field) for row in predictions if row.get(field) is not None}):
        subset = [row for row in predictions if row.get(field) == value]
        result[str(value)] = metric_block(subset)[metric]
    return result


def evaluate_stage_test(model, tokenizer, records, config, checkpoint_id, path: Path) -> dict:
    predictions = predict_records(model, tokenizer, records, config, checkpoint_id, "paired_stage_test")
    write_jsonl(path, predictions)
    malicious = [row for row in predictions if row["true_label"] == 1]
    benign = [row for row in predictions if row["true_label"] == 0]
    return {
        "paired": metric_block(predictions),
        "malicious": metric_block(malicious),
        "benign": metric_block(benign),
        "attack_tpr_by_position": sliced_metric(malicious, "injection_position", "attack_tpr"),
        "benign_fpr_by_group": sliced_metric(benign, "benign_group", "benign_fpr"),
        "benign_fpr_by_partition": sliced_metric(benign, "benign_partition", "benign_fpr"),
        "benign_fpr_by_language": sliced_metric(benign, "language", "benign_fpr"),
    }


def evaluate_external(model, tokenizer, records, config, checkpoint_id, dataset, path: Path) -> dict:
    predictions = predict_records(model, tokenizer, records, config, checkpoint_id, dataset)
    write_jsonl(path, predictions)
    result = {"overall": metric_block(predictions)}
    if dataset == "NotInject":
        result["fpr_by_hb_group"] = sliced_metric(predictions, "hb_group", "benign_fpr")
        result["fpr_by_category"] = sliced_metric(predictions, "category", "benign_fpr")
    else:
        english = [row for row in predictions if row.get("language") == "en"]
        multilingual = [row for row in predictions if row.get("language") != "en"]
        result["English"] = metric_block(english)
        result["multilingual_non_English"] = metric_block(multilingual)
    return result


def validation_loss(model, tokenizer, records, config) -> float:
    total_loss = 0.0
    total = 0
    with isolated_evaluation(model):
        for offset in range(0, len(records), 4):
            batch = records[offset: offset + 4]
            inputs = batch_inputs(batch, tokenizer, 1280, "cuda")
            labels = torch.tensor([row["label"] for row in batch], dtype=torch.long, device="cuda")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                loss = model(**inputs, labels=labels).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite validation loss")
            total_loss += float(loss.item()) * len(batch)
            total += len(batch)
    return total_loss / total


def mean_training_loss(log_path: Path) -> float:
    rows = read_jsonl(log_path)
    count = sum(row["examples_in_step"] for row in rows)
    return sum(row["loss"] * row["examples_in_step"] for row in rows) / count


def save_boundary_checkpoint(path: Path, model, tokenizer, payload: dict) -> dict:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    torch.save({**payload, "rng": rng_state()}, path / "state.pt")
    hashes = {item.name: sha256_file(item) for item in sorted(path.iterdir()) if item.is_file()}
    atomic_json(path / "checkpoint_manifest.json", {"optimizer_state_saved": False, "reason": "optimizer resets at every stage", "files": hashes})
    return hashes


def load_boundary_checkpoint(path: Path):
    path = Path(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=False, trust_remote_code=False)
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    return model, tokenizer, state


def approved_config(root: Path, seed: int) -> dict:
    config = load_config(Path(root) / "configs/stronger_pilot_no_replay.yaml")
    config["seed"] = seed
    validate_config(config)
    return config


def run_preflight(root: Path) -> Path:
    root = Path(root).resolve()
    if root != Path(r"E:\ContinualGuard").resolve():
        raise ValueError(f"Wrong active project root: {root}")
    output = root / "results/stronger_pilot_no_replay/preflight"
    result_path = output / "preflight.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous.get("status") == "PASS" and previous.get("manifest_sha256") == MATERIALIZATION_MANIFEST_SHA256:
            return output
        raise ValueError("Existing preflight is not a reusable PASS artifact")
    output.mkdir(parents=True, exist_ok=False)
    integrity = artifact_integrity_preflight(root)
    atomic_json(output / "artifact_integrity_preflight.json", integrity)
    result = {"started_at": now(), "status": "RUNNING", "manifest_sha256": MATERIALIZATION_MANIFEST_SHA256}
    started = time.perf_counter()
    model = tokenizer = optimizer = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        config = approved_config(root, 42)
        datasets = {arm: CandidateBData(root, arm) for arm in ARMS}
        for stage in STAGES:
            malicious = [
                [row["example_id"] for row in datasets[arm].records(stage, "test") if row["label"] == 1]
                for arm in ARMS
            ]
            if any(value != malicious[0] for value in malicious[1:]):
                raise ValueError(f"Matched malicious test mismatch: {stage}")
        stage1_benign = [
            [row["example_id"] for row in datasets[arm].records("AF2", "test") if row["label"] == 0]
            for arm in ARMS
        ]
        if any(value != stage1_benign[0] for value in stage1_benign[1:]):
            raise ValueError("Stage-1 D0 anchor mismatch")
        short_stage2 = [row["example_id"] for row in datasets["short"].records("AF3", "test") if row["label"] == 0]
        hard_stage2 = [row["example_id"] for row in datasets["hard"].records("AF3", "test") if row["label"] == 0]
        if short_stage2 != hard_stage2:
            raise ValueError("Stage-2 B0 anchor mismatch")
        seed_everything(42, True, 4)
        tokenizer = make_tokenizer(config["model"], root)
        all_train = [row for stage in STAGES for row in datasets["doc"].records(stage, "train")]
        selected = sorted(all_train, key=lambda row: row["deberta_tokens"], reverse=True)[:4]
        encoded = [tokenizer.encode(row["text"], add_special_tokens=True, truncation=False) for row in selected]
        if any(len(ids) > 1280 for ids in encoded):
            raise ValueError("Preflight batch violates max_length")
        model = make_model(config["model"], root)
        initial = state_fingerprint(model)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.to("cuda").train()
        optimizer, scheduler = make_optimizer(model, config)
        inputs = batch_inputs(selected, tokenizer, 1280, "cuda")
        labels = torch.tensor([row["label"] for row in selected], dtype=torch.long, device="cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            loss = model(**inputs, labels=labels).loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            raise RuntimeError("Preflight produced nonfinite loss or gradient")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        torch.cuda.synchronize()
        checkpoint = output / "checkpoint"
        checkpoint_hashes = save_boundary_checkpoint(checkpoint, model, tokenizer, {
            "purpose": "software preflight only", "seed": 42, "global_step": 1,
            "manifest_sha256": MATERIALIZATION_MANIFEST_SHA256, "initialization": initial,
        })
        del model, tokenizer
        model = tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        loaded_model, loaded_tokenizer, state = load_boundary_checkpoint(checkpoint)
        loaded_model.to("cuda")
        sample_predictions = predict_records(loaded_model, loaded_tokenizer, selected, config, "preflight", "preflight")
        if len(sample_predictions) != 4 or state["global_step"] != 1:
            raise RuntimeError("Checkpoint save/load or evaluation path failed")
        result.update({
            "status": "PASS", "finished_at": now(), "gpu": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda, "batch_size": 4, "max_length": 1280,
            "selected_token_lengths": [len(ids) for ids in encoded], "loss": float(loss.item()),
            "gradient_norm": float(grad_norm.item()), "step_seconds": time.perf_counter() - step_started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "checkpoint_hashes": checkpoint_hashes, "checkpoint_reload": True,
            "evaluation_records": len(sample_predictions), "initialization": initial,
            "dataset_loading": True, "forward_backward": True, "cuda_oom": False,
            "artifact_integrity_preflight": "PASS",
        })
    except Exception as exc:
        result.update(status="FAIL", finished_at=now(), error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["runtime_seconds"] = time.perf_counter() - started
        atomic_json(result_path, result)
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return output


def _stage_directories(run_dir: Path) -> list[Path]:
    completed = []
    for index in range(1, 5):
        path = run_dir / f"stage{index}"
        if path.is_dir() and (path / "stage_metadata.json").exists() and (path / "checkpoint/state.pt").exists():
            completed.append(path)
        elif path.exists():
            raise ValueError(f"Incomplete published stage directory: {path}")
        elif completed:
            continue
    expected = list(range(1, len(completed) + 1))
    observed = [int(path.name.replace("stage", "")) for path in completed]
    if observed != expected:
        raise ValueError("Published stage directories are not a contiguous prefix")
    return completed


def _run_metadata_base(root: Path, arm: str, seed: int, config: dict, data: CandidateBData) -> dict:
    env = environment(root)
    return {
        "run_id": f"candidate-b-no-replay-{arm}-seed{seed}", "started_at": now(), "status": "RUNNING",
        "active_root": str(root), "interpreter": os.sys.executable, "arm": ARMS[arm], "arm_key": arm,
        "seed": seed, "stage_order": list(STAGES), "completed_stages": 0,
        "model_checkpoint": config["model"]["checkpoint"], "model_revision": REAL_REVISION,
        "model_weights_sha256": REAL_MODEL_SHA256, "max_length": 1280,
        "label_mapping": {"0": "benign", "1": "malicious_indirect_prompt_injection"},
        "threshold_rule": THRESHOLD_RULE, "training_policy": copy.deepcopy(config["training"]),
        "current_stage_only": True, "model_weights_carried": True, "optimizer_reset_each_stage": True,
        "replay": False, "selective_adaptation": False, "numerical_memory_budget": None,
        "materialization_manifest_sha256": MATERIALIZATION_MANIFEST_SHA256,
        "materialization_validation_sha256": MATERIALIZATION_VALIDATION_SHA256,
        "dataset_hashes": data.file_hashes, "environment": env,
        "warnings": [
            "AF5 is a compositional/combined shift, not a family-disjoint family",
            "BH-1/BH-2 are source-defined reviewed hard-benign partitions, not ordinal difficulty levels",
            "B-DOC, B-SHORT, and B-HARD differ in source/language/format after their declared anchors",
            "B-short language composition is unmatched; NotInject minus B-short is not a pure trigger-word causal effect",
            "Chen-derived malicious data are a deterministic project reconstruction, not an exact paper reproduction",
        ],
    }


def run_baseline(root: Path, arm: str, seed: int) -> Path:
    root = Path(root).resolve()
    if root != Path(r"E:\ContinualGuard").resolve():
        raise ValueError(f"Wrong active project root: {root}")
    preflight = root / "results/stronger_pilot_no_replay/preflight/preflight.json"
    if not preflight.exists() or json.loads(preflight.read_text(encoding="utf-8")).get("status") != "PASS":
        raise ValueError("PASS technical preflight is required before research execution")
    config = approved_config(root, seed)
    data = CandidateBData(root, arm)
    run_dir = root / "results/stronger_pilot_no_replay" / arm / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = run_dir / "resolved_config.yaml"
    if not resolved_config.exists():
        resolved_config.write_text(yaml.safe_dump({**config, "execution_arm": ARMS[arm]}, sort_keys=True), encoding="utf-8")
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else _run_metadata_base(root, arm, seed, config, data)
    if metadata.get("status") == "COMPLETE":
        return run_dir
    completed = _stage_directories(run_dir)
    metadata["status"] = "RUNNING"
    metadata["resumed_from_stage"] = len(completed) if completed else None
    atomic_json(metadata_path, metadata)
    model = tokenizer = None
    stage_started = None
    try:
        seed_everything(seed, True, 4)
        if completed:
            model, tokenizer, state = load_boundary_checkpoint(completed[-1] / "checkpoint")
            if state["seed"] != seed or state["arm"] != arm or state["completed_stages"] != len(completed):
                raise ValueError("Resume checkpoint lineage mismatch")
            restore_rng(state["rng"])
            global_step = state["global_step"]
            initialization = state["initialization"]
        else:
            tokenizer = make_tokenizer(config["model"], root)
            model = make_model(config["model"], root)
            initialization = state_fingerprint(model)
            atomic_json(run_dir / "initialization.json", initialization)
            global_step = 0
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.to("cuda")
        metadata["initialization"] = initialization
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
            optimizer, scheduler = make_optimizer(model, config)
            atomic_json(attempt / "stage_start.json", {
                "stage": stage, "stage_index": stage_index, "started_at": now(),
                "optimizer_state_entries_before_training": len(optimizer.state), "optimizer_reset": True,
                "model_weights_carried": stage_index > 1, "current_stage_training_records": 400,
                "historical_training_records": 0,
            })
            global_step = train_stage(
                model, tokenizer, data.records(stage, "train"), config, optimizer, scheduler,
                attempt, stage, global_step, allowed_task_ids=(stage,),
            )
            evaluations = {}
            checkpoint_id = f"{ARMS[arm]}:seed{seed}:stage{stage_index}"
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
                "arm": ARMS[arm], "seed": seed, "trained_stage": stage, "stage_index": stage_index,
                "evaluations": evaluations, "external": externals,
                "acquisition": {
                    "attack_tpr": current["malicious"]["attack_tpr"],
                    "benign_tnr": current["benign"]["benign_tnr"],
                    "benign_fpr": current["benign"]["benign_fpr"],
                    "paired_macro_f1": current["paired"]["macro_f1"],
                    "paired_accuracy": current["paired"]["accuracy"],
                },
                "training_loss": mean_training_loss(attempt / "train_log.jsonl"), "validation_loss": val_loss,
                "runtime_seconds": runtime, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            atomic_json(attempt / "metrics.json", stage_metrics)
            checkpoint_hashes = save_boundary_checkpoint(attempt / "checkpoint", model, tokenizer, {
                "experiment": "Candidate B stronger-pilot no-replay", "arm": arm, "arm_name": ARMS[arm], "seed": seed,
                "completed_stages": stage_index, "stage": stage, "global_step": global_step,
                "initialization": initialization, "config_sha256": sha256_file(root / "configs/stronger_pilot_no_replay.yaml"),
                "materialization_manifest_sha256": MATERIALIZATION_MANIFEST_SHA256,
                "dataset_hashes": data.file_hashes, "replay": False, "historical_training_records": 0,
            })
            stage_metadata = {
                "status": "COMPLETE", "completed_at": now(), "stage": stage, "stage_index": stage_index,
                "runtime_seconds": runtime, "global_step": global_step, "checkpoint_hashes": checkpoint_hashes,
                "optimizer_state_saved": False, "optimizer_reset_next_stage": True,
            }
            atomic_json(attempt / "stage_metadata.json", stage_metadata)
            final_stage_dir = run_dir / f"stage{stage_index}"
            os.replace(attempt, final_stage_dir)
            metadata["completed_stages"] = stage_index
            metadata["last_completed_stage"] = stage
            metadata["runtime_seconds"] = sum(json.loads((run_dir / f"stage{i}/stage_metadata.json").read_text(encoding="utf-8"))["runtime_seconds"] for i in range(1, stage_index + 1))
            atomic_json(metadata_path, metadata)
            print(json.dumps({"event": "stage_complete", "arm": arm, "seed": seed, "stage": stage, "runtime_seconds": runtime, "peak_gpu_memory_bytes": stage_metrics["peak_gpu_memory_bytes"]}), flush=True)
        metadata["status"] = "COMPLETE"
        metadata["finished_at"] = now()
        atomic_json(metadata_path, metadata)
        print(json.dumps({"event": "run_complete", "arm": arm, "seed": seed, "runtime_seconds": metadata["runtime_seconds"]}), flush=True)
    except Exception as exc:
        metadata["status"] = "FAILED"
        metadata["failed_at"] = now()
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        metadata["failed_stage_elapsed_seconds"] = time.perf_counter() - stage_started if stage_started is not None else None
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




