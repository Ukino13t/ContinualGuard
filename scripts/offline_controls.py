"""Route A offline union reference and compute-matched current-only control."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
import yaml

from continualguard.artifacts import append_jsonl, now
from continualguard.loss_aware_control import initialization_reference
from continualguard.model import batch_inputs, make_model, make_tokenizer
from continualguard.reproducibility import restore_rng, seed_everything
from continualguard.stronger_pilot_no_replay import (
    ARMS, SEEDS, STAGES, CandidateBData, MATERIALIZATION_MANIFEST_SHA256,
    atomic_json, evaluate_external, evaluate_stage_test, load_boundary_checkpoint,
    save_boundary_checkpoint, sha256_file, state_fingerprint, validation_loss,
)
from continualguard.train import make_optimizer, train_stage
from continualguard.stronger_pilot_no_replay import approved_config as baseline_config

OUTPUT = Path("results/extended/offline_controls")


def compute_matched_stage(model, tokenizer, rows, config, optimizer, scheduler,
                          out: Path, stage: str, step: int) -> dict:
    """100 updates, two current-stage batches/update; 800 current exposures.

    Replay trains on 400 current + 400 historical exposures at these same
    100 update steps. This control substitutes a second shuffled current-stage
    pass for historical replay and averages the two CE losses. It is an
    optimization-compute control, not a statistically identical sample budget.
    """
    if len(rows) != 400 or any(x["split"] != "train" or x["task_id"] != stage for x in rows):
        raise ValueError("Current-stage TRAIN only")
    model.train()
    first = torch.randperm(len(rows)).tolist()
    second = torch.randperm(len(rows)).tolist()
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)
    optimizer.zero_grad(set_to_none=True)
    total = 0.
    for offset in range(0, 400, 4):
        batches = [[rows[i] for i in order[offset:offset + 4]] for order in (first, second)]
        losses = []
        for batch in batches:
            inputs = batch_inputs(batch, tokenizer, 1280, "cuda")
            labels = torch.tensor([x["label"] for x in batch], dtype=torch.long, device="cuda")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                losses.append(model(**inputs, labels=labels).loss)
        loss = (losses[0] + losses[1]) / 2
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite gradient norm")
        scaler.step(optimizer); scaler.update(); scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        total += float(loss.item())
        append_jsonl(out / "train_log.jsonl", {
            "stage": stage, "global_step": step, "loss": float(loss.item()),
            "current_example_ids": [x["example_id"] for x in batches[0]],
            "repeated_current_example_ids": [x["example_id"] for x in batches[1]],
            "historical_examples": 0, "examples_in_step": 8})
    return {"global_step": step, "training_loss": total / 100,
            "optimizer_steps": 100, "current_examples_exposed": 800,
            "historical_examples_exposed": 0}


def run(root: Path, control: str, arm: str, seed: int) -> Path:
    root = root.resolve()
    if root != Path(r"E:\ContinualGuard").resolve():
        raise ValueError("Wrong active root")
    if control not in ("joint_union_1epoch", "compute_matched_current_only") or arm not in ARMS or seed not in SEEDS:
        raise ValueError("Unknown control/arm/seed")
    data = CandidateBData(root, arm)
    config = baseline_config(root, seed)
    out = root / OUTPUT / control / arm / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    resolved = out / "resolved_config.yaml"
    if not resolved.exists():
        resolved.write_text(yaml.safe_dump({**config, "control": control, "arm": ARMS[arm]}, sort_keys=True), encoding="utf-8")
    meta_path = out / "run_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {
        "control": control, "arm": ARMS[arm], "seed": seed, "status": "RUNNING",
        "started_at": now(), "materialization_manifest_sha256": MATERIALIZATION_MANIFEST_SHA256,
        "dataset_hashes": data.file_hashes,
        "runner_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(root / "configs/stronger_pilot_no_replay.yaml"),
        "full_history_train_access": control == "joint_union_1epoch",
        "future_stage_train_access": control == "joint_union_1epoch",
        "validation_test_external_train_access": False,
        "optimization_budget": ("400 steps, 1600 unique train examples, one epoch" if control == "joint_union_1epoch"
                                else "400 steps, 3200 current-stage exposures, no historical examples"),
    }
    if meta["status"] == "COMPLETE":
        return out
    atomic_json(meta_path, meta)
    model = tokenizer = None
    try:
        seed_everything(seed, True, 4)
        stages_done = 0
        if control == "compute_matched_current_only":
            stages_done = sum((out / f"stage{i}" / "checkpoint/state.pt").exists() for i in range(1, 5))
            if stages_done:
                model, tokenizer, state = load_boundary_checkpoint(out / f"stage{stages_done}" / "checkpoint")
                if (state["control"], state["arm"], state["seed"], state["completed_stages"]) != (control, arm, seed, stages_done):
                    raise ValueError("Resume lineage mismatch")
                restore_rng(state["rng"])
                step, initialization = state["global_step"], state["initialization"]
        if model is None:
            tokenizer = make_tokenizer(config["model"], root)
            model = make_model(config["model"], root)
            initialization = state_fingerprint(model)
            if initialization != initialization_reference(root, arm, seed):
                raise ValueError("Initialization differs from frozen seed")
            step = 0
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.to("cuda")
        if control == "joint_union_1epoch":
            if (out / "joint" / "checkpoint/state.pt").exists():
                raise ValueError("Existing unpublished joint artifact requires manual inspection")
            attempt = out / f"_attempt_joint_{os.getpid()}"
            attempt.mkdir(parents=False, exist_ok=False); (attempt / "predictions").mkdir()
            rows = [x for task in STAGES for x in data.records(task, "train")]
            if len(rows) != 1600 or len({x["example_id"] for x in rows}) != 1600:
                raise ValueError("Joint union train set identity contract failed")
            started = time.perf_counter(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            optimizer, scheduler = make_optimizer(model, config)
            step = train_stage(model, tokenizer, rows, config, optimizer, scheduler,
                               attempt, "JOINT_AF2_AF5", step, allowed_task_ids=STAGES)
            evaluations = {}
            checkpoint_id = f"joint:{ARMS[arm]}:seed{seed}"
            for task in STAGES:
                evaluations[task] = evaluate_stage_test(model, tokenizer, data.records(task, "test"), config,
                    checkpoint_id, attempt / "predictions" / f"{task}.jsonl")
            external = {
                "NotInject": evaluate_external(model, tokenizer, data.notinject, config, checkpoint_id,
                    "NotInject", attempt / "predictions/NotInject.jsonl"),
                "B-short": evaluate_external(model, tokenizer, data.bshort, config, checkpoint_id,
                    "B-short", attempt / "predictions/B-short.jsonl"),
            }
            torch.cuda.synchronize(); elapsed = time.perf_counter() - started
            atomic_json(attempt / "metrics.json", {"control": control, "arm": ARMS[arm], "seed": seed,
                "evaluations": evaluations, "external": external, "runtime_seconds": elapsed,
                "optimizer_steps": step, "train_examples_unique": len(rows),
                "train_examples_exposed": len(rows), "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()})
            hashes = save_boundary_checkpoint(attempt / "checkpoint", model, tokenizer, {
                "control": control, "arm": arm, "seed": seed, "initialization": initialization,
                "global_step": step, "materialization_manifest_sha256": MATERIALIZATION_MANIFEST_SHA256,
                "dataset_hashes": data.file_hashes, "runner_sha256": meta["runner_sha256"]})
            atomic_json(attempt / "stage_metadata.json", {"status": "COMPLETE", "runtime_seconds": elapsed,
                "checkpoint_hashes": hashes})
            os.replace(attempt, out / "joint")
            meta["runtime_seconds"] = elapsed
            print(json.dumps({"event": "joint_complete", "arm": arm, "seed": seed, "runtime_seconds": elapsed}), flush=True)
        else:
            for index in range(stages_done + 1, 5):
                task = STAGES[index - 1]
                attempt = out / f"_attempt_stage{index}_{os.getpid()}"
                attempt.mkdir(parents=False, exist_ok=False); (attempt / "predictions").mkdir()
                started = time.perf_counter(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                optimizer, scheduler = make_optimizer(model, config)
                training = compute_matched_stage(model, tokenizer, data.records(task, "train"), config,
                                                 optimizer, scheduler, attempt, task, step)
                step = training["global_step"]
                checkpoint_id = f"compute-matched:{ARMS[arm]}:seed{seed}:stage{index}"
                evaluations = {name: evaluate_stage_test(model, tokenizer, data.records(name, "test"), config,
                    checkpoint_id, attempt / "predictions" / f"{name}.jsonl") for name in STAGES[:index]}
                external = {
                    "NotInject": evaluate_external(model, tokenizer, data.notinject, config, checkpoint_id,
                        "NotInject", attempt / "predictions/NotInject.jsonl"),
                    "B-short": evaluate_external(model, tokenizer, data.bshort, config, checkpoint_id,
                        "B-short", attempt / "predictions/B-short.jsonl"),
                }
                val = validation_loss(model, tokenizer, data.records(task, "val"), config)
                torch.cuda.synchronize(); elapsed = time.perf_counter() - started
                acquired = evaluations[task]
                atomic_json(attempt / "metrics.json", {"control": control, "arm": ARMS[arm], "seed": seed,
                    "trained_stage": task, "stage_index": index, "evaluations": evaluations, "external": external,
                    "acquisition": {"attack_tpr": acquired["malicious"]["attack_tpr"],
                                    "benign_fpr": acquired["benign"]["benign_fpr"],
                                    "paired_macro_f1": acquired["paired"]["macro_f1"]},
                    "training": training, "validation_loss": val, "runtime_seconds": elapsed,
                    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()})
                hashes = save_boundary_checkpoint(attempt / "checkpoint", model, tokenizer, {
                    "control": control, "arm": arm, "seed": seed, "completed_stages": index,
                    "initialization": initialization, "global_step": step,
                    "materialization_manifest_sha256": MATERIALIZATION_MANIFEST_SHA256,
                    "dataset_hashes": data.file_hashes, "runner_sha256": meta["runner_sha256"]})
                atomic_json(attempt / "stage_metadata.json", {"status": "COMPLETE", "stage": task,
                    "stage_index": index, "runtime_seconds": elapsed, "checkpoint_hashes": hashes})
                os.replace(attempt, out / f"stage{index}")
                meta["completed_stages"] = index
                meta["runtime_seconds"] = sum(json.loads((out / f"stage{i}" / "stage_metadata.json").read_text(encoding="utf-8"))["runtime_seconds"] for i in range(1, index + 1))
                atomic_json(meta_path, meta)
                print(json.dumps({"event": "stage_complete", "control": control, "arm": arm,
                                  "seed": seed, "stage": task, "runtime_seconds": elapsed}), flush=True)
        meta.update(status="COMPLETE", finished_at=now())
        atomic_json(meta_path, meta)
    except Exception as exc:
        meta.update(status="FAILED", failed_at=now(), error=f"{type(exc).__name__}: {exc}")
        atomic_json(meta_path, meta)
        raise
    finally:
        if model is not None: del model
        if tokenizer is not None: del tokenizer
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"E:\ContinualGuard"))
    parser.add_argument("--control", choices=("joint_union_1epoch", "compute_matched_current_only"), required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    a = parser.parse_args()
    print(run(a.root, a.control, a.arm, a.seed))


if __name__ == "__main__":
    main()
