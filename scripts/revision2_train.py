"""Route A Revision 2 matched-rank replay training; frozen outputs untouched."""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import torch
import yaml

from continualguard.artifacts import now
from continualguard.loss_aware_control import (
    approved_config, initialization_reference, record_lookup, stage_directories,
)
from continualguard.model import make_model, make_tokenizer
from continualguard.replay_candidate_b import train_stage_with_replay
from continualguard.reproducibility import environment, restore_rng, seed_everything
from continualguard.stronger_pilot_no_replay import (
    ARMS, SEEDS, STAGES, CandidateBData, MATERIALIZATION_MANIFEST_SHA256,
    atomic_json, evaluate_external, evaluate_stage_test, load_boundary_checkpoint,
    save_boundary_checkpoint, sha256_file, state_fingerprint, validation_loss,
    write_jsonl,
)
from continualguard.train import make_optimizer
from revision2_selector import (
    MatchedBuffer, VARIANTS, ORDER_NAMESPACE, DROPOUT_NAMESPACE,
    ranked_entry, stream_seed,
)

CAPACITY = 200
OUTPUT = Path("results/revision2/matched_ablation_amp256")
AMP_INIT_SCALE = 256.0


def composition(buffer: MatchedBuffer, quota=None) -> dict:
    rows = buffer.entries
    return {"size": len(rows), "capacity": buffer.capacity,
            "by_label": dict(Counter(str(x["label"]) for x in rows)),
            "by_stage": dict(Counter(x["task_id"] for x in rows)),
            "by_label_stage": dict(Counter(f"{x['label']}/{x['task_id']}" for x in rows)),
            "coverage_keys": len({x["coverage_key"] for x in rows}),
            "quota_fulfillment": quota, "raw_text_stored": False}


def run(root: Path, variant: str, arm: str, seed: int, *, capacity: int = CAPACITY,
        smoke: bool = False, stop_after_stage: int = 1) -> Path:
    root = root.resolve()
    if root != Path(r"E:\ContinualGuard").resolve():
        raise ValueError("Wrong active project root")
    if variant not in VARIANTS or arm not in ARMS or seed not in SEEDS or capacity not in (50, 200, 400):
        raise ValueError("Unknown variant, arm or seed")
    config = approved_config(root, seed)
    # The frozen trainer accepts `loss_aware` as the name of the identical
    # (current CE + replay CE)/2 objective. Selection method is recorded apart.
    config["method"] = "loss_aware"
    config["replay"]["memory_capacity_examples"] = capacity
    data = CandidateBData(root, arm)
    lookup = record_lookup(data)
    run_dir = root / OUTPUT / f"M{capacity}" / variant / arm / f"seed{seed}"
    if smoke:
        run_dir = root / OUTPUT / "smoke" / f"M{capacity}" / variant / arm / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = run_dir / "resolved_config.yaml"
    if not resolved.exists():
        resolved.write_text(yaml.safe_dump({**config, "selection_method": variant,
                                            "execution_arm": ARMS[arm],
                                            "revision2_rank_namespace": "route-a-revision2-shared-rank-v1"},
                                           sort_keys=True), encoding="utf-8")
    meta_path = run_dir / "run_metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {
        "run_id": f"route-a-revision2-{variant}-M{capacity}-{arm}-seed{seed}", "status": "RUNNING",
        "started_at": now(), "active_root": str(root), "interpreter": os.sys.executable,
        "variant": variant, "arm": ARMS[arm], "arm_key": arm, "seed": seed,
        "stage_order": list(STAGES), "completed_stages": 0,
        "model_revision": config["model"]["revision"], "training_policy": config["training"],
        "replay_policy": config["replay"], "memory_capacity_examples": capacity,
        "materialization_manifest_sha256": MATERIALIZATION_MANIFEST_SHA256,
        "dataset_hashes": data.file_hashes, "environment": environment(root),
        "runner_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(resolved),
        "selection_pool": "previous buffer plus current-stage TRAIN only",
        "rng_specification": {"selection": "SHA256(seed, route-a-revision2-shared-rank-v1, example_id)",
                              "current_order": ORDER_NAMESPACE, "dropout": DROPOUT_NAMESPACE,
                              "replay_sampling": "independent Python Random route-a-revision2-replay-v1"},
        "amp_grad_scaler_initial_scale": AMP_INIT_SCALE,
    }
    if metadata.get("status") == "COMPLETE":
        return run_dir
    completed = stage_directories(run_dir)
    metadata.update(status="RUNNING", resumed_from_stage=len(completed))
    atomic_json(meta_path, metadata)
    model = tokenizer = None
    try:
        seed_everything(seed, True, 4)
        if completed:
            model, tokenizer, state = load_boundary_checkpoint(completed[-1] / "checkpoint")
            if (state["method"], state["arm"], state["seed"], state["completed_stages"]) != (variant, arm, seed, len(completed)):
                raise ValueError("Resume lineage mismatch")
            buffer = MatchedBuffer.from_state_dict(state["buffer_state"])
            restore_rng(state["rng"])
            global_step, initialization = state["global_step"], state["initialization"]
        else:
            tokenizer = make_tokenizer(config["model"], root)
            model = make_model(config["model"], root)
            initialization = state_fingerprint(model)
            if initialization != initialization_reference(root, arm, seed):
                raise ValueError("Initialization differs from frozen matched seed")
            atomic_json(run_dir / "initialization.json", initialization)
            buffer, global_step = MatchedBuffer(variant, seed, capacity), 0
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.to("cuda")
        for stage_index in range(len(completed) + 1, 5):
            if shutil.disk_usage(root).free < 3_500_000_000:
                raise RuntimeError("E: free-space guard: fewer than 3.5 billion bytes remain")
            stage = STAGES[stage_index - 1]
            attempt = run_dir / f"_attempt_stage{stage_index}_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{os.getpid()}"
            attempt.mkdir(parents=False, exist_ok=False)
            (attempt / "predictions").mkdir()
            started = time.perf_counter()
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            current = data.records(stage, "train")
            candidates = [ranked_entry(row, seed) for row in current]
            order_generator = torch.Generator(device="cpu")
            order_generator.manual_seed(stream_seed(seed, ORDER_NAMESPACE, stage))
            stochastic_seed = stream_seed(seed, DROPOUT_NAMESPACE, stage)
            torch.manual_seed(stochastic_seed)
            torch.cuda.manual_seed_all(stochastic_seed)
            optimizer, scheduler = make_optimizer(model, config)
            atomic_json(attempt / "stage_start.json", {"stage": stage, "stage_index": stage_index,
                        "optimizer_reset": len(optimizer.state) == 0, "current_stage_training_records": len(current),
                        "historical_memory_records": len(buffer.entries), "buffer_updated_after_training": True})
            global_step, _, training = train_stage_with_replay(
                model, tokenizer, current, config, optimizer, scheduler, buffer,
                lookup, attempt, stage, stage_index, global_step,
                order_generator=order_generator,
                grad_scaler_init_scale=AMP_INIT_SCALE)
            quota = buffer.update(candidates, tuple(STAGES[:stage_index]))
            write_jsonl(attempt / "buffer.jsonl", buffer.entries)
            after = composition(buffer, quota)
            atomic_json(attempt / "memory_composition.json", after)
            checkpoint_id = f"{variant}:{ARMS[arm]}:seed{seed}:stage{stage_index}"
            evaluations = {}
            for task in STAGES[:stage_index]:
                evaluations[task] = evaluate_stage_test(model, tokenizer, data.records(task, "test"),
                    config, checkpoint_id, attempt / "predictions" / f"{task}.jsonl")
            external = {
                "NotInject": evaluate_external(model, tokenizer, data.notinject, config, checkpoint_id,
                    "NotInject", attempt / "predictions/NotInject.jsonl"),
                "B-short": evaluate_external(model, tokenizer, data.bshort, config, checkpoint_id,
                    "B-short", attempt / "predictions/B-short.jsonl"),
            }
            val = validation_loss(model, tokenizer, data.records(stage, "val"), config)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            acquired = evaluations[stage]
            metrics = {"method": variant, "arm": ARMS[arm], "seed": seed,
                       "trained_stage": stage, "stage_index": stage_index,
                       "evaluations": evaluations, "external": external,
                       "acquisition": {"attack_tpr": acquired["malicious"]["attack_tpr"],
                                       "benign_fpr": acquired["benign"]["benign_fpr"],
                                       "benign_tnr": acquired["benign"]["benign_tnr"],
                                       "paired_macro_f1": acquired["paired"]["macro_f1"],
                                       "paired_accuracy": acquired["paired"]["accuracy"]},
                       "training": training, "training_loss": training["training_loss"],
                       "selector_diagnostics": quota,
                       "validation_loss": val, "buffer_after_training": after,
                       "runtime_seconds": elapsed, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()}
            atomic_json(attempt / "metrics.json", metrics)
            hashes = save_boundary_checkpoint(attempt / "checkpoint", model, tokenizer, {
                "experiment": "Route A structured ablation", "method": variant, "arm": arm,
                "seed": seed, "completed_stages": stage_index, "stage": stage,
                "global_step": global_step, "initialization": initialization,
                "config_sha256": metadata["config_sha256"],
                "runner_sha256": metadata["runner_sha256"],
                "materialization_manifest_sha256": MATERIALIZATION_MANIFEST_SHA256,
                "dataset_hashes": data.file_hashes, "buffer_state": buffer.state_dict(),
            })
            atomic_json(attempt / "stage_metadata.json", {"status": "COMPLETE", "stage": stage,
                "stage_index": stage_index, "runtime_seconds": elapsed, "global_step": global_step,
                "checkpoint_hashes": hashes, "buffer_size": len(buffer.entries),
                "replay_examples_exposed": training["replay_examples_exposed"]})
            os.replace(attempt, run_dir / f"stage{stage_index}")
            metadata["completed_stages"] = stage_index
            metadata["runtime_seconds"] = sum(json.loads((run_dir / f"stage{i}/stage_metadata.json").read_text(encoding="utf-8"))["runtime_seconds"] for i in range(1, stage_index + 1))
            atomic_json(meta_path, metadata)
            print(json.dumps({"event": "stage_complete", "variant": variant, "arm": arm,
                              "seed": seed, "stage": stage, "runtime_seconds": elapsed}), flush=True)
            if smoke and stage_index >= stop_after_stage:
                break
        metadata.update(status="SMOKE_COMPLETE" if smoke and metadata["completed_stages"] < 4 else "COMPLETE",
                        finished_at=now())
        atomic_json(meta_path, metadata)
    except Exception as exc:
        metadata.update(status="FAILED", failed_at=now(), error=f"{type(exc).__name__}: {exc}")
        atomic_json(meta_path, metadata)
        raise
    finally:
        if model is not None: del model
        if tokenizer is not None: del tokenizer
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"E:\ContinualGuard"))
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--capacity", type=int, choices=(50, 200, 400), default=200)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stop-after-stage", type=int, choices=(1, 2, 3, 4), default=1)
    args = parser.parse_args()
    print(run(args.root, args.variant, args.arm, args.seed, capacity=args.capacity,
              smoke=args.smoke, stop_after_stage=args.stop_after_stage))


if __name__ == "__main__":
    main()
