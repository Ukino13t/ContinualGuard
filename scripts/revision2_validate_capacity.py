"""Independent source/access and reservoir-selection check for P1 capacity runs."""
from __future__ import annotations

import json
import random
from pathlib import Path

from revision2_validate import (
    ROOT, STAGES, checkpoint, jsonl, load, predictions, sha256, source_rows,
    validate_matched,
)

BASE = ROOT / "results/revision2/memory_sensitivity_amp256"


def reservoir(arm: str, seed: int, cap: int, full_hash: bool) -> dict | None:
    folder = BASE / f"M{cap}/reservoir_capacity" / arm / f"seed{seed}"
    metadata = folder / "run_metadata.json"
    if not metadata.is_file() or load(metadata)["status"] != "COMPLETE":
        return None
    info = load(metadata)
    assert info["memory_capacity_examples"] == cap
    assert info["amp_grad_scaler_initial_scale"] == 256.0
    assert info["config_sha256"] == sha256(folder / "resolved_config.yaml")
    assert info["materialization_manifest_sha256"] == sha256(
        ROOT / "data/processed/stronger_pilot_candidate_b/materialization_manifest.json")
    runner = ROOT / "scripts/revision2_reservoir_capacity.py"
    assert info["runner_sha256"] == sha256(runner)
    train, test = source_rows(arm)
    update_rng = random.Random(seed * 1009 + 11)  # Core METHOD_CODES['reservoir'].
    expected: list[str] = []
    previous: set[str] = set()
    seen_count = checked_files = prediction_rows = 0
    for i, stage in enumerate(STAGES, 1):
        path = folder / f"stage{i}"
        metrics = load(path / "metrics.json")
        assert metrics["stage_index"] == i and metrics["trained_stage"] == stage
        checked_files += checkpoint(path, full_hash)
        logs = jsonl(path / "train_log.jsonl")
        assert len(logs) == 100
        current = [identity for row in logs for identity in row["example_ids"]]
        replay = [identity for row in logs for identity in row["replay_example_ids"]]
        assert len(current) == 400 and set(current) == set(train[stage])
        assert len(replay) == (0 if i == 1 else 400) and set(replay) <= previous
        assert metrics["training"]["optimizer_steps"] == 100
        for identity in current:
            seen_count += 1
            if len(expected) < cap:
                expected.append(identity)
            else:
                index = update_rng.randrange(seen_count)
                if index < cap:
                    expected[index] = identity
        buffer = jsonl(path / "buffer.jsonl")
        actual = [entry["example_id"] for entry in buffer]
        assert actual == expected and len(set(actual)) == cap
        assert all(entry["text_sha256"] == train[entry["task_id"]][entry["example_id"]]["text_sha256"]
                   and "text" not in entry for entry in buffer)
        previous = set(actual)
        for task in STAGES[:i]:
            prediction_rows += predictions(path, task, test[task])
    return {"method": "reservoir_capacity", "capacity": cap, "arm": arm,
            "seed": seed, "checkpoint_files_checked": checked_files,
            "prediction_rows": prediction_rows,
            "exact_online_reservoir_sequence": True}


def main() -> None:
    checked = []
    for cap in (50, 200, 400):
        for arm in ("doc", "short", "hard"):
            for seed in (17, 42, 2026):
                if cap != 200:
                    matched = validate_matched("a1_matched", cap, arm, seed, True)
                    if matched:
                        checked.append(matched)
                result = reservoir(arm, seed, cap, True)
                if result:
                    checked.append(result)
    expected = {(capacity, method, "hard", 42)
                for capacity in (50, 400)
                for method in ("a1_matched", "reservoir_capacity")}
    observed = {(row["capacity"], row["method"], row["arm"], row["seed"])
                for row in checked}
    if observed != expected or len(checked) != len(expected):
        raise RuntimeError(f"Incomplete or unexpected predeclared capacity subset: {observed}")
    target = ROOT / "results/revision2/analysis/capacity_validation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"status": "PASS", "full_checkpoint_hashes": True,
                                  "runs_checked": len(checked), "runs": checked}, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"status": "PASS", "runs_checked": len(checked)}))


if __name__ == "__main__":
    main()
