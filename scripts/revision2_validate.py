"""Independent Revision 2 stage, access, prediction and checkpoint validator."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
STAGES = ("AF2", "AF3", "AF4", "AF5")
SEEDS = (17, 42, 2026)
ARMS = ("doc", "short", "hard")
MATCHED = ROOT / "results/revision2/matched_ablation_amp256"
COMPUTE = ROOT / "results/revision2/offline_controls/compute_matched_current_only"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def sha256(path: Path):
    value = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def checkpoint(directory: Path, full_hash: bool):
    cp = directory / "checkpoint"
    expected = load(cp / "checkpoint_manifest.json")["files"]
    for name, digest in expected.items():
        path = cp / name
        assert path.is_file(), path
        if full_hash:
            assert sha256(path) == digest, path
    assert load(directory / "stage_metadata.json")["checkpoint_hashes"] == expected
    return len(expected)


def predictions(directory: Path, task: str, expected_ids: set[str]) -> int:
    rows = jsonl(directory / "predictions" / f"{task}.jsonl")
    assert len(rows) == 120
    assert {x["example_id"] for x in rows} == expected_ids
    assert Counter(x["true_label"] for x in rows) == Counter({0: 60, 1: 60})
    assert all(x["predicted_label"] in (0, 1) and x["task_id"] == task for x in rows)
    metrics = load(directory / "metrics.json")["evaluations"][task]
    attack = sum(x["true_label"] == 1 and x["predicted_label"] == 1 for x in rows) / 60
    benign = sum(x["true_label"] == 0 and x["predicted_label"] == 1 for x in rows) / 60
    assert abs(attack - metrics["malicious"]["attack_tpr"]) < 1e-9
    assert abs(benign - metrics["benign"]["benign_fpr"]) < 1e-9
    assert all("text" not in x for x in rows)
    return len(rows)


def source_rows(arm: str):
    base = ROOT / "data/processed/stronger_pilot_candidate_b" / f"b_{arm}"
    train, test = {}, {}
    for i, task in enumerate(STAGES, 1):
        train[task] = {x["example_id"]: x for x in jsonl(base / f"stage{i}/train.jsonl")}
        test[task] = {x["example_id"] for x in jsonl(base / f"stage{i}/test.jsonl")}
        assert len(train[task]) == 400 and len(test[task]) == 120
    return train, test


def validate_matched(variant: str, cap: int, arm: str, seed: int, full_hash: bool):
    base = MATCHED / f"M{cap}" / variant / arm / f"seed{seed}"
    meta = base / "run_metadata.json"
    if not meta.is_file() or load(meta)["status"] != "COMPLETE":
        return None
    info = load(meta)
    assert info["memory_capacity_examples"] == cap and info["amp_grad_scaler_initial_scale"] == 256.0
    assert info["config_sha256"] == sha256(base / "resolved_config.yaml")
    assert info["materialization_manifest_sha256"] == sha256(ROOT / "data/processed/stronger_pilot_candidate_b/materialization_manifest.json")
    runner_hash = info["runner_sha256"]
    current_runner = ROOT / "scripts/revision2_train.py"
    snapshot = ROOT / "results/revision2/code_snapshots" / f"{runner_hash}.py"
    assert sha256(current_runner) == runner_hash or (snapshot.is_file() and sha256(snapshot) == runner_hash)
    train, test = source_rows(arm)
    previous = set(); checked_files = 0; prediction_count = 0
    for i, task in enumerate(STAGES, 1):
        directory = base / f"stage{i}"
        metrics = load(directory / "metrics.json")
        assert metrics["stage_index"] == i and metrics["trained_stage"] == task
        checked_files += checkpoint(directory, full_hash)
        logs = jsonl(directory / "train_log.jsonl")
        assert len(logs) == 100
        current = [identity for row in logs for identity in row["example_ids"]]
        replay = [identity for row in logs for identity in row["replay_example_ids"]]
        assert len(current) == 400 and set(current) == set(train[task])
        assert len(replay) == (0 if i == 1 else 400) and set(replay) <= previous
        assert metrics["training"]["optimizer_steps"] == 100
        assert metrics["training"]["replay_examples_exposed"] == len(replay)
        buffer = jsonl(directory / "buffer.jsonl")
        ids = [entry["example_id"] for entry in buffer]
        assert len(ids) == len(set(ids)) == cap
        assert set(ids) <= previous | set(train[task])
        assert Counter(x["label"] for x in buffer) == Counter({0: cap // 2, 1: cap // 2})
        if variant != "a1_matched":
            base_quota, remainder = divmod(cap // 2, i)
            for label in (0, 1):
                for j, stage in enumerate(STAGES[:i]):
                    assert sum(x["label"] == label and x["task_id"] == stage for x in buffer) == base_quota + (j < remainder)
        for entry in buffer:
            source = train[entry["task_id"]][entry["example_id"]]
            assert entry["text_sha256"] == source["text_sha256"]
            assert "text" not in entry
        previous = set(ids)
        for seen in STAGES[:i]:
            prediction_count += predictions(directory, seen, test[seen])
        for bank, count in (("NotInject", 339), ("B-short", 120)):
            assert len(jsonl(directory / "predictions" / f"{bank}.jsonl")) == count
            prediction_count += count
    assert load(base / "initialization.json") == load(ROOT / "results/stronger_pilot_no_replay" / arm / f"seed{seed}/initialization.json")
    return {"method": variant, "capacity": cap, "arm": arm, "seed": seed,
            "checkpoint_files_checked": checked_files, "prediction_rows": prediction_count}


def validate_compute(arm: str, seed: int, full_hash: bool):
    base = COMPUTE / arm / f"seed{seed}"
    meta = base / "run_metadata.json"
    if not meta.is_file() or load(meta)["status"] != "COMPLETE":
        return None
    train, test = source_rows(arm)
    files = predictions_count = 0
    for i, task in enumerate(STAGES, 1):
        directory = base / f"stage{i}"
        files += checkpoint(directory, full_hash)
        logs = jsonl(directory / "train_log.jsonl")
        assert len(logs) == 100
        for field in ("current_example_ids", "repeated_current_example_ids"):
            ids = [identity for row in logs for identity in row[field]]
            assert len(ids) == 400 and set(ids) == set(train[task])
        assert all(row["historical_examples"] == 0 for row in logs)
        assert load(directory / "metrics.json")["training"]["optimizer_steps"] == 100
        for seen in STAGES[:i]:
            predictions_count += predictions(directory, seen, test[seen])
    return {"method": "compute_current_only", "arm": arm, "seed": seed,
            "checkpoint_files_checked": files, "prediction_rows": predictions_count}


def validate_pair_control(arm: str, seed: int) -> dict:
    left = MATCHED / "M200/a1_matched" / arm / f"seed{seed}"
    right = MATCHED / "M200/a2_matched" / arm / f"seed{seed}"
    if not all(load(p / "run_metadata.json")["status"] == "COMPLETE" for p in (left, right)):
        raise ValueError("Matched pair incomplete")
    assert load(left / "initialization.json") == load(right / "initialization.json")
    for stage_index in range(1, 5):
        left_log = jsonl(left / f"stage{stage_index}/train_log.jsonl")
        right_log = jsonl(right / f"stage{stage_index}/train_log.jsonl")
        assert [row["example_ids"] for row in left_log] == [row["example_ids"] for row in right_log]
    left_hashes = load(left / "stage1/checkpoint/checkpoint_manifest.json")["files"]
    right_hashes = load(right / "stage1/checkpoint/checkpoint_manifest.json")["files"]
    assert left_hashes["model.safetensors"] == right_hashes["model.safetensors"]
    assert jsonl(left / "stage1/buffer.jsonl") == jsonl(right / "stage1/buffer.jsonl")
    return {"arm": arm, "seed": seed, "identical_current_order_all_stages": True,
            "identical_first_stage_weights_and_buffer": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-hash", action="store_true")
    args = parser.parse_args()
    checked = []
    for variant in ("a1_matched", "a2_matched"):
        for arm in ARMS:
            for seed in SEEDS:
                result = validate_matched(variant, 200, arm, seed, args.full_hash)
                if result:
                    checked.append(result)
    for arm in ARMS:
        for seed in (17, 2026):
            result = validate_compute(arm, seed, args.full_hash)
            if result:
                checked.append(result)
    pairs = []
    if sum(row["method"] in ("a1_matched", "a2_matched") for row in checked) == 18:
        pairs = [validate_pair_control(arm, seed) for arm in ARMS for seed in SEEDS]
    out = ROOT / "results/revision2/analysis/validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"status": "PASS", "full_checkpoint_hashes": args.full_hash,
                               "runs_checked": len(checked), "runs": checked,
                               "matched_pair_controls": pairs}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "runs_checked": len(checked),
                      "full_checkpoint_hashes": args.full_hash}))


if __name__ == "__main__":
    main()
