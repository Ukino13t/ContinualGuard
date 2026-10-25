"""Independent integrity, data-access, exposure and checkpoint validation for Route A."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
STAGES = ("AF2", "AF3", "AF4", "AF5")
VARIANTS = ("a1_class", "a2_class_stage", "a3_coverage")
ARMS = ("doc", "short", "hard")
SEEDS = (17, 42, 2026)


def load(path): return json.loads(path.read_text(encoding="utf-8"))
def jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(s) for s in f]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint(stage_dir, full_hash):
    cp = stage_dir / "checkpoint"
    manifest = load(cp / "checkpoint_manifest.json")
    for name, expected in manifest["files"].items():
        file = cp / name
        assert file.exists(), file
        if full_hash:
            assert sha256(file) == expected, file
    return len(manifest["files"])


def validate_ablation(root, variant, arm, seed, full_hash):
    base = root / "results/extended/ablation" / variant / arm / f"seed{seed}"
    if not base.exists() or load(base / "run_metadata.json")["status"] != "COMPLETE":
        return None
    manifest = load(root / "data/processed/stronger_pilot_candidate_b/materialization_manifest.json")
    assert load(base / "run_metadata.json")["materialization_manifest_sha256"] == sha256(root / "data/processed/stronger_pilot_candidate_b/materialization_manifest.json")
    train = {}
    for i, task in enumerate(STAGES, 1):
        path = root / "data/processed/stronger_pilot_candidate_b" / f"b_{arm}" / f"stage{i}" / "train.jsonl"
        train[task] = {x["example_id"]: x for x in jsonl(path)}
        assert len(train[task]) == 400
    previous = set()
    files_hashed = prediction_rows = 0
    for i, task in enumerate(STAGES, 1):
        stage_dir = base / f"stage{i}"
        m = load(stage_dir / "metrics.json")
        assert m["stage_index"] == i and m["trained_stage"] == task
        files_hashed += checkpoint(stage_dir, full_hash)
        logs = jsonl(stage_dir / "train_log.jsonl")
        assert len(logs) == 100
        current_ids = [v for row in logs for v in row["example_ids"]]
        replay_ids = [v for row in logs for v in row["replay_example_ids"]]
        assert len(current_ids) == 400 and set(current_ids) == set(train[task])
        assert len(replay_ids) == (0 if i == 1 else 400)
        assert set(replay_ids) <= previous
        assert m["training"]["optimizer_steps"] == 100
        assert m["training"]["current_examples"] == 400
        assert m["training"]["replay_examples_exposed"] == len(replay_ids)
        buffer = jsonl(stage_dir / "buffer.jsonl")
        ids = [x["example_id"] for x in buffer]
        assert len(ids) == len(set(ids)) == 200
        assert set(ids) <= previous | set(train[task])
        assert Counter(x["label"] for x in buffer) == Counter({0:100, 1:100})
        if variant != "a1_class":
            for label in (0, 1):
                base_q, rem = divmod(100, i)
                for j, old in enumerate(STAGES[:i]):
                    assert sum(x["label"] == label and x["task_id"] == old for x in buffer) == base_q + (j < rem)
        for item in buffer:
            source = train[item["task_id"]][item["example_id"]]
            assert source["split"] == "train" and item["text_sha256"] == source["text_sha256"]
            assert "text" not in item
        previous = set(ids)
        for seen in STAGES[:i]:
            preds = jsonl(stage_dir / "predictions" / f"{seen}.jsonl")
            assert len(preds) == 120
            prediction_rows += len(preds)
        for bank, expected in (("NotInject",339), ("B-short",120)):
            preds = jsonl(stage_dir / "predictions" / f"{bank}.jsonl")
            assert len(preds) == expected
            prediction_rows += len(preds)
    assert load(base / "initialization.json") == load(root / "results/stronger_pilot_no_replay" / arm / f"seed{seed}" / "initialization.json")
    return {"variant": variant, "arm": arm, "seed": seed, "stages": 4,
            "checkpoint_files_checked": files_hashed, "prediction_rows": prediction_rows,
            "full_checkpoint_hashes": full_hash}


def validate_joint(root, arm, seed, full_hash):
    base = root / "results/extended/offline_controls/joint_union_1epoch" / arm / f"seed{seed}"
    if not base.exists() or load(base / "run_metadata.json")["status"] != "COMPLETE":
        return None
    d = base / "joint"
    count = checkpoint(d, full_hash)
    logs = jsonl(d / "train_log.jsonl")
    assert len(logs) == 400
    all_ids = [v for row in logs for v in row["example_ids"]]
    assert len(all_ids) == len(set(all_ids)) == 1600
    train_ids = set()
    for i in range(1, 5):
        train_ids.update(x["example_id"] for x in jsonl(root / "data/processed/stronger_pilot_candidate_b" / f"b_{arm}" / f"stage{i}" / "train.jsonl"))
    assert set(all_ids) == train_ids
    for task in STAGES:
        assert len(jsonl(d / "predictions" / f"{task}.jsonl")) == 120
    return {"control":"joint_union_1epoch", "arm":arm, "seed":seed,
            "optimizer_steps":400, "unique_train_ids":1600, "checkpoint_files_checked":count,
            "full_checkpoint_hashes":full_hash}


def validate_compute(root, arm, seed, full_hash):
    base = root / "results/extended/offline_controls/compute_matched_current_only" / arm / f"seed{seed}"
    if not base.exists() or load(base / "run_metadata.json")["status"] != "COMPLETE":
        return None
    count = 0
    for i, task in enumerate(STAGES, 1):
        d = base / f"stage{i}"
        count += checkpoint(d, full_hash)
        logs = jsonl(d / "train_log.jsonl")
        assert len(logs) == 100
        expected = {x["example_id"] for x in jsonl(root / "data/processed/stronger_pilot_candidate_b" / f"b_{arm}" / f"stage{i}" / "train.jsonl")}
        for field in ("current_example_ids", "repeated_current_example_ids"):
            ids = [v for row in logs for v in row[field]]
            assert len(ids) == 400 and set(ids) == expected
        assert all(row["historical_examples"] == 0 for row in logs)
        assert load(d / "metrics.json")["training"]["optimizer_steps"] == 100
    return {"control":"compute_matched_current_only", "arm":arm, "seed":seed,
            "optimizer_steps":400, "current_exposures":3200, "historical_exposures":0,
            "checkpoint_files_checked":count, "full_checkpoint_hashes":full_hash}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--full-hash", action="store_true")
    a = p.parse_args()
    root = a.root.resolve()
    assert root == ROOT.resolve()
    results = []
    for variant in VARIANTS:
        for arm in ARMS:
            for seed in SEEDS:
                result = validate_ablation(root, variant, arm, seed, a.full_hash)
                if result: results.append(result)
    for arm in ARMS:
        for seed in SEEDS:
            for fn in (validate_joint, validate_compute):
                result = fn(root, arm, seed, a.full_hash)
                if result: results.append(result)
    output = root / "results/extended/analysis/validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"status":"PASS", "runs_checked":len(results), "runs":results}, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status":"PASS", "runs_checked":len(results), "full_hash":a.full_hash}))


if __name__ == "__main__": main()
