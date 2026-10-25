"""Independent identity, training-access, metric, and checkpoint audit of A0 runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
ZROOT = Path(r"Z:\ContinualGuard_Revision3")
BASE = ZROOT / "results/revision3/matched_ablation_amp256"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from continualguard.stronger_pilot_no_replay import CandidateBData, SEEDS, STAGES  # noqa: E402
from scripts.revision3_selector import A0MatchedBuffer, ranked_entry  # noqa: E402


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def sha(path: Path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate(arm: str, seed: int, capacity: int, full_hash: bool) -> dict:
    protocol = load(ROOT / "results/revision3/protocol_v1.json")
    data = CandidateBData(ROOT, arm)
    run = BASE / f"M{capacity}/a0_matched/{arm}/seed{seed}"
    frozen_a1 = ROOT / "results/revision2/matched_ablation_amp256/M200/a1_matched" / arm / f"seed{seed}"
    metadata = load(run / "run_metadata.json")
    assert metadata["status"] == "COMPLETE" and metadata["completed_stages"] == 4
    assert metadata["memory_capacity_examples"] == capacity
    assert metadata["amp_grad_scaler_initial_scale"] == 256.0
    assert metadata["runner_sha256"] == protocol["source_sha256"]["scripts/revision3_train.py"]
    assert sha(run / "resolved_config.yaml") == protocol["run_config_sha256"][
        f"a0_matched/{arm}/seed{seed}/M{capacity}"]
    assert metadata["materialization_manifest_sha256"] == protocol["source_sha256"][
        "data/processed/stronger_pilot_candidate_b/materialization_manifest.json"]
    assert load(run / "initialization.json") == load(frozen_a1 / "initialization.json")
    buffer = A0MatchedBuffer("a0_matched", seed, capacity)
    all_train = {stage: {r["example_id"]: r for r in data.records(stage, "train")}
                 for stage in STAGES}
    all_test = {stage: {r["example_id"] for r in data.records(stage, "test")}
                for stage in STAGES}
    checked_files = 0
    prediction_rows = 0
    prior_memory = set()
    identical_order = True
    for index, stage in enumerate(STAGES, 1):
        folder = run / f"stage{index}"
        metrics = load(folder / "metrics.json")
        stage_meta = load(folder / "stage_metadata.json")
        assert (metrics["stage_index"], metrics["trained_stage"]) == (index, stage)
        assert stage_meta["status"] == "COMPLETE"
        assert load(folder / "stage_start.json")["optimizer_reset"] is True
        manifest = load(folder / "checkpoint/checkpoint_manifest.json")
        assert manifest["optimizer_state_saved"] is False
        assert stage_meta["checkpoint_hashes"] == manifest["files"]
        for name, digest in manifest["files"].items():
            path = folder / "checkpoint" / name
            assert path.is_file(), path
            if full_hash:
                assert sha(path) == digest, path
            checked_files += 1
        logs = jsonl(folder / "train_log.jsonl")
        assert len(logs) == 100
        current = [identity for row in logs for identity in row["example_ids"]]
        replay = [identity for row in logs for identity in row["replay_example_ids"]]
        assert len(current) == 400 and set(current) == set(all_train[stage])
        assert len(replay) == (0 if index == 1 else 400)
        assert set(replay) <= prior_memory
        assert metrics["training"]["optimizer_steps"] == 100
        assert metrics["training"]["replay_examples_exposed"] == len(replay)
        a1_current = [identity for row in jsonl(frozen_a1 / f"stage{index}/train_log.jsonl")
                      for identity in row["example_ids"]]
        identical_order &= current == a1_current
        assert current == a1_current
        ranked = [ranked_entry(row, seed) for row in data.records(stage, "train")]
        buffer.update(ranked, STAGES[:index])
        saved = jsonl(folder / "buffer.jsonl")
        assert [e["example_id"] for e in saved] == [e["example_id"] for e in buffer.entries]
        assert len(saved) == capacity and len({e["example_id"] for e in saved}) == capacity
        for entry in saved:
            source = all_train[entry["task_id"]][entry["example_id"]]
            assert entry["text_sha256"] == source["text_sha256"] and "text" not in entry
        prior_memory = {e["example_id"] for e in saved}
        for task in STAGES[:index]:
            predictions = jsonl(folder / "predictions" / f"{task}.jsonl")
            assert len(predictions) == 120
            assert {r["example_id"] for r in predictions} == all_test[task]
            assert Counter(r["true_label"] for r in predictions) == Counter({0: 60, 1: 60})
            assert all("text" not in row for row in predictions)
            tpr = sum(r["true_label"] == 1 and r["predicted_label"] == 1 for r in predictions) / 60
            fpr = sum(r["true_label"] == 0 and r["predicted_label"] == 1 for r in predictions) / 60
            assert abs(tpr - metrics["evaluations"][task]["malicious"]["attack_tpr"]) < 1e-9
            assert abs(fpr - metrics["evaluations"][task]["benign"]["benign_fpr"]) < 1e-9
            prediction_rows += len(predictions)
        for bank, expected in (("NotInject", 339), ("B-short", 120)):
            assert len(jsonl(folder / "predictions" / f"{bank}.jsonl")) == expected
            prediction_rows += expected
    first_a0 = load(run / "stage1/checkpoint/checkpoint_manifest.json")["files"]["model.safetensors"]
    first_a1 = load(frozen_a1 / "stage1/checkpoint/checkpoint_manifest.json")["files"]["model.safetensors"]
    assert first_a0 == first_a1
    return {"arm": arm, "seed": seed, "capacity": capacity,
            "checkpoint_files_checked": checked_files, "prediction_rows_checked": prediction_rows,
            "first_stage_model_matches_a1": True, "all_current_orders_match_a1": identical_order,
            "buffer_reconstruction": "exact", "full_hash": full_hash}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("doc", "short", "hard"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--capacity", type=int, choices=(50, 200, 400), default=200)
    parser.add_argument("--full-hash", action="store_true")
    args = parser.parse_args()
    cells = [(args.arm, args.seed)] if args.arm is not None and args.seed is not None else [
        (arm, seed) for arm in ("doc", "short", "hard") for seed in SEEDS]
    results = []
    for arm, seed in cells:
        run = BASE / f"M{args.capacity}/a0_matched/{arm}/seed{seed}/run_metadata.json"
        if run.is_file() and load(run).get("status") == "COMPLETE":
            results.append(validate(arm, seed, args.capacity, args.full_hash))
    report = {"status": "PASS", "runs_checked": len(results),
              "full_checkpoint_hashes": args.full_hash, "runs": results}
    target = ROOT / "results/revision3/analysis" / f"validation_M{args.capacity}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
