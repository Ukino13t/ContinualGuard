"""Verify the completed no-replay real-baseline artifact contract."""
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 42, 2026)
DATA_SHA256 = "4d5a3e4f98a9df2211d8c2658db68c6ce7bf6d5284ec482a746c10ad9b394fb0"
MODEL_SHA256 = "691d48a2800b926a19e3051def466fc2cca4f59a15e42ce4a0cf7f1b380b5e33"
MATRICES = ("accuracy", "macro_f1", "attack_recall", "benign_fpr")
PREDICTION_KEYS = {
    "example_id", "label", "prediction", "attack_probability", "task_id",
    "attack_family", "injection_position", "domain", "checkpoint_id",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def latest_complete(pattern, regime):
    candidates = []
    for directory in (ROOT / "results").glob(pattern):
        metadata = read_json(directory / "metadata.json")
        if metadata.get("status") == "complete" and metadata.get("experiment_regime") == regime:
            candidates.append((metadata["finished_at"], directory, metadata))
    if not candidates:
        raise AssertionError(f"No complete run matches {pattern} ({regime})")
    return max(candidates, key=lambda item: item[0])[1:]


def verify_common(directory, metadata, expected_predictions, expected_checkpoint, expected_metric_rows, expected_matrix_rows):
    assert metadata["model_checkpoint"] == "microsoft/deberta-v3-base"
    assert metadata["data_manifest_hash"] == DATA_SHA256
    assert metadata["model_weights_sha256"] == MODEL_SHA256
    assert metadata["seed"] in SEEDS and metadata["precision"] == "float16"
    assert metadata["effective_batch_size"] == 4 and metadata["exclusions"] == []
    assert (directory / "checkpoints" / expected_checkpoint / "model.safetensors").is_file()
    assert (directory / "checkpoints" / expected_checkpoint / "state.pt").is_file()
    assert (directory / "summary.json").is_file()
    assert len(read_jsonl(directory / "metrics.jsonl")) == expected_metric_rows
    assert len(read_jsonl(directory / "confusion_counts.jsonl")) == expected_metric_rows

    observed = {}
    for path in sorted((directory / "predictions").glob("*.jsonl")):
        rows = read_jsonl(path)
        observed[path.name] = len(rows)
        assert rows and all(set(row) == PREDICTION_KEYS for row in rows)
        assert all(isinstance(row["example_id"], str) and "text" not in row for row in rows)
    assert observed == expected_predictions

    for metric in MATRICES:
        with (directory / f"matrix_{metric}.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert rows[0] == ["trained_stage", "S1", "S2"]
        assert len(rows) == expected_matrix_rows + 1


def main():
    verified = []
    sequential_predictions = {
        "post_S1_S1.jsonl": 3600,
        "post_S2_S1.jsonl": 3600,
        "post_S2_S2.jsonl": 3600,
        "pre_S1_S1.jsonl": 3600,
        "pre_S2_S2.jsonl": 3600,
        "val_S1_S1.jsonl": 1889,
        "val_S2_S2.jsonl": 1745,
    }
    joint_predictions = {
        "post_JOINT_S1.jsonl": 3600,
        "post_JOINT_S2.jsonl": 3600,
        "pre_JOINT_S1.jsonl": 3600,
        "pre_JOINT_S2.jsonl": 3600,
        "val_JOINT_S1.jsonl": 1889,
        "val_JOINT_S2.jsonl": 1745,
    }
    for seed in SEEDS:
        seq_dir, seq_meta = latest_complete(f"real-sequential-seed{seed}-*", "sequential")
        verify_common(seq_dir, seq_meta, sequential_predictions, "S2", 7, 2)
        assert (seq_dir / "checkpoints" / "S1" / "model.safetensors").is_file()
        assert seq_meta["optimizer_transition"] == "reset"
        events = read_jsonl(seq_dir / "stage_events.jsonl")
        starts = [event for event in events if event["event"] == "stage_start"]
        assert [event["stage"] for event in starts] == ["S1", "S2"]
        assert all(event["optimizer_state_entries_before_training"] == 0 for event in starts)
        verified.append({"run_id": seq_dir.name, "regime": "sequential", "seed": seed})

        joint_dir, joint_meta = latest_complete(f"real-joint-seed{seed}-*", "joint_offline_reference")
        verify_common(joint_dir, joint_meta, joint_predictions, "JOINT", 6, 1)
        verified.append({"run_id": joint_dir.name, "regime": "joint_offline_reference", "seed": seed})

    aggregate = read_json(ROOT / "results" / "real_baseline_aggregate.json")
    assert aggregate["seeds"] == list(SEEDS) and aggregate["max_length"] == 1280
    assert len(aggregate["sequential_per_seed"]) == 3 and len(aggregate["joint_per_seed"]) == 3
    report = {
        "status": "passed",
        "scope": "six complete no-replay real-baseline runs",
        "verified_runs": verified,
        "checks": [
            "fixed model/data hashes and approved seed/config fields",
            "stage-end checkpoints and optimizer resets for sequential stages",
            "raw prediction schemas and exact row counts without source text",
            "confusion counts, metrics, four matrices, summaries, and aggregate",
        ],
    }
    output = ROOT / "results" / "real_baseline_artifact_validation.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"verification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
