"""Verify artifacts and lineage for the completed baseline controls."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_SHA = "4d5a3e4f98a9df2211d8c2658db68c6ce7bf6d5284ec482a746c10ad9b394fb0"
MODEL_SHA = "691d48a2800b926a19e3051def466fc2cca4f59a15e42ce4a0cf7f1b380b5e33"
PREDICTION_KEYS = {"example_id", "label", "prediction", "attack_probability", "task_id", "attack_family", "injection_position", "domain", "checkpoint_id"}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def latest(control, seed):
    found = []
    for directory in (ROOT / "results").glob(f"control-*-seed{seed}-*"):
        metadata = load(directory / "metadata.json")
        if metadata.get("control_type") == control and metadata.get("status") == "complete":
            found.append((metadata["finished_at"], directory, metadata))
    assert found, (control, seed)
    return max(found, key=lambda item: item[0])[1:]


def verify_run(control, seed, prediction_count, matrix_rows, checkpoints):
    directory, metadata = latest(control, seed)
    assert metadata["data_manifest_hash"] == DATA_SHA and metadata["model_weights_sha256"] == MODEL_SHA
    assert metadata["max_length"] == 1280 and metadata["effective_batch_size"] == 4 and metadata["learning_rate"] == 1e-5
    resolved = (directory / "resolved_config.yaml").read_text(encoding="utf-8")
    assert "replay: false" in resolved
    assert len(jsonl(directory / "metrics.jsonl")) == prediction_count
    assert len(jsonl(directory / "confusion_counts.jsonl")) == prediction_count
    predictions = sorted((directory / "predictions").glob("*.jsonl"))
    assert len(predictions) == prediction_count
    for path in predictions:
        rows = jsonl(path)
        assert len(rows) == 3600 and all(set(row) == PREDICTION_KEYS and "text" not in row for row in rows)
    for checkpoint in checkpoints:
        assert (directory / "checkpoints" / checkpoint / "model.safetensors").is_file()
        assert (directory / "checkpoints" / checkpoint / "state.pt").is_file()
    for metric in ("accuracy", "macro_f1", "attack_recall", "benign_fpr"):
        with (directory / f"matrix_{metric}.csv").open(newline="", encoding="utf-8") as handle:
            assert len(list(csv.reader(handle))) == matrix_rows + 1
    return directory, metadata


def main():
    verified = []
    for seed in (17, 42, 2026):
        directory, _ = verify_run("s2_only", seed, 3, 3, ("S2ONLY-E1", "S2ONLY-E2", "S2ONLY-E3"))
        events = jsonl(directory / "stage_events.jsonl")
        starts = [event for event in events if event["event"] == "epoch_start"]
        assert starts[0]["optimizer_state_entries_before_training"] == 0 and all(event["optimizer_state_entries_before_training"] > 0 for event in starts[1:])
        verified.append(directory.name)
        directory, _ = verify_run("sequential_s2_epochs", seed, 6, 4, ("S2-E1", "S2-E2", "S2-E3"))
        events = jsonl(directory / "stage_events.jsonl")
        starts = [event for event in events if event["event"] == "epoch_start"]
        assert starts[0]["optimizer_reset_at_s2_boundary"] and starts[0]["optimizer_state_entries_before_training"] == 0
        assert all(event["optimizer_state_entries_before_training"] > 0 for event in starts[1:])
        verified.append(directory.name)
    reverse_dir, _ = verify_run("reverse_order", 42, 3, 2, ("S2", "S1"))
    reverse_starts = [event for event in jsonl(reverse_dir / "stage_events.jsonl") if event["event"] == "stage_start"]
    assert [event["stage"] for event in reverse_starts] == ["S2", "S1"] and all(event["optimizer_state_entries_before_training"] == 0 for event in reverse_starts)
    verified.append(reverse_dir.name)
    joint_dir, _ = verify_run("joint_epochs", 42, 6, 3, ("JOINT-E1", "JOINT-E2", "JOINT-E3"))
    joint_starts = [event for event in jsonl(joint_dir / "stage_events.jsonl") if event["event"] == "epoch_start"]
    assert joint_starts[0]["optimizer_state_entries_before_training"] == 0 and all(event["optimizer_state_entries_before_training"] > 0 for event in joint_starts[1:])
    verified.append(joint_dir.name)

    baseline = load(ROOT / "results/real_baseline_aggregate.json")
    aggregate = load(ROOT / "results/baseline_sanity_aggregate.json")
    for seed in (17, 42, 2026):
        original = next(row for row in baseline["sequential_per_seed"] if row["seed"] == seed)
        control = next(row for row in aggregate["sequential_per_seed_epoch"] if row["seed"] == seed and row["epoch"] == 1)
        assert control["per_stage"]["S2"]["attack_recall"] == original["s2_attack_recall_acquisition"]
        assert control["per_stage"]["S1"]["attack_recall"] == original["s1_attack_recall_after_s2"]
    original_joint = next(row for row in baseline["joint_per_seed"] if row["seed"] == 42)
    joint_e1 = aggregate["joint_seed42_by_epoch"][0]
    assert joint_e1["per_stage"]["S1"]["attack_recall"] == original_joint["s1_attack_recall"]
    assert joint_e1["per_stage"]["S2"]["attack_recall"] == original_joint["s2_attack_recall"]
    reverse = aggregate["reverse_order_seed42"]
    s2_e1 = next(row for row in aggregate["s2_only_per_seed_epoch"] if row["seed"] == 42 and row["epoch"] == 1)
    assert reverse["s2_acquisition"]["attack_recall"] == s2_e1["per_stage"]["S2"]["attack_recall"]

    report = {"status": "passed", "verified_runs": verified, "checks": ["fixed model/data/config and no-replay gates", "prediction schemas/counts without source text", "checkpoints, confusion counts, four matrices, and summaries", "optimizer reset/carry behavior across stages/epochs", "exact E1 reproduction and checkpoint lineage"]}
    write = ROOT / "results/baseline_sanity_artifact_validation.json"
    write.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(write)


if __name__ == "__main__":
    main()
