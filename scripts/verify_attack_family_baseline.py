"""Verify the three-seed AF1-AF4 no-replay artifact and lineage contract."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("AF1", "AF2", "AF3", "AF4")
SEEDS = (17, 42, 2026)
DATA_SHA = "4b4efaf7640f31f5f12af2f2b3a40b91bbb71789c91c059d21ce4af1e6a9e43a"
PREDICTION_KEYS = {"example_id", "label", "prediction", "attack_probability", "task_id", "attack_family", "injection_position", "domain", "checkpoint_id"}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def latest(seed):
    found = []
    for directory in (ROOT / "results").glob(f"attack-family-sequential-seed{seed}-*"):
        metadata = load(directory / "metadata.json")
        if metadata.get("status") == "complete":
            found.append((metadata["finished_at"], directory, metadata))
    assert found, seed
    return max(found, key=lambda item: item[0])[1:]


def main():
    source_rows = jsonl(ROOT / "data/processed/chen_attack_family_shift.jsonl")
    id_stage = {row["example_id"]: row["task_id"] for row in source_rows}
    verified = []
    for seed in SEEDS:
        directory, metadata = latest(seed)
        assert metadata["stage_order"] == list(STAGES) and metadata["data_manifest_hash"] == DATA_SHA
        assert metadata["replay"] is False and metadata["AF5_included"] is False and metadata["variant_b_constructed"] is False
        assert metadata["optimizer_transition"] == "reset" and metadata["model_weights_carried"] is True and metadata["current_stage_only"] is True
        resolved = (directory / "resolved_config.yaml").read_text(encoding="utf-8")
        assert "replay: false" in resolved and "historical_training_samples: 0" in resolved
        manifest = load(directory / "data_manifest.json")
        assert manifest["source_sha256"] == DATA_SHA and len(manifest["records"]) == 14_640
        assert all("text" not in row and row["task_id"] in STAGES for row in manifest["records"])
        starts = [row for row in jsonl(directory / "stage_events.jsonl") if row["event"] == "stage_start"]
        assert [row["stage"] for row in starts] == list(STAGES)
        assert all(row["optimizer_state_entries_before_training"] == 0 and row["historical_training_records"] == 0 and row["allowed_task_ids"] == [row["stage"]] for row in starts)
        train = jsonl(directory / "train_log.jsonl")
        assert len(train) == 3_000 // 4 * 4
        for entry in train:
            assert all(id_stage[example_id] == entry["stage"] for example_id in entry["example_ids"])
        metrics = jsonl(directory / "metrics.jsonl")
        confusion = jsonl(directory / "confusion_counts.jsonl")
        assert len(metrics) == len(confusion) == 10
        predictions = sorted((directory / "predictions").glob("post_*.jsonl"))
        assert len(predictions) == 10
        for path in predictions:
            rows = jsonl(path)
            assert len(rows) == 360 and all(set(row) == PREDICTION_KEYS and "text" not in row for row in rows)
        for stage in STAGES:
            checkpoint = directory / "checkpoints" / stage
            assert (checkpoint / "model.safetensors").is_file() and (checkpoint / "state.pt").is_file()
        for name in ("accuracy", "macro_f1", "attack_tpr", "attack_recall", "benign_fpr", "attack_tpr_head", "attack_tpr_middle", "attack_tpr_tail"):
            with (directory / f"matrix_{name}.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            assert rows[0] == ["trained_stage", *STAGES] and len(rows) == 5
        summary = load(directory / "summary.json")
        assert len(summary["stage_acquisition_diagnostics"]) == 4 and summary["AF5"] == "excluded from primary baseline"
        assert all(row["training_loss"] >= 0 and row["validation_loss"] >= 0 and row["runtime_seconds"] > 0 and row["peak_gpu_memory_bytes"] > 0 for row in summary["stage_acquisition_diagnostics"])
        verified.append(directory.name)
    aggregate = load(ROOT / "results/attack_family_baseline_aggregate.json")
    assert aggregate["status"] == "complete" and aggregate["seeds"] == list(SEEDS)
    assert aggregate["scope"] == {"AF5_included": False, "variant_b_constructed": False, "replay": False, "secondary_order_run": False}
    output = ROOT / "results/attack_family_baseline_artifact_validation.json"
    output.write_text(json.dumps({"status": "passed", "verified_runs": verified, "checks": ["fixed audited data/model/config", "current-stage-only train-log lineage", "optimizer reset and weight carry", "10 triangular evaluation cells per seed", "raw predictions/confusions/four matrices/position diagnostics", "four stage checkpoints and acquisition diagnostics", "AF5/Variant B/replay exclusions"]}, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
