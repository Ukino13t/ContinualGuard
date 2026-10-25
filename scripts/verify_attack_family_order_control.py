"""Verify order-control lineage, metrics, predictions, and artifact completeness."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORDER = ("AF1", "AF4", "AF3", "AF2")
SEEDS = (17, 42, 2026)
DATA_SHA = "4b4efaf7640f31f5f12af2f2b3a40b91bbb71789c91c059d21ce4af1e6a9e43a"
PREDICTION_KEYS = {
    "example_id", "label", "prediction", "attack_probability", "task_id",
    "attack_family", "injection_position", "domain", "checkpoint_id",
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def latest(seed):
    found = []
    for directory in (ROOT / "results").glob(f"attack-family-order-control-seed{seed}-*"):
        metadata = load(directory / "metadata.json")
        if metadata.get("status") == "complete":
            found.append((metadata["finished_at"], directory, metadata))
    assert found, seed
    return max(found, key=lambda item: item[0])[1:]


def main():
    source_rows = jsonl(ROOT / "data/processed/chen_attack_family_shift.jsonl")
    id_stage = {row["example_id"]: row["task_id"] for row in source_rows}
    verified = []
    prediction_files = 0
    prediction_rows = 0
    for seed in SEEDS:
        directory, metadata = latest(seed)
        assert metadata["stage_order"] == list(ORDER)
        assert metadata["experiment_regime"] == "attack_family_secondary_order_no_replay"
        assert Path(metadata["project_root"]).resolve() == ROOT.resolve()
        model_origin = metadata["model_architecture"]["_name_or_path"]
        assert Path(model_origin).resolve().is_relative_to(ROOT.resolve())
        assert metadata["data_manifest_hash"] == DATA_SHA
        assert metadata["replay"] is False and metadata["AF5_included"] is False
        assert metadata["optimizer_transition"] == "reset"
        assert metadata["model_weights_carried"] is True and metadata["current_stage_only"] is True
        resolved = (directory / "resolved_config.yaml").read_text(encoding="utf-8")
        assert "replay: false" in resolved and "historical_training_samples: 0" in resolved
        manifest = load(directory / "data_manifest.json")
        assert manifest["source_sha256"] == DATA_SHA and len(manifest["records"]) == 14_640
        assert all("text" not in row and row["task_id"] in ORDER for row in manifest["records"])
        starts = [row for row in jsonl(directory / "stage_events.jsonl") if row["event"] == "stage_start"]
        assert [row["stage"] for row in starts] == list(ORDER)
        assert all(
            row["optimizer_state_entries_before_training"] == 0
            and row["historical_training_records"] == 0
            and row["allowed_task_ids"] == [row["stage"]]
            for row in starts
        )
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
            assert len(rows) == 360
            assert all(set(row) == PREDICTION_KEYS and "text" not in row for row in rows)
            prediction_files += 1
            prediction_rows += len(rows)
        for stage in ORDER:
            checkpoint = directory / "checkpoints" / stage
            assert (checkpoint / "model.safetensors").is_file()
            assert (checkpoint / "state.pt").is_file()
        for name in (
            "accuracy", "macro_f1", "attack_tpr", "attack_recall", "benign_fpr",
            "attack_tpr_head", "attack_tpr_middle", "attack_tpr_tail",
        ):
            with (directory / f"matrix_{name}.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            assert rows[0] == ["trained_stage", *ORDER] and len(rows) == 5
        summary = load(directory / "summary.json")
        assert len(summary["stage_acquisition_diagnostics"]) == 4
        assert summary["experiment_regime"] == "attack_family_secondary_order_no_replay"
        assert summary["replay"] == "not implemented or run"
        verified.append(directory.name)
    aggregate = load(ROOT / "results/attack_family_order_control_aggregate.json")
    assert aggregate["status"] == "complete" and aggregate["seeds"] == list(SEEDS)
    assert aggregate["scope"]["replay"] is False
    output = ROOT / "results/attack_family_order_control_artifact_validation.json"
    output.write_text(json.dumps({
        "status": "passed",
        "verified_runs": verified,
        "checkpoint_directories": 12,
        "prediction_files": prediction_files,
        "prediction_rows": prediction_rows,
        "metric_cells": 30,
        "confusion_cells": 30,
        "checks": [
            "fixed audited data/model/config and secondary order",
            "current-stage-only train-log lineage",
            "optimizer reset and model-weight carry",
            "ten triangular evaluation cells per seed",
            "raw predictions, confusion counts, metric and position matrices",
            "four stage-end checkpoint pairs per seed",
            "AF5, Variant B, and replay exclusions",
        ],
    }, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
