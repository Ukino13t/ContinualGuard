"""Check shared initialization and current-data order across P1 conditions."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
ANALYSIS = ROOT / "results/revision2/analysis"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def logs(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    validation = load(ANALYSIS / "capacity_validation.json")
    if (validation.get("status"), validation.get("runs_checked"),
            validation.get("full_checkpoint_hashes")) != ("PASS", 4, True):
        raise RuntimeError("P1 full-hash validation is required")
    runs = []
    for capacity, variant in ((50, "a1_matched"), (200, "a1_matched"),
                              (400, "a1_matched"), (50, "reservoir_capacity"),
                              (400, "reservoir_capacity")):
        base = (ROOT / "results/revision2/matched_ablation_amp256"
                if variant == "a1_matched" else
                ROOT / "results/revision2/memory_sensitivity_amp256")
        runs.append((variant, capacity, base / f"M{capacity}" / variant / "hard/seed42"))
    first = runs[0][2]
    initialization = load(first / "initialization.json")
    weight_hash = load(first / "stage1/checkpoint/checkpoint_manifest.json")["files"]["model.safetensors"]
    current_order = [[r["example_ids"] for r in logs(first / f"stage{i}/train_log.jsonl")]
                     for i in range(1, 5)]
    for variant, capacity, folder in runs[1:]:
        assert load(folder / "initialization.json") == initialization, (variant, capacity)
        actual_hash = load(folder / "stage1/checkpoint/checkpoint_manifest.json")["files"]["model.safetensors"]
        assert actual_hash == weight_hash, (variant, capacity)
        for i in range(1, 5):
            assert [r["example_ids"] for r in logs(folder / f"stage{i}/train_log.jsonl")] == current_order[i - 1], (variant, capacity, i)
    result = {"status": "PASS", "runs_checked": len(runs),
              "shared_initialization": True, "identical_first_stage_weights": True,
              "identical_current_order_all_four_stages": True,
              "stage1_model_sha256": weight_hash,
              "run_keys": [{"method": variant, "capacity": capacity, "arm": "hard", "seed": 42}
                           for variant, capacity, _ in runs]}
    (ANALYSIS / "capacity_pair_controls.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "runs_checked": len(runs)}))


if __name__ == "__main__":
    main()
