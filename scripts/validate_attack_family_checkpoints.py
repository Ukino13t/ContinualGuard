"""Load every AF1-AF4 stage-end model and trusted local resume state."""

import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from continualguard.reproducibility import load_checkpoint


STAGES = ("AF1", "AF2", "AF3", "AF4")
DATA_SHA = "4b4efaf7640f31f5f12af2f2b3a40b91bbb71789c91c059d21ce4af1e6a9e43a"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    validated = []
    for seed in (17, 42, 2026):
        found = []
        for directory in (ROOT / "results").glob(f"attack-family-sequential-seed{seed}-*"):
            metadata = load_json(directory / "metadata.json")
            if metadata.get("status") == "complete":
                found.append((metadata["finished_at"], directory))
        assert found, seed
        run = max(found)[1]
        for index, stage in enumerate(STAGES, 1):
            checkpoint = run / "checkpoints" / stage
            model, state = load_checkpoint(checkpoint)
            assert state["experiment_regime"] == "attack_family_sequential_no_replay"
            assert state["completed_stages"] == index and state["data_hash"] == DATA_SHA
            assert state["replay"] is False and state["historical_training_samples"] == 0
            validated.append({
                "seed": seed, "run_id": run.name, "stage": stage,
                "model_class": type(model).__name__,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "resume_state": "passed", "status": "passed",
            })
            del model, state
            gc.collect()
    output = ROOT / "results/attack_family_checkpoint_validation.json"
    output.write_text(json.dumps({"status": "passed", "checkpoint_count": len(validated), "validated": validated}, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
