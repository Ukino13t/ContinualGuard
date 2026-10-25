"""Finalize H2-D1 validation after saved 24-checkpoint inference."""
import json
from pathlib import Path
from continualguard.notinject_evaluation import (
    RESULT_STEM,
    atomic_text,
    checkpoint_fingerprint,
    discover_checkpoints,
    infer_checkpoint,
    load_notinject,
    read_jsonl,
    verify_saved,
    write_json,
)

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    descriptors = discover_checkpoints(root)
    records = load_notinject(root)
    print("Fingerprinting before final validation", flush=True)
    before = checkpoint_fingerprint(descriptors)
    sample = records[:16]
    repeated, elapsed = infer_checkpoint(descriptors[0], sample)
    saved = [row for row in read_jsonl(root / f"results/{RESULT_STEM}_predictions.jsonl") if row["checkpoint"] == descriptors[0]["checkpoint_id"]][:16]
    differences = [abs(a["p_attack"] - b["p_attack"]) for a, b in zip(saved, repeated)]
    reproducibility = {
        "checkpoint": descriptors[0]["checkpoint_id"],
        "sample_records": 16,
        "repeat_runtime_seconds": elapsed,
        "labels_identical": [row["predicted_label"] for row in saved] == [row["predicted_label"] for row in repeated],
        "max_abs_p_attack_difference": max(differences),
        "tolerance": 1e-5,
    }
    reproducibility["passed"] = reproducibility["labels_identical"] and reproducibility["max_abs_p_attack_difference"] <= reproducibility["tolerance"]
    print("Fingerprinting after final validation", flush=True)
    after = checkpoint_fingerprint(descriptors)
    validation = verify_saved(root, before, after, reproducibility)
    write_json(root / f"results/{RESULT_STEM}_artifact_validation.json", validation)
    print(json.dumps({"status": validation["status"], "checks": validation["checks"], "reproducibility": reproducibility}, indent=2), flush=True)
