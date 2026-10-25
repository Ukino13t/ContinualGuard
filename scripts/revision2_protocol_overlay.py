"""Record actual selector semantics without rewriting frozen run configurations.

The matched runner inherited inactive selector-description keys from the CE
control's base YAML. This overlay makes the code-level protocol explicit while
preserving each run's original hashed resolved_config.yaml and metadata.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
BASE = ROOT / "results/revision2/matched_ablation_amp256/M200"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    selector_hash = sha(ROOT / "scripts/revision2_selector.py")
    trainer_hash = sha(ROOT / "src/continualguard/replay_candidate_b.py")
    output = []
    for variant in ("a1_matched", "a2_matched"):
        for arm in ("doc", "short", "hard"):
            for seed in (17, 42, 2026):
                folder = BASE / variant / arm / f"seed{seed}"
                meta = json.loads((folder / "run_metadata.json").read_text(encoding="utf-8"))
                if meta["status"] != "COMPLETE" or meta["completed_stages"] != 4:
                    raise RuntimeError(f"Incomplete trajectory: {folder}")
                overlay = {
                    "record_type": "post-run non-mutating protocol clarification",
                    "run_id": meta["run_id"], "variant": variant, "arm": arm, "seed": seed,
                    "capacity": 200, "eligible_update_pool": "previous retained buffer union current TRAIN only",
                    "update_timing": "after current-stage training",
                    "selection_score": "none; deterministic SHA-256 random rank",
                    "rank_definition": "SHA256(seed|route-a-revision2-shared-rank-v1|example_id)",
                    "label_quota": "100 benign and 100 attack records",
                    "stage_quota": "none" if variant == "a1_matched" else "equal within each label, earlier-stage remainder",
                    "coverage_preference": "none; A3 parity was audited separately",
                    "replay_sampling": "separate Python Random route-a-revision2-replay-v1",
                    "current_order": "stage-specific Torch generator route-a-revision2-order-v1",
                    "model_stochasticity": "stage-specific Torch/CUDA seed route-a-revision2-dropout-v1",
                    "loss": "(current cross-entropy + replay cross-entropy)/2 after stage 1",
                    "amp_grad_scaler_initial_scale": 256.0,
                    "source_hashes": {"runner": meta["runner_sha256"],
                                      "selector": selector_hash, "trainer": trainer_hash,
                                      "original_resolved_config": meta["config_sha256"]},
                    "inactive_inherited_resolved_config_keys": [
                        "experiment", "replay.selection_score", "replay.score_timing",
                        "replay.selection_score_precision", "replay.coverage",
                        "replay.stage_quota", "replay.tie_break"],
                    "evidence": "selected buffer JSONL, train logs, source code, selector audit, full-hash validation",
                }
                path = folder / "actual_selector_protocol.json"
                if path.exists() and json.loads(path.read_text(encoding="utf-8")) != overlay:
                    raise RuntimeError(f"Existing overlay differs: {path}")
                path.write_text(json.dumps(overlay, indent=2) + "\n", encoding="utf-8")
                output.append({"run_id": meta["run_id"], "relative_path": str(path.relative_to(ROOT)),
                               "sha256": sha(path)})
    summary = {"status": "PASS", "count": len(output), "overlays": output}
    target = ROOT / "results/revision2/analysis/protocol_overlays.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "count": len(output)}))


if __name__ == "__main__":
    main()
