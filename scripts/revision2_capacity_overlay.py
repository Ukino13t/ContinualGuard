"""Write non-mutating, hash-linked protocol clarifications for P1 runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
ANALYSIS = ROOT / "results/revision2/analysis"
MATCHED = ROOT / "results/revision2/matched_ablation_amp256"
RESERVOIR = ROOT / "results/revision2/memory_sensitivity_amp256"


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    validation = json.loads((ANALYSIS / "capacity_validation.json").read_text(encoding="utf-8"))
    if (validation.get("status"), validation.get("full_checkpoint_hashes"),
            validation.get("runs_checked")) != ("PASS", True, 4):
        raise RuntimeError("P1 overlays require four full-hash-validated runs")
    selector = sha(ROOT / "scripts/revision2_selector.py")
    trainer = sha(ROOT / "src/continualguard/replay_candidate_b.py")
    output = []
    for capacity in (50, 400):
        for variant in ("a1_matched", "reservoir_capacity"):
            base = MATCHED if variant == "a1_matched" else RESERVOIR
            folder = base / f"M{capacity}" / variant / "hard/seed42"
            meta = json.loads((folder / "run_metadata.json").read_text(encoding="utf-8"))
            if meta["status"] != "COMPLETE" or meta["completed_stages"] != 4:
                raise RuntimeError(f"Incomplete P1 run: {folder}")
            rank = ("SHA256(seed|route-a-revision2-shared-rank-v1|example_id)"
                    if variant == "a1_matched" else "unused by reservoir")
            protocol = {
                "record_type": "post-run non-mutating protocol clarification",
                "run_id": meta["run_id"], "variant": variant, "arm": "hard", "seed": 42,
                "capacity": capacity,
                "selection": ("class-balanced hash rank, no stage quota"
                              if variant == "a1_matched" else
                              "unstratified per-arrival online reservoir"),
                "label_quota": (f"{capacity // 2} benign, {capacity // 2} attack"
                                if variant == "a1_matched" else "none"),
                "selection_rank": rank,
                "eligible_update_pool": "previous retained buffer plus current TRAIN only",
                "update_timing": "after each current-stage training pass",
                "reservoir_update_rng": ("not applicable" if variant == "a1_matched"
                                         else "Python Random(seed*1009+11), actual current presentation order"),
                "replay_sampling": "separate Python Random route-a-revision2-replay-v1",
                "current_order": "stage-specific Torch generator route-a-revision2-order-v1",
                "model_stochasticity": "stage-specific Torch/CUDA seed route-a-revision2-dropout-v1",
                "amp_grad_scaler_initial_scale": 256.0,
                "source_hashes": {"runner": meta["runner_sha256"],
                                  "selector": selector if variant == "a1_matched" else None,
                                  "trainer": trainer,
                                  "original_resolved_config": meta["config_sha256"]},
                "inactive_inherited_resolved_config_keys": (
                    ["experiment", "replay.selection_score", "replay.score_timing",
                     "replay.selection_score_precision", "replay.coverage",
                     "replay.stage_quota", "replay.tie_break"]
                    if variant == "a1_matched" else []),
                "evidence": "selected buffers, train logs, source code, exact reservoir reconstruction, full-hash validation",
            }
            path = folder / "actual_selector_protocol.json"
            if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
                raise RuntimeError(f"Existing P1 overlay differs: {path}")
            path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
            output.append({"relative_path": str(path.relative_to(ROOT)), "sha256": sha(path)})
    target = ANALYSIS / "capacity_protocol_overlays.json"
    target.write_text(json.dumps({"status": "PASS", "count": len(output),
                                  "overlays": output}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "count": len(output)}))


if __name__ == "__main__":
    main()
