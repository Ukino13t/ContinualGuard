"""Freeze A0 matched training identities/configurations before GPU work."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(r"E:\ContinualGuard")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from continualguard.loss_aware_control import approved_config  # noqa: E402
from continualguard.stronger_pilot_no_replay import ARMS, SEEDS  # noqa: E402

OUTPUT = ROOT / "results/revision3/protocol_v1.json"
SOURCE_FILES = (
    "scripts/revision3_train.py", "scripts/revision3_selector.py",
    "scripts/revision2_selector.py", "src/continualguard/replay_candidate_b.py",
    "src/continualguard/loss_aware_control.py",
    "data/processed/stronger_pilot_candidate_b/materialization_manifest.json",
)


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    source = {name: sha(ROOT / name) for name in SOURCE_FILES}
    configs = {}
    paired_a1 = {}
    for arm in ARMS:
        for seed in SEEDS:
            frozen_a1 = ROOT / "results/revision2/matched_ablation_amp256/M200/a1_matched" / arm / f"seed{seed}/run_metadata.json"
            record = json.loads(frozen_a1.read_text(encoding="utf-8"))
            assert record["status"] == "COMPLETE" and record["amp_grad_scaler_initial_scale"] == 256.0
            paired_a1[f"{arm}/seed{seed}"] = {
                "run_id": record["run_id"], "config_sha256": record["config_sha256"],
                "initialization": json.loads((frozen_a1.parent / "initialization.json").read_text(encoding="utf-8")),
            }
            for capacity in (50, 200, 400):
                config = approved_config(ROOT, seed)
                config["method"] = "loss_aware"
                config["replay"]["memory_capacity_examples"] = capacity
                payload = {**config, "selection_method": "a0_matched",
                           "execution_arm": ARMS[arm],
                           "revision2_rank_namespace": "route-a-revision2-shared-rank-v1"}
                encoded = yaml.safe_dump(payload, sort_keys=True).encode("utf-8")
                configs[f"a0_matched/{arm}/seed{seed}/M{capacity}"] = hashlib.sha256(encoded).hexdigest()
    protocol = {
        "revision": 3, "status": "FROZEN_BEFORE_TRAINING", "active_project_root": str(ROOT),
        "large_artifact_root": r"Z:\ContinualGuard_Revision3",
        "matched_comparison": "A0 unstratified random priority vs frozen A1 class-balanced random priority",
        "method_distinction": "Original M1 online Reservoir is contextual and not an A0-matched comparator",
        "fixed": {"arms": list(ARMS), "seeds": list(SEEDS),
                  "stage_order": ["AF2", "AF3", "AF4", "AF5"],
                  "capacity_primary": 200, "capacity_exploratory": [50, 400],
                  "rank": "SHA256(seed|route-a-revision2-shared-rank-v1|example_id)",
                  "eligibility": "previous retained buffer union current-stage TRAIN, after each stage",
                  "a0_selector": "global ascending priority then example_id; no label or stage quota",
                  "a1_selector": "ascending priority within 100 benign and 100 malicious slots",
                  "replay_rng": "route-a-revision2-replay-v1",
                  "current_order_rng": "route-a-revision2-order-v1",
                  "dropout_rng": "route-a-revision2-dropout-v1",
                  "amp_grad_scaler_initial_scale": 256.0,
                  "optimizer_steps_per_stage": 100,
                  "replay_examples_per_step_after_first_stage": 4,
                  "threshold": "fixed argmax over binary logits"},
        "source_sha256": source, "run_config_sha256": configs,
        "paired_frozen_a1": paired_a1,
        "planned_primary_cells": [f"a0_matched/{arm}/seed{seed}/M200" for arm in ARMS for seed in SEEDS],
        "analysis_rule": "paired three-seed descriptive mean and sample SD; no significance claim",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    if OUTPUT.exists() and OUTPUT.read_text(encoding="utf-8") != content:
        raise RuntimeError("Frozen protocol already exists and differs")
    OUTPUT.write_text(content, encoding="utf-8")
    print(json.dumps({"status": "FROZEN", "run_config_count": len(configs),
                      "protocol_sha256": sha(OUTPUT),
                      "materialization_manifest_sha256": source[SOURCE_FILES[-1]]}))


if __name__ == "__main__":
    main()
