"""Freeze coherent B-SHORT/B-HARD 3x3 A0/A1 capacity frontier before P1."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(r"E:\ContinualGuard")
sys.path.insert(0, str(ROOT / "src"))
from continualguard.loss_aware_control import approved_config  # noqa: E402
from continualguard.stronger_pilot_no_replay import ARMS, SEEDS  # noqa: E402

OUTPUT = ROOT / "results/revision3/capacity_protocol_v1.json"
SOURCE_FILES = ("scripts/revision3_capacity_a1.py", "scripts/revision2_train.py",
                "scripts/revision2_selector.py", "scripts/revision3_train.py",
                "scripts/revision3_selector.py", "src/continualguard/replay_candidate_b.py",
                "data/processed/stronger_pilot_candidate_b/materialization_manifest.json",
                "results/revision3/protocol_v1.json")


def sha(path: Path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    p0 = json.loads((ROOT / "results/revision3/analysis/validation_M200.json").read_text(encoding="utf-8"))
    if p0.get("status") != "PASS" or p0.get("runs_checked") != 9 or not p0.get("full_checkpoint_hashes"):
        raise RuntimeError("Complete validated P0 A0/A1 prerequisite missing")
    p0_protocol = json.loads((ROOT / "results/revision3/protocol_v1.json").read_text(encoding="utf-8"))
    source = {name: sha(ROOT / name) for name in SOURCE_FILES}
    configs, reuse, planned = {}, [], []
    for arm in ("short", "hard"):
        for seed in SEEDS:
            for capacity in (50, 200, 400):
                for variant in ("a0_matched", "a1_matched"):
                    config = approved_config(ROOT, seed)
                    config["method"] = "loss_aware"
                    config["replay"]["memory_capacity_examples"] = capacity
                    payload = {**config, "selection_method": variant,
                               "execution_arm": ARMS[arm],
                               "revision2_rank_namespace": "route-a-revision2-shared-rank-v1"}
                    encoded = yaml.safe_dump(payload, sort_keys=True).encode("utf-8")
                    if variant == "a1_matched":
                        encoded = encoded.replace(b"\n", b"\r\n")
                    key = f"{variant}/{arm}/seed{seed}/M{capacity}"
                    digest = hashlib.sha256(encoded).hexdigest()
                    configs[key] = digest
                    if variant == "a0_matched" and digest != p0_protocol["run_config_sha256"][key]:
                        raise ValueError("A0 capacity config differs from P0 freeze")
                    if variant == "a1_matched" and (capacity == 200 or
                            (arm, seed, capacity) in (("hard", 42, 50), ("hard", 42, 400))):
                        existing = (ROOT / "results/revision2/matched_ablation_amp256" /
                                    f"M{capacity}" / variant / arm / f"seed{seed}")
                        metadata = json.loads((existing / "run_metadata.json").read_text(encoding="utf-8"))
                        if metadata["status"] != "COMPLETE" or sha(existing / "resolved_config.yaml") != digest:
                            raise ValueError(f"Incompatible frozen A1 capacity cell: {key}")
                        reuse.append({"key": key, "path": str(existing.relative_to(ROOT)),
                                      "config_sha256": digest})
                    elif variant == "a0_matched" and capacity == 200:
                        reuse.append({"key": key, "path": str(Path(r"Z:\ContinualGuard_Revision3") /
                                      "results/revision3/matched_ablation_amp256" /
                                      f"M{capacity}/{variant}/{arm}/seed{seed}"),
                                      "config_sha256": digest})
                    else:
                        planned.append(key)
    protocol = {"revision": 3, "status": "FROZEN_BEFORE_CAPACITY_TRAINING",
                "primary_arms": ["short", "hard"], "methods": ["a0_matched", "a1_matched"],
                "capacities": [50, 200, 400], "seeds": list(SEEDS),
                "source_sha256": source, "run_config_sha256": configs,
                "reuse": reuse, "planned_new_cells": planned,
                "policy": "Same detector, stage data, optimizer, 100 updates/stage, current/replay 4:4, AMP scale 256, init/current order/dropout streams, frozen tests and threshold; only capacity and class-quota policy differ by design.",
                "analysis": "Separate malicious BWT and benign FPR curves; paired three-seed mean and sample SD; no monotonic claim by default.",
                "original_M1_reservoir": "contextual only, not a same-policy M200 capacity cell"}
    OUTPUT.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "FROZEN", "cells": len(configs), "reuse": len(reuse),
                      "new": len(planned), "protocol_sha256": sha(OUTPUT)}))


if __name__ == "__main__":
    main()
