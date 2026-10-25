"""Freeze corrected capacity pairing before A0-v2 M=50/400 training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
OUT = ROOT / "results/revision3/capacity_protocol_v2.json"


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    a0 = ROOT / "results/revision3/protocol_v2.json"
    a1 = ROOT / "results/revision3/capacity_protocol_v1.json"
    a0_data = json.loads(a0.read_text(encoding="utf-8"))
    a1_data = json.loads(a1.read_text(encoding="utf-8"))
    assert a0_data["status"] == "FROZEN_BEFORE_V2_TRAINING"
    assert a1_data["status"] == "FROZEN_BEFORE_CAPACITY_TRAINING"
    cells = {}
    for arm in ("short", "hard"):
        for capacity in (50, 200, 400):
            for seed in (17, 42, 2026):
                for method in ("a0_matched", "a1_matched"):
                    key = f"{method}/{arm}/seed{seed}/M{capacity}"
                    protocol = a0_data if method == "a0_matched" else a1_data
                    cells[key] = {"config_sha256": protocol["run_config_sha256"][key],
                                  "protocol": "A0_v2" if method == "a0_matched" else "frozen_A1"}
    document = {
        "status": "FROZEN_BEFORE_CORRECTED_CAPACITY_TRAINING",
        "active_root": str(ROOT),
        "large_artifact_root": r"Z:\ContinualGuard_Revision3",
        "corrected_A0_protocol_sha256": sha(a0),
        "frozen_A1_capacity_protocol_sha256": sha(a1),
        "arms": ["short", "hard"], "capacities": [50, 200, 400],
        "seeds": [17, 42, 2026],
        "comparison": "A1 minus A0-v2 under shared ranking, numerical policy, canonical buffer order and matched exposure",
        "cells": cells,
        "capacity_policy": "M=200 A0-v2 primary cells and frozen A1; M=50/400 A0-v2 new cells and existing or pending frozen A1 cells",
        "exclusion": "Do not include A0-v1 capacity trajectories because buffer ordering confounds replay identity",
        "outcomes": ["attack_TPR_BWT", "macro_F1_BWT", "benign_FPR_increase", "acquisition_TPR",
                     "final_attack_TPR", "final_benign_FPR", "memory_composition", "runtime", "peak_GPU_memory"],
    }
    OUT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{OUT} {sha(OUT)} {len(cells)} cells")


if __name__ == "__main__":
    main()
