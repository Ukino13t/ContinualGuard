"""Freeze the corrected A0 protocol before any v2 model training."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from continualguard.stronger_pilot_no_replay import sha256_file

ROOT = Path(r"E:\ContinualGuard")
OLD = ROOT / "results/revision3/protocol_v1.json"
NEW = ROOT / "results/revision3/protocol_v2.json"


def main() -> None:
    if NEW.exists():
        raise FileExistsError(f"Frozen protocol already exists: {NEW}")
    old = json.loads(OLD.read_text(encoding="utf-8"))
    protocol = copy.deepcopy(old)
    protocol["revision"] = "3-v2-canonical-buffer-order"
    protocol["supersedes_protocol_sha256"] = sha256_file(OLD)
    protocol["status"] = "FROZEN_BEFORE_V2_TRAINING"
    protocol["matched_comparison"] = (
        "A0 global bottom-k membership versus frozen A1 per-label bottom-k; "
        "both replay from canonical (label, rank, ID) buffer order"
    )
    protocol["fixed"]["a0_selector"] = (
        "Global ascending (rank, ID) bottom-k membership, followed by canonical "
        "(label, rank, ID) storage order before replay sampling"
    )
    protocol["fixed"]["replay_index_parity_rule"] = (
        "If A0 and A1 retain identical ID sets at a boundary, their ordered "
        "buffers and future sampled replay ID sequences must match exactly"
    )
    protocol["v1_exclusion"] = (
        "Old A0 v1 stores global-rank order; frozen A1 stores label-then-rank "
        "order. Its efficacy outcomes are not a class-quota-only contrast. "
        "Preserve as historical provenance, exclude from controlled inference."
    )
    protocol["source_sha256"].pop("scripts/revision3_train.py")
    protocol["source_sha256"].pop("scripts/revision3_selector.py")
    for source in ("scripts/revision3_train_v2.py", "scripts/revision3_selector_v2.py"):
        protocol["source_sha256"][source] = sha256_file(ROOT / source)
    NEW.parent.mkdir(parents=True, exist_ok=True)
    NEW.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{NEW} {sha256_file(NEW)}")


if __name__ == "__main__":
    main()
