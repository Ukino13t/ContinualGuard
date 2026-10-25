"""Versioned Revision 2 wrapper for the unchanged current-only control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import offline_controls
from continualguard.stronger_pilot_no_replay import atomic_json

ROOT = Path(r"E:\ContinualGuard")
offline_controls.OUTPUT = Path("results/revision2/offline_controls")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("doc", "short", "hard"), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 2026), required=True)
    args = parser.parse_args()
    out = offline_controls.run(ROOT, "compute_matched_current_only", args.arm, args.seed)
    metadata_path = out / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["revision2_wrapper_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata["reused_control_definition"] = "scripts/offline_controls.py"
    atomic_json(metadata_path, metadata)
    print(out)


if __name__ == "__main__":
    main()
