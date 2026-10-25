"""Run frozen A1 matched selector at new capacities with Z-only large output.

The scientific runner remains scripts/revision2_train.py, unchanged.  This
wrapper only changes its output base and verifies the predeclared capacity
protocol before invoking it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import revision2_train as runner  # noqa: E402
from continualguard.loss_aware_control import approved_config  # noqa: E402
from continualguard.stronger_pilot_no_replay import ARMS, SEEDS  # noqa: E402
import yaml  # noqa: E402

OUTPUT = Path(r"Z:\ContinualGuard_Revision3\results\revision3\matched_ablation_amp256")


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("short", "hard"), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--capacity", type=int, choices=(50, 400), required=True)
    args = parser.parse_args()
    protocol_file = ROOT / "results/revision3/capacity_protocol_v1.json"
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    if sha(Path(__file__)) != protocol["source_sha256"]["scripts/revision3_capacity_a1.py"]:
        raise ValueError("Capacity wrapper differs from frozen source")
    if sha(ROOT / "scripts/revision2_train.py") != protocol["source_sha256"]["scripts/revision2_train.py"]:
        raise ValueError("Frozen A1 scientific runner changed")
    config = approved_config(ROOT, args.seed)
    config["method"] = "loss_aware"
    config["replay"]["memory_capacity_examples"] = args.capacity
    payload = {**config, "selection_method": "a1_matched", "execution_arm": ARMS[args.arm],
               "revision2_rank_namespace": "route-a-revision2-shared-rank-v1"}
    # The unchanged Revision 2 runner uses Path.write_text on Windows (CRLF).
    encoded = yaml.safe_dump(payload, sort_keys=True).replace("\n", "\r\n").encode("utf-8")
    key = f"a1_matched/{args.arm}/seed{args.seed}/M{args.capacity}"
    if hashlib.sha256(encoded).hexdigest() != protocol["run_config_sha256"][key]:
        raise ValueError("A1 resolved configuration differs from frozen capacity protocol")
    runner.OUTPUT = OUTPUT
    result = runner.run(ROOT, "a1_matched", args.arm, args.seed, capacity=args.capacity)
    overlay = {"revision": 3, "purpose": "capacity sensitivity", "scientific_runner": "scripts/revision2_train.py",
               "scientific_runner_sha256": protocol["source_sha256"]["scripts/revision2_train.py"],
               "wrapper_sha256": sha(Path(__file__)), "capacity_protocol_sha256": sha(protocol_file),
               "output_root": str(OUTPUT), "config_sha256": protocol["run_config_sha256"][key]}
    (result / "revision3_capacity_overlay.json").write_text(
        json.dumps(overlay, indent=2) + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
