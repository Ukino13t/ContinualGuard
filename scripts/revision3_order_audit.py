"""Audit index-based replay equivalence for frozen A1 versus A0 v1/v2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from continualguard.stronger_pilot_no_replay import sha256_file

ROOT = Path(r"E:\ContinualGuard")
BASE = Path(r"Z:\ContinualGuard_Revision3\results\revision3")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def compare(a: Path, b: Path, before: int, after: int) -> dict:
    left = rows(a / f"stage{before}/buffer.jsonl")
    right = rows(b / f"stage{before}/buffer.jsonl")
    la = [r["example_id"] for r in left]
    lb = [r["example_id"] for r in right]
    a_log = rows(a / f"stage{after}/train_log.jsonl")
    b_log = rows(b / f"stage{after}/train_log.jsonl")
    if len(a_log) != len(b_log):
        raise ValueError("Unequal number of training steps")
    replay_a = [x for r in a_log for x in r["replay_example_ids"]]
    replay_b = [x for r in b_log for x in r["replay_example_ids"]]
    current_a = [x for r in a_log for x in r["example_ids"]]
    current_b = [x for r in b_log for x in r["example_ids"]]
    return {
        "buffer_same_id_set": set(la) == set(lb),
        "buffer_ordered_position_delta": sum(x != y for x, y in zip(la, lb)),
        "current_order_position_delta": sum(x != y for x, y in zip(current_a, current_b)),
        "replay_position_delta": sum(x != y for x, y in zip(replay_a, replay_b)),
        "replay_positions": len(replay_a),
        "model_weights_sha256_a": sha256_file(a / f"stage{after}/checkpoint/model.safetensors"),
        "model_weights_sha256_b": sha256_file(b / f"stage{after}/checkpoint/model.safetensors"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=("v1", "v2"), required=True)
    parser.add_argument("--arm", choices=("doc", "short", "hard"), default="doc")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--before", type=int, default=1)
    parser.add_argument("--after", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    a = ROOT / f"results/revision2/matched_ablation_amp256/M200/a1_matched/{args.arm}/seed{args.seed}"
    suffix = "matched_ablation_amp256_v2" if args.version == "v2" else "matched_ablation_amp256"
    b = BASE / suffix / f"M200/a0_matched/{args.arm}/seed{args.seed}"
    if args.smoke:
        if args.version != "v2":
            raise ValueError("Smoke parity applies only to corrected v2")
        b = BASE / suffix / f"smoke/M200/a0_matched/{args.arm}/seed{args.seed}"
    result = compare(a, b, args.before, args.after)
    result.update(version=args.version, arm=args.arm, seed=args.seed,
                  before_stage=args.before, after_stage=args.after)
    tag = "_smoke" if args.smoke else ""
    destination = ROOT / f"results/revision3/order_audit_{args.version}{tag}_{args.arm}_seed{args.seed}_s{args.before}_s{args.after}.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
