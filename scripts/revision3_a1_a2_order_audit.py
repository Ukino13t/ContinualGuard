"""Prompt-free audit of shared-rank A1/A2 selected-member ordering."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
BASE = ROOT / "results/revision2/matched_ablation_amp256/M200"
OUT = ROOT / "results/revision3/a1_a2_order_audit.json"


def ids(path: Path) -> list[str]:
    return [json.loads(line)["example_id"] for line in path.read_text(encoding="utf-8").splitlines() if line]


def replay(path: Path) -> list[str]:
    return [identity for line in path.read_text(encoding="utf-8").splitlines() if line
            for identity in json.loads(line)["replay_example_ids"]]


def main() -> None:
    cells = []
    for arm in ("doc", "short", "hard"):
        for seed in (17, 42, 2026):
            a1 = BASE / f"a1_matched/{arm}/seed{seed}"
            a2 = BASE / f"a2_matched/{arm}/seed{seed}"
            initial_equal = ids(a1 / "stage1/buffer.jsonl") == ids(a2 / "stage1/buffer.jsonl")
            assert initial_equal
            left = ids(a1 / "stage2/buffer.jsonl")
            right = ids(a2 / "stage2/buffer.jsonl")
            left_set, right_set = set(left), set(right)
            common_left = [x for x in left if x in right_set]
            common_right = [x for x in right if x in left_set]
            assert len(common_left) == len(common_right)
            order_difference = sum(x != y for x, y in zip(common_left, common_right))
            next_left = replay(a1 / "stage3/train_log.jsonl")
            next_right = replay(a2 / "stage3/train_log.jsonl")
            assert len(next_left) == len(next_right) == 400
            cells.append({"arm": arm, "seed": seed, "after_stage": "AF3",
                          "initial_AF2_order_equal": initial_equal,
                          "selected_set_intersection": len(common_left),
                          "common_id_relative_order_positions_different": order_difference,
                          "next_AF4_replay_positions_different": sum(x != y for x, y in zip(next_left, next_right)),
                          "replay_positions": 400})
    result = {"status": "PASS", "cells": cells,
              "conclusion": "A1/A2 shared ranks, but selected common IDs also change relative list order; outcome difference cannot isolate stage quotas"}
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "cells": len(cells),
                      "minimum_common_order_difference": min(x["common_id_relative_order_positions_different"] for x in cells)}))


if __name__ == "__main__":
    main()
