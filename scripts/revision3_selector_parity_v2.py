"""Fail-closed parity check of A0 v2 and frozen A1 before GPU training."""
from __future__ import annotations

import json
from pathlib import Path

from continualguard.stronger_pilot_no_replay import CandidateBData, STAGES
from scripts.revision2_selector import MatchedBuffer, ranked_entry
from scripts.revision3_selector_v2 import A0MatchedBuffer

ROOT = Path(r"E:\ContinualGuard")
ZROOT = Path(r"Z:\ContinualGuard_Revision3")


def ids(items: list[dict]) -> list[str]:
    return [x["example_id"] for x in items]


def main() -> None:
    seed, arm = 2026, "doc"
    data = CandidateBData(ROOT, arm)
    candidates = [ranked_entry(row, seed) for row in data.records(STAGES[0], "train")]
    a0 = A0MatchedBuffer("a0_matched", seed, 200)
    a1 = MatchedBuffer("a1_matched", seed, 200)
    a0.update(candidates, (STAGES[0],))
    a1.update(candidates, (STAGES[0],))
    frozen_a0 = [json.loads(line)["example_id"] for line in (
        ZROOT / "results/revision3/matched_ablation_amp256/M200/a0_matched/doc/seed2026/stage1/buffer.jsonl"
    ).read_text(encoding="utf-8").splitlines()]
    frozen_a1 = [json.loads(line)["example_id"] for line in (
        ROOT / "results/revision2/matched_ablation_amp256/M200/a1_matched/doc/seed2026/stage1/buffer.jsonl"
    ).read_text(encoding="utf-8").splitlines()]
    assert set(ids(a0.entries)) == set(frozen_a0) == set(frozen_a1)
    assert ids(a0.entries) == ids(a1.entries) == frozen_a1
    sampled_a0 = [x["example_id"] for _ in range(100) for x in a0.sample(4, 2)]
    sampled_a1 = [x["example_id"] for _ in range(100) for x in a1.sample(4, 2)]
    assert sampled_a0 == sampled_a1
    result = {"status": "PASS", "seed": seed, "arm": arm, "stage": STAGES[0],
              "same_membership_as_v1": True, "same_order_as_frozen_a1": True,
              "identical_replay_sequence": True, "replay_positions": len(sampled_a0)}
    destination = ROOT / "results/revision3/selector_parity_v2.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
