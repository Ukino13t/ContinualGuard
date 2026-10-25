"""Unstratified random-priority control with canonical replay ordering.

A0 differs from frozen A1 only in omitting label quotas. Membership is global
bottom-k; selected entries are stored in A1's (label, rank, ID) order before
index-based replay sampling. The v1 selector used global rank order, which
confounded quota effects with sampled replay identity even when sets matched.
"""
from __future__ import annotations

import copy
import random
from collections import Counter

from scripts.revision2_selector import (
    DROPOUT_NAMESPACE, ORDER_NAMESPACE, REPLAY_NAMESPACE, RANK_NAMESPACE,
    MatchedBuffer, digest, ranked_entry, stream_seed,
)
from continualguard.stronger_pilot_no_replay import STAGES

VARIANTS = ("a0_matched",)


class A0MatchedBuffer(MatchedBuffer):
    def __init__(self, variant: str, seed: int, capacity: int = 200):
        if variant != "a0_matched" or capacity < 2 or capacity % 2:
            raise ValueError("Unknown selector or invalid even capacity")
        self.variant, self.seed, self.capacity = variant, seed, capacity
        self.entries: list[dict] = []
        self.seen_count = 0
        self.sample_rng = random.Random(stream_seed(seed, REPLAY_NAMESPACE))

    @classmethod
    def from_state_dict(cls, state: dict) -> "A0MatchedBuffer":
        obj = cls(state["variant"], state["seed"], state["capacity"])
        obj.entries = copy.deepcopy(state["entries"])
        obj.seen_count = state["seen_count"]
        obj.sample_rng.setstate(state["sample_rng_state"])
        obj.validate()
        return obj

    def update(self, candidates: list[dict], stages: tuple[str, ...]) -> dict:
        if not stages or tuple(STAGES[:len(stages)]) != tuple(stages):
            raise ValueError("Stage prefix violation")
        if len({x["example_id"] for x in candidates}) != len(candidates):
            raise ValueError("Duplicate current candidate")
        if any(x["task_id"] != stages[-1] or x["stage_index"] != len(stages)
               or x["random_rank_sha256"] != digest(self.seed, RANK_NAMESPACE, x["example_id"])
               for x in candidates):
            raise ValueError("Current-stage candidate/rank violation")
        self.validate()
        if {x["example_id"] for x in self.entries} & {x["example_id"] for x in candidates}:
            raise ValueError("Current candidate repeats previous memory")
        pool = [*self.entries, *candidates]
        selected = sorted(pool, key=lambda x: (x["random_rank_sha256"], x["example_id"]))[:self.capacity]
        self.entries = sorted(selected, key=lambda x: (x["label"], x["random_rank_sha256"], x["example_id"]))
        self.seen_count += len(candidates)
        self.validate()
        if len(selected) != min(self.capacity, len(pool)):
            raise ValueError("Unfilled buffer")
        return {"quotas": None, "eligible_count": len(pool),
                "eligible_unique_coverage_keys": len({x["coverage_key"] for x in pool}),
                "duplicate_coverage_key_count": sum(n > 1 for n in Counter(x["coverage_key"] for x in pool).values()),
                "coverage_reordered_positions": 0, "selected_count": len(selected),
                "unstratified_priority": True}
