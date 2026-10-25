"""Controlled Candidate B memory selectors for Route A Revision 2.

All matched variants use one rank per (seed, source ID). Only previous memory
and current-stage TRAIN records may compete at an update.
"""
from __future__ import annotations

import copy
import hashlib
import random
from collections import Counter

from continualguard.replay_candidate_b import entry_from_row
from continualguard.stronger_pilot_no_replay import STAGES

VARIANTS = ("a1_matched", "a2_matched", "a3_matched")
RANK_NAMESPACE = "route-a-revision2-shared-rank-v1"
REPLAY_NAMESPACE = "route-a-revision2-replay-v1"
ORDER_NAMESPACE = "route-a-revision2-order-v1"
DROPOUT_NAMESPACE = "route-a-revision2-dropout-v1"


def digest(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def stream_seed(seed: int, namespace: str, stage: str = "") -> int:
    return int(digest(seed, namespace, stage)[:16], 16) % (2**63 - 1)


def ranked_entry(row: dict, seed: int) -> dict:
    if row.get("split") != "train" or row.get("task_id") not in STAGES:
        raise ValueError("Only Candidate B TRAIN records can enter memory")
    item = entry_from_row(row, selection_seed=seed)
    item["random_rank_sha256"] = digest(seed, RANK_NAMESPACE, row["example_id"])
    return item


class MatchedBuffer:
    def __init__(self, variant: str, seed: int, capacity: int = 200):
        if variant not in VARIANTS or capacity < 2 or capacity % 2:
            raise ValueError("Unknown selector or invalid even capacity")
        self.variant, self.seed, self.capacity = variant, seed, capacity
        self.entries: list[dict] = []
        self.seen_count = 0
        self.sample_rng = random.Random(stream_seed(seed, REPLAY_NAMESPACE))

    def state_dict(self) -> dict:
        return {"variant": self.variant, "seed": self.seed, "capacity": self.capacity,
                "entries": copy.deepcopy(self.entries), "seen_count": self.seen_count,
                "sample_rng_state": self.sample_rng.getstate()}

    @classmethod
    def from_state_dict(cls, state: dict) -> "MatchedBuffer":
        obj = cls(state["variant"], state["seed"], state["capacity"])
        obj.entries = copy.deepcopy(state["entries"])
        obj.seen_count = state["seen_count"]
        obj.sample_rng.setstate(state["sample_rng_state"])
        obj.validate()
        return obj

    def validate(self, current_stage_index: int | None = None) -> None:
        if len(self.entries) > self.capacity or len({x["example_id"] for x in self.entries}) != len(self.entries):
            raise ValueError("Memory capacity/identity violation")
        for item in self.entries:
            stage = item["task_id"]
            if stage not in STAGES or item["stage_index"] != STAGES.index(stage) + 1:
                raise ValueError("Invalid stage metadata")
            if current_stage_index is not None and item["stage_index"] >= current_stage_index:
                raise ValueError("Current/future replay")
            if item["random_rank_sha256"] != digest(self.seed, RANK_NAMESPACE, item["example_id"]):
                raise ValueError("Rank mismatch")

    def sample(self, count: int, current_stage_index: int) -> list[dict]:
        self.validate(current_stage_index)
        return self.sample_rng.sample(self.entries, min(count, len(self.entries)))

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
        key_counts = Counter(x["coverage_key"] for x in pool)
        selected, quotas = [], {}
        coverage_reordered = 0
        for label in (0, 1):
            if self.variant == "a1_matched":
                buckets = [("all", self.capacity // 2, [x for x in pool if x["label"] == label])]
            else:
                base, rem = divmod(self.capacity // 2, len(stages))
                buckets = [(stage, base + (i < rem),
                            [x for x in pool if x["label"] == label and x["task_id"] == stage])
                           for i, stage in enumerate(stages)]
            for stage, target, bucket in buckets:
                bucket.sort(key=lambda x: (x["random_rank_sha256"], x["example_id"]))
                if self.variant == "a3_matched":
                    first, later, seen = [], [], set()
                    for item in bucket:
                        if item["coverage_key"] in seen:
                            later.append(item)
                        else:
                            first.append(item)
                            seen.add(item["coverage_key"])
                    reordered = first + later
                    coverage_reordered += sum(a["example_id"] != b["example_id"] for a, b in zip(bucket, reordered))
                    bucket = reordered
                selected.extend(bucket[:target])
                quotas[f"label{label}/{stage}"] = {"target": target, "available": len(bucket), "selected": min(target, len(bucket))}
        if len(selected) != min(self.capacity, len(pool)):
            raise ValueError("Unfilled feasible quota; explicit redistribution required")
        self.entries = selected
        self.seen_count += len(candidates)
        self.validate()
        return {"quotas": quotas, "eligible_count": len(pool),
                "eligible_unique_coverage_keys": len(key_counts),
                "duplicate_coverage_key_count": sum(n > 1 for n in key_counts.values()),
                "coverage_reordered_positions": coverage_reordered,
                "selected_count": len(selected)}
