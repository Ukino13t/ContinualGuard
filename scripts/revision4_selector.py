"""Canonical-order A2 stage-quota selector; delegates membership to frozen A2."""
from __future__ import annotations
from revision2_selector import (
    MatchedBuffer, RANK_NAMESPACE, REPLAY_NAMESPACE, ORDER_NAMESPACE,
    DROPOUT_NAMESPACE, ranked_entry, stream_seed, digest,
)

VARIANTS = ("a2_canonical",)

class A2CanonicalBuffer(MatchedBuffer):
    def __init__(self, variant: str, seed: int, capacity: int = 200):
        if variant not in VARIANTS:
            raise ValueError("Only A2_canonical is valid")
        super().__init__("a2_matched", seed, capacity)
        self.variant = variant

    def update(self, candidates: list[dict], stages: tuple[str, ...]) -> dict:
        self.variant = "a2_matched"
        try:
            diagnostics = super().update(candidates, stages)
        finally:
            self.variant = "a2_canonical"
        self.entries.sort(key=lambda x: (x["label"], x["random_rank_sha256"], x["example_id"]))
        self.validate()
        return diagnostics

