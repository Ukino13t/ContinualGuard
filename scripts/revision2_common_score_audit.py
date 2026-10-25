"""Single-state selector diagnostic for A3 random, A4 CE and A5 risk."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(r"E:\ContinualGuard")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from continualguard.model import batch_inputs  # noqa: E402
from continualguard.stronger_pilot_no_replay import CandidateBData, load_boundary_checkpoint  # noqa: E402
from revision2_selector import MatchedBuffer, ranked_entry  # noqa: E402

ARM, SEED = "short", 42
STAGE = "AF3"
BASE = ROOT / "results/revision2/matched_ablation_amp256/M200/a2_matched/short/seed42"
OUT = ROOT / "results/revision2/selector_audit/common_model_state_af3_short_seed42.json"


def choose_by_score(entries: list[dict], key: str) -> list[str]:
    selected = []
    for label in (0, 1):
        for stage in ("AF2", "AF3"):
            bucket = [x for x in entries if x["label"] == label and x["task_id"] == stage]
            bucket.sort(key=lambda x: (-x[key], x["random_rank_sha256"], x["example_id"]))
            first, later, seen = [], [], set()
            for item in bucket:
                if item["coverage_key"] in seen:
                    later.append(item)
                else:
                    first.append(item); seen.add(item["coverage_key"])
            selected.extend(x["example_id"] for x in (first + later)[:50])
    assert len(selected) == len(set(selected)) == 200
    return selected


def overlap(a: list[str], b: list[str]) -> dict:
    sa, sb = set(a), set(b)
    return {"intersection": len(sa & sb), "symmetric_difference": len(sa ^ sb),
            "jaccard": len(sa & sb) / len(sa | sb), "ordered_equal": a == b}


def main() -> None:
    data = CandidateBData(ROOT, ARM)
    old = [json.loads(line) for line in (BASE / "stage1/buffer.jsonl").open(encoding="utf-8")]
    new = [ranked_entry(row, SEED) for row in data.records(STAGE, "train")]
    pool = [*old, *new]
    assert len(pool) == 600 and len({x["example_id"] for x in pool}) == 600
    lookup = {x["example_id"]: x for stage in ("AF2", "AF3") for x in data.records(stage, "train")}
    model, tokenizer, _ = load_boundary_checkpoint(BASE / "stage1/checkpoint")
    model.to("cuda"); model.eval()
    model.gradient_checkpointing_disable()
    with torch.inference_mode():
        for start in range(0, len(pool), 4):
            batch = pool[start:start + 4]
            source = [lookup[x["example_id"]] for x in batch]
            inputs = batch_inputs(source, tokenizer, 1280, "cuda")
            logits = model(**inputs).logits.float()
            labels = torch.tensor([x["label"] for x in batch], dtype=torch.long, device="cuda")
            ce = F.cross_entropy(logits, labels, reduction="none").cpu().tolist()
            probs = torch.softmax(logits, dim=-1)
            risk = (1 - probs[torch.arange(len(batch), device="cuda"), labels]).cpu().tolist()
            for item, c, r in zip(batch, ce, risk):
                item["common_state_ce"] = float(c)
                item["common_state_risk"] = float(r)
    buffer = MatchedBuffer("a3_matched", SEED)
    buffer.entries = old
    buffer.seen_count = 400
    buffer.update(new, ("AF2", "AF3"))
    random_ids = [x["example_id"] for x in buffer.entries]
    ce_ids = choose_by_score(pool, "common_state_ce")
    risk_ids = choose_by_score(pool, "common_state_risk")
    finite = [abs(x["common_state_ce"] + math.log1p(-x["common_state_risk"]))
              for x in pool if x["common_state_risk"] < 1]
    result = {
        "status": "PASS", "purpose": "selector-only common-state diagnostic; not a training outcome",
        "arm": ARM, "seed": SEED, "checkpoint": str(BASE / "stage1/checkpoint"),
        "eligible_count": len(pool), "unique_coverage_keys": len({x["coverage_key"] for x in pool}),
        "label_counts": dict(Counter(str(x["label"]) for x in pool)),
        "common_state": "A2 matched AF2 stage-end model, before AF3 training",
        "ce_distinct_values": len({x["common_state_ce"] for x in pool}),
        "risk_distinct_values": len({x["common_state_risk"] for x in pool}),
        "risk_saturated_at_one": sum(x["common_state_risk"] == 1 for x in pool),
        "max_abs_ce_minus_negative_log_one_minus_risk": max(finite),
        "A3_random_vs_A4_CE": overlap(random_ids, ce_ids),
        "A4_CE_vs_A5_risk": overlap(ce_ids, risk_ids),
        "selected_ids": {"A3_random": random_ids, "A4_CE": ce_ids, "A5_risk": risk_ids},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "selected_ids"}))


if __name__ == "__main__":
    main()
