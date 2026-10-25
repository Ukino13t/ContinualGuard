"""Selector-only audit on frozen Candidate B TRAIN records; no model training."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from continualguard.stronger_pilot_no_replay import CandidateBData, STAGES, SEEDS  # noqa: E402
from revision2_selector import MatchedBuffer, VARIANTS, ranked_entry  # noqa: E402

OUT = ROOT / "results/revision2/selector_audit"


def counts(entries: list[dict], key: str) -> dict:
    return dict(sorted(Counter(str(x.get(key)) for x in entries).items()))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, identities = [], []
    for arm in ("doc", "short", "hard"):
        data = CandidateBData(ROOT, arm)
        for seed in SEEDS:
            buffers = {v: MatchedBuffer(v, seed) for v in VARIANTS}
            for index, stage in enumerate(STAGES, 1):
                new = [ranked_entry(row, seed) for row in data.records(stage, "train")]
                snapshots = {}
                for variant, buffer in buffers.items():
                    old = list(buffer.entries)
                    pool = [*old, *new]
                    key_counts = Counter(x["coverage_key"] for x in pool)
                    result = buffer.update(new, tuple(STAGES[:index]))
                    snapshots[variant] = {x["example_id"] for x in buffer.entries}
                    rows.append({
                        "arm": arm, "seed": seed, "stage": stage, "variant": variant,
                        "current_count": len(new), "previous_memory_count": len(old),
                        "eligible_count": len(pool), "eligible_unique_keys": len(key_counts),
                        "duplicate_key_count": sum(n > 1 for n in key_counts.values()),
                        "max_key_multiplicity": max(key_counts.values()),
                        "keys_competing_for_capacity": len(key_counts),
                        "selected_count": len(buffer.entries),
                        "selected_unique_keys": len({x["coverage_key"] for x in buffer.entries}),
                        "coverage_reordered_positions": result["coverage_reordered_positions"],
                        "label_counts": json.dumps(counts(buffer.entries, "label"), sort_keys=True),
                        "stage_counts": json.dumps(counts(buffer.entries, "task_id"), sort_keys=True),
                        "benign_group_counts": json.dumps(counts(buffer.entries, "benign_group"), sort_keys=True),
                        "attack_family_counts": json.dumps(counts(buffer.entries, "attack_family"), sort_keys=True),
                        "quota": json.dumps(result["quotas"], sort_keys=True),
                    })
                    identities.append({"arm": arm, "seed": seed, "stage": stage,
                                       "variant": variant,
                                       "selected_ids": [x["example_id"] for x in buffer.entries]})
                if snapshots["a2_matched"] != snapshots["a3_matched"]:
                    raise AssertionError(f"Coverage changed selected IDs: {arm}/{seed}/{stage}")
                for a, b in (("a1_matched", "a2_matched"), ("a2_matched", "a3_matched")):
                    common = {"arm": arm, "seed": seed, "stage": stage, "variant": f"{a}_vs_{b}",
                              "jaccard": jaccard(snapshots[a], snapshots[b]),
                              "symmetric_difference_count": len(snapshots[a] ^ snapshots[b])}
                    rows.append(common)
    columns = sorted({key for item in rows for key in item})
    path = OUT / "actual_candidate_pools.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    with (OUT / "selected_identities.jsonl").open("w", encoding="utf-8") as handle:
        for item in identities:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    summaries = {"status": "PASS", "arm_seed_stage_cells": 36,
                 "all_a2_a3_selected_sets_equal": True,
                 "max_eligible_duplicate_keys": max(int(x["duplicate_key_count"]) for x in rows if x.get("duplicate_key_count") is not None),
                 "a1_a2_changed_cells": sum(x.get("symmetric_difference_count", 0) > 0 for x in rows if x.get("variant") == "a1_matched_vs_a2_matched"),
                 "a2_a3_changed_cells": sum(x.get("symmetric_difference_count", 0) > 0 for x in rows if x.get("variant") == "a2_matched_vs_a3_matched")}
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries))


if __name__ == "__main__":
    main()
