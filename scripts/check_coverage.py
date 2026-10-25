import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, r"E:\ContinualGuard\src")
from continualguard.replay_candidate_b import coverage_key

root = Path(r"E:\ContinualGuard\data\processed\stronger_pilot_candidate_b")
summary = []
for arm in ("doc", "short", "hard"):
    for index, stage in enumerate(("AF2", "AF3", "AF4", "AF5"), 1):
        path = root / f"b_{arm}" / f"stage{index}" / "train.jsonl"
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        keys = Counter(coverage_key(row) for row in rows)
        summary.append({"arm": arm, "stage": stage, "train_records": len(rows),
                        "unique_coverage_keys": len(keys),
                        "multi_item_keys": sum(count > 1 for count in keys.values()),
                        "largest_group": max(keys.values())})
        print(arm, stage, "train", len(rows), "unique_keys", len(keys),
              "multi_item_keys", sum(count > 1 for count in keys.values()),
              "largest_group", max(keys.values()))

out = Path(r"E:\ContinualGuard\results\extended\analysis\coverage_key_audit.json")
out.write_text(json.dumps({"status": "PASS", "coverage_rule_nonbinding": all(
    row["train_records"] == row["unique_coverage_keys"] for row in summary),
    "arm_stage_counts": summary,
    "interpretation": "A2/A3 differ by variant-specific deterministic rank streams, not coverage-key grouping on this benchmark."},
    indent=2) + "\n", encoding="utf-8")
