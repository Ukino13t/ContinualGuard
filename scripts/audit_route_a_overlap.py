"""Quantify A2/A3 selected-memory overlap without reading prompt text."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
BASE = ROOT / "results/extended/ablation"
OUT = ROOT / "results/extended/analysis/a2_a3_memory_overlap.csv"
rows = []
for arm in ("doc", "short", "hard"):
    for seed in (17, 42, 2026):
        for index, stage in enumerate(("AF2", "AF3", "AF4", "AF5"), 1):
            buffers = []
            for method in ("a2_class_stage", "a3_coverage"):
                file = BASE / method / arm / f"seed{seed}" / f"stage{index}" / "buffer.jsonl"
                buffers.append({json.loads(line)["example_id"] for line in file.open(encoding="utf-8")})
            assert len(buffers[0]) == len(buffers[1]) == 200
            shared = len(buffers[0] & buffers[1])
            rows.append({"arm": arm, "seed": seed, "stage": stage,
                         "a2_size": 200, "a3_size": 200, "shared_ids": shared,
                         "jaccard": shared / len(buffers[0] | buffers[1])})

with OUT.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
for stage in ("AF2", "AF3", "AF4", "AF5"):
    values = [row["jaccard"] for row in rows if row["stage"] == stage]
    print(stage, f"mean Jaccard={statistics.mean(values):.3f}",
          f"sample SD={statistics.stdev(values):.3f}", "n=9")
