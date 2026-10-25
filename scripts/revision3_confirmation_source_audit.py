"""Metadata-only overlap audit for proposed independent confirmation sources."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
OUT = ROOT / "reports/revision3/confirmation_source_audit.json"
KEYS = ("example_id", "text_sha256", "normalized_text_sha256", "model_view_sha256",
        "source_native_id", "duplicate_group_id", "base_document_id", "template_id")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def fields(rows):
    return {key: {str(row[key]) for row in rows if row.get(key) is not None} for key in KEYS}


def compare(a, b):
    left, right = fields(a), fields(b)
    return {key: len(left[key] & right[key]) for key in KEYS}


def main() -> None:
    candidate = {split: [] for split in ("train", "val", "test")}
    for arm in ("b_doc", "b_short", "b_hard"):
        for stage in range(1, 5):
            for split in candidate:
                candidate[split].extend(read_jsonl(ROOT / "data/processed/stronger_pilot_candidate_b" / arm /
                                                  f"stage{stage}/{split}.jsonl"))
    original = read_jsonl(ROOT / "data/processed/chen_attack_family_shift.jsonl")
    h2 = {split: [] for split in candidate}
    for arm in ("h2m_c0", "h2m_f"):
        for stage in range(1, 5):
            for split in h2:
                h2[split].extend(read_jsonl(ROOT / "data/processed/h2_mini" / arm /
                                             f"stage{stage}/{split}.jsonl"))
    original_by_split = {split: [r for r in original if r.get("split") == split]
                         for split in candidate}
    result = {"candidate_b_counts": {split: len(rows) for split, rows in candidate.items()},
              "original_family_counts": {split: len(rows) for split, rows in original_by_split.items()},
              "h2_mini_arm_membership_counts": {split: len(rows) for split, rows in h2.items()},
              "candidate_test_vs_sources": {
                  "original_family": {split: compare(candidate["test"], rows)
                                      for split, rows in original_by_split.items()},
                  "h2_mini": {split: compare(candidate["test"], rows)
                              for split, rows in h2.items()}},
              "candidate_train_vs_sources": {
                  "original_family": {split: compare(candidate["train"], rows)
                                      for split, rows in original_by_split.items()},
                  "h2_mini": {split: compare(candidate["train"], rows)
                              for split, rows in h2.items()}},
              "source_composition": {
                  "original_family": dict(Counter((r.get("task_id"), r.get("label")) for r in original)),
              }}
    # Tuple dictionary keys are not JSON keys; preserve composition as rows.
    result["source_composition"]["original_family"] = [
        {"stage": stage, "label": label, "count": count}
        for (stage, label), count in sorted(Counter((r.get("task_id"), r.get("label")) for r in original).items())]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_test_vs_sources": result["candidate_test_vs_sources"],
                      "candidate_train_vs_sources": result["candidate_train_vs_sources"]}))


if __name__ == "__main__":
    main()
