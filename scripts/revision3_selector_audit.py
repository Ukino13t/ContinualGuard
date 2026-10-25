"""Pre-training A0/A1 audit on the frozen Candidate B TRAIN identities."""
from __future__ import annotations

import copy
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from continualguard.stronger_pilot_no_replay import CandidateBData, SEEDS, STAGES  # noqa: E402
from scripts.revision2_selector import MatchedBuffer, ranked_entry  # noqa: E402
from scripts.revision3_selector import A0MatchedBuffer  # noqa: E402

OUT = ROOT / "results/revision3/selector_audit"


def counts(entries: list[dict], key: str) -> str:
    return json.dumps(dict(sorted(Counter(str(e.get(key)) for e in entries).items())), sort_keys=True)


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def rank_median(entries: list[dict]) -> float:
    return statistics.median(int(e["random_rank_sha256"], 16) / 2**256 for e in entries)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, selected = [], []
    for arm in ("doc", "short", "hard"):
        data = CandidateBData(ROOT, arm)
        train_ids = {stage: {r["example_id"] for r in data.records(stage, "train")}
                     for stage in STAGES}
        disallowed = {r["example_id"] for stage in STAGES for split in ("val", "test")
                      for r in data.records(stage, split)}
        for seed in SEEDS:
            a0 = A0MatchedBuffer("a0_matched", seed)
            a1 = MatchedBuffer("a1_matched", seed)
            for index, stage in enumerate(STAGES, 1):
                current = [ranked_entry(r, seed) for r in data.records(stage, "train")]
                a0_before = copy.deepcopy(a0.entries)
                a1_before = copy.deepcopy(a1.entries)
                assert {e["example_id"] for e in current} == train_ids[stage]
                assert not ({e["example_id"] for e in [*a0_before, *a1_before, *current]} & disallowed)
                a0_update = a0.update(current, STAGES[:index])
                a1_update = a1.update(current, STAGES[:index])
                common = A0MatchedBuffer("a0_matched", seed)
                common.entries = a1_before
                common.update(current, STAGES[:index])
                assert a0.sample_rng.getstate() == a1.sample_rng.getstate()
                assert len(a0.entries) == len(a1.entries) == 200
                frozen = ROOT / "results/revision2/matched_ablation_amp256/M200/a1_matched" / arm / f"seed{seed}" / f"stage{index}/buffer.jsonl"
                actual_a1 = [json.loads(line)["example_id"] for line in frozen.read_text(encoding="utf-8").splitlines()]
                assert [e["example_id"] for e in a1.entries] == actual_a1, frozen
                a0_ids = {e["example_id"] for e in a0.entries}
                a1_ids = {e["example_id"] for e in a1.entries}
                common_ids = {e["example_id"] for e in common.entries}
                rows.append({"arm": arm, "seed": seed, "stage": stage,
                             "a0_eligible": a0_update["eligible_count"],
                             "a1_eligible": a1_update["eligible_count"],
                             "actual_pool_jaccard": jaccard(
                                 {e["example_id"] for e in [*a0_before, *current]},
                                 {e["example_id"] for e in [*a1_before, *current]}),
                             "actual_selected_jaccard": jaccard(a0_ids, a1_ids),
                             "same_pool_counterfactual_jaccard": jaccard(common_ids, a1_ids),
                             "actual_quota_affected_count": len(a0_ids ^ a1_ids) // 2,
                             "same_pool_quota_affected_count": len(common_ids ^ a1_ids) // 2,
                             "a0_label_counts": counts(a0.entries, "label"),
                             "a1_label_counts": counts(a1.entries, "label"),
                             "a0_stage_counts": counts(a0.entries, "task_id"),
                             "a1_stage_counts": counts(a1.entries, "task_id"),
                             "a0_median_rank": rank_median(a0.entries),
                             "a1_median_rank": rank_median(a1.entries),
                             "frozen_a1_ids_matched": True,
                             "shared_rank_and_independent_rng": True,
                             "no_validation_test_future_leakage": True})
                for name, buffer in (("a0_matched", a0), ("a1_matched", a1)):
                    selected.append({"arm": arm, "seed": seed, "stage": stage,
                                     "variant": name,
                                     "selected_ids": [e["example_id"] for e in buffer.entries]})
    with (OUT / "selector_cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (OUT / "selected_ids.jsonl").open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    summary = {"status": "PASS", "cells": len(rows),
               "cells_with_actual_selection_change": sum(r["actual_quota_affected_count"] > 0 for r in rows),
               "cells_with_same_pool_quota_intervention": sum(r["same_pool_quota_affected_count"] > 0 for r in rows),
               "mean_actual_selected_jaccard": statistics.mean(r["actual_selected_jaccard"] for r in rows),
               "mean_same_pool_jaccard": statistics.mean(r["same_pool_counterfactual_jaccard"] for r in rows),
               "all_frozen_a1_buffers_reproduced": True,
               "note": "After an intervention changes memory, actual later candidate pools diverge by design; the same-pool counterfactual isolates the immediate quota effect."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
