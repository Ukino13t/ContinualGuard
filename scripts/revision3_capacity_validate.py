"""Full identity, data-access, checkpoint, and matched-pair audit for P1."""
from __future__ import annotations

import json
from pathlib import Path

from scripts import revision2_validate as r2
from scripts import revision3_validate as r3
from scripts.revision2_selector import MatchedBuffer, ranked_entry

ROOT = Path(r"E:\ContinualGuard")
ZROOT = Path(r"Z:\ContinualGuard_Revision3")
ZBASE = ZROOT / "results/revision3/matched_ablation_amp256"
EBASE = ROOT / "results/revision2/matched_ablation_amp256"
ARMS = ("short", "hard")
SEEDS = (17, 42, 2026)
CAPACITIES = (50, 200, 400)
OUT = ROOT / "results/revision3/analysis/capacity_validation.json"


def source(variant: str, arm: str, seed: int, cap: int) -> Path:
    if variant == "a0_matched":
        return ZBASE / f"M{cap}" / variant / arm / f"seed{seed}"
    if cap == 200 or (arm == "hard" and seed == 42 and cap in (50, 400)):
        return EBASE / f"M{cap}" / variant / arm / f"seed{seed}"
    return ZBASE / f"M{cap}" / variant / arm / f"seed{seed}"


def main() -> None:
    protocol = r2.load(ROOT / "results/revision3/capacity_protocol_v1.json")
    assert protocol["status"] == "FROZEN_BEFORE_CAPACITY_TRAINING"
    results, pairs = [], []
    try:
        for arm in ARMS:
            for seed in SEEDS:
                for cap in CAPACITIES:
                    left = source("a0_matched", arm, seed, cap)
                    right = source("a1_matched", arm, seed, cap)
                    for variant, run in (("a0_matched", left), ("a1_matched", right)):
                        key = f"{variant}/{arm}/seed{seed}/M{cap}"
                        assert r2.sha256(run / "resolved_config.yaml") == protocol["run_config_sha256"][key]
                        if variant == "a0_matched":
                            row = r3.validate(arm, seed, cap, True)
                        else:
                            r2.MATCHED = run.parents[3]
                            row = r2.validate_matched(variant, cap, arm, seed, True)
                            assert row is not None
                            buffer = MatchedBuffer("a1_matched", seed, cap)
                            data = r3.CandidateBData(ROOT, arm)
                            for i, stage in enumerate(r3.STAGES, 1):
                                buffer.update([ranked_entry(x, seed) for x in data.records(stage, "train")],
                                              r3.STAGES[:i])
                                saved = r2.jsonl(run / f"stage{i}/buffer.jsonl")
                                assert [x["example_id"] for x in saved] == [
                                    x["example_id"] for x in buffer.entries]
                        for i in range(1, 5):
                            assert r2.load(run / f"stage{i}/stage_start.json")["optimizer_reset"] is True
                        row["source"] = str(run)
                        results.append(row)
                    assert r2.load(left / "initialization.json") == r2.load(right / "initialization.json")
                    current_orders_equal = True
                    for stage in range(1, 5):
                        l = r2.jsonl(left / f"stage{stage}/train_log.jsonl")
                        r = r2.jsonl(right / f"stage{stage}/train_log.jsonl")
                        current_orders_equal &= [x["example_ids"] for x in l] == [x["example_ids"] for x in r]
                    assert current_orders_equal
                    assert (r2.load(left / "stage1/checkpoint/checkpoint_manifest.json")["files"]["model.safetensors"] ==
                            r2.load(right / "stage1/checkpoint/checkpoint_manifest.json")["files"]["model.safetensors"])
                    pairs.append({"arm": arm, "seed": seed, "capacity": cap,
                                  "same_initialization": True, "same_current_order": True,
                                  "same_first_stage_model": True})
    finally:
        r2.MATCHED = EBASE
    assert len(results) == 36 and len(pairs) == 18
    report = {"status": "PASS", "full_checkpoint_hashes": True,
              "runs_checked": len(results), "matched_pairs": pairs,
              "new_runs": 22, "reused_runs": 14, "runs": results}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "runs_checked": len(results),
                      "matched_pairs": len(pairs), "full_checkpoint_hashes": True}))


if __name__ == "__main__":
    main()
