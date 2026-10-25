"""Reproducible matched-selector vector plots from validated stage CSVs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import plot_route_a as plotter

ROOT = Path(r"E:\ContinualGuard")


def main():
    validation = json.loads((ROOT / "results/revision2/analysis/validation.json").read_text(encoding="utf-8"))
    if (validation.get("status"), validation.get("full_checkpoint_hashes"),
            validation.get("runs_checked"), len(validation.get("matched_pair_controls", []))) != ("PASS", True, 24, 9):
        raise RuntimeError("Incomplete full-hash or paired-control validation")
    path = ROOT / "results/revision2/analysis/stage_curves.csv"
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    required = ("M0_NoReplay", "A0_Reservoir", "A1_matched", "A2_matched")
    for method in required:
        for arm in ("doc", "short", "hard"):
            seeds = {row["seed"] for row in rows if row["method"] == method and row["arm"] == arm}
            if seeds != {"17", "42", "2026"}:
                raise RuntimeError(f"Incomplete three-seed plot source: {method}/{arm}: {seeds}")
    for row in rows:
        row["eval_stage"] = row["tested_stage"]
    plotter.OUT = ROOT / "figures/revision2"
    plotter.COLORS = {
        "M0_NoReplay": (0.08, 0.08, 0.08),
        "A0_Reservoir": (0.0, 0.45, 0.70),
        "A1_matched": (0.90, 0.62, 0.0),
        "A2_matched": (0.0, 0.62, 0.45),
    }
    plotter.LABELS = {"M0_NoReplay": "No Replay", "A0_Reservoir": "Reservoir ER",
                      "A1_matched": "A1 matched", "A2_matched": "A2 matched"}
    plotter.ORDER = tuple(plotter.COLORS)
    plotter.plot(rows, "short", "AF3", "attack_tpr_pct",
                 "B-SHORT: AF3 attack recall under matched ranking", "b_short_af3_matched")
    plotter.plot(rows, "hard", "AF2", "benign_fpr_pct",
                 "B-HARD: AF2 benign FPR under matched ranking", "b_hard_af2_matched")
    print(plotter.OUT)


if __name__ == "__main__":
    main()
