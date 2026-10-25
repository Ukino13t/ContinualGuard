"""Plot the validated matched capacity grid with separate security axes."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"E:\ContinualGuard")
DATA = ROOT / "results/revision3/analysis"
OUT = ROOT / "figures/revision3"


def main():
    check = json.loads((DATA / "capacity_validation.json").read_text(encoding="utf-8"))
    assert check["status"] == "PASS" and check["runs_checked"] == 36
    with (DATA / "capacity_aggregate.csv").open(encoding="utf-8", newline="") as handle:
        data = list(csv.DictReader(handle))
    for endpoint, title, filename in (
        ("tpr_bwt_pp", "Attack-TPR backward transfer", "capacity_attack_retention.pdf"),
        ("benign_fpr_increase_pp", "Benign-FPR increase", "capacity_benign_fpr.pdf"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.9), sharey=False,
                                 constrained_layout=True)
        for ax, arm in zip(axes, ("short", "hard")):
            for method, color, label in (("a0_matched", "#e69f00", "A0 priority"),
                                         ("a1_matched", "#0072b2", "A1 class-balanced")):
                subset = sorted((r for r in data if r["arm"] == arm and
                                 r["method"] == method), key=lambda r: int(r["capacity"]))
                assert [int(r["capacity"]) for r in subset] == [50, 200, 400]
                x = [int(r["capacity"]) for r in subset]
                y = [float(r[endpoint + "_mean"]) for r in subset]
                err = [float(r[endpoint + "_sample_sd"]) for r in subset]
                ax.errorbar(x, y, yerr=err, marker="o", linewidth=1.6,
                            capsize=3, color=color, label=label)
            ax.axhline(0, color="#666666", linewidth=.7, linestyle=":")
            ax.set_xticks((50, 200, 400))
            ax.set_xlabel("Memory capacity (examples)")
            ax.set_ylabel("Percentage points")
            ax.set_title(f"B-{arm.upper()}")
            ax.grid(alpha=.20)
        axes[0].legend(fontsize=8, frameon=False)
        fig.suptitle(title + " (mean +/- sample SD; three seeds)", fontsize=11)
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / filename, bbox_inches="tight")
        plt.close(fig)
    print("PASS: separate malicious-retention and benign-FPR capacity plots")


if __name__ == "__main__":
    main()
