"""Publication-ready descriptive figures for matched Revision 3 evidence."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"E:\ContinualGuard")
DATA = ROOT / "results/revision3/analysis"
SELECTOR = ROOT / "results/revision3/selector_audit/selector_cells.csv"
OUT = ROOT / "figures/revision3"
STAGES = ("AF2", "AF3", "AF4", "AF5")
COLORS = {"A0_matched": "#e69f00", "A1_matched": "#0072b2",
          "M1_Reservoir_original": "#666666"}
LABELS = {"A0_matched": "A0 priority", "A1_matched": "A1 class-balanced",
          "M1_Reservoir_original": "M1 Reservoir (context)"}


def rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_trajectories():
    data = rows(DATA / "stage_curves.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.0), constrained_layout=True)
    specs = (("short", "AF3", "attack_tpr_pct", "B-SHORT: AF3 attack TPR", axes[0]),
             ("hard", "AF2", "benign_fpr_pct", "B-HARD: AF2 benign FPR", axes[1]))
    for arm, task, metric, title, ax in specs:
        ticks = STAGES[STAGES.index(task):]
        plotted_bounds = []
        for method in COLORS:
            means, sds = [], []
            for trained in ticks:
                values = [float(r[metric]) for r in data if r["arm"] == arm and
                          r["eval_stage"] == task and r["trained_stage"] == trained and
                          r["method"] == method]
                assert len(values) == 3
                means.append(statistics.mean(values))
                sds.append(statistics.stdev(values))
            x = list(range(len(ticks)))
            ax.plot(x, means, marker="o", markersize=4, linewidth=1.8,
                    linestyle="--" if method.startswith("M1") else "-",
                    label=LABELS[method], color=COLORS[method])
            ax.fill_between(x, [m - s for m, s in zip(means, sds)],
                            [m + s for m, s in zip(means, sds)],
                            color=COLORS[method], alpha=.10)
            plotted_bounds.extend(v for m, s in zip(means, sds) for v in (m - s, m + s))
        ax.set_xticks(range(len(ticks)), ticks)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Percent")
        ax.grid(alpha=.2)
        lower = max(0.0, min(plotted_bounds) - 3.0)
        upper = min(100.0, max(plotted_bounds) + 3.0)
        if upper - lower < 8.0:
            lower = max(0.0, lower - 4.0)
            upper = min(100.0, upper + 4.0)
        ax.set_ylim(lower, upper)
    axes[0].legend(loc="lower left", fontsize=7, frameon=False)
    fig.savefig(OUT / "matched_retention_and_fpr.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_memory():
    data = rows(DATA / "memory_composition.csv")
    selector = rows(SELECTOR)
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.0), constrained_layout=True)
    arms = ("doc", "short", "hard")
    for j, method in enumerate(("A0_matched", "A1_matched")):
        means, sds = [], []
        for arm in arms:
            values = [json.loads(r["by_label"])["0"] for r in data if r["arm"] == arm and
                      r["method"] == method and r["stage"] == "AF5"]
            assert len(values) == 3
            means.append(statistics.mean(values))
            sds.append(statistics.stdev(values))
        x = [i + (j - .5) * .28 for i in range(3)]
        axes[0].bar(x, means, width=.28, yerr=sds, capsize=2,
                    color=COLORS[method], label=LABELS[method])
    axes[0].axhline(100, color="black", linewidth=.7, linestyle=":")
    axes[0].set_xticks(range(3), ("B-DOC", "B-SHORT", "B-HARD"))
    axes[0].set_ylabel("Benign records in 200-item final buffer")
    axes[0].set_title("Memory label composition", fontsize=10)
    axes[0].legend(fontsize=7, frameon=False)
    for arm, color in zip(arms, ("#009e73", "#d55e00", "#cc79a7")):
        means = []
        for stage in STAGES:
            values = [float(r["actual_selected_jaccard"]) for r in selector if
                      r["arm"] == arm and r["stage"] == stage]
            assert len(values) == 3
            means.append(statistics.mean(values))
        axes[1].plot(range(4), means, marker="o", label=f"B-{arm.upper()}",
                     color=color, linewidth=1.6)
    axes[1].set_xticks(range(4), STAGES)
    axes[1].set_ylim(.89, 1.005)
    axes[1].set_ylabel("A0/A1 selected-ID Jaccard")
    axes[1].set_title("Selected-buffer overlap", fontsize=10)
    axes[1].legend(fontsize=7, frameon=False)
    axes[1].grid(alpha=.2)
    fig.savefig(OUT / "matched_memory_and_overlap.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    validation = json.loads((DATA / "validation_M200.json").read_text(encoding="utf-8"))
    assert validation["status"] == "PASS" and validation["runs_checked"] == 9
    OUT.mkdir(parents=True, exist_ok=True)
    plot_trajectories(); plot_memory()
    print("PASS: two vector figures generated from validated matched evidence")


if __name__ == "__main__":
    main()
