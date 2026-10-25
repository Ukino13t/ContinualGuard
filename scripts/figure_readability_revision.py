"""Redraw paper Figures 1, 2, 3 and 5 from frozen aggregate artifacts.

Only writes to the separate acl_figure_readability source tree. It does not
change the frozen experiments or the Revision 5 manuscript/package.
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "paper" / "acl_figure_readability" / "figures"
PREVIEW = ROOT / "reports" / "figure_readability_20261024" / "figures"
STAGES = ("AF2", "AF3", "AF4", "AF5")
SEEDS = (17, 42, 2026)
METHODS = ("no_replay", "reservoir", "derpp", "security_aware", "loss_aware")
LABELS = {
    "no_replay": "No Replay", "reservoir": "Reservoir ER", "derpp": "DER++",
    "security_aware": "Security-aware", "loss_aware": "Loss-aware",
}
COLORS = {
    "no_replay": "#4C566A", "reservoir": "#0072B2", "derpp": "#009E73",
    "security_aware": "#D55E00", "loss_aware": "#CC79A7",
}
MARKERS = {"no_replay": "o", "reservoir": "o", "derpp": "s", "security_aware": "^", "loss_aware": "D"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 7.5,
    "axes.titlesize": 8.2, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_rows() -> tuple[list[dict], list[dict], list[dict]]:
    validation = json.loads((ROOT / "results/revision4/validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "PASS" and validation["runs_checked"] == 9
    older = ROOT / "results/replay_comparison_aggregate.json"
    newer = ROOT / "results/loss_aware_control_aggregate.json"
    with older.open(encoding="utf-8") as handle:
        a = json.load(handle)
    with newer.open(encoding="utf-8") as handle:
        b = json.load(handle)
    trajectory = a["trajectory_per_seed"] + b["trajectory_per_seed"]
    retention = a["retention_per_seed"] + b["retention_per_seed"]
    runs = a["run_summary_per_seed"] + b["run_summary_per_seed"]
    for row in trajectory + retention + runs:
        row.setdefault("method", "loss_aware")
    assert set(r["method"] for r in trajectory) == set(METHODS)
    return trajectory, retention, runs


def one(rows: list[dict], **where) -> dict:
    found = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
    assert len(found) == 1, (where, len(found))
    return found[0]


def mean_sd(values: list[float]) -> tuple[float, float]:
    assert len(values) == 3
    return statistics.mean(values), statistics.stdev(values)


def save(fig: plt.Figure, stem: str) -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    fig.savefig(DEST / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(PREVIEW / f"{stem}.png", dpi=240, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def fig1(trajectory: list[dict]) -> None:
    """Full-scale no-replay seed variation plus a marked replay-only zoom."""
    stages = ("AF3", "AF4", "AF5")
    xs = np.arange(3)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.34, 4.18), sharex=True,
                                      gridspec_kw={"height_ratios": [1.16, 1]})
    for method in METHODS[1:]:
        data = np.asarray([[100 * one(trajectory, method=method, arm="B-SHORT", seed=seed,
                                    checkpoint_stage=stage, task="AF3")["attack_tpr"]
                            for stage in stages] for seed in SEEDS])
        top.plot(xs, data.mean(axis=0), marker=MARKERS[method], markersize=4,
                 linewidth=1.35, color=COLORS[method], label=LABELS[method])
    top.set_ylim(94, 101)
    top.set_yticks([94, 96, 98, 100])
    top.set_ylabel("Attack TPR (%)")
    top.set_title("Replay means (zoom: 94-101%)", loc="left", pad=3)
    top.grid(axis="y", alpha=.2, linewidth=.5)
    top.legend(loc="lower left", ncol=2, frameon=False, handlelength=1.6,
               columnspacing=.7, borderaxespad=.3)

    data = np.asarray([[100 * one(trajectory, method="no_replay", arm="B-SHORT", seed=seed,
                                checkpoint_stage=stage, task="AF3")["attack_tpr"]
                        for stage in stages] for seed in SEEDS])
    for i, seed in enumerate(SEEDS):
        bottom.plot(xs, data[i], marker=".", markersize=3, linewidth=.85,
                    color="#9DA3AF", alpha=.8, linestyle=(0, (i + 2, 2)))
    bottom.plot(xs, data.mean(axis=0), marker="o", markersize=4.3,
                linewidth=1.8, color=COLORS["no_replay"])
    bottom.set_ylim(0, 101)
    bottom.set_yticks([0, 25, 50, 75, 100])
    bottom.set_ylabel("Attack TPR (%)")
    bottom.set_title("No Replay: mean and three seeds (full scale)", loc="left", pad=3)
    bottom.set_xticks(xs, ["After AF3", "After AF4", "After AF5"])
    bottom.grid(axis="y", alpha=.2, linewidth=.5)
    fig.subplots_adjust(left=.17, right=.98, top=.97, bottom=.09, hspace=.26)
    save(fig, "b_short_af3_retention")


def fig2(trajectory: list[dict]) -> None:
    """Partition-specific scales; BH-1 is explicitly a zero-only result."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.47),
                             gridspec_kw={"width_ratios": [1.18, 1.1, .78]})
    partitions = (("D0-1", "AF2", (0, 125), [0, 25, 50, 75, 100, 125]),
                  ("B0-2", "AF3", (0, 8), [0, 2, 4, 6, 8]))
    handles = []
    for ax, (part, task, limits, ticks) in zip(axes, partitions):
        stages = STAGES[STAGES.index(task):]
        for method in METHODS:
            means, sds = [], []
            for stage in stages:
                vals = [100 * one(trajectory, method=method, arm="B-HARD", seed=seed,
                                  checkpoint_stage=stage, task=task)
                        ["benign_fpr_by_partition"][part] for seed in SEEDS]
                m, sd = mean_sd(vals)
                means.append(m)
                sds.append(sd)
            handle = ax.errorbar(range(len(stages)), means, yerr=sds, marker=MARKERS[method],
                                 markersize=3.5, capsize=1.8, linewidth=1.15,
                                 elinewidth=.85, color=COLORS[method], label=LABELS[method])
            if ax is axes[0]:
                handles.append(handle)
        ax.set_title(f"{part} (from {task})", pad=2)
        ax.set_xticks(range(len(stages)), stages)
        ax.set_ylim(*limits)
        ax.set_yticks(ticks)
        ax.grid(axis="y", alpha=.2, linewidth=.5)
    axes[0].set_ylabel("Benign FPR (%)")
    axes[1].set_ylabel("Benign FPR (%)")
    ax = axes[2]
    for method in METHODS:
        for seed in SEEDS:
            for stage in ("AF4", "AF5"):
                value = one(trajectory, method=method, arm="B-HARD", seed=seed,
                            checkpoint_stage=stage, task="AF4")
                assert value["benign_fpr_by_partition"]["BH-1"] == 0
    ax.set_title("BH-1 (from AF4)", pad=2)
    ax.set_axis_off()
    ax.text(.5, .54, "0% FPR", fontsize=13, weight="bold", ha="center", va="center",
            transform=ax.transAxes)
    ax.text(.5, .33, "AF4 and AF5\nall 5 methods x 3 seeds", fontsize=7.3,
            ha="center", va="center", transform=ax.transAxes)
    fig.legend(handles, [LABELS[m] for m in METHODS], loc="upper center", ncol=5,
               bbox_to_anchor=(.5, 1.03), frameon=False, columnspacing=.85,
               handlelength=1.5)
    fig.subplots_adjust(left=.07, right=.99, bottom=.17, top=.76, wspace=.36)
    save(fig, "b_hard_benign_retention")


def fig3(retention: list[dict], runs: list[dict]) -> None:
    """Show the outlying no-replay mean separately from a replay-only zoom."""
    def point(method: str) -> tuple[float, float, float, float]:
        x = [100 * one(retention, method=method, arm="B-SHORT", seed=seed,
                       task="AF3")["attack_tpr_loss"] for seed in SEEDS]
        y = [100 * one(runs, method=method, arm="B-HARD", seed=seed)
             ["benign_fpr_increase"] for seed in SEEDS]
        xm, xs = mean_sd(x)
        ym, ys = mean_sd(y)
        return xm, xs, ym, ys

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.96),
                                      gridspec_kw={"width_ratios": [1, 1.07]})
    xm, xs, ym, ys = point("no_replay")
    left.errorbar(xm, ym, xerr=xs, yerr=ys, fmt="o", color=COLORS["no_replay"],
                  capsize=2.8, markersize=5, linewidth=1.4)
    left.set_xlim(-6, 77)
    left.set_ylim(-3, 42)
    left.set_title("No Replay: full range", pad=3)
    left.text(.04, .94, f"mean ({xm:.1f}, {ym:.1f}) pp\nwhiskers: sample SD", transform=left.transAxes,
              va="top", fontsize=7, color=COLORS["no_replay"])
    for method in METHODS[1:]:
        xm, xs, ym, ys = point(method)
        right.errorbar(xm, ym, xerr=xs, yerr=ys, fmt=MARKERS[method],
                       color=COLORS[method], capsize=2.5, markersize=5,
                       linewidth=1.1, label=LABELS[method])
    right.set_xlim(-3.6, 2.7)
    right.set_ylim(-2.1, 5.1)
    right.set_title("Replay methods: zoom", pad=3)
    for ax in (left, right):
        ax.axhline(0, color="#888", linewidth=.65, zorder=0)
        ax.axvline(0, color="#888", linewidth=.65, zorder=0)
        ax.grid(alpha=.15, linewidth=.5)
        ax.set_xlabel("AF3 attack-TPR loss (pp)")
    left.set_ylabel("Benign-FPR increase (pp)")
    right.set_ylabel("Benign-FPR increase (pp)")
    right.legend(loc="upper right", frameon=False, fontsize=6.7,
                 borderaxespad=.25, handlelength=1.2)
    fig.subplots_adjust(left=.075, right=.99, bottom=.2, top=.86, wspace=.32)
    save(fig, "malicious_benign_tradeoff")


def fig5() -> None:
    with (ROOT / "results/revision4/analysis/memory_composition.csv").open(
            encoding="utf-8", newline="") as handle:
        memory = list(csv.DictReader(handle))
    with (ROOT / "results/revision4/selector_audit/candidate_pool_audit.csv").open(
            encoding="utf-8", newline="") as handle:
        audit = list(csv.DictReader(handle))
    arms = ("doc", "short", "hard")
    methods = ("A1_matched", "A2_canonical")
    positions, labels, values = [], [], []
    for arm in arms:
        for method in methods:
            records = [r for r in memory if r["arm"] == arm and r["method"] == method
                       and r["stage"] == "AF5"]
            assert len(records) == 3
            vals = [statistics.mean(int(json.loads(r["by_stage"]).get(stage, 0))
                                    for r in records) for stage in STAGES]
            positions.append(len(positions))
            labels.append(f"{arm.upper()}\n{'A1' if method == methods[0] else 'A2'}")
            values.append(vals)
    fig, ax = plt.subplots(figsize=(3.35, 2.85))
    bottom = np.zeros(len(positions))
    for j, (stage, color) in enumerate(zip(STAGES,
                                           ("#0072B2", "#E69F00", "#009E73", "#D55E00"))):
        heights = np.asarray([v[j] for v in values])
        ax.bar(positions, heights, bottom=bottom, label=stage, width=.7, color=color)
        bottom += heights
    assert all(abs(v - 200) < 1e-9 for v in bottom)
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 210)
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_ylabel("Selected records (M=200)")
    ax.grid(axis="y", alpha=.18, linewidth=.5)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.025), ncol=4,
              frameon=False, columnspacing=.65, handlelength=1.1, borderaxespad=0)
    fig.subplots_adjust(left=.2, right=.99, bottom=.16, top=.79)
    save(fig, "revision4_stage_memory_composition")

    fig, ax = plt.subplots(figsize=(3.35, 2.85))
    for arm, color, marker in (("doc", "#0072B2", "o"), ("short", "#D55E00", "s"),
                               ("hard", "#009E73", "^")):
        means = [statistics.mean(float(r["actual_selected_jaccard"]) for r in audit
                                 if r["arm"] == arm and r["stage"] == stage)
                 for stage in STAGES]
        assert all(.84 <= v <= 1 for v in means)
        ax.plot(STAGES, means, marker=marker, markersize=4.5, linewidth=1.4,
                color=color, label=f"B-{arm.upper()}")
    ax.set_ylabel("A1/A2 selected-ID Jaccard")
    ax.set_ylim(.84, 1.01)
    ax.set_yticks([.85, .9, .95, 1])
    ax.grid(axis="y", alpha=.2, linewidth=.5)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.025), ncol=3,
              frameon=False, columnspacing=.55, handlelength=1.1, borderaxespad=0)
    fig.subplots_adjust(left=.2, right=.99, bottom=.16, top=.79)
    save(fig, "revision4_selected_buffer_overlap")


def main() -> None:
    trajectory, retention, runs = source_rows()
    fig1(trajectory)
    fig2(trajectory)
    fig3(retention, runs)
    fig5()
    sources = ["results/replay_comparison_aggregate.json",
               "results/loss_aware_control_aggregate.json",
               "results/revision4/analysis/memory_composition.csv",
               "results/revision4/selector_audit/candidate_pool_audit.csv",
               "results/revision4/validation.json"]
    manifest = {"source_sha256": {rel: sha256(ROOT / rel) for rel in sources},
                "output_sha256": {p.name: sha256(p) for p in sorted(DEST.glob("*.pdf"))
                                  if p.name in {"b_short_af3_retention.pdf",
                                                "b_hard_benign_retention.pdf",
                                                "malicious_benign_tradeoff.pdf",
                                                "revision4_stage_memory_composition.pdf",
                                                "revision4_selected_buffer_overlap.pdf"}}}
    report = PREVIEW.parent / "figure_data_manifest.json"
    report.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "figures": len(manifest["output_sha256"]),
                      "manifest": str(report)}))


if __name__ == "__main__":
    main()
