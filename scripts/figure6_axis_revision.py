"""Redraw only paper Figure 6 with independent middle/right y-axis ranges."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "acl_figure6_readability"
CSV = PAPER / "data" / "appendix_E_external_diagnostics.csv"
OUT = PAPER / "figures" / "external_overdefense.pdf"
REPORT = ROOT / "reports" / "figure6_readability_20261024"
METHODS = ("no_replay", "reservoir", "derpp", "security_aware", "loss_aware")
ARMS = ("B-DOC", "B-SHORT", "B-HARD")
SEEDS = (17, 42, 2026)
COLORS = {"no_replay": "#4C566A", "reservoir": "#0072B2", "derpp": "#009E73",
          "security_aware": "#D55E00", "loss_aware": "#CC79A7"}
NAMES = {"no_replay": "No Replay", "reservoir": "Reservoir ER", "derpp": "DER++",
         "security_aware": "Security-aware", "loss_aware": "Loss-aware"}


def hashfile(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from scripts import build_final_analysis_package as frozen

    with CSV.open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["checkpoint_stage"] == "AF5"]
    assert len(rows) == 45
    lookup = {(r["method"], r["arm"], int(r["seed"])): r for r in rows}
    assert len(lookup) == 45
    for (method, arm, seed), row in lookup.items():
        original = frozen.find_rows(frozen.EXTERNAL, method=method, arm=arm,
                                    checkpoint_stage="AF5", seed=seed)
        assert len(original) == 1
        assert abs(float(row["notinject_fpr_pct"]) - frozen.pct(original[0]["notinject_fpr"])) < 1e-9
        assert abs(float(row["bshort_fpr_pct"]) - frozen.pct(original[0]["bshort_fpr"])) < 1e-9

    frozen.set_plot_style(plt)
    # Match the original three-panel size, order, palette and left-arm content.
    # Only remove shared y-scaling and set informative ranges for middle/right.
    fig, axes = plt.subplots(1, 3, figsize=(11.1, 3.8), sharey=False)
    width = .16
    x = np.arange(2)
    summary = {}
    for ax, arm in zip(axes, ARMS):
        summary[arm] = {}
        for i, method in enumerate(METHODS):
            ni = [float(lookup[(method, arm, seed)]["notinject_fpr_pct"]) for seed in SEEDS]
            bs = [float(lookup[(method, arm, seed)]["bshort_fpr_pct"]) for seed in SEEDS]
            means = [statistics.mean(ni), statistics.mean(bs)]
            sds = [statistics.stdev(ni), statistics.stdev(bs)]
            summary[arm][method] = {"mean": means, "sample_sd": sds}
            ax.bar(x + (i - 2) * width, means, width, yerr=sds,
                   capsize=2, color=COLORS[method], label=NAMES[method])
        ax.set_xticks(x, ["NotInject", "B-short"])
        ax.set_title(arm)
    axes[0].set_ylabel("Final false-positive rate (%)")
    # The left axis remains Matplotlib's original auto-range. The middle and
    # right ranges cover each mean+sample-SD whisker without truncation.
    middle_upper = max(summary["B-SHORT"][m]["mean"][0] +
                       summary["B-SHORT"][m]["sample_sd"][0] for m in METHODS)
    right_upper = max(summary["B-HARD"][m]["mean"][0] +
                      summary["B-HARD"][m]["sample_sd"][0] for m in METHODS)
    assert middle_upper < 5 and right_upper < 4
    assert all(summary[a][m]["mean"][1] == 0 and summary[a][m]["sample_sd"][1] == 0
               for a in ARMS[1:] for m in METHODS)
    axes[1].set_ylim(0, 5)
    axes[1].set_yticks(range(0, 6))
    axes[1].set_ylabel("FPR (%)")
    axes[2].set_ylim(0, 4)
    axes[2].set_yticks(range(0, 5))
    axes[2].set_ylabel("FPR (%)")
    axes[-1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Frozen external over-defense diagnostics after AF5", y=1.02)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    fig.savefig(REPORT / "external_overdefense_preview.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    manifest = {"source_csv_sha256": hashfile(CSV), "figure_pdf_sha256": hashfile(OUT),
                "middle_mean_plus_sd_max": middle_upper,
                "right_mean_plus_sd_max": right_upper,
                "axes_middle": [0, 5], "axes_right": [0, 4],
                "left_axis": "original auto-range and identical data/style"}
    (REPORT / "figure6_data_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", **manifest}))


if __name__ == "__main__":
    main()
