"""Write a traceable descriptive Markdown report from validated capacity CSVs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
DATA = ROOT / "results/revision3/analysis"
OUT = ROOT / "reports/revision3/04_memory_capacity.md"


def rows(name):
    with (DATA / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ms(row, metric, signed=True):
    mean = float(row[metric + "_mean"])
    value = f"{mean:+.2f}" if signed else f"{mean:.2f}"
    return f"{value} +/- {float(row[metric + '_sample_sd']):.2f}"


def main():
    validation = json.loads((DATA / "capacity_validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "PASS" and validation["runs_checked"] == 36
    per_seed = rows("capacity_per_seed.csv")
    aggregate = rows("capacity_aggregate.csv")
    paired = rows("capacity_paired_per_seed.csv")
    paired_aggregate = rows("capacity_paired_aggregate.csv")
    assert (len(per_seed), len(aggregate), len(paired), len(paired_aggregate)) == (36, 12, 18, 6)
    lines = ["# Matched A0/A1 memory-capacity sensitivity", "",
             "Status: complete, 36/36 trajectories full-hash validated, including 22 new Revision 3 runs and 14 compatible frozen/reused cells. All 18 A0/A1 capacity pairs have identical initialization, current-stage ordering, and first-stage model weights. Each trajectory uses the frozen DeBERTa/AMP-256 policy and four 100-update stages. The original online Reservoir M=200 baseline is excluded from the matched capacity grid.", "",
             "Candidate B stage-test attack TPR and benign FPR change in 1.667-pp increments; a 20-example attack-position slice changes in 5-pp increments. Values below are descriptive over seeds 17, 42, and 2026.", "",
             "## Per-seed results", "",
             "| Arm | M | Method | Seed | TPR BWT (pp) | F1 BWT (pp) | FPR increase (pp) | Acquisition TPR (%) | Final attack TPR (%) | Final benign FPR (%) | Worst benign-group drift (pp) | Buffer size / benign | Runtime (min) | Peak GPU (GiB) |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    for r in per_seed:
        lines.append(f"| {r['arm']} | {r['capacity']} | {r['method']} | {r['seed']} | "
                     f"{float(r['tpr_bwt_pp']):+.2f} | {float(r['macro_f1_bwt_pp']):+.2f} | "
                     f"{float(r['benign_fpr_increase_pp']):+.2f} | "
                     f"{float(r['acquisition_tpr_pct']):.2f} | "
                     f"{float(r['final_attack_tpr_pct']):.2f} | "
                     f"{float(r['final_benign_fpr_pct']):.2f} | "
                     f"{float(r['worst_benign_group_fpr_drift_pp']):.2f} | "
                     f"{int(float(r['final_buffer_occupancy']))} / "
                     f"{int(float(r['final_benign_memory_count']))} | "
                     f"{float(r['runtime_seconds']) / 60:.2f} | "
                     f"{float(r['peak_gpu_memory_bytes']) / (1024 ** 3):.3f} |")
    lines += ["", "## Mean +/- sample SD", "",
              "| Arm | M | Method | TPR BWT (pp) | F1 BWT (pp) | FPR increase (pp) | Acquisition TPR (%) | Final attack TPR (%) | Final benign FPR (%) |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in aggregate:
        lines.append(f"| {r['arm']} | {r['capacity']} | {r['method']} | "
                     f"{ms(r, 'tpr_bwt_pp')} | {ms(r, 'macro_f1_bwt_pp')} | "
                     f"{ms(r, 'benign_fpr_increase_pp')} | {ms(r, 'acquisition_tpr_pct', False)} | "
                     f"{ms(r, 'final_attack_tpr_pct', False)} | {ms(r, 'final_benign_fpr_pct', False)} |")
    lines += ["", "## Paired A1 minus A0 by seed", "",
              "Positive TPR BWT and negative FPR-increase contrasts favor class balancing. No significance test is claimed.", "",
              "| Arm | M | Seed | TPR BWT difference (pp) | FPR-increase difference (pp) |",
              "|---|---:|---:|---:|---:|"]
    for r in paired:
        lines.append(f"| {r['arm']} | {r['capacity']} | {r['seed']} | "
                     f"{float(r['a1_minus_a0_tpr_bwt_pp']):+.2f} | "
                     f"{float(r['a1_minus_a0_benign_fpr_increase_pp']):+.2f} |")
    lines += ["", "## Paired means", "",
              "| Arm | M | TPR BWT difference (pp) | FPR-increase difference (pp) |",
              "|---|---:|---:|---:|"]
    for r in paired_aggregate:
        lines.append(f"| {r['arm']} | {r['capacity']} | "
                     f"{ms(r, 'a1_minus_a0_tpr_bwt_pp')} | "
                     f"{ms(r, 'a1_minus_a0_benign_fpr_increase_pp')} |")
    lines += ["", "## Interpretation boundary", "",
              "The separate vector plots `figures/revision3/capacity_attack_retention.pdf` and `figures/revision3/capacity_benign_fpr.pdf` show malicious and benign axes without imposing a composite score. All trained-stage/evaluation-stage cells are in `capacity_stage_curves.csv`; selected-buffer Jaccard by stage/capacity/seed is in `capacity_memory_overlap.csv`. Finite test resolution, three seeds, and a single fixed stream/order limit inference. A larger memory can change acquisition and retention in different directions; check individual rows before claiming saturation or monotonicity. The earlier Revision 2 single-seed online-Reservoir capacity subset remains a historical contextual analysis, not a cell in this grid.", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "PASS", "per_seed_rows": len(per_seed),
                      "paired_rows": len(paired)}))


if __name__ == "__main__":
    main()
