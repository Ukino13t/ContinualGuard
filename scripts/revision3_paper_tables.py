"""Generate source-traceable ACL tables from validated Revision 3 CSV outputs."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
if HERE.name == "scripts":
    ROOT = HERE.parent
    DATA = ROOT / "results/revision3/analysis_v2"
    TABLES = ROOT / "paper/acl/tables"
else:
    # The same standalone script is packaged as paper/acl/build_revision3_tables.py.
    ROOT = HERE
    DATA = ROOT / "data/revision3"
    TABLES = ROOT / "tables"


def rows(name):
    with (DATA / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(row, key):
    return f"{float(row[key]):+.2f}"


def mean_sd(row, key, signed=True):
    value = float(row[key + "_mean"])
    mean = f"{value:+.2f}" if signed else f"{value:.2f}"
    return f"${mean}\\pm{float(row[key + '_sample_sd']):.2f}$"


def matched():
    summary = {r["arm"]: r for r in rows("paired_aggregate.csv")}
    out = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
           r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
           r"Arm & $\Delta$TPR BWT & $\Delta$F1 BWT & $\Delta$FPR incr. \\",
           r"\midrule"]
    for arm, label in (("doc", "B-DOC"), ("short", "B-SHORT"), ("hard", "B-HARD")):
        row = summary[arm]
        out.append(f"{label} & {mean_sd(row, 'a1_minus_a0_tpr_bwt_pp')} & "
                   f"{mean_sd(row, 'a1_minus_a0_macro_f1_bwt_pp')} & "
                   f"{mean_sd(row, 'a1_minus_a0_benign_fpr_increase_pp')} " + r"\\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Corrected matched A1 minus A0-v2 effects in percentage points, mean $\pm$ sample SD over three paired seeds. A0 uses global-priority membership and the same canonical replay order as A1; A1 enforces 100/100 label slots. Positive backward-transfer differences and negative FPR-increase differences favor A1. Original online Reservoir is contextual.}",
            r"\label{tab:r3-class-balance}", r"\end{table}", ""]
    (TABLES / "revision3_class_balance.tex").write_text("\n".join(out), encoding="utf-8")

    paired = rows("paired_per_seed.csv")
    out = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
           r"Arm & Seed & $\Delta$TPR BWT & $\Delta$F1 BWT & $\Delta$FPR increase \\",
           r"\midrule"]
    for r in paired:
        out.append(f"B-{r['arm'].upper()} & {r['seed']} & "
                   f"{fmt(r, 'a1_minus_a0_tpr_bwt_pp')} & "
                   f"{fmt(r, 'a1_minus_a0_macro_f1_bwt_pp')} & "
                   f"{fmt(r, 'a1_minus_a0_benign_fpr_increase_pp')} " + r"\\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Every seed-level corrected matched A1-minus-A0-v2 contrast at $M=200$, in percentage points. The signs are descriptive; no confidence intervals or significance claims are implied.}",
            r"\label{tab:r3-class-balance-seeds}", r"\end{table*}", ""]
    (TABLES / "revision3_class_balance_seeds.tex").write_text("\n".join(out), encoding="utf-8")


def capacity():
    validation = json.loads((DATA / "capacity_validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "PASS" and validation["runs_checked"] == 36
    data = rows("capacity_aggregate.csv")
    out = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular}{@{}lllr rrr@{}}", r"\toprule",
           r"Arm & $M$ & Method & TPR BWT & FPR increase & Acq. TPR & Final FPR \\",
           r"\midrule"]
    for arm in ("short", "hard"):
        for cap in (50, 200, 400):
            for method in ("a0_matched", "a1_matched"):
                row = next(r for r in data if r["arm"] == arm and
                           int(r["capacity"]) == cap and r["method"] == method)
                out.append(f"B-{arm.upper()} & {cap} & {'A0' if method.startswith('a0') else 'A1'} & "
                           f"{mean_sd(row, 'tpr_bwt_pp')} & "
                           f"{mean_sd(row, 'benign_fpr_increase_pp')} & "
                           f"{mean_sd(row, 'acquisition_tpr_pct', False)} & "
                           f"{mean_sd(row, 'final_benign_fpr_pct', False)} " + r"\\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Corrected matched capacity grid, mean $\pm$ sample SD over three seeds, all values in percentage points except memory capacity $M$. A0-v2/A1 share numerical and replay-index policy within each capacity. Original online Reservoir is excluded.}",
            r"\label{tab:r3-capacity}", r"\end{table*}", ""]
    (TABLES / "revision3_capacity.tex").write_text("\n".join(out), encoding="utf-8")

    data = rows("capacity_paired_per_seed.csv")
    out = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
           r"Arm & $M$ & Seed & $\Delta$TPR BWT & $\Delta$FPR increase \\",
           r"\midrule"]
    for arm in ("short", "hard"):
        for cap in (50, 200, 400):
            for seed in (17, 42, 2026):
                row = next(r for r in data if r["arm"] == arm and
                           int(r["capacity"]) == cap and int(r["seed"]) == seed)
                out.append(f"B-{arm.upper()} & {cap} & {seed} & "
                           f"{fmt(row, 'a1_minus_a0_tpr_bwt_pp')} & "
                           f"{fmt(row, 'a1_minus_a0_benign_fpr_increase_pp')} " + r"\\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Every capacity-specific seed-paired A1-minus-A0 contrast in percentage points. Positive TPR backward transfer and negative FPR increase favor A1. Three-seed descriptives do not identify a population effect.}",
            r"\label{tab:r3-capacity-pairs}", r"\end{table*}", ""]
    (TABLES / "revision3_capacity_pairs.tex").write_text("\n".join(out), encoding="utf-8")

    per_seed = rows("capacity_per_seed.csv")
    out = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\begin{tabular}{@{}llrr@{}}", r"\toprule",
           r"Method & $M$ & Runtime (min) & Peak GPU (GiB) \\", r"\midrule"]
    for method in ("a0_matched", "a1_matched"):
        for cap in (50, 200, 400):
            group = [r for r in per_seed if r["method"] == method and int(r["capacity"]) == cap]
            assert len(group) == 6
            runtimes = [float(r["runtime_seconds"]) / 60 for r in group]
            memory = [float(r["peak_gpu_memory_bytes"]) / (1024 ** 3) for r in group]
            out.append(f"{'A0' if method.startswith('a0') else 'A1'} & {cap} & "
                       f"${statistics.mean(runtimes):.2f}\\pm{statistics.stdev(runtimes):.2f}$ & "
                       f"${statistics.mean(memory):.3f}\\pm{statistics.stdev(memory):.3f}$ " + r"\\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Revision 3 matched capacity resources over B-SHORT/B-HARD (six arm--seed runs per row). Sample SD mixes arm and seed variation; reported runtime is summed stage training/evaluation time and excludes process startup.}",
            r"\label{tab:r3-resources}", r"\end{table}", ""]
    (TABLES / "revision3_resources.tex").write_text("\n".join(out), encoding="utf-8")


def main():
    validation = json.loads((DATA / "validation_M200.json").read_text(encoding="utf-8"))
    assert validation["status"] == "PASS" and validation["runs_checked"] == 9
    TABLES.mkdir(parents=True, exist_ok=True)
    matched()
    if (DATA / "capacity_validation.json").is_file():
        capacity()
    print("PASS: Revision 3 tables generated from validated CSVs")


if __name__ == "__main__":
    main()
