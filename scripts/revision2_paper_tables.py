"""Generate ACL tables only from validated complete Revision 2 CSV rows."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
DATA = ROOT / "results/revision2/analysis"
OUT = ROOT / "paper/acl/tables"


def read(name):
    with (DATA / name).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def fmt(row, key, digits=2):
    return f"{float(row[key + '_mean']):.{digits}f} $\\pm$ {float(row[key + '_sample_sd']):.{digits}f}"


def save(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body.rstrip() + "\n", encoding="utf-8")


def main():
    validation = json.loads((DATA / "validation.json").read_text(encoding="utf-8"))
    if (validation.get("status"), validation.get("full_checkpoint_hashes"),
            validation.get("runs_checked"), len(validation.get("matched_pair_controls", []))) != ("PASS", True, 24, 9):
        raise RuntimeError("All P0 trajectories must pass full-hash and matched-pair validation")
    aggregate = {(r["method"], r["arm"]): r for r in read("aggregate.csv")}
    per_seed = read("per_seed.csv")
    paired = {(r["before"], r["after"], r["arm"]): r for r in read("paired_aggregate.csv")}
    for method in ("A1_matched", "A2_matched", "compute_current_only"):
        for arm in ("doc", "short", "hard"):
            row = aggregate.get((method, arm))
            if row is None or int(row["n_seeds"]) != 3:
                raise RuntimeError(f"Three validated seeds required before paper table: {method}/{arm}")
    labels = {"M0_NoReplay": "No Replay", "A0_Reservoir": "Reservoir ER",
              "A1_matched": "A1 matched (class)", "A2_matched": "A2 matched (class+stage)",
              "A4_CE": "Generic CE", "A5_Risk": "Security risk",
              "compute_current_only": "Current-only 2$\\times$"}
    lines = [r"\begin{table*}[t]", r"\centering\small\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r"Method & B-SHORT TPR BWT & B-HARD FPR change & B-HARD F1 BWT & B-SHORT acq. TPR \\",
             r"\midrule"]
    for method in ("M0_NoReplay", "A0_Reservoir", "A1_matched", "A2_matched", "A4_CE", "A5_Risk"):
        short, hard = aggregate[(method, "short")], aggregate[(method, "hard")]
        lines.append(f"{labels[method]} & {fmt(short, 'tpr_bwt_pp')} & {fmt(hard, 'benign_fpr_increase_pp')} & {fmt(hard, 'macro_f1_bwt_pp')} & {fmt(short, 'acquisition_tpr_pct')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Matched-rank Revision 2 selector comparison at $M=200$; historical frozen controls are contextual. Values are mean $\pm$ sample SD over three seeds in percentage points. TPR and macro-F1 BWT are higher-is-better; positive benign-FPR change is harmful. A1/A2 share rank and numerical policy, while frozen methods do not.}",
              r"\label{tab:r2-matched}", r"\end{table*}"]
    save("revision2_matched.tex", "\n".join(lines))

    lines = [r"\begin{table*}[t]", r"\centering\small", r"\begin{tabular}{llcccc}",
             r"\toprule", r"Arm & Method & Final TPR & Final FPR & Final macro-F1 & TPR BWT \\", r"\midrule"]
    for arm, arm_label in (("doc", "B-DOC"), ("short", "B-SHORT"), ("hard", "B-HARD")):
        for method in ("M0_NoReplay", "compute_current_only", "A0_Reservoir"):
            row = aggregate[(method, arm)]
            lines.append(f"{arm_label} & {labels[method]} & {fmt(row, 'final_attack_tpr_pct')} & {fmt(row, 'final_benign_fpr_pct')} & {fmt(row, 'final_macro_f1_pct')} & {fmt(row, 'tpr_bwt_pp')} \\\\")
        if arm != "hard":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Three-seed current-only exposure control. All methods have 100 optimizer updates per stage. Current-only processes two independent current minibatches per update and never sees historical training examples. Replay substitutes historical records after the first stage; the conditions are not fully wall-clock or data-access equivalent. Values are mean $\pm$ sample SD in percent or percentage points.}",
              r"\label{tab:r2-compute}", r"\end{table*}"]
    save("revision2_compute.tex", "\n".join(lines))

    lines = [r"\begin{table*}[t]", r"\centering\small", r"\begin{tabular}{lccc}",
             r"\toprule", r"Arm & $\Delta$ TPR BWT & $\Delta$ macro-F1 BWT & $\Delta$ FPR change \\", r"\midrule"]
    for arm, label in (("doc", "B-DOC"), ("short", "B-SHORT"), ("hard", "B-HARD")):
        row = paired[("A1_matched", "A2_matched", arm)]
        lines.append(f"{label} & {fmt(row, 'delta_tpr_bwt_pp')} & {fmt(row, 'delta_macro_f1_bwt_pp')} & {fmt(row, 'delta_benign_fpr_increase_pp')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Paired A2-minus-A1 differences under shared ranking, mean $\pm$ sample SD in percentage points. Positive $\Delta$BWT favors A2; negative $\Delta$FPR change favors A2.}",
              r"\label{tab:r2-paired}", r"\end{table*}"]
    save("revision2_paired.tex", "\n".join(lines))

    lines = [r"\begin{table*}[t]", r"\centering\small", r"\begin{tabular}{llrcccc}",
             r"\toprule", r"Arm & Method & Seed & TPR BWT & macro-F1 BWT & FPR change & Acquisition TPR \\", r"\midrule"]
    for arm, label in (("doc", "B-DOC"), ("short", "B-SHORT"), ("hard", "B-HARD")):
        for method in ("A1_matched", "A2_matched"):
            for row in sorted((r for r in per_seed if r["arm"] == arm and r["method"] == method), key=lambda r: int(r["seed"])):
                lines.append(f"{label} & {labels[method]} & {row['seed']} & {float(row['tpr_bwt_pp']):.2f} & {float(row['macro_f1_bwt_pp']):.2f} & {float(row['benign_fpr_increase_pp']):.2f} & {float(row['acquisition_tpr_pct']):.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{All seed-level matched-selector outcomes, percentage points except acquisition TPR (percent). These descriptive values underlie Table~\ref{tab:r2-matched}.}",
              r"\label{tab:r2-per-seed}", r"\end{table*}"]
    save("revision2_per_seed.tex", "\n".join(lines))

    lines = [r"\begin{table*}[t]", r"\centering\small\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llcccc}",
             r"\toprule", r"Arm & Method & Worst benign drift & Worst position loss & Peak TPR forgetting & Acquisition TPR \\", r"\midrule"]
    for arm, label in (("doc", "B-DOC"), ("short", "B-SHORT"), ("hard", "B-HARD")):
        for method in ("A1_matched", "A2_matched"):
            row = aggregate[(method, arm)]
            lines.append(f"{label} & {labels[method]} & {fmt(row, 'worst_benign_group_fpr_drift_pp')} & {fmt(row, 'worst_attack_position_tpr_loss_pp')} & {fmt(row, 'attack_peak_forgetting_pp')} & {fmt(row, 'acquisition_tpr_pct')} \\\\")
        if arm != "hard":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Revision 2 worst-group and worst-position diagnostics at $M=200$, mean $\pm$ sample SD over three seeds in percentage points except acquisition TPR (percent). Maxima are taken within each frozen run before averaging seeds; they are descriptive extrema over tested groups, not population worst-case bounds.}",
              r"\label{tab:r2-worst}", r"\end{table*}"]
    save("revision2_worst_groups.tex", "\n".join(lines))
    print("Revision 2 ACL tables generated from complete three-seed CSVs")


if __name__ == "__main__":
    main()
