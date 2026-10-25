"""Build Route A numeric reports and LaTeX tables from stored CSVs only."""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
AN = ROOT / "results/extended/analysis"
REPORTS = ROOT / "reports"
TABLES = ROOT / "paper/acl/tables"
METHODS = ("m0_no_replay", "a0_reservoir", "a1_class", "a2_class_stage",
           "a3_coverage", "a4_ce", "a5_security")
NAMES = {"m0_no_replay": "No Replay", "a0_reservoir": "A0 Reservoir",
         "a1_class": "A1 Class", "a2_class_stage": "A2 Class+Stage",
         "a3_coverage": "A3 +Coverage", "a4_ce": "A4 +CE",
         "a5_security": "A5 Security", "compute_matched": "Compute-matched",
         "joint_union_1epoch": "Joint offline"}
ARMS = ("doc", "short", "hard")


def read(name):
    path = AN / name
    if not path.exists(): return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean_sd(values):
    values = list(values)
    if not values: return "pending"
    if len(values) == 1: return f"{values[0]:.2f} (n=1)"
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"


def main():
    per = read("per_run.csv")
    paired = read("paired_per_seed.csv")
    joint = read("joint_reference.csv")
    by = defaultdict(list)
    for row in per: by[(row["method"], row["arm"])].append(row)
    lines = ["# Route A structured replay ablation", "",
             "Status is determined from stored four-stage runs only. The frozen A0/A4/A5 artifacts are reused; A1–A3 live under `results/extended/ablation`. Source tables: `results/extended/analysis/{per_run,aggregate,paired_per_seed,paired_aggregate,per_stage_curves,memory_composition}.csv`. The adjacent contrasts below are after-minus-before in percentage points. TPR and macro-F1 BWT are higher-is-better; benign-FPR increase, peak forgetting and worst benign-group drift are lower-is-better. Three seeds are required for an unqualified aggregate.", "",
             "| Variant | Complete arm×seed runs | B-SHORT TPR BWT (pp) | B-HARD FPR increase (pp) | B-HARD macro-F1 BWT (pp) | Mean runtime/run (min) |",
             "|---|---:|---:|---:|---:|---:|"]
    latex = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Method & B-SHORT TPR BWT & B-HARD FPR change & B-HARD macro-F1 BWT & Runs \\",
             r"\midrule"]
    for method in METHODS:
        rows = [r for r in per if r["method"] == method]
        short = by[(method, "short")]
        hard = by[(method, "hard")]
        runtime = [float(r["runtime_seconds"]) / 60 for r in rows]
        vals = [mean_sd(float(r["tpr_bwt_pp"]) for r in short),
                mean_sd(float(r["benign_fpr_increase_pp"]) for r in hard),
                mean_sd(float(r["macro_f1_bwt_pp"]) for r in hard)]
        lines.append(f"| {NAMES[method]} | {len(rows)}/9 | {vals[0]} | {vals[1]} | {vals[2]} | {mean_sd(runtime)} |")
        latex_vals = [x.replace("±", r"$\pm$") if x != "pending" else "---" for x in vals]
        latex.append(f"{NAMES[method]} & {latex_vals[0]} & {latex_vals[1]} & {latex_vals[2]} & {len(rows)}/9 " + r"\\")
    latex += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Route A replay-memory ablation. Means $\pm$ sample SD across seeds when three runs are complete. Signed BWT is higher-is-better; positive FPR change is harmful. Incomplete cells are not estimates.}",
              r"\label{tab:route-ablation}", r"\end{table*}"]
    lines += ["", "## Adjacent paired contrasts", "",
              "| Contrast | Arm | Matched seeds | Δ attack-TPR BWT (pp) | Δ benign-FPR increase (pp) | Δ acquisition TPR (pp) |",
              "|---|---|---:|---:|---:|---:|"]
    contrasts = (("a0_reservoir", "a1_class"), ("a1_class", "a2_class_stage"),
                 ("a2_class_stage", "a3_coverage"), ("a3_coverage", "a4_ce"),
                 ("a4_ce", "a5_security"))
    for before, after in contrasts:
        for arm in ARMS:
            group = [r for r in paired if r["before"] == before and r["after"] == after and r["arm"] == arm]
            if not group: continue
            lines.append(f"| {NAMES[before]} → {NAMES[after]} | {arm} | {len(group)}/3 | "
                         f"{mean_sd(float(r['delta_tpr_bwt_pp']) for r in group)} | "
                         f"{mean_sd(float(r['delta_benign_fpr_increase_pp']) for r in group)} | "
                         f"{mean_sd(float(r['delta_acquisition_tpr_pct']) for r in group)} |")
    lines += ["", "The exact binary relation `CE=-log(1-risk)` holds for a common model state. Frozen A4/A5 differ in FP32/FP16 scoring and may diverge through ties and subsequent training states. The original `results/loss_aware_score_relationship.json` records same-forward diagnostics; selected-memory Jaccard overlap averaged 0.932 at AF2, 0.935 at AF3, 0.695 at AF4, and 0.545 at AF5 across the nine arm×seed pairs. At AF2 in B-DOC/seed 17, the FP32 CE-side forward produced 398 unique risks among 400 candidates versus only 51 from the FP16 security forward, illustrating numerical ties. Later actual-method correlations also mix score precision with diverged model states. Operational differences must not be described as semantic superiority of the risk formula.",
              "", "Initialization control: A1 seed-17 Stage-1 `model.safetensors` hashes are identical across B-DOC, B-SHORT and B-HARD (`530633be...646560a4`) and match the frozen M3 Stage-1 model hash for the same seed. Stage-1 has no replay, so this checks the common model/data/training path before selector interventions can matter.",
              "", "**Coverage audit and identification limit.** Every Candidate B TRAIN arm-stage has 400 distinct `coverage_key` values among 400 records, and every observed 200-example final buffer has 200 unique keys. The one-per-key rule is nonbinding here. A2 and A3 also use variant-specific deterministic random ranks: their selected-buffer Jaccard overlap averages 0.335 / 0.151 / 0.100 / 0.068 at AF2 / AF3 / AF4 / AF5, respectively, across the nine arm-seed pairs. Their outcome differences therefore reflect rank-stream and downstream trajectory variation, not a coverage effect. Adjacent A1--A4 contrasts inherit this random-stream difference and cannot be read as pure causal component effects. See `results/extended/analysis/{coverage_key_audit.json,a2_a3_memory_overlap.csv}`.",
              "", "Memory composition and coverage counts are available per stage in the CSV. Complete plots are regenerated by `scripts/plot_route_a.py`. The human-selected Candidate B stream and three-seed sample constrain generalization. No multiple-testing significance claim is made.", ""]
    (REPORTS / "01_ablation_results.md").write_text("\n".join(lines), encoding="utf-8")
    (TABLES / "route_a_ablation.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    appendix = [r"\begin{table*}[t]", r"\centering", r"\small",
                r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
                r"Method & Seed & B-SHORT attack-TPR BWT & B-HARD benign-FPR increase & B-HARD macro-F1 BWT \\",
                r"\midrule"]
    for method in METHODS:
        for seed in (17, 42, 2026):
            short_rows = [r for r in by[(method, "short")] if int(r["seed"]) == seed]
            hard_rows = [r for r in by[(method, "hard")] if int(r["seed"]) == seed]
            if not (short_rows and hard_rows):
                continue
            a, b = short_rows[0], hard_rows[0]
            appendix.append(f"{NAMES[method]} & {seed} & {float(a['tpr_bwt_pp']):.2f} & "
                            f"{float(b['benign_fpr_increase_pp']):.2f} & {float(b['macro_f1_bwt_pp']):.2f} " + r"\\")
    appendix += [r"\bottomrule", r"\end{tabular}",
                 r"\caption{Route A per-seed signed outcomes in percentage points. This table lists only fully observed B-SHORT/B-HARD seed pairs; source CSVs contain every stage and arm.}",
                 r"\label{tab:route-per-seed}", r"\end{table*}"]
    (TABLES / "route_a_per_seed.tex").write_text("\n".join(appendix) + "\n", encoding="utf-8")

    # Offline reference: average final performance across the four frozen tests.
    joint_by = defaultdict(list)
    for row in joint: joint_by[(row["arm"], row["seed"])].append(row)
    joint_summary = []
    for (arm, seed), tasks in joint_by.items():
        if len(tasks) != 4: continue
        joint_summary.append({"arm": arm, "seed": seed,
            "tpr": statistics.mean(float(r["attack_tpr_pct"]) for r in tasks),
            "fpr": statistics.mean(float(r["benign_fpr_pct"]) for r in tasks),
            "f1": statistics.mean(float(r["macro_f1_pct"]) for r in tasks)})
    offline = ["# Route A joint and compute controls", "",
               "Joint/offline uses all 1,600 unique AF2–AF5 TRAIN records at once, one epoch/400 updates, and tests only on frozen held-out splits. It has full-history/future training access by design and is a finite-optimization reference, not an upper bound. Compute-matched current-only uses two independently shuffled current-stage 4-example minibatches per update, averages their CE, and performs 100 updates per stage: 800 current-stage exposures per stage, zero historical. It matches update count and approximately the two-forward-pass cost of replay, but intentionally has different unique and total exposure counts.", "",
               "| Arm | Method | Complete seeds | Mean final attack TPR across four tests (%) | Mean final benign FPR (%) | Mean final macro-F1 (%) |",
               "|---|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        for method in ("m0_no_replay", "compute_matched", "a0_reservoir", "a5_security", "joint_union_1epoch"):
            if method == "joint_union_1epoch":
                selected = [r for r in joint_summary if r["arm"] == arm]
                tpr = [r["tpr"] for r in selected]
                fpr = [r["fpr"] for r in selected]
                f1 = [r["f1"] for r in selected]
            else:
                selected = by[(method, arm)]
                tpr, fpr, f1 = [], [], []
                # Final four-test averages come from per_stage_curves for each seed.
                curves = read("per_stage_curves.csv")
                for run in selected:
                    subset = [r for r in curves if r["method"] == method and r["arm"] == arm
                              and int(r["seed"]) == int(run["seed"]) and r["trained_stage"] == "AF5"]
                    if len(subset) == 4:
                        tpr.append(statistics.mean(float(r["attack_tpr_pct"]) for r in subset))
                        fpr.append(statistics.mean(float(r["benign_fpr_pct"]) for r in subset))
                        f1.append(statistics.mean(float(r["macro_f1_pct"]) for r in subset))
            offline.append(f"| {arm} | {NAMES[method]} | {len(tpr)}/3 | {mean_sd(tpr)} | {mean_sd(fpr)} | {mean_sd(f1)} |")
    offline += ["", "**Matched seed-42 interpretation.** For B-SHORT, final mean benign FPR over four tests is 20.83% for standard No Replay, 11.67% for compute-matched current-only, 0.42% for Reservoir, 3.33% for Security-aware, and 0.42% for joint. For B-HARD the corresponding FPRs are 25.42%, 24.17%, 4.17%, 0.42%, and 0.42%. Compute-matched current-only does not recover replay's benign retention in this seed. Its B-DOC mean attack TPR is 95.42% versus 98.75% for standard No Replay. The control uses one seed per arm, so these are matched case studies, not three-seed estimates.",
                "", "Comparison of offline and continual methods must account for different temporal access and optimization trajectories. A single-epoch joint result can be below a replay method on a given endpoint; that is finite optimization, not evidence that access to all training data is intrinsically harmful. New checkpoint, prediction, runtime and manifest files are under `results/extended/offline_controls`. Pending seed combinations are not imputed.", ""]
    (REPORTS / "02_joint_and_compute_controls.md").write_text("\n".join(offline), encoding="utf-8")


if __name__ == "__main__":
    main()
