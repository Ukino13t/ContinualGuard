"""Report only the predeclared, validated B-HARD seed-42 capacity subset."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
ANALYSIS = ROOT / "results/revision2/analysis"
REPORT = ROOT / "reports/revision2/04_memory_sensitivity.md"
TABLE = ROOT / "paper/acl/tables/revision2_capacity.tex"
START = "<!-- REVISION2_CAPACITY_RESULTS:BEGIN -->"
END = "<!-- REVISION2_CAPACITY_RESULTS:END -->"


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    validation = json.loads((ANALYSIS / "capacity_validation.json").read_text(encoding="utf-8"))
    expected = {(capacity, method, "hard", 42)
                for capacity in (50, 400)
                for method in ("a1_matched", "reservoir_capacity")}
    observed = {(r["capacity"], r["method"], r["arm"], r["seed"])
                for r in validation.get("runs", [])}
    if (validation.get("status"), validation.get("full_checkpoint_hashes"),
            validation.get("runs_checked"), observed) != ("PASS", True, 4, expected):
        raise RuntimeError("All four predeclared capacity trajectories require full-hash validation")
    controls = json.loads((ANALYSIS / "capacity_pair_controls.json").read_text(encoding="utf-8"))
    if (controls.get("status"), controls.get("runs_checked"),
            controls.get("shared_initialization"),
            controls.get("identical_first_stage_weights"),
            controls.get("identical_current_order_all_four_stages")) != ("PASS", 5, True, True, True):
        raise RuntimeError("Capacity initialization/order controls failed")
    with (ANALYSIS / "per_seed.csv").open(encoding="utf-8", newline="") as handle:
        rows = {(r["method"], r["arm"], int(r["seed"])): r for r in csv.DictReader(handle)}
    ordered = [("a1_matched_M50", "A1 matched", 50),
               ("A1_matched", "A1 matched", 200),
               ("a1_matched_M400", "A1 matched", 400),
               ("reservoir_capacity_M50", "Reservoir", 50),
               ("reservoir_capacity_M400", "Reservoir", 400)]
    selected = []
    for name, label, capacity in ordered:
        row = rows.get((name, "hard", 42))
        if row is None:
            raise RuntimeError(f"Missing B-HARD seed-42 aggregate row: {name}")
        selected.append((label, capacity, row))

    old = REPORT.read_text(encoding="utf-8")
    protocol = old.split(START)[0].rstrip().replace(
        "No P1 result is available yet. Replay-ratio and independent-stream robustness remain separate conditional experiments rather than implicit claims of this capacity check.",
        "The validated P1 subset is reported below. Replay-ratio and independent-stream robustness remain separate conditional experiments rather than implicit claims of this capacity check.")
    lines = [START, "", "## Validated exploratory results", "",
             "All four new trajectories passed exact training-access, reservoir-update, prediction, and full-checkpoint-hash validation. Across all five displayed cells, initialization, first-stage model weights, and current-example order at every stage match. The A1 M=200 cell is the separately validated matched-rank main experiment with the same seed and AMP policy. All values below are a single seed, with no sample SD or inference of monotonic capacity response.", "",
             "| Method | M | TPR BWT (pp) | macro-F1 BWT (pp) | Benign-FPR increase (pp) | Final four-test FPR (%) | Final four-test TPR (%) | Peak forgetting (pp) | Runtime (min) | Peak GPU (GiB) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    tex = [r"\begin{table*}[t]", r"\centering\small", r"\begin{tabular}{llrrrrr}",
           r"\toprule", r"Method & $M$ & TPR BWT & macro-F1 BWT & FPR change & Final FPR & Final TPR \\",
           r"\midrule"]
    for label, capacity, row in selected:
        lines.append("| {label} | {capacity} | {tpr:+.2f} | {f1:+.2f} | {change:+.2f} | {fpr:.2f} | {final_tpr:.2f} | {peak:.2f} | {runtime:.2f} | {memory:.2f} |".format(
            label=label, capacity=capacity, tpr=f(row, "tpr_bwt_pp"),
            f1=f(row, "macro_f1_bwt_pp"), change=f(row, "benign_fpr_increase_pp"),
            fpr=f(row, "final_benign_fpr_pct"), final_tpr=f(row, "final_attack_tpr_pct"),
            peak=f(row, "attack_peak_forgetting_pp"),
            runtime=f(row, "runtime_seconds") / 60,
            memory=f(row, "peak_gpu_memory_bytes") / 2**30))
        tex.append("{label} & {capacity} & {tpr:+.2f} & {f1:+.2f} & {change:+.2f} & {fpr:.2f} & {final_tpr:.2f} \\\\".format(
            label=label, capacity=capacity, tpr=f(row, "tpr_bwt_pp"),
            f1=f(row, "macro_f1_bwt_pp"), change=f(row, "benign_fpr_increase_pp"),
            fpr=f(row, "final_benign_fpr_pct"), final_tpr=f(row, "final_attack_tpr_pct")))
    for label, lo_name, hi_name in (("A1 matched", "a1_matched_M50", "a1_matched_M400"),
                                    ("Reservoir", "reservoir_capacity_M50", "reservoir_capacity_M400")):
        lo, hi = rows[(lo_name, "hard", 42)], rows[(hi_name, "hard", 42)]
        lines.append("\n{label} M=400 minus M=50: TPR BWT {tpr:+.2f} pp; macro-F1 BWT {f1:+.2f} pp; benign-FPR increase {fpr:+.2f} pp. This is one matched-seed contrast, not a capacity trend.".format(
            label=label, tpr=f(hi, "tpr_bwt_pp") - f(lo, "tpr_bwt_pp"),
            f1=f(hi, "macro_f1_bwt_pp") - f(lo, "macro_f1_bwt_pp"),
            fpr=f(hi, "benign_fpr_increase_pp") - f(lo, "benign_fpr_increase_pp")))
    lines += ["", "The frozen historical M=200 Reservoir point used its earlier numerical policy and training order and is excluded from this within-policy capacity table. Other arms, seeds, capacities, replay ratios, and independent stream orders were not tested in this subset.", "", END]
    REPORT.write_text(protocol + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Exploratory B-HARD seed-42 capacity subset. TPR and macro-F1 BWT and FPR change are percentage points; final rates are percent. A1 M=200 is the validated main matched-rank run. M=50/400 A1 and Reservoir runs share the Revision 2 AMP policy, but there is no same-policy Reservoir M=200 cell. These single-seed observations do not establish a capacity frontier.}",
            r"\label{tab:r2-capacity}", r"\end{table*}"]
    TABLE.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "capacity_runs": 4, "table_rows": len(selected)}))


if __name__ == "__main__":
    main()
