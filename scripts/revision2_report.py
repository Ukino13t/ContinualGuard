"""Write descriptive Revision 2 P0 reports from completed, validated CSVs.

This script refuses incomplete three-seed cells. It never reads raw prompt text.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
DATA = ROOT / "results/revision2/analysis"
REPORT = ROOT / "reports/revision2"
ARMS = ("doc", "short", "hard")
SEEDS = (17, 42, 2026)


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def num(row: dict, key: str) -> str:
    return f"{float(row[key]):+.2f}"


def mean_sd(row: dict, key: str) -> str:
    return f"{float(row[key + '_mean']):+.2f} +/- {float(row[key + '_sample_sd']):.2f}"


def require_complete(per_seed: list[dict], methods: tuple[str, ...]) -> None:
    available = {(r["method"], r["arm"], int(r["seed"])) for r in per_seed}
    missing = [(m, a, s) for m in methods for a in ARMS for s in SEEDS
               if (m, a, s) not in available]
    if missing:
        raise RuntimeError(f"Incomplete validated result cells: {missing}")


def save(name: str, text: str) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def matched(per_seed: list[dict], agg: dict, paired: dict, diagnostics: list[dict]) -> None:
    lines = [
        "# Revision 2 matched-rank ablation results",
        "",
        "A1 and A2 use the same seeded SHA-256 rank function, current-stage arrivals, frozen tests, training schedule, capacity M=200, and AMP initial scale 256. Their eligible historical pools can diverge after memory selection, as part of the intervention. The failed AMP-1024 pilot is excluded. Original A0/A4/A5 are frozen historical context with a different numerical policy. No significance test is claimed from three seeds.",
        "",
        "## Per-seed complete trajectories",
        "",
        "| Arm | Method | Seed | Attack-TPR BWT (pp) | Macro-F1 BWT (pp) | Benign-FPR increase (pp) | Worst benign-group FPR drift (pp) | Worst attack-position TPR loss (pp) | Acquisition TPR (%) | Runtime (min) | Peak GPU (GiB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    selected = [r for r in per_seed if r["method"] in ("A1_matched", "A2_matched")]
    for r in sorted(selected, key=lambda x: (x["arm"], x["method"], int(x["seed"]))):
        lines.append("| {arm} | {method} | {seed} | {tpr} | {f1} | {fpr} | {worst_benign} | {worst_attack} | {acq:.2f} | {runtime:.2f} | {mem:.2f} |".format(
            arm=r["arm"], method=r["method"], seed=r["seed"],
            tpr=num(r, "tpr_bwt_pp"), f1=num(r, "macro_f1_bwt_pp"),
            fpr=num(r, "benign_fpr_increase_pp"),
            worst_benign=num(r, "worst_benign_group_fpr_drift_pp"),
            worst_attack=num(r, "worst_attack_position_tpr_loss_pp"),
            acq=float(r["acquisition_tpr_pct"]),
            runtime=float(r["runtime_seconds"]) / 60,
            mem=float(r["peak_gpu_memory_bytes"]) / 2**30))
    lines += ["", "## Mean +/- sample SD across seeds", "",
              "| Arm | Method | TPR BWT (pp) | Macro-F1 BWT (pp) | FPR increase (pp) | Peak forgetting (pp) | Worst benign-group FPR drift (pp) | Worst attack-position TPR loss (pp) | Acquisition TPR (%) |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for method in ("A1_matched", "A2_matched"):
            r = agg[(method, arm)]
            lines.append(f"| {arm} | {method} | {mean_sd(r, 'tpr_bwt_pp')} | {mean_sd(r, 'macro_f1_bwt_pp')} | {mean_sd(r, 'benign_fpr_increase_pp')} | {mean_sd(r, 'attack_peak_forgetting_pp')} | {mean_sd(r, 'worst_benign_group_fpr_drift_pp')} | {mean_sd(r, 'worst_attack_position_tpr_loss_pp')} | {mean_sd(r, 'acquisition_tpr_pct')} |")
    lines += ["", "## Paired A2 minus A1", "",
              "| Arm | TPR BWT difference (pp) | Macro-F1 BWT difference (pp) | FPR increase difference (pp) | Worst-group FPR drift difference (pp) | Worst-position TPR loss difference (pp) | Better TPR seeds | Lower FPR seeds |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        r = paired[("A1_matched", "A2_matched", arm)]
        lines.append(f"| {arm} | {mean_sd(r, 'delta_tpr_bwt_pp')} | {mean_sd(r, 'delta_macro_f1_bwt_pp')} | {mean_sd(r, 'delta_benign_fpr_increase_pp')} | {mean_sd(r, 'delta_worst_benign_group_fpr_drift_pp')} | {mean_sd(r, 'delta_worst_attack_position_tpr_loss_pp')} | {r['delta_tpr_bwt_pp_positive_seeds']}/3 | {r['delta_benign_fpr_increase_pp_negative_seeds']}/3 |")
    lines += ["", "## Stage acquisition and optimization diagnostics", "",
              "Each cell below is mean +/- sample SD over three seeds. Runtime is stage-local minutes and GPU memory is peak allocated GiB.", "",
              "| Arm | Method | Stage | Attack TPR (%) | Benign FPR (%) | Macro-F1 (%) | Train loss | Validation loss | Runtime (min) | Peak GPU (GiB) |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    def stat(group: list[dict], key: str, scale: float = 1.0, digits: int = 2) -> str:
        values = [float(row[key]) / scale for row in group]
        return f"{statistics.mean(values):.{digits}f} +/- {statistics.stdev(values):.{digits}f}"
    for arm in ARMS:
        for method in ("A1_matched", "A2_matched"):
            for stage in ("AF2", "AF3", "AF4", "AF5"):
                group = [r for r in diagnostics if r["arm"] == arm and r["method"] == method
                         and r["trained_stage"] == stage]
                if len(group) != 3:
                    raise RuntimeError(f"Incomplete stage diagnostics: {arm}/{method}/{stage}")
                lines.append(f"| {arm} | {method} | {stage} | {stat(group, 'acquisition_attack_tpr_pct')} | {stat(group, 'acquisition_benign_fpr_pct')} | {stat(group, 'acquisition_macro_f1_pct')} | {stat(group, 'training_loss', digits=4)} | {stat(group, 'validation_loss', digits=4)} | {stat(group, 'runtime_seconds', scale=60)} | {stat(group, 'peak_gpu_memory_bytes', scale=2**30)} |")
    lines += ["", "The 36 actual eligible pools have no duplicate coverage key; A2 and A3 select identical IDs and order in every cell. A two-stage independent parity smoke run produced byte-identical model weights, logs, and buffers. Thus the coverage preference did not intervene on this benchmark and a full A3 training rerun is redundant. The original A2/A3 outcome contrast is not evidence of coverage benefit.",
              "", "Acquisition is measured at first learning. A negative TPR or macro-F1 BWT indicates later decline; positive benign-FPR increase indicates worsening benign acceptance. Attack and benign test sets have 60 examples each per stage, giving 1.667-pp rate resolution. These are descriptive results on a constructed benchmark, not population estimates."]
    save("01_matched_ablation.md", "\n".join(lines))


def compute(per_seed: list[dict], agg: dict, paired: dict) -> None:
    lines = ["# Revision 2 three-seed current-only compute control", "",
             "The current-only control makes 100 optimizer updates per stage and processes two independent current minibatches per update, with zero historical training exposure. Replay processes one current and one historical minibatch per update after AF2. This matches update and forward-batch counts, not sample identities, information access, or elapsed time.", "",
             "| Arm | Method | Final four-test TPR (%) | Final four-test FPR (%) | Final four-test macro-F1 (%) | TPR BWT (pp) | Runtime (min) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for method in ("M0_NoReplay", "compute_current_only", "A0_Reservoir"):
            r = agg[(method, arm)]
            lines.append(f"| {arm} | {method} | {mean_sd(r, 'final_attack_tpr_pct')} | {mean_sd(r, 'final_benign_fpr_pct')} | {mean_sd(r, 'final_macro_f1_pct')} | {mean_sd(r, 'tpr_bwt_pp')} | {float(r['runtime_seconds_mean'])/60:.2f} +/- {float(r['runtime_seconds_sample_sd'])/60:.2f} |")
    lines += ["", "## Per-seed current-only outcomes", "",
              "| Arm | Seed | Final TPR (%) | Final FPR (%) | Macro-F1 (%) | TPR BWT (pp) | Peak GPU (GiB) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted((r for r in per_seed if r["method"] == "compute_current_only"),
                    key=lambda x: (x["arm"], int(x["seed"]))):
        lines.append(f"| {r['arm']} | {r['seed']} | {float(r['final_attack_tpr_pct']):.2f} | {float(r['final_benign_fpr_pct']):.2f} | {float(r['final_macro_f1_pct']):.2f} | {num(r, 'tpr_bwt_pp')} | {float(r['peak_gpu_memory_bytes'])/2**30:.2f} |")
    lines += ["", "## Paired differences", "",
              "| Arm | Current-only minus No Replay: final FPR (pp) | Reservoir minus current-only: final FPR (pp) | Reservoir minus current-only: TPR BWT (pp) |",
              "|---|---:|---:|---:|"]
    for arm in ARMS:
        a = paired[("M0_NoReplay", "compute_current_only", arm)]
        b = paired[("compute_current_only", "A0_Reservoir", arm)]
        lines.append(f"| {arm} | {mean_sd(a, 'delta_final_benign_fpr_pct')} | {mean_sd(b, 'delta_final_benign_fpr_pct')} | {mean_sd(b, 'delta_tpr_bwt_pp')} |")
    lines += ["", "A descriptive paired difference in three seeds does not establish that computation alone is or is not causally sufficient for replay's effect. Current-only's extra examples are drawn from the present stage; replay's are historical, and wall-clock cost differs."]
    save("03_compute_matched.md", "\n".join(lines))


def main() -> None:
    validation = json.loads((DATA / "validation.json").read_text(encoding="utf-8"))
    if (validation.get("status"), validation.get("full_checkpoint_hashes"),
            validation.get("runs_checked"), len(validation.get("matched_pair_controls", []))) != ("PASS", True, 24, 9):
        raise RuntimeError("Full-hash validation and all paired controls required before reporting")
    per_seed = rows("per_seed.csv")
    require_complete(per_seed, ("A1_matched", "A2_matched", "compute_current_only"))
    agg = {(r["method"], r["arm"]): r for r in rows("aggregate.csv")}
    paired = {(r["before"], r["after"], r["arm"]): r for r in rows("paired_aggregate.csv")}
    diagnostics = rows("stage_diagnostics.csv")
    matched(per_seed, agg, paired, diagnostics)
    compute(per_seed, agg, paired)
    print("Wrote complete P0 reports")


if __name__ == "__main__":
    main()
