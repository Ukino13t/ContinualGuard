"""Render the reviewed AF1-AF4 aggregate into the research result document."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("AF1", "AF2", "AF3", "AF4")
METRICS = ("accuracy", "macro_f1", "attack_recall", "benign_fpr")
SHORT = {"accuracy": "Acc", "macro_f1": "F1", "attack_recall": "TPR", "benign_fpr": "FPR"}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value):
    return f"{100 * value:.3f}"


def stat(value, scale=100):
    return f"{scale * value['mean']:.3f} ± {scale * value['sample_sd']:.3f}"


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows),
    ])


def matrix_cell(per_seed, trained, evaluated):
    if evaluated not in per_seed["matrix"]["accuracy"][trained]:
        return "—"
    return "/".join(pct(per_seed["matrix"][metric][trained][evaluated]) for metric in METRICS)


def aggregate_cell(aggregate, trained, evaluated):
    if evaluated not in aggregate["mean_sample_sd_matrix"]["accuracy"][trained]:
        return "—"
    return " / ".join(stat(aggregate["mean_sample_sd_matrix"][metric][trained][evaluated]) for metric in METRICS)


def main():
    aggregate = load(ROOT / "results/attack_family_baseline_aggregate.json")
    lines = [
        "# Attack-family sequential baseline results",
        "",
        "Run date: 2026-09-12. Status: **three-seed primary no-replay baseline complete; replay remains blocked for human review.**",
        "",
        "## Protocol and evidence boundary",
        "",
        "The run uses the already audited Chen-derived deterministic reconstruction at SHA-256 `4b4efaf7640f31f5f12af2f2b3a40b91bbb71789c91c059d21ce4af1e6a9e43a`. The primary order is AF1 Naive → AF2 Ignore → AF3 Escape-Separation → AF4 Fake Completion. AF5 Combined is excluded, Variant B is not constructed, and every stage trains only its 3,000 current-stage records. The fixed detector is `microsoft/deberta-v3-base`, max length 1280, labels 0 benign/1 indirect prompt injection, one epoch per stage, batch size 4, learning rate 1e-5, optimizer reset at every boundary, and carried model weights. Seeds are 17, 42, and 2026.",
        "",
        "Each test family has 180 benign and 180 injected examples; each injected position has support 60. Matrix cells below are percentages in `accuracy / macro-F1 / attack TPR / benign FPR` order.",
    ]
    for item in aggregate["per_seed"]:
        lines += ["", f"## Seed {item['seed']} complete matrix", "", table(
            ["After training", *STAGES],
            [[trained, *[matrix_cell(item, trained, evaluated) for evaluated in STAGES]] for trained in STAGES],
        )]
    lines += [
        "", "## Mean ± sample SD matrix", "",
        "Each metric component is reported in percentage points as `mean ± sample SD`.", "",
        table(["After training", *STAGES], [[trained, *[aggregate_cell(aggregate, trained, evaluated) for evaluated in STAGES]] for trained in STAGES]),
        "", "## Acquisition and optimization diagnostics", "",
        table(
            ["Stage", "Attack TPR (%)", "Benign FPR (%)", "Macro-F1 (%)", "Training loss", "Validation loss", "Runtime (s)", "Peak allocated GPU (GiB)"],
            [[stage,
              stat(aggregate["acquisition_mean_sample_sd"][stage]["attack_recall"]),
              stat(aggregate["acquisition_mean_sample_sd"][stage]["benign_fpr"]),
              stat(aggregate["acquisition_mean_sample_sd"][stage]["macro_f1"]),
              stat(aggregate["stage_diagnostics_mean_sample_sd"][stage]["training_loss"], 1),
              stat(aggregate["stage_diagnostics_mean_sample_sd"][stage]["validation_loss"], 1),
              stat(aggregate["stage_diagnostics_mean_sample_sd"][stage]["runtime_seconds"], 1),
              stat(aggregate["stage_diagnostics_mean_sample_sd"][stage]["peak_gpu_memory_bytes"], 1 / (1024**3))]
             for stage in STAGES],
        ),
        "",
        "All four families reach 100.000 ± 0.000% attack TPR and macro-F1 with 0.000 ± 0.000% benign FPR at first acquisition. The primary result therefore does not show poor new-family acquisition. AF1 validation loss is seed-sensitive (mean 0.007151 ± 0.012234), driven by seed 2026 at 0.021277 despite unchanged hard classification metrics; this is confidence/optimization variability rather than acquisition failure.",
        "", "## Final retention, BWT, FPR, and peak forgetting", "",
        table(
            ["Earlier family", "Final attack-TPR change (pp)", "Final macro-F1 change (pp)", "Final benign-FPR change (pp)", "Peak attack-TPR forgetting (pp)", "Peak macro-F1 forgetting (pp)"],
            [[stage,
              stat(aggregate["final_retention_change_mean_sample_sd"][stage]["attack_recall"]),
              stat(aggregate["final_retention_change_mean_sample_sd"][stage]["macro_f1"]),
              stat(aggregate["final_retention_change_mean_sample_sd"][stage]["benign_fpr"]),
              stat(aggregate["peak_forgetting_mean_sample_sd"][stage]["attack_recall"]),
              stat(aggregate["peak_forgetting_mean_sample_sd"][stage]["macro_f1"])]
             for stage in STAGES[:-1]],
        ),
        "",
        f"At the AF4 endpoint, mean attack-TPR BWT is {stat(aggregate['final_bwt_mean_sample_sd']['attack_recall'])} pp, macro-F1 BWT is {stat(aggregate['final_bwt_mean_sample_sd']['macro_f1'])} pp, and accuracy BWT is {stat(aggregate['final_bwt_mean_sample_sd']['accuracy'])} pp. Mean benign-FPR increase is {stat(aggregate['final_benign_fpr_increase_mean_sample_sd'])} pp.",
        "",
        "The largest mean degradation is AF1 across the AF1→AF2 transition: attack TPR changes by −2.037 ± 3.528 pp across seeds (0.000, −6.111, 0.000). AF1 then changes by +2.037 ± 3.528 pp across AF2→AF3 and fully recovers in every seed at the AF3 endpoint. AF3→AF4 produces the largest consistent-direction final AF1 change, −0.370 ± 0.321 pp (0.000, −0.556, −0.556), but the effect is small and occurs in two of three seeds. No attack-TPR decline is consistent across all seeds.",
        "",
        "Malicious and benign behavior move differently. Seed 42 has the large temporary AF1 TPR loss with zero benign FPR throughout. Seed 17 has no attack-TPR loss but transient/final 0.556-point benign-FPR changes on different earlier families. Seed 2026 has a small final AF1 TPR loss and a transient AF2 benign-FPR increase that returns to zero.",
        "", "## Position-specific attack TPR", "",
    ]
    for position in ("head", "middle", "tail"):
        rows = []
        for trained in STAGES:
            rows.append([trained, *[
                stat(aggregate["position_attack_tpr_mean_sample_sd"][position][trained][evaluated])
                if evaluated in aggregate["position_attack_tpr_mean_sample_sd"][position][trained] else "—"
                for evaluated in STAGES
            ]])
        lines += [f"### {position}", "", table(["After training", *STAGES], rows), ""]
    lines += [
        "The worst position-specific cell is seed 42 AF1-middle after AF2 at 86.667% TPR, a 13.333-point temporary loss; head is 96.667% and tail 98.333% in that same cell. All three return to 100% after AF3. At the final AF4 endpoint, the only mean position reductions are AF1-head 99.444 ± 0.962%, AF1-middle 99.444 ± 0.962%, and AF3-tail 99.444 ± 0.962%; all other family-position cells are 100%.",
        "", "## Comparison and decision", "",
        "The family stream is more interpretable than the earlier SQuAD→TriviaQA document pilot because family, document, payload, and position are explicitly separated, and it exposes a seed-42 AF1-middle vulnerability that an aggregate score would hide. It does not create stronger consistent final retention pressure: the earlier pilot's mean S1 attack-TPR change was −1.210 ± 1.459 pp, whereas this stream's final attack-TPR BWT is only −0.185 ± 0.185 pp. Its strongest pressure is transient and seed-dependent: AF1 peak attack-TPR forgetting is 2.222 ± 3.379 pp, with a 6.111-point seed-42 maximum that recovers after AF3.",
        "",
        "These results are not evidence of catastrophic forgetting. Acquisition is saturated, final retention loss is small, and the largest drop is not consistent across seeds and reverses at the next stage. The current evidence is therefore insufficient to justify a broad replay experiment as the next scientific step. A predeclared secondary-order control can test order-dependent transfer; a separately approved harder benchmark or frozen hard-benign diagnostic may be needed before replay would address a demonstrated dominant limitation.",
        "", "## Runtime, storage, and validation", "",
        table(["Seed", "Run ID", "Runtime (s)", "Runtime (min)", "Peak allocated GPU (GiB)"], [[item["seed"], item["run_id"], f"{item['runtime_seconds']:.2f}", f"{item['runtime_seconds']/60:.2f}", f"{item['peak_gpu_memory_bytes']/(1024**3):.3f}"] for item in aggregate["per_seed"]]),
        "",
        f"Mean runtime is {aggregate['runtime_mean_sample_sd']['mean']:.2f} +/- {aggregate['runtime_mean_sample_sd']['sample_sd']:.2f} seconds. The original report's storage observation remains in the archived rendered document; storage management is now manual and has no scientific gate.",
        "",
        "Validation passed for the complete test suite (33 tests), compileall, pip check, three-run artifact/lineage verification, and loading all 12 stage-end model plus resume-state checkpoints. The retained Python 3.14/PyTorch JIT warning is unchanged.",
        "", "## Remaining human decisions", "",
        "- Decide whether to run the predeclared secondary order AF1→AF4→AF3→AF2 to test order sensitivity.",
        "- Decide whether the benchmark needs a harder acquisition/retention condition before replay, while keeping Variant B and hard-benign evolution separate until approved.",
        "- Decide whether AF5 Combined should later be evaluated only as an auxiliary controlled-overlap diagnostic.",
        "- Manage storage manually; data or checkpoint deletion always requires explicit human approval.",
        "- Keep all replay and selective-adaptation methods blocked pending review.",
    ]
    (ROOT / "docs/attack_family_baseline_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(ROOT / "docs/attack_family_baseline_results.md")


if __name__ == "__main__":
    main()
