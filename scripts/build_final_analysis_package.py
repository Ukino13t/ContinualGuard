from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(r"E:\ContinualGuard")
RESULTS = ROOT / "results"
TABLES = RESULTS / "final_tables"
FIGURES = RESULTS / "final_figures"
DOCS = ROOT / "docs"

METHODS = ["no_replay", "reservoir", "derpp", "security_aware", "loss_aware"]
METHOD_NAMES = {
    "no_replay": "M0 No Replay",
    "reservoir": "M1 Reservoir ER",
    "derpp": "M2 DER++",
    "security_aware": "M3 Security-aware",
    "loss_aware": "M4 Generic Loss-aware",
}
SHORT_NAMES = {
    "no_replay": "No Replay",
    "reservoir": "Reservoir ER",
    "derpp": "DER++",
    "security_aware": "Security-aware",
    "loss_aware": "Loss-aware",
}
COLORS = {
    "no_replay": "#4C566A",
    "reservoir": "#0072B2",
    "derpp": "#009E73",
    "security_aware": "#D55E00",
    "loss_aware": "#CC79A7",
}
ARMS = ["B-DOC", "B-SHORT", "B-HARD"]
SEEDS = [17, 42, 2026]
STAGES = ["AF2", "AF3", "AF4", "AF5"]


def load(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


OLD = load("replay_comparison_aggregate.json")
LOSS = load("loss_aware_control_aggregate.json")
OLD_MEMORY = load("replay_memory_composition.json")["rows"]
LOSS_MEMORY = load("loss_aware_memory_composition.json")["rows"]
SCORES = load("loss_aware_score_relationship.json")
OLD_FAIR = load("replay_fairness_validation.json")
LOSS_FAIR = load("loss_aware_fairness_validation.json")
OLD_ART = load("replay_artifact_validation.json")
LOSS_ART = load("loss_aware_artifact_validation.json")


def combined(key: str) -> list[dict]:
    rows = [dict(x) for x in OLD[key]] + [dict(x) for x in LOSS[key]]
    for row in rows:
        if "method" not in row:
            row["method"] = "loss_aware"
        row.setdefault("method_name", SHORT_NAMES[row["method"]])
    return rows


RUN = combined("run_summary_per_seed")
RETENTION = combined("retention_per_seed")
TRAJECTORY = combined("trajectory_per_seed")
ACQUISITION = combined("acquisition_per_seed")
EXTERNAL = combined("external_per_seed")
MEMORY = [dict(x) for x in OLD_MEMORY] + [dict(x) for x in LOSS_MEMORY]


def mean_sd(values):
    vals = [float(v) for v in values]
    return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0


def pct(x):
    return 100.0 * float(x)


def fmt_pm(values, scale=100.0, digits=3):
    m, s = mean_sd(values)
    return f"{m * scale:.{digits}f} ± {s * scale:.{digits}f}"


def fnum(x, digits=6):
    return f"{float(x):.{digits}f}"


def find_rows(rows, **criteria):
    return [r for r in rows if all(r.get(k) == v for k, v in criteria.items())]


def write_csv(path: Path, rows: list[dict], fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def final_external(method, arm, field):
    rows = find_rows(EXTERNAL, method=method, arm=arm, checkpoint_stage="AF5")
    return [r[field] for r in rows]


def run_values(method, arm, field):
    return [r[field] for r in find_rows(RUN, method=method, arm=arm)]


def retention_values(method, arm, task, field):
    return [r[field] for r in find_rows(RETENTION, method=method, arm=arm, task=task)]


def b_hard_d01_change(method):
    vals = []
    for seed in SEEDS:
        a = find_rows(TRAJECTORY, method=method, arm="B-HARD", seed=seed, checkpoint_stage="AF2", task="AF2")[0]
        z = find_rows(TRAJECTORY, method=method, arm="B-HARD", seed=seed, checkpoint_stage="AF5", task="AF2")[0]
        vals.append(z["benign_fpr_by_partition"]["D0-1"] - a["benign_fpr_by_partition"]["D0-1"])
    return vals


def aggregate_all_arms(method, field):
    return [r[field] for r in RUN if r["method"] == method]


def build_master_table():
    rows = []
    for method in METHODS:
        row = {
            "method_id": method,
            "method": METHOD_NAMES[method],
            "memory_capacity_examples": 0 if method == "no_replay" else 200,
            "auxiliary_logit_bytes_at_full_buffer": 1600 if method == "derpp" else 0,
            "acquisition_gates_passed": 12,
            "acquisition_gates_total": 12,
        }
        for arm in ARMS:
            tag = arm.lower().replace("-", "_")
            for field in ["attack_tpr_bwt", "macro_f1_bwt", "benign_fpr_increase", "attack_peak_forgetting_mean"]:
                values = run_values(method, arm, field)
                m, s = mean_sd(values)
                row[f"{tag}_{field}_mean_pp"] = 100 * m
                row[f"{tag}_{field}_sample_sd_pp"] = 100 * s
        for field in ["attack_tpr_loss", "attack_peak_forgetting", "macro_f1_loss"]:
            values = retention_values(method, "B-SHORT", "AF3", field)
            m, s = mean_sd(values)
            row[f"b_short_af3_{field}_mean_pp"] = 100 * m
            row[f"b_short_af3_{field}_sample_sd_pp"] = 100 * s
        vals = b_hard_d01_change(method)
        m, s = mean_sd(vals)
        row["b_hard_d0_1_final_fpr_change_mean_pp"] = 100 * m
        row["b_hard_d0_1_final_fpr_change_sample_sd_pp"] = 100 * s
        for arm in ARMS:
            tag = arm.lower().replace("-", "_")
            for field in ["notinject_fpr", "bshort_fpr", "bshort_english_fpr", "bshort_multilingual_fpr"]:
                vals = final_external(method, arm, field)
                m, s = mean_sd(vals)
                row[f"{tag}_final_{field}_mean_pp"] = 100 * m
                row[f"{tag}_final_{field}_sample_sd_pp"] = 100 * s
        runtimes = aggregate_all_arms(method, "runtime_seconds")
        peaks = aggregate_all_arms(method, "peak_gpu_memory_bytes")
        m, s = mean_sd(runtimes)
        row["runtime_minutes_mean"] = m / 60
        row["runtime_minutes_sample_sd"] = s / 60
        m, s = mean_sd(peaks)
        row["peak_gpu_memory_gib_mean"] = m / (1024 ** 3)
        row["peak_gpu_memory_gib_sample_sd"] = s / (1024 ** 3)
        rows.append(row)
    write_csv(TABLES / "five_method_master_table.csv", rows)
    return rows


def build_appendix_tables():
    acq = []
    for r in ACQUISITION:
        acq.append({
            "method": r["method"], "arm": r["arm"], "seed": r["seed"], "stage": r["stage"],
            "attack_tpr_pct": pct(r["attack_tpr"]), "benign_fpr_pct": pct(r["benign_fpr"]),
            "macro_f1_pct": pct(r["paired_macro_f1"]), "accuracy_pct": pct(r["paired_accuracy"]),
            "gate_pass": r["gate_pass"],
        })
    write_csv(TABLES / "appendix_A_acquisition_per_seed.csv", acq)

    bwt = []
    for r in RUN:
        bwt.append({
            "method": r["method"], "arm": r["arm"], "seed": r["seed"],
            "attack_tpr_bwt_pp": pct(r["attack_tpr_bwt"]), "macro_f1_bwt_pp": pct(r["macro_f1_bwt"]),
            "benign_fpr_increase_pp": pct(r["benign_fpr_increase"]),
            "mean_attack_peak_forgetting_pp": pct(r["attack_peak_forgetting_mean"]),
        })
    write_csv(TABLES / "appendix_B_bwt_per_seed.csv", bwt)

    short = []
    for r in TRAJECTORY:
        if r["arm"] == "B-SHORT" and r["task"] == "AF3" and STAGES.index(r["checkpoint_stage"]) >= 1:
            short.append({"method": r["method"], "seed": r["seed"], "checkpoint_stage": r["checkpoint_stage"],
                          "attack_tpr_pct": pct(r["attack_tpr"]), "macro_f1_pct": pct(r["macro_f1"]),
                          "benign_fpr_pct": pct(r["benign_fpr"])})
    write_csv(TABLES / "appendix_C_b_short_af3_trajectories.csv", short)

    hard = []
    for r in TRAJECTORY:
        if r["arm"] != "B-HARD":
            continue
        for partition, value in r.get("benign_fpr_by_partition", {}).items():
            hard.append({"method": r["method"], "seed": r["seed"], "checkpoint_stage": r["checkpoint_stage"],
                         "task": r["task"], "partition": partition, "benign_fpr_pct": pct(value)})
    write_csv(TABLES / "appendix_D_b_hard_benign_trajectories.csv", hard)

    ext = []
    for r in EXTERNAL:
        ext.append({"method": r["method"], "arm": r["arm"], "seed": r["seed"], "checkpoint_stage": r["checkpoint_stage"],
                    "notinject_fpr_pct": pct(r["notinject_fpr"]), "bshort_fpr_pct": pct(r["bshort_fpr"]),
                    "bshort_english_fpr_pct": pct(r["bshort_english_fpr"]),
                    "bshort_multilingual_fpr_pct": pct(r["bshort_multilingual_fpr"])})
    write_csv(TABLES / "appendix_E_external_diagnostics.csv", ext)

    mem = []
    for r in MEMORY:
        mem.append({
            "method": r["method"], "arm": r["arm"], "seed": r["seed"], "stage": r["stage"],
            "capacity_examples": r["capacity_examples"], "buffer_size": r["buffer_size"],
            "label_counts": r["label_counts"], "stage_distribution": r["stage_distribution"],
            "unique_coverage_keys": r["unique_coverage_keys"], "raw_text_stored": r["raw_text_stored"],
            "historical_logit_entries": r["historical_logit_entries"],
            "auxiliary_logit_bytes_float32": r["auxiliary_logit_bytes_float32"],
        })
    write_csv(TABLES / "appendix_F_memory_composition.csv", mem)

    runtime = []
    for r in RUN:
        runtime.append({"method": r["method"], "arm": r["arm"], "seed": r["seed"],
                        "runtime_seconds": r["runtime_seconds"], "runtime_minutes": r["runtime_seconds"] / 60,
                        "peak_gpu_memory_bytes": r["peak_gpu_memory_bytes"],
                        "peak_gpu_memory_gib": r["peak_gpu_memory_bytes"] / 1024 ** 3,
                        "replay_forward_passes": r.get("replay_forward_passes", ""),
                        "buffer_scoring_runtime_seconds": r.get("buffer_scoring_runtime_seconds", "")})
    write_csv(TABLES / "appendix_G_runtime_gpu.csv", runtime)

    fairness = [
        {"scope": "M1-M3 primary replay", "status": OLD_FAIR["status"], "capacity": OLD_FAIR["capacity"],
         "replay_batch_size": OLD_FAIR["replay_batch_size"], "fixed_common_factors": True,
         "only_intended_difference": "replay method and its required stored state/selection rule"},
        {"scope": "M3-M4 selector control", "status": LOSS_FAIR["status"], "capacity": LOSS_FAIR["memory_capacity_examples"],
         "replay_batch_size": LOSS_FAIR["replay_batch_size"], "fixed_common_factors": LOSS_FAIR["fixed_common_factors"],
         "only_intended_difference": LOSS_FAIR["only_intended_difference"]},
    ]
    write_csv(TABLES / "appendix_H_fairness_checks.csv", fairness)

    validation = [
        {"scope": "M0-M3", **{k: OLD_ART.get(k, "") for k in ["status", "new_runs", "new_checkpoints", "checkpoint_files_hashed", "prediction_files", "prediction_rows", "failed_runs", "resumed_runs", "checkpoint_lineage_verified"]}},
        {"scope": "M4", **{k: LOSS_ART.get(k, "") for k in ["status", "new_runs", "new_checkpoints", "checkpoint_files_hashed", "prediction_files", "prediction_rows", "failed_runs", "resumed_runs", "checkpoint_lineage_verified"]}},
    ]
    write_csv(TABLES / "appendix_I_artifact_validation.csv", validation)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_reproducibility_table():
    rows = []
    common = {
        "model": "microsoft/deberta-v3-base",
        "model_revision": "8ccc9b6f36199bec6961081d44eb72fb3f7353f3",
        "max_length": 1280,
        "seeds": "17|42|2026",
        "stage_order": "AF2>AF3>AF4>AF5",
        "epochs_per_stage": 1,
        "batch_size": 4,
        "learning_rate": "1e-5",
        "optimizer": "AdamW (betas=0.9/0.999, epsilon=1e-8, weight_decay=0)",
        "optimizer_reset_each_stage": True,
        "replay_exposure_ratio": "1:1 current:replay minibatch when memory nonempty",
        "decision_threshold": "argmax binary logits; p_attack=0.5 boundary; ties class 0",
        "precision": "float16",
        "cuda": "12.6 (PyTorch build cu126)",
        "device": "NVIDIA GeForce RTX 2060",
        "training_python": r"E:\ContinualGuard\.venv\Scripts\python.exe (3.14.5)",
        "analysis_python": r"E:\ContinualGuard\.venv\Scripts\python.exe (3.14.5)",
        "figure_renderer": r"D:\APP\ana\python.exe / Matplotlib 3.8.4 (rendering only; no model or metric computation)",
        "torch": "2.14.0+cu126",
        "transformers": "4.57.6",
        "candidate_b_materialization_manifest_sha256": "81962e092e1f8502a537d675582549df94f128f5ebf493b1ed0b59540ec3e8f4",
        "candidate_b_validation_sha256": "15f2fae9b34595bf586124092eb77b01ffbda44b93f6dd8fc0e78c25581a376a",
    }
    configs = {
        "no_replay": "configs/stronger_pilot_no_replay.yaml",
        "reservoir": "configs/replay_comparison_candidate_b.yaml",
        "derpp": "configs/replay_comparison_candidate_b.yaml",
        "security_aware": "configs/replay_comparison_candidate_b.yaml",
        "loss_aware": "configs/loss_aware_control.yaml",
    }
    for method in METHODS:
        config = ROOT / configs[method]
        rows.append({
            "method": METHOD_NAMES[method], **common,
            "config": configs[method], "config_sha256": sha256(config),
            "memory_capacity_examples": 0 if method == "no_replay" else 200,
            "replay_batch_size": 0 if method == "no_replay" else 4,
            "artifact_validation": "PASS",
            "fairness_validation": "PASS" if method != "no_replay" else "reference",
            "frozen_aggregate": "results/loss_aware_control_aggregate.json" if method == "loss_aware" else "results/replay_comparison_aggregate.json",
        })
    write_csv(TABLES / "reproducibility_table.csv", rows)


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(lines)


def per_method_summary(method):
    loss = retention_values(method, "B-SHORT", "AF3", "attack_tpr_loss")
    hard = run_values(method, "B-HARD", "benign_fpr_increase")
    d01 = b_hard_d01_change(method)
    return loss, hard, d01


def build_docs(master):
    primary_rows = []
    for method in METHODS:
        loss, hard, d01 = per_method_summary(method)
        primary_rows.append([SHORT_NAMES[method], fmt_pm(loss), fmt_pm(hard), fmt_pm(d01),
                             fmt_pm(final_external(method, "B-HARD", "notinject_fpr"))])
    primary_table = markdown_table(
        ["Method", "B-SHORT AF3 final TPR loss (pp)", "B-HARD FPR increase (pp)", "D0-1 final FPR change (pp)", "B-HARD final NotInject FPR (pp)"],
        primary_rows,
    )

    seed_rows = []
    for method in METHODS:
        loss, hard, d01 = per_method_summary(method)
        for i, seed in enumerate(SEEDS):
            seed_rows.append([SHORT_NAMES[method], seed, f"{100*loss[i]:.3f}", f"{100*hard[i]:.3f}", f"{100*d01[i]:.3f}"])
    seed_table = markdown_table(["Method", "Seed", "AF3 final TPR loss", "B-HARD FPR increase", "D0-1 final FPR change"], seed_rows)

    desc = f"""# Final descriptive statistics

Status: **EXPERIMENTS COMPLETE / FINAL ANALYSIS / REPORT WRITING**  
Frozen methods: M0 No Replay, M1 Reservoir ER, M2 DER++, M3 Security-aware Replay, M4 Generic Loss-aware Replay. All values below are descriptive means ± sample SD over seeds 17, 42, and 2026. Percentage-point changes are signed unless explicitly labeled as loss.

## Primary outcomes

{primary_table}

The decisive retention stressor is B-SHORT AF3 after AF5. No Replay loses **35.000 ± 38.115 pp** of attack TPR. Reservoir ER and Generic Loss-aware each have a mean final change of +0.556 pp (reported as a loss of -0.556 pp); DER++ is 0.000 ± 1.667 pp; Security-aware loses 1.111 ± 0.962 pp. The three replay methods from the primary comparison and the generic selector control all sharply reduce the large, seed-unstable No Replay loss.

On B-HARD, No Replay increases benign FPR by **20.370 ± 18.736 pp**. Reservoir ER reduces that to 1.481 ± 3.060 pp, DER++ to 0.370 ± 1.156 pp, Security-aware to -0.556 ± 0.962 pp, and Generic Loss-aware to 0.185 ± 0.321 pp. Negative change means the final benign FPR is lower than at acquisition.

## Per-seed primary values

{seed_table}

## Acquisition and stability

All five methods pass all 12 predeclared acquisition gates (three arms × four stages). This rules out failed current-stage learning as the explanation for the primary retention differences. The No Replay B-SHORT AF3 effect is directionally harmful in all three seeds, but its magnitude ranges from 6.667 to 78.333 pp. Replay methods remove most of that variability.

## Security-aware versus generic loss-aware

Generic Loss-aware has lower mean B-SHORT AF3 final TPR loss than Security-aware by 1.667 pp, with paired seed differences in Security-aware minus Loss-aware loss of +1.667, 0.000, and +3.333 pp. On B-HARD overall benign-FPR change, Security-aware is lower by 0.741 pp. On D0-1 final FPR change, the loss-aware values are 3.333, 0, and 0 pp while Security-aware is 0 in every seed.

These are small n=3 descriptive differences. They do not establish a unique security-aware selection benefit. For binary true-label cross-entropy, the loss-aware score is mathematically `-log(1 - security_risk)` under a common model state, so exact-arithmetic rankings are identical. Recorded differences arise from numeric precision/ties and then-diverged training states.

## Source tables

The complete machine-readable results are in `results/final_tables/five_method_master_table.csv` and Appendix Tables A–I. No inferential tests are reported because the experiment used three fixed seeds and was designed for controlled descriptive comparison.
"""

    claims = """# Final claims audit

Status: **EXPERIMENTS COMPLETE / FINAL ANALYSIS / REPORT WRITING**

| ID | Intended report claim | Classification | Exact evidence | Reporting limit |
|---|---|---|---|---|
| A | Continual prompt-injection detection can suffer retention failure under distribution shift. | **SUPPORTED within Candidate B** | No Replay loses B-SHORT AF3 attack TPR by 20.000, 6.667, and 78.333 pp after AF5; mean 35.000 ± 38.115 pp. | Candidate B is constructed; do not generalize prevalence or magnitude to deployment. |
| B | Replay materially reduces malicious-task forgetting. | **SUPPORTED** | The four replay means for B-SHORT AF3 final TPR loss are -0.556, 0.000, 1.111, and -0.556 pp, versus 35.000 pp for No Replay; every method passes 12/12 acquisition gates. | Descriptive n=3 result at M=200, not a universal method ranking. |
| C | Replay materially reduces benign false-positive drift. | **SUPPORTED** | B-HARD overall FPR increase falls from 20.370 ± 18.736 pp for No Replay to between -0.556 and 1.481 pp for replay methods. | Report overall and D0-1 partition changes separately. |
| D | Security-aware Replay is better than No Replay. | **SUPPORTED** | AF3 loss is 1.111 ± 0.962 versus 35.000 ± 38.115 pp; B-HARD drift is -0.556 ± 0.962 versus 20.370 ± 18.736 pp; acquisition is preserved. | Supported on frozen Candidate B endpoints, without a significance claim. |
| E | Security-aware Replay is better than Reservoir ER. | **PARTIALLY SUPPORTED / MIXED** | Security-aware is better on B-HARD drift (-0.556 versus 1.481 pp) and D0-1 (0 versus 5.000 pp), while Reservoir is better on B-SHORT AF3 loss (-0.556 versus 1.111 pp). | No uniform superiority. |
| F | Security-aware Replay is better than DER++. | **PARTIALLY SUPPORTED / MIXED** | Security-aware is better on B-HARD drift (-0.556 versus 0.370 pp) and D0-1 (0 versus 1.667 pp); DER++ is better on AF3 loss (0 versus 1.111 pp). | Differences are small and seed-sensitive. |
| G | Security-aware Replay is better than Generic Loss-aware Replay. | **PARTIALLY SUPPORTED / MIXED** | Security-aware is better on B-HARD drift (-0.556 versus 0.185 pp) and D0-1 (0 versus 1.111 pp); Loss-aware is better on AF3 loss (-0.556 versus 1.111 pp). | Neither dominates both security directions. |
| H | Security-specific scoring has an independent semantic advantage. | **NOT SUPPORTED** | Under a common binary model state, true-label CE = `-log(1-security_risk)`, a strictly increasing transform. Actual overlap declines with FP16/FP32 ties and diverged trajectories. | Do not interpret operational selection differences as semantic uniqueness. |
| I | Security-aware Replay improves bidirectional security stability. | **PARTIALLY SUPPORTED** | It strongly improves both primary directions versus No Replay and has the best observed B-HARD benign retention, but generic methods also repair both failures. | Use “promising” and benchmark-specific wording. |
| J | Candidate B demonstrates universal real-world deployment robustness. | **NOT SUPPORTED** | One detector, three seeds, one memory budget, constructed data, and diagnostic external sets cannot establish deployment robustness. | Avoid universal, production, or deployment-validity claims. |

Historical documents retain the status that was true when they were written. Current summary documents and the final package use the final frozen interpretation.
"""

    results = f"""# Final results narrative

## 1. Benchmark qualification

The earlier H2-mini stream remained at R0 because acquisition was near ceiling and no predeclared primary retention metric crossed the replay-entry threshold. Candidate B retained the frozen detector and malicious stream while introducing matched document, short-form, and reviewed hard-benign arms. Its no-replay run reached R2 through a large B-SHORT AF3 malicious-retention failure and a separate B-HARD benign-FPR failure. Benchmark escalation then stopped and the predeclared replay comparison began.

The five-method comparison separates current-stage acquisition, malicious-family retention, and benign over-defense. Every method passed every acquisition gate, so the central differences appear after later stages rather than at first learning.

{primary_table}

## 2. No-replay failure modes

The strongest malicious-retention result occurs on B-SHORT AF3. No Replay lost 35.000 ± 38.115 percentage points by the AF5 endpoint, with losses in every seed and a range of 6.667–78.333 points. In B-HARD, No Replay increased overall benign FPR by 20.370 ± 18.736 points and D0-1 FPR by 62.222 ± 54.297 points.

## 3. Replay comparison

All replay methods held B-SHORT AF3 mean loss between -0.556 and 1.111 points and B-HARD overall FPR drift within 1.481 points of zero. Reservoir ER achieved -0.556 ± 2.546 pp AF3 loss and 1.481 ± 3.060 pp B-HARD drift. DER++ achieved 0.000 ± 1.667 and 0.370 ± 1.156 pp. Security-aware achieved 1.111 ± 0.962 and -0.556 ± 0.962 pp. Generic Loss-aware achieved -0.556 ± 0.962 and 0.185 ± 0.321 pp.

## 4. Main replay finding

Reservoir ER, DER++, Security-aware, and Generic Loss-aware all resolve the dominant No Replay failures under the fixed 200-example memory budget without failing acquisition.

## 5. Security-aware finding

Security-aware provides the best observed B-HARD benign-retention control. Its overall mean change is -0.556 ± 0.962 pp, and D0-1 is unchanged in every seed. It is not uniformly superior: Reservoir and Loss-aware have lower mean AF3 attack-TPR loss, and DER++ is also slightly lower on that endpoint.

## 6. Generic Loss-aware control

Generic Loss-aware narrows the causal interpretation. It slightly improves the AF3 mean relative to Security-aware but slightly worsens the benign-boundary mean. For a fixed binary model state, true-label CE is a monotone transform of security risk. The control therefore supports structured hard-example retention while preventing a claim that the security-specific formula has independently demonstrated semantic value.

## 7. External diagnostics

B-DOC remains severely over-defensive across methods, while B-SHORT and B-HARD are near zero. All replay methods reach zero final B-short FPR on B-HARD; final NotInject FPR is zero for Security-aware and Loss-aware and small for Reservoir and DER++. These sets are diagnostic-only and were not used for method selection. Full per-seed values are in Appendix E.

## 8. Overall interpretation

Replay strategy changes the malicious-retention versus benign-retention trade-off. The class of replay interventions is strongly supported relative to No Replay, while method-specific differences remain mixed and descriptive at n=3.
"""

    discussion = """# Final discussion narrative

Static prompt-injection evaluation is insufficient when the deployment distribution evolves. A detector can retain malicious recall yet reject many benign inputs, or preserve benign acceptance while forgetting an earlier attack family. Continual evaluation must measure both malicious retention and benign over-defense at every stage.

Simple attack-family streams can ceiling-saturate. The AF1–AF4 and H2-mini phases largely did so, which made small score movements difficult to interpret. Candidate B is more informative because its short-form arm exposes an earlier-family recall loss and its hard-benign arm exposes false-positive drift under the same frozen malicious assignment. Benchmark escalation stopped prospectively once Candidate B met the frozen R2 criteria.

The project’s main empirical result is that a small fixed replay memory removes most of the retention pressure exposed by Candidate B. Replay works here because later optimization continues to expose the model to a bounded sample of earlier decision boundaries. This conclusion is stronger than a claim about any one replay selector: Reservoir ER, DER++, Security-aware, and Generic Loss-aware all perform well, while No Replay is unstable on B-SHORT AF3 and drifts on B-HARD benign FPR.

The controlled selector comparison narrows the interpretation of Security-aware Replay. In binary classification, its risk score and true-label cross-entropy are linked by a strictly increasing transform. The actual implementation still produced different buffers because scores were frozen with different numeric precision and later model states diverged, but that operational difference is not evidence of a distinct semantic ranking principle. The fairest conclusion is that class/stage-balanced hard-example replay works here; the current study does not isolate an additional security-specific advantage.

Candidate B is more informative than the earlier SQuAD-to-TriviaQA document/domain pilot because it creates a reproducible, interpretable retention failure in one malicious family and a large benign-boundary drift in the hard-benign arm. It still remains a constructed research benchmark. The attack families come from a Chen-derived pipeline, AF5 is compositional, benign partitions combine source and language effects, and external benign sets do not represent deployment prevalence.

The malicious-versus-benign pattern is a concrete stability-plasticity problem. Current-stage acquisition remains strong, while the decision boundary for earlier examples can move in either security direction. Future RAG guardrails should monitor both missed injections and blocked benign context over time, retain auditable historical examples, and treat selector design as a trade-off rather than a single-score optimization. These are design implications from the benchmark, not deployment validation.

The fixed 200-example budget is adequate for this benchmark but does not establish a memory-performance frontier. Likewise, three deterministic seeds support paired descriptive comparisons, not population-level uncertainty or claims of equivalence. The study should therefore be reported as a completed controlled comparison with clear benchmark-specific effects and explicit limits.
"""

    method = """# Final method description

We evaluated a fixed `microsoft/deberta-v3-base` binary indirect-prompt-injection detector on the frozen Candidate B stream `AF2 → AF3 → AF4 → AF5`. Labels were 0 for benign and 1 for indirect prompt injection. All methods used maximum sequence length 1280, one epoch per stage, batch size 4, learning rate 1e-5, constant scheduling, FP16 CUDA training, optimizer reset at every stage, weights carried across stages, and seeds 17, 42, and 2026.

The three benchmark arms share the malicious assignment and vary benign pressure: B-DOC uses document controls, B-SHORT uses short benign controls, and B-HARD uses the reviewed hard-benign progression. Evaluation after each stage covers every seen stage, with attack TPR, benign FPR, macro-F1, accuracy, per-position diagnostics, acquisition gates, BWT, and peak forgetting. Frozen NotInject and B-short sets provide external over-defense diagnostics.

M0 trains only on current-stage data. M1 is the published Reservoir ER baseline implemented as a standard unstratified reservoir with 200 examples. M2 is the published DER++ baseline adapted to the common harness, with the same memory capacity, historical logits, and alpha=beta=0.5. M3 is the project’s proposed Security-aware Replay method. It selects a 200-example buffer with exactly 100 benign and 100 malicious examples, deterministic stage-balanced quotas, coverage-key diversity, and hash tie-breaking. Scores are computed once, before the stage update, on current-stage TRAIN data only. Security risk is `1 - p_attack` for malicious records and `p_attack` for benign records. Its replay classification loss is the same CE rule as Reservoir: `(current CE + replay CE) / 2`.

M4 is the project’s matched Generic Loss-aware control. It keeps the M3 exposure, replay CE loss, class/stage quotas, coverage behavior, tie-breaking, and capacity fixed, changing only selection to pre-update true-label cross-entropy. Under a common binary model state and fixed label, `true-label CE = -log(1 - security_risk)`. The transform is strictly increasing, so exact rankings coincide. Operational differences in this study begin with FP16/FP32 score precision and ties and then compound as model trajectories diverge.

All reported results come from frozen run artifacts. No checkpoint was retrained or re-evaluated during final analysis. Machine-readable tables were derived from the validated aggregate JSON files listed in the reproducibility table.
"""

    limitations = """# Final limitations

- The comparison has three fixed seeds. Means and sample SDs describe these runs; no statistical significance claim is made.
- Candidate B is a controlled constructed benchmark. It does not estimate real-world attack or benign prevalence.
- The Chen-derived attack pipeline is a deterministic reconstruction rather than an exact reproduction of the original study.
- AF5 is compositional and reuses earlier attack components, so later-stage effects cannot be interpreted as fully unseen attack-family evolution.
- B-DOC, B-SHORT, and B-HARD differ in source, length, language, and semantic properties. Arm effects are package effects rather than isolated causal effects of one factor.
- NotInject and B-short are external over-defense diagnostics; zero FPR on these finite sets does not establish deployment safety or calibration.
- The single 200-example memory budget does not establish a capacity curve or optimal budget.
- The study uses one replay exposure ratio: one replay minibatch per current minibatch when memory is nonempty.
- Each stage test has 60 examples per class, so one example changes a class-conditional rate by 1.667 percentage points.
- DER++ uses 1,600 bytes of auxiliary FP32 logits at full capacity in addition to the common example memory. Runtime and storage comparisons should retain this distinction.
- Security-risk and true-label CE scores are monotonic transforms for binary labels under the same model state. This design cannot establish an independent semantic advantage for the security-aware score.
- Numeric precision, tie behavior, and model-state divergence affect selector comparisons. Recorded score correlations and buffer overlaps expose these implementation effects.
- Score saturation is substantial at later stages, especially for malicious AF5 examples, reducing rank resolution and making tie handling consequential.
- The external B-short set is language-mismatched across slices (42 English and 78 multilingual records), so aggregate FPR is composition-dependent.
- The detector architecture, tokenizer, threshold, stage order, and one-epoch schedule are fixed. Conclusions may change under other architectures, thresholds, orders, or optimization budgets.
- There is no deployment-scale validation.
- Raw source datasets remain local under their source terms and are not redistributed through Git.
"""

    timeline = """# Final experiment timeline

| Phase | Development step | Outcome |
|---|---|---|
| 1 | AF1–AF4 baseline | Near-ceiling acquisition and retention showed that a family-disjoint stream alone was too easy for replay evaluation. |
| 2 | Order control | A secondary order exposed transient, order-dependent interference without a strong endpoint replay case. |
| 3 | NotInject hard-benign diagnosis | Established severe over-defense motivation while preserving NotInject as a diagnostic rather than a causal trigger test. |
| 4 | H2-mini construction and human review | Built a provenance-aware benign stream from independently reviewed material. |
| 5 | H2-mini no-replay negative result | Acquisition ceiling and sub-threshold retention changes kept H2-mini at R0. |
| 6 | Candidate B stronger benchmark | Froze matched B-DOC, B-SHORT, and B-HARD arms with shared malicious assignments. |
| 7 | Candidate B R2 no-replay result | B-SHORT AF3 forgetting and B-HARD benign drift met frozen replay-entry criteria; R2 was reached. |
| 8 | Primary replay comparison | Reservoir ER, DER++, and Security-aware Replay completed for 3 arms × 3 seeds at M=200. |
| 9 | Generic Loss-aware control | Completed the matched selector control and clarified the algebraic/numeric score relationship. |
| 10 | Final analysis | Experiments froze; tables, figures, claims audit, and report-ready narratives were generated from validated artifacts only. |

Benchmark escalation stopped once Candidate B satisfied the frozen replay-entry criteria. Replay methods were then evaluated without changing the detector, benchmark, seeds, order, or budget. This prospective sequence avoids post-hoc benchmark tuning to favor a replay method.
"""

    contribution = """# Final contribution statement

1. **Benchmark and methodology.** We provide a controlled security-bidirectional continual prompt-injection benchmark with matched document, short-form, and hard-benign arms that exposes both malicious forgetting and benign over-defense.
2. **Empirical replay result.** Under a fixed DeBERTa-v3-base detector and a fixed 200-example memory, replay strongly mitigates the dominant retention failures without materially harming new-stage acquisition.
3. **Replay-selection result.** Structured class/stage-balanced hard-example replay changes the malicious-versus-benign stability trade-off. Security-aware Replay gives the strongest observed B-HARD benign protection, while the Generic Loss-aware control shows that current evidence does not isolate a uniquely security-specific ranking advantage.
"""

    consistency = """# Final documentation consistency audit

Status: **PASS WITH HISTORICAL RECORDS PRESERVED**

The final package was checked against the prohibited outdated claims. Current final-report documents use the frozen interpretation below:

- Security-aware is not universally best; its advantage is mixed across malicious and benign endpoints.
- B1/B2/B3 are descriptive source strata, not ordinal difficulty levels.
- H2-mini remained R0 and did not prove forgetting.
- AF5 is compositional and is not family-disjoint.
- NotInject is an external over-defense diagnostic, not a pure causal trigger test.
- The replay budget is fixed at M=200 for the final comparison.
- Security-aware risk has no demonstrated unique semantic advantage over true-label CE under a common model state.

Dated protocol, audit, and experiment-history documents were not rewritten when their wording accurately represented the state at that time. The current sources of truth are `final_project_summary.md`, `final_claims_audit.md`, `final_results_narrative.md`, and the machine-readable final tables.
"""

    summary = """# Final project summary

Status: **EXPERIMENTS COMPLETE / FINAL ANALYSIS / REPORT WRITING**

The project developed and audited a continual-learning evaluation for indirect prompt-injection detection, then used it to compare No Replay, Reservoir ER, DER++, Security-aware Replay, and a matched Generic Loss-aware control. Candidate B supplied two complementary stressors: malicious-family retention on B-SHORT AF3 and benign over-defense on B-HARD.

At the fixed 200-example budget, every replay method sharply improved both stressors relative to No Replay while preserving current-stage acquisition. Reservoir ER was already strong; DER++ and Security-aware did not resolve a clear decisive advantage over it. Generic Loss-aware matched or slightly improved malicious retention relative to Security-aware but was slightly worse on the benign boundary. Because binary security risk and true-label cross-entropy induce the same exact ranking under a shared model state, the study does not establish an independent semantic benefit for Security-aware selection.

The completed contribution is a provenance-aware, human-reviewed benchmark and a fully paired five-method descriptive comparison that separates acquisition, malicious retention, benign false positives, external over-defense, memory composition, runtime, and numeric selector behavior. The result supports replay as a class of interventions for this benchmark and limits method-specific claims to the observed fixed setting.

Primary artifacts:

- `results/final_tables/five_method_master_table.csv`
- `results/final_tables/reproducibility_table.csv`
- `results/final_tables/appendix_A_acquisition_per_seed.csv` through `appendix_I_artifact_validation.csv`
- `results/final_figures/` PNG/PDF figure pairs
- `docs/final_descriptive_statistics.md`
- `docs/final_claims_audit.md`
- `docs/final_results_narrative.md`
- `docs/final_discussion_narrative.md`
- `docs/final_method_description.md`
- `docs/final_limitations.md`
- `docs/final_experiment_timeline.md`
- `docs/final_contribution_statement.md`
- `docs/final_consistency_audit.md`

No further experiment is part of the approved project state.
"""

    docs = {
        "final_descriptive_statistics.md": desc,
        "final_claims_audit.md": claims,
        "final_results_narrative.md": results,
        "final_discussion_narrative.md": discussion,
        "final_method_description.md": method,
        "final_limitations.md": limitations,
        "final_experiment_timeline.md": timeline,
        "final_contribution_statement.md": contribution,
        "final_consistency_audit.md": consistency,
        "final_project_summary.md": summary,
    }
    for name, text in docs.items():
        (DOCS / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def set_plot_style(plt):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
        "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 150,
        "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def savefig(fig, stem, plt):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_figures():
    import matplotlib.pyplot as plt
    import numpy as np
    set_plot_style(plt)

    # 1. B-SHORT AF3 trajectories.
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    xs = np.arange(3)
    for method in METHODS:
        by_seed = []
        for seed in SEEDS:
            vals = []
            for stage in ["AF3", "AF4", "AF5"]:
                vals.append(pct(find_rows(TRAJECTORY, method=method, arm="B-SHORT", seed=seed, checkpoint_stage=stage, task="AF3")[0]["attack_tpr"]))
            by_seed.append(vals)
            ax.plot(xs, vals, color=COLORS[method], alpha=.18, linewidth=.9)
        arr = np.asarray(by_seed)
        ax.plot(xs, arr.mean(0), marker="o", linewidth=2.4, color=COLORS[method], label=SHORT_NAMES[method])
    ax.set_xticks(xs, ["After AF3\n(acquisition)", "After AF4", "After AF5"])
    ax.set_ylabel("AF3 attack TPR (%)")
    ax.set_ylim(0, 104)
    ax.set_title("B-SHORT: AF3 retention across later stages")
    ax.legend(ncol=2, frameon=False, loc="lower left")
    ax.text(.995, .01, "Thin lines: individual seeds; bold: mean", transform=ax.transAxes, ha="right", va="bottom", color="#555")
    savefig(fig, "b_short_af3_retention", plt)

    # 2. B-HARD benign partition retention.
    partitions = [("D0-1", "AF2"), ("B0-2", "AF3"), ("BH-1", "AF4")]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=True)
    for ax, (partition, task) in zip(axes, partitions):
        start = STAGES.index(task)
        stages = STAGES[start:]
        for method in METHODS:
            means = []
            sds = []
            for stage in stages:
                vals = []
                for seed in SEEDS:
                    row = find_rows(TRAJECTORY, method=method, arm="B-HARD", seed=seed, checkpoint_stage=stage, task=task)[0]
                    vals.append(pct(row["benign_fpr_by_partition"][partition]))
                m, s = mean_sd(vals)
                means.append(m); sds.append(s)
            ax.errorbar(range(len(stages)), means, yerr=sds, marker="o", capsize=2, linewidth=1.8, color=COLORS[method], label=SHORT_NAMES[method])
        ax.set_xticks(range(len(stages)), stages)
        ax.set_title(f"{partition} (introduced {task})")
        ax.set_xlabel("Checkpoint")
    axes[0].set_ylabel("Benign FPR (%)")
    axes[-1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("B-HARD: benign-boundary retention by partition", y=1.02)
    savefig(fig, "b_hard_benign_retention", plt)

    # 3. Malicious-retention / benign-boundary trade-off.
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for method in METHODS:
        x = [pct(v) for v in retention_values(method, "B-SHORT", "AF3", "attack_tpr_loss")]
        y = [pct(v) for v in run_values(method, "B-HARD", "benign_fpr_increase")]
        xm, xsdev = mean_sd(x); ym, ysdev = mean_sd(y)
        ax.errorbar(xm, ym, xerr=xsdev, yerr=ysdev, fmt="o", markersize=7, capsize=3, color=COLORS[method], label=SHORT_NAMES[method])
    ax.axhline(0, color="#777", linewidth=.8)
    ax.axvline(0, color="#777", linewidth=.8)
    ax.set_xlabel("B-SHORT AF3 final attack-TPR loss (pp; lower is better)")
    ax.set_ylabel("B-HARD benign-FPR increase (pp; lower is better)")
    ax.set_title("Malicious retention versus benign-boundary drift")
    ax.legend(frameon=False)
    savefig(fig, "malicious_benign_tradeoff", plt)

    # 4. Memory selection/composition.
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))
    replay_methods = METHODS[1:]
    final_rows = [r for r in MEMORY if r["arm"] == "B-SHORT" and r["stage"] == "AF5"]
    for i, method in enumerate(replay_methods):
        rows = [r for r in final_rows if r["method"] == method]
        benign = statistics.mean(r["label_counts"].get("0", 0) for r in rows)
        mal = statistics.mean(r["label_counts"].get("1", 0) for r in rows)
        axes[0].bar(i, benign, color="#56B4E9")
        axes[0].bar(i, mal, bottom=benign, color="#E69F00")
    axes[0].set_xticks(range(4), ["ER", "DER++", "Sec", "Loss"], rotation=0)
    axes[0].set_ylabel("Examples")
    axes[0].set_title("Final label composition")
    axes[0].legend(["Benign", "Injection"], frameon=False, fontsize=7)
    bottoms = np.zeros(4)
    stage_colors = ["#0072B2", "#009E73", "#F0E442", "#D55E00"]
    for stage, color in zip(STAGES, stage_colors):
        vals = []
        for method in replay_methods:
            rows = [r for r in final_rows if r["method"] == method]
            vals.append(statistics.mean(r["stage_distribution"].get(stage, 0) for r in rows))
        axes[1].bar(range(4), vals, bottom=bottoms, color=color, label=stage)
        bottoms += np.asarray(vals)
    axes[1].set_xticks(range(4), ["ER", "DER++", "Sec", "Loss"])
    axes[1].set_title("Final stage composition")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    overlap = [r for r in SCORES["overlap_aggregate"] if r["label"] == "all"]
    axes[2].plot(range(4), [100*r["jaccard"]["mean"] for r in overlap], marker="o", color="#6A3D9A")
    axes[2].fill_between(range(4), [100*(r["jaccard"]["mean"]-r["jaccard"]["sample_sd"]) for r in overlap],
                         [100*(r["jaccard"]["mean"]+r["jaccard"]["sample_sd"]) for r in overlap], color="#CAB2D6", alpha=.45)
    axes[2].set_xticks(range(4), STAGES)
    axes[2].set_ylim(0, 105)
    axes[2].set_ylabel("Jaccard overlap (%)")
    axes[2].set_title("Security vs Loss buffers")
    fig.suptitle("Replay memory composition and selector overlap", y=1.02)
    savefig(fig, "memory_selection_comparison", plt)

    # 5. Score relationship.
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    risk = np.linspace(0, .99, 300)
    axes[0].plot(risk, -np.log(1-risk), color="#6A3D9A", linewidth=2.3)
    axes[0].set_xlabel("Security risk")
    axes[0].set_ylabel("True-label CE = -log(1-risk)")
    axes[0].set_title("Exact binary-score relation")
    corr = [r for r in SCORES["correlation_aggregate"] if r["label"] == "all"]
    jac = [r for r in SCORES["overlap_aggregate"] if r["label"] == "all"]
    axes[1].plot(range(4), [r["spearman"]["mean"] for r in corr], marker="o", label="Spearman ρ", color="#0072B2")
    axes[1].plot(range(4), [r["jaccard"]["mean"] for r in jac], marker="s", label="Selected-buffer Jaccard", color="#D55E00")
    axes[1].set_xticks(range(4), STAGES)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Relationship")
    axes[1].set_title("Recorded FP16/FP32 implementation")
    axes[1].legend(frameon=False)
    fig.suptitle("Security-risk and generic-loss selector scores", y=1.02)
    savefig(fig, "security_loss_score_relationship", plt)

    # 6. External over-defense at final checkpoint.
    fig, axes = plt.subplots(1, 3, figsize=(11.1, 3.8), sharey=True)
    width = .16
    x = np.arange(2)
    for ax, arm in zip(axes, ARMS):
        for i, method in enumerate(METHODS):
            ni = [pct(v) for v in final_external(method, arm, "notinject_fpr")]
            bs = [pct(v) for v in final_external(method, arm, "bshort_fpr")]
            means = [statistics.mean(ni), statistics.mean(bs)]
            sds = [statistics.stdev(ni), statistics.stdev(bs)]
            ax.bar(x + (i-2)*width, means, width, yerr=sds, capsize=2, color=COLORS[method], label=SHORT_NAMES[method])
        ax.set_xticks(x, ["NotInject", "B-short"])
        ax.set_title(arm)
    axes[0].set_ylabel("Final false-positive rate (%)")
    axes[-1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Frozen external over-defense diagnostics after AF5", y=1.02)
    savefig(fig, "external_overdefense", plt)


def validate():
    expected = [
        TABLES / "five_method_master_table.csv", TABLES / "reproducibility_table.csv",
        *[TABLES / f"appendix_{letter}_{name}.csv" for letter, name in [
            ("A", "acquisition_per_seed"), ("B", "bwt_per_seed"), ("C", "b_short_af3_trajectories"),
            ("D", "b_hard_benign_trajectories"), ("E", "external_diagnostics"),
            ("F", "memory_composition"), ("G", "runtime_gpu"), ("H", "fairness_checks"),
            ("I", "artifact_validation")]],
        *[DOCS / name for name in ["final_descriptive_statistics.md", "final_claims_audit.md", "final_results_narrative.md",
                                   "final_discussion_narrative.md", "final_method_description.md", "final_limitations.md",
                                   "final_project_summary.md", "final_experiment_timeline.md", "final_contribution_statement.md",
                                   "final_consistency_audit.md"]],
    ]
    if FIGURES.exists():
        for stem in ["b_short_af3_retention", "b_hard_benign_retention", "malicious_benign_tradeoff",
                     "memory_selection_comparison", "security_loss_score_relationship", "external_overdefense"]:
            expected += [FIGURES / f"{stem}.png", FIGURES / f"{stem}.pdf"]
    missing = [str(p) for p in expected if not p.exists() or p.stat().st_size == 0]
    if missing:
        raise AssertionError("missing or empty outputs: " + ", ".join(missing))
    with (TABLES / "five_method_master_table.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5 and [r["method_id"] for r in rows] == METHODS
    assert len(ACQUISITION) == 5 * 3 * 3 * 4
    assert all(r["gate_pass"] for r in ACQUISITION)
    assert OLD_ART["status"] == LOSS_ART["status"] == "PASS"
    print(json.dumps({"status": "PASS", "validated_outputs": len(expected), "methods": 5,
                      "acquisition_rows": len(ACQUISITION), "frozen_input_files": 8}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Build final analysis artifacts from frozen aggregate JSON files only.")
    parser.add_argument("--mode", choices=["all", "tables-docs", "figures", "validate"], default="all")
    args = parser.parse_args()
    if ROOT.resolve() != Path.cwd().resolve():
        raise SystemExit(f"run from canonical root {ROOT}")
    if args.mode in {"all", "tables-docs"}:
        TABLES.mkdir(parents=True, exist_ok=True)
        master = build_master_table()
        build_appendix_tables()
        build_reproducibility_table()
        build_docs(master)
    if args.mode in {"all", "figures"}:
        build_figures()
    if args.mode in {"all", "validate"}:
        validate()


if __name__ == "__main__":
    main()
