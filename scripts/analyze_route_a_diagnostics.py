"""Frozen external and qualitative diagnostic analysis; no raw text exported."""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from continualguard.notinject_evaluation import load_notinject

ROOT = Path(r"E:\ContinualGuard")
OUT = ROOT / "results/extended/analysis"
METHODS = {
    "M0 No Replay": ROOT / "results/stronger_pilot_no_replay",
    "M1 Reservoir": ROOT / "results/replay_comparison_candidate_b/reservoir",
    "M3 Security-aware": ROOT / "results/replay_comparison_candidate_b/security_aware",
    "M4 Structured CE": ROOT / "results/replay_comparison_candidate_b/loss_aware",
}
ARMS = ("doc", "short", "hard")
SEEDS = (17, 42, 2026)
STAGES = ("AF2", "AF3", "AF4", "AF5")


def records(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def csv_out(name, rows):
    if not rows:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def external():
    rows = []
    notinject_length = {r["record_id"]: len(r["prompt"]) for r in load_notinject(ROOT)}
    bshort_length = {r["example_id"]: len(r["text"]) for r in records(
        ROOT / "data/processed/h2_mini/external/b_short.jsonl")}
    length_maps = {"NotInject": notinject_length, "B-short": bshort_length}
    def length_bin(n):
        return "0-128 chars" if n <= 128 else ("129-256 chars" if n <= 256 else ">256 chars")
    for method, base in METHODS.items():
        for arm in ARMS:
            for seed in SEEDS:
                p = base / arm / f"seed{seed}" / "stage4/predictions"
                for bank in ("NotInject", "B-short"):
                    items = records(p / f"{bank}.jsonl")
                    subsets = {"all": items}
                    assert all(x["example_id"] in length_maps[bank] for x in items)
                    for bin_name in ("0-128 chars", "129-256 chars", ">256 chars"):
                        chosen = [x for x in items if length_bin(length_maps[bank][x["example_id"]]) == bin_name]
                        if chosen:
                            subsets[f"length={bin_name}"] = chosen
                    for field in (("hb_group", "category") if bank == "NotInject" else ("language",)):
                        for value in sorted({str(x.get(field)) for x in items}):
                            subsets[f"{field}={value}"] = [x for x in items if str(x.get(field)) == value]
                    for group, examples in subsets.items():
                        rows.append({"method": method, "arm": arm, "seed": seed, "bank": bank,
                                     "slice": group, "n": len(examples),
                                     "false_positives": sum(x["predicted_label"] == 1 for x in examples),
                                     "fpr_pct": 100 * statistics.mean(x["predicted_label"] == 1 for x in examples)})
    csv_out("external_slices_per_seed.csv", rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["arm"], row["bank"], row["slice"])].append(row)
    aggregate = []
    for (method, arm, bank, sl), group in grouped.items():
        vals = [x["fpr_pct"] for x in group]
        aggregate.append({"method": method, "arm": arm, "bank": bank, "slice": sl,
                          "n_per_seed": group[0]["n"], "n_seeds": len(vals),
                          "fpr_pct_mean": statistics.mean(vals),
                          "fpr_pct_sample_sd": statistics.stdev(vals) if len(vals) > 1 else ""})
    csv_out("external_slices_aggregate.csv", aggregate)
    return rows, aggregate


def qualitative():
    cases = []
    for arm in ARMS:
        for seed in SEEDS:
            for task_i, task in enumerate(STAGES[:3], 1):
                by_method = {}
                for method, base in METHODS.items():
                    initial = {x["example_id"]: x for x in records(base / arm / f"seed{seed}" / f"stage{task_i}/predictions/{task}.jsonl")}
                    final = {x["example_id"]: x for x in records(base / arm / f"seed{seed}" / f"stage4/predictions/{task}.jsonl")}
                    by_method[method] = (initial, final)
                m0_init, m0_final = by_method["M0 No Replay"]
                er_final = by_method["M1 Reservoir"][1]
                ce_final = by_method["M4 Structured CE"][1]
                sec_final = by_method["M3 Security-aware"][1]
                for example_id, old in m0_init.items():
                    last = m0_final[example_id]
                    if old["true_label"] == 1 and old["predicted_label"] == 1 and last["predicted_label"] == 0:
                        category = "forgotten_attack"
                    elif old["true_label"] == 0 and old["predicted_label"] == 0 and last["predicted_label"] == 1:
                        category = "new_benign_false_positive"
                    else:
                        continue
                    prevented_er = er_final[example_id]["predicted_label"] == old["true_label"]
                    prevented_ce = ce_final[example_id]["predicted_label"] == old["true_label"]
                    prevented_sec = sec_final[example_id]["predicted_label"] == old["true_label"]
                    cases.append({"category": category, "arm": arm, "seed": seed, "task": task,
                                  "example_id": example_id, "position": old.get("injection_position"),
                                  "benign_group": old.get("benign_group"),
                                  "m0_acquisition_p_attack": old["p_attack"],
                                  "m0_final_p_attack": last["p_attack"],
                                  "er_final_p_attack": er_final[example_id]["p_attack"],
                                  "ce_final_p_attack": ce_final[example_id]["p_attack"],
                                  "security_final_p_attack": sec_final[example_id]["p_attack"],
                                  "er_prevents_error": prevented_er,
                                  "ce_prevents_error": prevented_ce,
                                  "security_prevents_error": prevented_sec,
                                  "persists_all_three_replay": not (prevented_er or prevented_ce or prevented_sec),
                                  "ce_security_disagree": ce_final[example_id]["predicted_label"] != sec_final[example_id]["predicted_label"]})
    csv_out("qualitative_error_candidates.csv", cases)
    summary = defaultdict(list)
    for row in cases:
        summary[(row["category"], row["arm"])].append(row)
    out = []
    for (category, arm), group in summary.items():
        out.append({"category": category, "arm": arm, "n_seed_example_pairs": len(group),
                    "er_prevented": sum(x["er_prevents_error"] for x in group),
                    "ce_prevented": sum(x["ce_prevents_error"] for x in group),
                    "security_prevented": sum(x["security_prevents_error"] for x in group),
                    "persists_all_three": sum(x["persists_all_three_replay"] for x in group),
                    "ce_security_disagree": sum(x["ce_security_disagree"] for x in group)})
    csv_out("qualitative_error_summary.csv", out)
    return cases, out


if __name__ == "__main__":
    a, b = external()
    c, d = qualitative()
    print(json.dumps({"external_seed_slice_rows": len(a), "external_aggregate_rows": len(b),
                      "qualitative_seed_example_pairs": len(c), "qualitative_summary_rows": len(d)}))
