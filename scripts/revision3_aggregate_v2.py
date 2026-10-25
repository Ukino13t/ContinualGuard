"""Aggregate corrected A0 v2 against frozen A1 without overwriting v1."""
from __future__ import annotations

import csv
from pathlib import Path

from scripts import revision3_aggregate as base

base.OUT = Path(r"E:\ContinualGuard\results\revision3\analysis_v2")
base.REPORT = Path(r"E:\ContinualGuard\reports\revision3\02_class_balance_ablation_v2.md")
base.SELECTOR_SUMMARY = Path(r"E:\ContinualGuard\results\revision3\selector_audit_v2\summary.json")
base.METHODS["A0_matched"] = Path(
    r"Z:\ContinualGuard_Revision3\results\revision3\matched_ablation_amp256_v2\M200\a0_matched"
)

if __name__ == "__main__":
    base.main()
    report = base.REPORT.read_text(encoding="utf-8")
    report = report.split("## Interpretation", 1)[0]
    report = report.replace(
        "Status: complete, full-hash validated; Candidate B only.",
        "Status: corrected A0-v2, nine trajectories full-hash validated; Candidate B only.",
    ).replace(" 卤 ", " +/- ").replace("results/revision3/selector_audit/", "results/revision3/selector_audit_v2/")
    report = report.replace("\u5364", "+/-")
    report = report.replace("Selected IDs, buffer Jaccard, rank medians, and quota-affected counts",
                            "Selected IDs, buffer Jaccard, and quota-affected counts")
    with (base.OUT / "paired_aggregate.csv").open(encoding="utf-8", newline="") as handle:
        pairs = list(csv.DictReader(handle))
    lines = [report.rstrip(), "", "## Interpretation", "",
             "A0-v2 uses the same global bottom-M selected-ID membership as A0-v1 but stores selected items in frozen A1's canonical `(label, rank, ID)` order. Thus A0-v2/A1 share the replay-index protocol. The old A0-v1 outcomes are excluded: in a same-ID-set case they differed at all 400 replay positions. The corrected two-stage smoke check restored exact replay and model-weight parity. Full v2 validation confirms selected-ID membership parity with v1 and hashes every stage-end checkpoint.", "",
             "Paired A1 minus A0-v2 effects below are percentage points, mean +/- sample SD over seeds 17, 42, and 2026. Positive TPR/macro-F1 BWT and negative benign-FPR increase favor A1. Signs and magnitudes should be read beside the 1.667-pp stage-test class-conditional resolution; no significance or population claim is implied.", ""]
    for row in pairs:
        arm = row["arm"]
        tpr, fpr = (f"a1_minus_a0_{name}" for name in ("tpr_bwt_pp", "benign_fpr_increase_pp"))
        lines.append(
            f"- B-{arm.upper()}: TPR BWT {float(row[tpr + '_mean']):+.2f} +/- "
            f"{float(row[tpr + '_sample_sd']):.2f} pp "
            f"({row[tpr + '_positive_seeds']} positive, {row[tpr + '_negative_seeds']} negative seeds); "
            f"FPR increase {float(row[fpr + '_mean']):+.2f} +/- "
            f"{float(row[fpr + '_sample_sd']):.2f} pp "
            f"({row[fpr + '_positive_seeds']} positive, {row[fpr + '_negative_seeds']} negative seeds)."
        )
    lines += ["", "The original online Reservoir is a contextual method with a different selection and numerical policy. The shared-rank A1/A2 pair also changes buffer ordering and does not isolate stage quotas. Stage/evaluation cells, acquisition diagnostics, position and benign-group slices, and memory composition remain in sibling CSVs. The prompt-free ordering correction is in `reports/revision3/05_selector_order_correction.md`.", ""]
    base.REPORT.write_text("\n".join(lines), encoding="utf-8")
