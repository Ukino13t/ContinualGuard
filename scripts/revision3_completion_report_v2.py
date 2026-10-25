"""Render the corrected Revision 3 evidence and delivery report from validated outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
DATA = ROOT / "results/revision3/analysis_v2"
OUT = ROOT / "reports/revision3/RESEARCH_COMPLETION_REPORT_V3.md"


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def rows(name: str) -> list[dict]:
    with (DATA / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ms(row: dict, key: str, signed: bool = True) -> str:
    mean = float(row[key + "_mean"])
    return f"{mean:+.2f} +/- {float(row[key + '_sample_sd']):.2f}" if signed else (
        f"{mean:.2f} +/- {float(row[key + '_sample_sd']):.2f}")


def main() -> None:
    p0, p1 = load("validation_M200.json"), load("capacity_validation.json")
    assert p0["status"] == p1["status"] == "PASS"
    assert p0["runs_checked"] == 9 and p0["full_checkpoint_hashes"]
    assert p1["runs_checked"] == 36 and p1["full_checkpoint_hashes"]
    assert all(row["v1_membership_parity_stages"] == 4 for row in p0["runs"])
    assert len(p1["matched_pairs"]) == 18
    paired, capacity, capacity_pair = (rows(name) for name in
        ("paired_aggregate.csv", "capacity_aggregate.csv", "capacity_paired_aggregate.csv"))
    package = json.loads((ROOT / "reports/revision3/overleaf_package_manifest.json")
                         .read_text(encoding="utf-8"))
    qa = json.loads((ROOT / "reports/revision3/visual_qa/page_audit.json")
                    .read_text(encoding="utf-8"))
    assert qa["all_word_boxes_inside_page"] and not qa["blank_text_pages"]
    lines = [
        "# ContinualGuard Route A Revision 3 — corrected research completion report", "",
        "Status: experiments and final analyses complete; scientific interpretation and manuscript await human review. The active project root is `E:\\ContinualGuard`; large new checkpoints remain under `Z:\\ContinualGuard_Revision3`. The archived former project copy was not accessed or modified.", "",
        "## Scientific correction", "",
        "A0 v1 shared selected-ID ranks with A1 but stored its buffer in global-rank order rather than A1's label-then-rank order. Index-based replay therefore changed sampled identities independently of quotas. In the B-DOC seed-2026 negative control, the selected sets were identical after AF2, yet all 400 AF3 replay positions differed. A0 v1 results are preserved but excluded from class-quota inference. A0 v2 retains global bottom-M membership and stores selected items in A1's canonical `(label, rank, ID)` order. The same-set two-stage smoke test had 0/400 replay mismatches and identical AF3 model weights. Full v2 validation confirmed A0-v1/v2 membership parity in all 36 M=200 stage boundaries.", "",
        "A1/A2 share ranks but also differ in buffer ordering: after AF3, their nine pairs shared 184–195 selected IDs yet differed at 178–190 positions of common-ID relative order and at 391–399 of 400 subsequent replay positions. Their outcome difference is descriptive of the implemented selectors, not an isolated stage-quota effect. The A2/A3 coverage rule remains nonbinding on the actual candidate pools.", "",
        "## Corrected M=200 class-quota comparison", "",
        "Nine A0-v2 trajectories and nine frozen A1 comparators cover B-DOC, B-SHORT, and B-HARD with seeds 17, 42, and 2026. All A0-v2 checkpoints, raw predictions, training-access logs, current orders, and canonical buffers passed full-hash validation. Signed values below are A1 minus A0-v2, mean +/- sample SD in percentage points; positive BWT and negative FPR increase favor A1.", "",
        "| Arm | Attack-TPR BWT difference | Macro-F1 BWT difference | Benign-FPR increase difference |",
        "|---|---:|---:|---:|",
    ]
    for r in paired:
        lines.append(f"| B-{r['arm'].upper()} | {ms(r, 'a1_minus_a0_tpr_bwt_pp')} | "
                     f"{ms(r, 'a1_minus_a0_macro_f1_bwt_pp')} | "
                     f"{ms(r, 'a1_minus_a0_benign_fpr_increase_pp')} |")
    lines += ["", "All nine seed-paired effects, every stage-test cell, acquisition loss/recall/FPR, position slices, and memory composition are in `results/revision3/analysis_v2/`. Candidate B stage-test class rates resolve in 1.667-pp steps; 20-item position slices resolve in 5-pp steps. These three seeds support descriptive direction and stability checks, not significance or population inference.", "",
              "## B-SHORT/B-HARD memory-capacity sensitivity", "",
              "The corrected grid has 36 validated method-arm-capacity-seed trajectories and 18 matched A0-v2/A1 pairs at M=50/200/400. A1 uses its frozen scientific runner; original online Reservoir is not a grid cell. Attack retention and benign FPR remain separate axes.", "",
              "| Arm | M | Method | Attack-TPR BWT (pp) | Benign-FPR increase (pp) | Acquisition TPR (%) |",
              "|---|---:|---|---:|---:|---:|"]
    for arm in ("short", "hard"):
        for memory in (50, 200, 400):
            for method in ("a0_matched", "a1_matched"):
                r = next(row for row in capacity if row["arm"] == arm and
                         int(row["capacity"]) == memory and row["method"] == method)
                lines.append(f"| B-{arm.upper()} | {memory} | {'A0-v2' if method.startswith('a0') else 'A1'} | "
                             f"{ms(r, 'tpr_bwt_pp')} | {ms(r, 'benign_fpr_increase_pp')} | "
                             f"{ms(r, 'acquisition_tpr_pct', False)} |")
    lines += ["", "Paired A1-minus-A0-v2 capacity contrasts (pp):", "",
              "| Arm | M | Attack-TPR BWT difference | Benign-FPR increase difference |",
              "|---|---:|---:|---:|"]
    for r in capacity_pair:
        lines.append(f"| B-{r['arm'].upper()} | {r['capacity']} | "
                     f"{ms(r, 'a1_minus_a0_tpr_bwt_pp')} | "
                     f"{ms(r, 'a1_minus_a0_benign_fpr_increase_pp')} |")
    lines += ["", "Per-seed capacity values, full curves, selected-buffer overlap, runtime, and peak GPU allocation are in `results/revision3/analysis_v2/` and `reports/revision3/04_memory_capacity_v2.md`. Do not infer a monotonic memory frontier from three capacities and one replay exposure ratio.", "",
              "## Independent confirmation and limits", "",
              "No defensible independent confirmation stream passed the existing-source audit. Candidate B overlaps previously explored family/H2-mini sources, including 1,600 unique TRAIN text hashes and 480 unique TEST text hashes. No confirmation training was run. `reports/revision3/03_independent_confirmation.md` records the official-source availability review and an executable, fail-closed future lineage gate. Candidate B remains one constructed detector/stream evaluation; no cross-detector, live-RAG, adaptive-attack, or deployment claim follows.", "",
              "## Delivered artifacts and validation", "",
              f"The ACL review PDF has {qa['page_count']} rendered pages, no blank text page, no out-of-page word boxes, no undefined-reference/citation warning, and no LaTeX error. PDF SHA-256: `{package['pdf_sha256']}`.", "",
              f"Overleaf ZIP: `{package['package']}`; SHA-256 `{package['sha256']}`; {package['members']} allowlisted members. It contains source, vector figures, aggregate CSV/JSON, hashes, and IDs only. It excludes raw third-party content, reviewer free text, predictions, and model checkpoints.", "",
              "The remaining decision is human scientific review of the corrected class-quota and capacity effects and the final ACL manuscript. Independent confirmation requires a separately audited source before any training.", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"PASS: {OUT}")


if __name__ == "__main__":
    main()
