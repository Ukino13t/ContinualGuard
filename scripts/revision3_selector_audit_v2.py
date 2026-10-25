"""Export prompt-free corrected A0-v2/A1 selected-ID and overlap evidence."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
ZROOT = Path(r"Z:\ContinualGuard_Revision3")
NEW = ZROOT / "results/revision3/matched_ablation_amp256_v2/M200/a0_matched"
A1 = ROOT / "results/revision2/matched_ablation_amp256/M200/a1_matched"
OLD = ZROOT / "results/revision3/matched_ablation_amp256/M200/a0_matched"
OUT = ROOT / "results/revision3/selector_audit_v2"
STAGES = ("AF2", "AF3", "AF4", "AF5")


def ids(path: Path) -> list[str]:
    return [json.loads(line)["example_id"] for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    validation = json.loads((ROOT / "results/revision3/analysis_v2/validation_M200.json")
                            .read_text(encoding="utf-8"))
    assert validation["status"] == "PASS" and validation["runs_checked"] == 9
    assert all(r["v1_membership_parity_stages"] == 4 for r in validation["runs"])
    OUT.mkdir(parents=True, exist_ok=True)
    cells, selected = [], []
    for arm in ("doc", "short", "hard"):
        for seed in (17, 42, 2026):
            for index, stage in enumerate(STAGES, 1):
                left = ids(NEW / arm / f"seed{seed}/stage{index}/buffer.jsonl")
                right = ids(A1 / arm / f"seed{seed}/stage{index}/buffer.jsonl")
                historical = ids(OLD / arm / f"seed{seed}/stage{index}/buffer.jsonl")
                assert len(left) == len(right) == len(historical) == 200
                assert set(left) == set(historical)
                intersection = len(set(left) & set(right))
                cells.append({"arm": arm, "seed": seed, "stage": stage,
                              "a0_v2_selected": len(left), "a1_selected": len(right),
                              "intersection": intersection,
                              "actual_selected_jaccard": intersection / (400 - intersection),
                              "selected_sets_differ": set(left) != set(right),
                              "a0_v2_membership_equal_to_v1": True})
                for variant, chosen in (("a0_matched_v2", left), ("a1_matched", right)):
                    selected.append({"arm": arm, "seed": seed, "stage": stage,
                                     "variant": variant, "selected_ids": chosen})
    with (OUT / "selector_cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    with (OUT / "selected_ids.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {"status": "PASS", "protocol": "corrected A0-v2 versus frozen A1",
               "cells": len(cells), "cells_with_actual_selection_change": sum(r["selected_sets_differ"] for r in cells),
               "mean_actual_selected_jaccard": sum(r["actual_selected_jaccard"] for r in cells) / len(cells),
               "selected_id_rows": len(selected), "a0_v2_v1_membership_parity_cells": len(cells),
               "source_validation_sha256": sha(ROOT / "results/revision3/analysis_v2/validation_M200.json"),
               "ordered_ids_reconstructed_from_actual_buffer_files": True,
               "raw_text_stored": False}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
