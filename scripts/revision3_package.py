"""Build source-only ACL Overleaf ZIP from validated aggregate artifacts."""
from __future__ import annotations

import hashlib
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from scripts import revision2_package as r2

ROOT = Path(r"E:\ContinualGuard")
PAPER = ROOT / "paper/acl"
ANALYSIS = ROOT / "results/revision3/analysis_v2"
DEST = PAPER / "data/revision3"
OUT = ROOT / "paper/ContinualGuard_RouteA_Revision3_Overleaf.zip"

ANALYSIS_FILES = (
    "per_seed.csv", "aggregate.csv", "paired_per_seed.csv", "paired_aggregate.csv",
    "stage_curves.csv", "stage_acquisition.csv", "memory_composition.csv",
    "validation_M200.json", "capacity_per_seed.csv", "capacity_aggregate.csv",
    "capacity_paired_per_seed.csv", "capacity_paired_aggregate.csv",
    "capacity_stage_curves.csv", "capacity_memory_overlap.csv", "capacity_validation.json",
)
EXTRAS = {
    "selector_summary.json": "results/revision3/selector_audit_v2/summary.json",
    "selector_cells.csv": "results/revision3/selector_audit_v2/selector_cells.csv",
    "selected_ids.jsonl": "results/revision3/selector_audit_v2/selected_ids.jsonl",
    "matched_protocol_v1.json": "results/revision3/protocol_v1.json",
    "matched_protocol_v2.json": "results/revision3/protocol_v2.json",
    "capacity_protocol_v1.json": "results/revision3/capacity_protocol_v1.json",
    "capacity_protocol_v2.json": "results/revision3/capacity_protocol_v2.json",
    "order_audit_v1.json": "results/revision3/order_audit_v1_doc_seed2026_s1_s2.json",
    "order_audit_v2_smoke.json": "results/revision3/order_audit_v2_smoke_doc_seed2026_s1_s2.json",
    "selector_parity_v2.json": "results/revision3/selector_parity_v2.json",
    "a1_a2_order_audit.json": "results/revision3/a1_a2_order_audit.json",
    "confirmation_source_audit.json": "reports/revision3/confirmation_source_audit.json",
}


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require_complete():
    r2.require_complete()
    p0 = json.loads((ANALYSIS / "validation_M200.json").read_text(encoding="utf-8"))
    p1 = json.loads((ANALYSIS / "capacity_validation.json").read_text(encoding="utf-8"))
    assert (p0["status"], p0["runs_checked"], p0["full_checkpoint_hashes"]) == ("PASS", 9, True)
    assert (p1["status"], p1["runs_checked"], p1["full_checkpoint_hashes"]) == ("PASS", 36, True)
    assert len(p1["matched_pairs"]) == 18
    assert all(row["v1_membership_parity_stages"] == 4 for row in p0["runs"])
    assert p1["a0_protocol"] == "results/revision3/protocol_v2.json"
    qa = json.loads((ROOT / "reports/revision3/visual_qa/page_audit.json").read_text(encoding="utf-8"))
    assert qa["all_word_boxes_inside_page"] and not qa["blank_text_pages"]
    assert not qa["undefined_reference_or_citation_warnings"] and qa["latex_error_count"] == 0
    assert sha(ROOT / qa["pdf"]) == qa["pdf_sha256"]


def copy_curated():
    r2.copy_curated()
    DEST.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "scripts/revision3_paper_tables.py",
                 PAPER / "build_revision3_tables.py")
    for name in ANALYSIS_FILES:
        shutil.copy2(ANALYSIS / name, DEST / name)
    for name, relative in EXTRAS.items():
        if name == "selected_ids.jsonl":
            with (ROOT / relative).open(encoding="utf-8") as handle:
                for line in handle:
                    if set(json.loads(line)) != {"arm", "seed", "stage", "variant", "selected_ids"}:
                        raise RuntimeError("Selected-ID audit contains unexpected fields")
        shutil.copy2(ROOT / relative, DEST / name)
    for name in ("matched_retention_and_fpr.pdf", "matched_memory_and_overlap.pdf",
                 "capacity_attack_retention.pdf", "capacity_benign_fpr.pdf"):
        source = ROOT / "figures/revision3" / name
        target = PAPER / "figures" / ("revision3_" + name)
        shutil.copy2(source, target)


def source_files():
    files = r2.source_files()
    source = (PAPER / "continualguard_acl.tex").read_text(encoding="utf-8")
    referenced_figures = {
        name + ".pdf" if not Path(name).suffix else name
        for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source)
    }
    files = [file for file in files if file.parent != PAPER / "figures"
             or file.relative_to(PAPER).as_posix() in referenced_figures]
    files.append(PAPER / "build_revision3_tables.py")
    files.append(PAPER / "build_pdf_local.ps1")
    files.extend(DEST / name for name in ANALYSIS_FILES)
    files.extend(DEST / name for name in EXTRAS)
    files.append(PAPER / "lineno.sty")
    for file in files:
        if not file.is_file() or not file.resolve().is_relative_to(PAPER.resolve()):
            raise RuntimeError(f"Missing or unsafe package member: {file}")
    return sorted(set(files))


def main():
    require_complete()
    copy_curated()
    table_names = ("revision3_class_balance.tex", "revision3_class_balance_seeds.tex",
                   "revision3_capacity.tex", "revision3_capacity_pairs.tex",
                   "revision3_resources.tex")
    before = {name: sha(PAPER / "tables" / name) for name in table_names}
    subprocess.run([sys.executable, str(PAPER / "build_revision3_tables.py")],
                   cwd=PAPER, check=True)
    after = {name: sha(PAPER / "tables" / name) for name in table_names}
    if before != after:
        raise RuntimeError("Portable packaged CSVs do not regenerate identical Revision 3 tables")
    from scripts.revision3_paper_check import main as static_check
    static_check()
    files = source_files()
    forbidden = ("raw", "checkpoint", "predictions", "reviewer", "_attempt", ".safetensors")
    for file in files:
        rel = file.relative_to(PAPER).as_posix()
        if any(term in rel.lower() for term in forbidden):
            raise RuntimeError(f"Disallowed source package path: {rel}")
        if file.suffix in (".jsonl", ".csv"):
            header = file.read_text(encoding="utf-8").splitlines()[0].lower()
            if "raw_text_stored" in header:
                if file.suffix != ".csv":
                    raise RuntimeError(f"Unexpected raw-text flag format: {rel}")
                with file.open(encoding="utf-8", newline="") as handle:
                    if any(row["raw_text_stored"].strip().lower() != "false"
                           for row in csv.DictReader(handle)):
                        raise RuntimeError(f"Raw text may be stored: {rel}")
                header = header.replace("raw_text_stored", "")
            if any(token in header for token in ("prompt_text", "raw_text", "reviewer_note")):
                raise RuntimeError(f"Disallowed source package column: {rel}")
    temporary = OUT.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for file in files:
            archive.write(file, file.relative_to(PAPER).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist()))
    os.replace(temporary, OUT)
    manifest = {"package": str(OUT), "sha256": sha(OUT), "bytes": OUT.stat().st_size,
                "members": len(files), "pdf_sha256": sha(PAPER / "continualguard_acl.pdf"),
                "source_allowlist": [p.relative_to(PAPER).as_posix() for p in files],
                "contents": "ACL sources, figures, aggregate CSV/JSON, IDs and hashes only"}
    (ROOT / "reports/revision3/overleaf_package_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("package", "sha256", "bytes", "members")}))


if __name__ == "__main__":
    main()
