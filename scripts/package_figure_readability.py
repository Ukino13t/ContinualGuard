"""Package the figure-readability paper as an independent Overleaf project."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "acl_figure_readability"
BASE = ROOT / "paper" / "ContinualGuard_RouteA_Revision5_Overleaf.zip"
BASE_MANIFEST = ROOT / "reports" / "revision5" / "overleaf_package_manifest.json"
OUT = ROOT / "paper" / "ContinualGuard_Figure_Readability_Overleaf.zip"
MANIFEST = ROOT / "reports" / "figure_readability_20261024" / "overleaf_package_manifest.json"
EXPECTED_CHANGED = {
    "continualguard_acl.tex", "main.tex", "README.md", "build_pdf_local.ps1",
    "figures/b_short_af3_retention.pdf",
    "figures/b_hard_benign_retention.pdf",
    "figures/malicious_benign_tradeoff.pdf",
    "figures/revision4_stage_memory_composition.pdf",
    "figures/revision4_selected_buffer_overlap.pdf",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    old = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    assert old["status"] == "PASS" and old["members"] == 145
    names = old["source_allowlist"]
    assert len(names) == len(set(names)) == 145
    assert (SOURCE / "main.tex").read_bytes() == (SOURCE / "continualguard_acl.tex").read_bytes(), (
        "main.tex must be a complete source copy so review line numbers render consistently")
    with zipfile.ZipFile(BASE) as base:
        assert base.testzip() is None and set(base.namelist()) == set(names)
        changed = {name for name in names if (SOURCE / name).read_bytes() != base.read(name)}
    assert changed == EXPECTED_CHANGED, (sorted(changed - EXPECTED_CHANGED),
                                         sorted(EXPECTED_CHANGED - changed))
    for name in names:
        path = SOURCE / name
        assert path.is_file() and path.resolve().is_relative_to(SOURCE.resolve())
        assert not any(part in ("..", "") for part in Path(name).parts)
    temp = OUT.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for name in names:
            archive.write(SOURCE / name, name)
    with zipfile.ZipFile(temp) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == names
        assert {"main.tex", "continualguard_acl.tex", "acl.sty", "references.bib"}.issubset(names)
    temp.replace(OUT)
    result = {
        "status": "PASS", "zip": str(OUT), "zip_sha256": digest(OUT.read_bytes()),
        "bytes": OUT.stat().st_size, "members": len(names),
        "changed_from_revision5": sorted(changed),
        "paper_pdf_sha256": digest((SOURCE / "continualguard_acl.pdf").read_bytes()),
        "provenance": "Same 145-member source-only allowlist as Revision 5; five vector PDFs, both synchronized LaTeX entrypoints, README and local build script changed",
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
