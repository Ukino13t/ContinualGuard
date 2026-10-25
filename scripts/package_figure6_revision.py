"""Package Figure-6-only source revision as a separate Overleaf ZIP."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "acl_figure6_readability"
BASE = ROOT / "paper" / "ContinualGuard_Figure_Readability_Overleaf.zip"
BASE_MANIFEST = ROOT / "reports" / "figure_readability_20261024" / "overleaf_package_manifest.json"
OUT = ROOT / "paper" / "ContinualGuard_Figure6_Readability_Overleaf.zip"
MANIFEST = ROOT / "reports" / "figure6_readability_20261024" / "overleaf_package_manifest.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    old = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    assert old["status"] == "PASS" and old["members"] == 145
    with zipfile.ZipFile(BASE) as archive:
        names = archive.namelist()
        assert archive.testzip() is None and len(names) == len(set(names)) == 145
        changed = {name for name in names if (SOURCE / name).read_bytes() != archive.read(name)}
    assert changed == {"figures/external_overdefense.pdf"}, sorted(changed)
    for name in names:
        path = SOURCE / name
        assert path.is_file() and path.resolve().is_relative_to(SOURCE.resolve())
    temporary = OUT.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for name in names:
            archive.write(SOURCE / name, name)
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None and archive.namelist() == names
        assert archive.read("main.tex") == archive.read("continualguard_acl.tex")
    temporary.replace(OUT)
    manifest = {"status": "PASS", "zip": str(OUT), "zip_sha256": sha(OUT),
                "bytes": OUT.stat().st_size, "members": len(names),
                "changed_from_previous_project": sorted(changed),
                "paper_pdf_sha256": sha(SOURCE / "continualguard_acl.pdf"),
                "contents": "Same vetted source-only 145-member allowlist; only Figure 6 vector PDF changed"}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
