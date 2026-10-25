"""Build a source-only, anonymized Overleaf archive from frozen Revision 4 data."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import revision3_package as r3  # noqa: E402
from scripts import revision4_package as r4  # noqa: E402

PAPER = ROOT / "paper" / "acl"
OUT = ROOT / "paper" / "ContinualGuard_RouteA_Revision5_Overleaf.zip"
MANIFEST = ROOT / "reports" / "revision5" / "overleaf_package_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    validation = json.loads((ROOT / "results/revision4/validation.json").read_text(encoding="utf-8"))
    if not (validation.get("status") == "PASS"
            and validation.get("full_checkpoint_hashes")
            and validation.get("runs_checked") == 9):
        raise RuntimeError("Frozen Revision 4 validation gate failed")

    files = set(r3.source_files())
    for name in r4.FILES:
        source = ROOT / "results" / "revision4" / "analysis" / name
        target = PAPER / "data" / "revision4" / name
        if sha256(source) != sha256(target):
            raise RuntimeError(f"Curated table input differs from frozen result: {name}")
        files.add(target)
    for name, relative in r4.EXTRAS.items():
        source = ROOT / relative
        target = PAPER / "data" / "revision4" / name
        if sha256(source) != sha256(target):
            raise RuntimeError(f"Curated audit input differs from frozen result: {name}")
        files.add(target)
    files.add(PAPER / "build_revision4_tables.py")

    tex = (PAPER / "continualguard_acl.tex").read_text(encoding="utf-8")
    for stem in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex):
        figure = PAPER / (stem if Path(stem).suffix else stem + ".pdf")
        files.add(figure)
    for stem in re.findall(r"\\input\{([^}]+)\}", tex):
        files.add(PAPER / (stem if Path(stem).suffix else stem + ".tex"))

    members = sorted(files)
    for path in members:
        if not path.is_file() or not path.resolve().is_relative_to(PAPER.resolve()):
            raise RuntimeError(f"Missing or unsafe source: {path}")
        rel = path.relative_to(PAPER).as_posix()
        lowered = rel.lower()
        if any(term in lowered for term in ("raw", "checkpoint", "predictions", "reviewer", "_attempt", ".safetensors")):
            raise RuntimeError(f"Forbidden member: {rel}")
        if path.suffix in (".csv", ".jsonl"):
            header = path.read_text(encoding="utf-8").splitlines()[0].lower()
            if "raw_text_stored" in header:
                if path.suffix != ".csv":
                    raise RuntimeError(f"Unexpected raw-text flag: {rel}")
                with path.open(encoding="utf-8", newline="") as handle:
                    if any(row["raw_text_stored"].strip().lower() != "false"
                           for row in csv.DictReader(handle)):
                        raise RuntimeError(f"Raw text present: {rel}")
                header = header.replace("raw_text_stored", "")
            if any(term in header for term in ("prompt_text", "reviewer_note", "raw_text")):
                raise RuntimeError(f"Forbidden source column: {rel}")

    temporary = OUT.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in members:
            archive.write(path, path.relative_to(PAPER).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(set(archive.namelist())):
            raise RuntimeError("ZIP integrity or duplicate-member failure")
        if not {"main.tex", "continualguard_acl.tex", "acl.sty", "references.bib"}.issubset(archive.namelist()):
            raise RuntimeError("Core Overleaf sources missing")
    temporary.replace(OUT)

    result = {
        "status": "PASS",
        "package": str(OUT),
        "sha256": sha256(OUT),
        "bytes": OUT.stat().st_size,
        "members": len(members),
        "pdf_sha256": sha256(PAPER / "continualguard_acl.pdf"),
        "source_allowlist": [path.relative_to(PAPER).as_posix() for path in members],
        "contents": "Anonymous ACL source, vector figures, aggregate/hash/ID-only inputs; no raw data or checkpoints",
    }
    MANIFEST.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "sha256", "bytes", "members")}))


if __name__ == "__main__":
    main()
