"""Bundle the ACL manuscript and aggregate-only artifacts for Overleaf."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

SRC = Path(r"E:\ContinualGuard\paper\acl")
OUT = Path(r"E:\ContinualGuard\paper\ContinualGuard_RouteA_Overleaf.zip")
ROOT_FILES = ("main.tex", "continualguard_acl.tex", "acl.sty", "acl_natbib.bst",
              "references.bib", "README.md", "build_tables.py")

files = [SRC / name for name in ROOT_FILES]
files += sorted((SRC / "figures").glob("*.pdf"))
files += sorted((SRC / "tables").glob("*.tex"))
files += sorted((SRC / "data").glob("*.csv"))
files += sorted((SRC / "data/route_a").glob("*.csv"))
files += sorted((SRC / "data/route_a").glob("*.json"))
assert len(files) == len(set(files))
assert all(file.is_file() for file in files)
assert (SRC / "figures/memory_composition.pdf") in files
assert (SRC / "tables/route_a_ablation.tex") in files

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for file in files:
        archive.write(file, file.relative_to(SRC).as_posix())

with zipfile.ZipFile(OUT) as archive:
    bad = archive.testzip()
    assert bad is None, bad
    names = set(archive.namelist())
    assert all(name in names for name in ROOT_FILES)
    assert all(not name.startswith("../") and not name.startswith("data/raw/") for name in names)
    assert "data/route_a/per_run.csv" in names
    assert "data/route_a/coverage_key_audit.json" in names

digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
print(json.dumps({"zip": str(OUT), "bytes": OUT.stat().st_size,
                  "sha256": digest, "files": len(files)}))
