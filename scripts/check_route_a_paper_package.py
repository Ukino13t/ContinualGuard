"""Static ACL package integrity check when local TeX packages are unavailable."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard\paper\acl")
ZIP = ROOT.parent / "ContinualGuard_RouteA_Overleaf.zip"
MAIN = ROOT / "continualguard_acl.tex"
source = MAIN.read_text(encoding="utf-8")
bib = (ROOT / "references.bib").read_text(encoding="utf-8")

assert (ROOT / "main.tex").read_text(encoding="utf-8").strip() == r"\input{continualguard_acl.tex}"
assert r"\usepackage[review]{acl}" in source
assert r"\input{tables/route_a_ablation.tex}" in source
assert r"\input{tables/route_a_per_seed.tex}" in source
assert "nonbinding" in source

keys = set(re.findall(r"@\w+\s*\{\s*([^,]+),", bib))
cited = {key.strip() for group in re.findall(r"\\cite[pt]?\{([^}]+)\}", source)
         for key in group.split(",")}
assert cited <= keys, sorted(cited - keys)

required = {"main.tex", "continualguard_acl.tex", "acl.sty", "acl_natbib.bst",
            "references.bib", "README.md"}
for expression in (r"\\includegraphics(?:\[[^]]+\])?\{([^}]+)\}",
                   r"\\input\{([^}]+)\}"):
    for target in re.findall(expression, source):
        assert (ROOT / target).exists(), target
        required.add(target)

for file in [MAIN, ROOT / "tables/route_a_ablation.tex",
             ROOT / "tables/route_a_per_seed.tex"]:
    stack: list[str] = []
    for command, env in re.findall(r"\\(begin|end)\{([^}]+)\}", file.read_text(encoding="utf-8")):
        if command == "begin":
            stack.append(env)
        else:
            assert stack and stack.pop() == env, (file, command, env)
    assert not stack, (file, stack)
    if file.name.startswith("route_a_"):
        fragment = file.read_text(encoding="utf-8")
        assert fragment.index(r"\begin{tabular}") < fragment.index(r"\bottomrule")
        assert fragment.index(r"\bottomrule") < fragment.index(r"\end{tabular}")

with zipfile.ZipFile(ZIP) as archive:
    assert archive.testzip() is None
    names = set(archive.namelist())
    assert required <= names, sorted(required - names)
    assert all(not name.startswith(("data/raw/", "results/", "history/")) for name in names)
    assert archive.read("continualguard_acl.tex") == MAIN.read_bytes()
print(f"PASS: {len(cited)} citations, {len(required)} required assets, "
      f"{len(names)} ZIP members, balanced manuscript/table environments")
