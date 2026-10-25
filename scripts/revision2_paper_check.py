"""Static reproducibility checks for the ACL/Overleaf paper package.

This is intentionally not advertised as a substitute for a TeX compile.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
PAPER = ROOT / "paper/acl"


def main() -> None:
    tex_path = PAPER / "continualguard_acl.tex"
    source = tex_path.read_text(encoding="utf-8")
    errors: list[str] = []
    inputs = re.findall(r"\\input\{([^}]+)\}", source)
    graphics = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source)
    references = re.findall(r"\\(?:citep|citet|cite)\{([^}]+)\}", source)
    bib_text = (PAPER / "references.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib_text))
    included = []
    for relative in inputs:
        path = PAPER / relative
        if not path.is_file():
            errors.append(f"missing input: {relative}")
        elif path.suffix == ".tex":
            included.append(path.read_text(encoding="utf-8"))
    for relative in graphics:
        if not (PAPER / relative).is_file():
            errors.append(f"missing figure: {relative}")
    for group in references:
        for key in group.split(","):
            if key.strip() not in bib_keys:
                errors.append(f"missing BibTeX key: {key.strip()}")
    complete = source + "\n" + "\n".join(included)
    for env in ("document", "table", "table*", "tabular", "figure", "figure*", "equation", "enumerate"):
        begins = len(re.findall(r"\\begin\{" + re.escape(env) + r"\}", complete))
        ends = len(re.findall(r"\\end\{" + re.escape(env) + r"\}", complete))
        if begins != ends:
            errors.append(f"environment imbalance: {env} {begins}/{ends}")
    for path in [tex_path, *(PAPER / p for p in inputs if (PAPER / p).is_file())]:
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\\bottomrule", text):
            earlier = text[: match.start()]
            if earlier.rfind(r"\begin{tabular}") < earlier.rfind(r"\end{tabular}"):
                errors.append(f"bottomrule outside tabular: {path.name}")
            lines = earlier.splitlines()
            previous = next((line.strip() for line in reversed(lines) if line.strip()
                             and not line.lstrip().startswith("%")), "")
            if previous and not previous.endswith(r"\\"):
                errors.append(f"row before bottomrule lacks LaTeX line break: {path.name}")
    output = {"status": "PASS" if not errors else "FAIL", "input_count": len(inputs),
              "figure_count": len(graphics), "citation_count": len(references),
              "bib_keys": len(bib_keys), "errors": errors,
              "limitation": "Static checks only; Overleaf pdfLaTeX/BibTeX compilation remains required."}
    destination = ROOT / "reports/revision2/paper_static_validation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
