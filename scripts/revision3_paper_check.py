"""Static ACL source and result-table consistency check for Revision 3."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
PAPER = ROOT / "paper/acl"
ANALYSIS = ROOT / "results/revision3/analysis_v2"
OUT = ROOT / "reports/revision3/paper_static_validation.json"


def main() -> None:
    source = (PAPER / "continualguard_acl.tex").read_text(encoding="utf-8")
    inputs = re.findall(r"\\input\{([^}]+)\}", source)
    graphics = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source)
    citations = re.findall(r"\\(?:citep|citet|cite)\{([^}]+)\}", source)
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
    errors = []
    included = []
    for name in inputs:
        path = PAPER / name
        if not path.is_file():
            errors.append(f"Missing input: {name}")
        else:
            included.append(path.read_text(encoding="utf-8"))
    for name in graphics:
        if not (PAPER / name).is_file():
            errors.append(f"Missing figure: {name}")
    for name in re.findall(r"\\path\{([^}]+)\}", source):
        if name.startswith("data/") and not (PAPER / name).is_file():
            errors.append(f"Missing mapped artifact: {name}")
    for group in citations:
        for key in group.split(","):
            if key.strip() not in bib_keys:
                errors.append(f"Missing citation: {key.strip()}")
    complete = source + "\n" + "\n".join(included)
    for env in ("document", "table", "table*", "tabular", "figure", "figure*", "equation"):
        starts = len(re.findall(r"\\begin\{" + re.escape(env) + r"\}", complete))
        ends = len(re.findall(r"\\end\{" + re.escape(env) + r"\}", complete))
        if starts != ends:
            errors.append(f"Unbalanced {env}: {starts}/{ends}")
    for path in [PAPER / "continualguard_acl.tex", *(PAPER / p for p in inputs if (PAPER / p).is_file())]:
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\\bottomrule", text):
            earlier = text[:match.start()]
            if earlier.rfind(r"\begin{tabular}") < earlier.rfind(r"\end{tabular}"):
                errors.append(f"Bottomrule outside tabular: {path.name}")
            previous = next((line.strip() for line in reversed(earlier.splitlines())
                             if line.strip() and not line.lstrip().startswith("%")), "")
            if previous and not previous.endswith(r"\\"):
                errors.append(f"Missing row break before bottomrule: {path.name}")
    with (ANALYSIS / "paired_aggregate.csv").open(encoding="utf-8", newline="") as handle:
        paired = list(csv.DictReader(handle))
    matched = (PAPER / "tables/revision3_class_balance.tex").read_text(encoding="utf-8")
    for row in paired:
        for key in ("tpr_bwt_pp", "macro_f1_bwt_pp", "benign_fpr_increase_pp"):
            token = f"{float(row['a1_minus_a0_' + key + '_mean']):+.2f}\\pm{float(row['a1_minus_a0_' + key + '_sample_sd']):.2f}"
            if token not in matched:
                errors.append(f"Matched table differs from CSV: {row['arm']}/{key}")
    capacity_table = PAPER / "tables/revision3_capacity.tex"
    if capacity_table.is_file():
        with (ANALYSIS / "capacity_aggregate.csv").open(encoding="utf-8", newline="") as handle:
            capacity = list(csv.DictReader(handle))
        tex = capacity_table.read_text(encoding="utf-8")
        for row in capacity:
            for key in ("tpr_bwt_pp", "benign_fpr_increase_pp"):
                token = f"{float(row[key + '_mean']):+.2f}\\pm{float(row[key + '_sample_sd']):.2f}"
                if token not in tex:
                    errors.append(f"Capacity table differs from CSV: {row['arm']}/{row['capacity']}/{row['method']}/{key}")
    if "no independent" not in source.lower() and "not executed" not in source.lower():
        errors.append("Independent confirmation status is missing")
    report = {"status": "PASS" if not errors else "FAIL", "input_count": len(inputs),
              "figure_count": len(graphics), "citation_count": len(citations),
              "bib_keys": len(bib_keys), "errors": errors,
              "source": "Revision 3 validated CSVs and LaTeX source"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
