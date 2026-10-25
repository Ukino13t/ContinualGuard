"""Audit the revised paper PDF and render every page for visual review."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "acl_figure_readability"
PDF = PAPER / "continualguard_acl.pdf"
OUT = ROOT / "reports" / "figure_readability_20261024" / "pdf_pages"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(PDF)
    rows = []
    for number, page in enumerate(doc, 1):
        words = page.get_text("words")
        outside = [word[:5] for word in words if word[0] < -.5 or word[1] < -.5
                   or word[2] > page.rect.width + .5 or word[3] > page.rect.height + .5]
        page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).save(
            OUT / f"page-{number:02}.png")
        text = page.get_text()
        rows.append({"page": number, "words": len(words), "outside": outside,
                     "conclusion": "Conclusion" in text,
                     "limitations": "Limitations" in text,
                     "figure1": "Figure 1:" in text,
                     "figure2": "Figure 2:" in text,
                     "figure3": "Figure 3:" in text,
                     "figure5": "Figure 5:" in text})
    for start in range(0, len(rows), 3):
        panels = []
        for row in rows[start:start + 3]:
            im = Image.open(OUT / f"page-{row['page']:02}.png").convert("RGB")
            im.thumbnail((560, 790))
            panel = Image.new("RGB", (575, 820), "white")
            panel.paste(im, ((575 - im.width) // 2, 22))
            ImageDraw.Draw(panel).text((10, 3), f"page {row['page']}", fill="black")
            panels.append(panel)
        contact = Image.new("RGB", (575 * len(panels), 820), "#ddd")
        for index, panel in enumerate(panels):
            contact.paste(panel, (575 * index, 0))
        contact.save(OUT / f"contact-{start // 3 + 1:02}.png")
    log = (PAPER / "continualguard_acl.log").read_text(encoding="utf-8", errors="replace")
    report = {
        "pdf": str(PDF), "sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
        "pages": len(doc), "figure_pages": {
            f"figure{n}": [r["page"] for r in rows if r[f"figure{n}"]]
            for n in (1, 2, 3, 5)},
        "conclusion_pages": [r["page"] for r in rows if r["conclusion"]],
        "limitations_pages": [r["page"] for r in rows if r["limitations"]],
        "blank_text_pages": [r["page"] for r in rows if r["words"] < 15],
        "outside_word_boxes": sum(len(r["outside"]) for r in rows),
        "latex_errors": len(re.findall(r"^! ", log, re.M)),
        "unresolved_citations_or_references": bool(re.search(
            r"undefined references|Citation .* undefined|Reference .* undefined", log, re.I)),
        "overfull_boxes": len(re.findall(r"Overfull \\[hv]box", log)),
        "rows": rows,
    }
    report["status"] = "PASS" if (
        14 <= report["pages"] <= 17 and
        all(len(report["figure_pages"][f"figure{n}"]) == 1 for n in (1, 2, 3, 5)) and
        report["conclusion_pages"] == [8] and
        not report["blank_text_pages"] and report["outside_word_boxes"] == 0 and
        report["latex_errors"] == 0 and not report["unresolved_citations_or_references"] and
        report["overfull_boxes"] == 0) else "FAIL"
    (OUT.parent / "pdf_audit.json").write_text(json.dumps(report, indent=2) + "\n",
                                                 encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "status", "pages", "figure_pages", "conclusion_pages", "limitations_pages",
        "blank_text_pages", "outside_word_boxes", "latex_errors",
        "unresolved_citations_or_references", "overfull_boxes")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
