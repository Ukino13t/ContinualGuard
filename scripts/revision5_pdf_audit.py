"""Objective PDF QA for the anonymous Revision 5 manuscript."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "acl" / "continualguard_acl.pdf"
LOG = PDF.with_suffix(".log")
OUT = ROOT / "reports" / "revision5" / "visual_qa" / "page_audit.json"


def main() -> None:
    doc = pymupdf.open(PDF)
    pages = []
    for index, page in enumerate(doc, 1):
        words = page.get_text("words")
        out = [word[:4] for word in words
               if word[0] < -0.5 or word[1] < -0.5
               or word[2] > page.rect.width + 0.5
               or word[3] > page.rect.height + 0.5]
        text = page.get_text()
        pages.append({
            "page": index,
            "words": len(words),
            "out_of_page_word_boxes": len(out),
            "has_conclusion": bool(re.search(r"\bConclusion\b", text)),
            "has_limitations": bool(re.search(r"\bLimitations\b", text)),
            "has_references": bool(re.search(r"\bReferences\b", text)),
            "has_appendix_a": "Human Review and Taxonomy" in text,
        })
    log = LOG.read_text(encoding="utf-8", errors="replace")
    report = {
        "status": "PASS",
        "pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
        "page_count": len(doc),
        "pages": pages,
        "all_word_boxes_inside_page": all(row["out_of_page_word_boxes"] == 0 for row in pages),
        "blank_text_pages": [row["page"] for row in pages if row["words"] < 15],
        "latex_error_count": len(re.findall(r"^! ", log, re.M)),
        "undefined_reference_or_citation_warnings": bool(re.search(
            r"undefined references|Citation .* undefined|Reference .* undefined", log, re.I)),
        "overfull_hbox_count": len(re.findall(r"Overfull \\hbox", log)),
        "overfull_vbox_count": len(re.findall(r"Overfull \\vbox", log)),
    }
    if not (len(doc) == 15 and report["all_word_boxes_inside_page"]
            and not report["blank_text_pages"] and report["latex_error_count"] == 0
            and not report["undefined_reference_or_citation_warnings"]
            and report["overfull_hbox_count"] == 0 and report["overfull_vbox_count"] == 0):
        report["status"] = "FAIL"
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "page_count", "all_word_boxes_inside_page", "blank_text_pages",
        "latex_error_count", "undefined_reference_or_citation_warnings",
        "overfull_hbox_count", "overfull_vbox_count"
    )}))


if __name__ == "__main__":
    main()
