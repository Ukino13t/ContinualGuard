"""Render every ACL page and record objective page-boundary checks.

Run from the project root with the project Python environment.  All generated
artifacts stay under reports/revision3/visual_qa.
"""
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw


ROOT = Path(r"E:\ContinualGuard")
PDF = ROOT / "paper/acl/continualguard_acl.pdf"
OUT = ROOT / "reports/revision3/visual_qa"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(PDF)
    tile_width = 340
    tiles = []
    rows = []
    for index, page in enumerate(document, 1):
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.4, 1.4), alpha=False)
        output = OUT / f"page_{index:02d}.png"
        pixmap.save(output)
        words = page.get_text("words")
        outside = [w[:4] for w in words if w[0] < -0.5 or w[1] < -0.5
                   or w[2] > page.rect.width + 0.5 or w[3] > page.rect.height + 0.5]
        image = Image.open(output).convert("RGB")
        thumb_height = round(image.height * tile_width / image.width)
        thumb = image.resize((tile_width, thumb_height), Image.Resampling.LANCZOS)
        tiles.append(thumb)
        text = page.get_text()
        rows.append({"page": index, "word_count": len(words),
                     "out_of_page_word_boxes": len(outside),
                     "width_pt": page.rect.width, "height_pt": page.rect.height,
                     "table_20_caption": "Revision 2 complete-trajectory runtime" in text,
                     "table_21_caption": "Machine-readable appendix" in text,
                     "render": str(output.relative_to(ROOT))})
    columns = 4
    padding = 12
    tile_height = max(tile.height for tile in tiles)
    height = ((len(tiles) + columns - 1) // columns) * (tile_height + 32 + padding) + padding
    sheet = Image.new("RGB", (columns * (tile_width + padding) + padding, height), "#d8d8d8")
    draw = ImageDraw.Draw(sheet)
    for index, tile in enumerate(tiles):
        x = padding + (index % columns) * (tile_width + padding)
        y = padding + (index // columns) * (tile_height + 32 + padding)
        sheet.paste(tile, (x, y + 24))
        draw.text((x + 4, y + 4), f"Page {index + 1}", fill="black")
    sheet.save(OUT / "contact_sheet.png")
    log = (ROOT / "paper/acl/continualguard_acl.log").read_text(encoding="utf-8", errors="replace")
    report = {"pdf": str(PDF.relative_to(ROOT)),
              "pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
              "page_count": len(document),
              "pages": rows, "all_word_boxes_inside_page": all(not r["out_of_page_word_boxes"] for r in rows),
              "blank_text_pages": [r["page"] for r in rows if r["word_count"] < 15],
              "undefined_reference_or_citation_warnings": bool(re.search(
                  r"undefined references|Citation .* undefined|Reference .* undefined", log, re.I)),
              "latex_error_count": len(re.findall(r"^! ", log, re.M)),
              "overfull_hbox_count": len(re.findall(r"Overfull \\hbox", log))}
    (OUT / "page_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"page_count": len(document),
                      "table20_pages": [r["page"] for r in rows if r["table_20_caption"]],
                      "table21_pages": [r["page"] for r in rows if r["table_21_caption"]],
                      "all_word_boxes_inside_page": report["all_word_boxes_inside_page"],
                      "blank_text_pages": report["blank_text_pages"],
                      "undefined_warnings": report["undefined_reference_or_citation_warnings"],
                      "latex_errors": report["latex_error_count"],
                      "overfull_hboxes": report["overfull_hbox_count"]}))


if __name__ == "__main__":
    main()
