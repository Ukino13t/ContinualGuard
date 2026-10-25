"""Vector composition figure from frozen Route A memory CSV; no prompt text."""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, r"E:\ContinualGuard\scripts")
from plot_route_a import pdf_text, pdf_write

ROOT = Path(r"E:\ContinualGuard")
SOURCE = ROOT / "results/extended/analysis/memory_composition.csv"
OUT = ROOT / "figures/extended"
NAMES = {
    "a0_reservoir": "A0 Reservoir",
    "a1_class": "A1 Class",
    "a2_class_stage": "A2 Class+Stage",
    "a3_coverage": "A3 +Coverage",
    "a4_ce": "A4 +CE",
    "a5_security": "A5 Security",
}
STAGES = ("AF2", "AF3", "AF4", "AF5")
COLORS = ((0.12, 0.47, 0.71), (0.27, 0.70, 0.64),
          (0.95, 0.65, 0.20), (0.72, 0.39, 0.61))


def main() -> None:
    rows = [r for r in csv.DictReader(SOURCE.open(encoding="utf-8", newline=""))
            if r["arm"] == "hard" and r["stage"] == "AF5" and r["method"] in NAMES]
    by = defaultdict(list)
    for row in rows:
        by[row["method"]].append(row)
    assert all(len(by[method]) == 3 for method in NAMES)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="520" viewBox="0 0 900 520">',
           '<rect width="900" height="520" fill="white"/>',
           '<text x="45" y="35" font-family="Arial" font-size="20" font-weight="bold">B-HARD memory composition after AF5</text>',
           '<text x="250" y="74" font-family="Arial" font-size="13">Stage memberships (200 slots)</text>',
           '<text x="672" y="74" font-family="Arial" font-size="13">Benign slots (of 200)</text>']
    pdf = ["1 1 1 rg 0 0 900 520 re f\n0 0 0 rg ",
           pdf_text(45, 35, "B-HARD memory composition after AF5", 18),
           pdf_text(250, 74, "Stage memberships (200 slots)", 12),
           pdf_text(672, 74, "Benign slots (of 200)", 12)]
    for index, method in enumerate(NAMES):
        y = 105 + index * 54
        sample = by[method]
        stage_counts = [statistics.mean(json.loads(r["by_stage"]).get(stage, 0)
                                        for r in sample) for stage in STAGES]
        benign = statistics.mean(json.loads(r["by_label"]).get("0", 0) for r in sample)
        assert abs(sum(stage_counts) - 200) < 1e-8
        svg.append(f'<text x="45" y="{y+18}" font-family="Arial" font-size="13">{NAMES[method]}</text>')
        pdf.append(pdf_text(45, y+18, NAMES[method], 12))
        x = 250.0
        for stage, count, (red, green, blue) in zip(STAGES, stage_counts, COLORS):
            width = 360 * count / 200
            color = f'#{round(255*red):02x}{round(255*green):02x}{round(255*blue):02x}'
            svg.append(f'<rect x="{x:.2f}" y="{y}" width="{width:.2f}" height="25" fill="{color}"/>')
            pdf.append(f"{red:.3f} {green:.3f} {blue:.3f} rg {x:.2f} {520-y-25:.2f} {width:.2f} 25 re f\n")
            if width >= 43:
                svg.append(f'<text x="{x+width/2:.2f}" y="{y+17}" text-anchor="middle" font-family="Arial" font-size="11" fill="white">{count:.0f}</text>')
                pdf.append("1 1 1 rg " + pdf_text(x+width/2-7, y+17, f"{count:.0f}", 10))
            x += width
        bx = 672.0
        bw = 144 * benign / 200
        svg.append(f'<rect x="{bx}" y="{y}" width="144" height="25" fill="#eeeeee"/>')
        svg.append(f'<rect x="{bx}" y="{y}" width="{bw:.2f}" height="25" fill="#25816b"/>')
        svg.append(f'<text x="824" y="{y+18}" font-family="Arial" font-size="12">{benign:.1f}</text>')
        pdf.append(f"0.93 0.93 0.93 rg {bx} {520-y-25} 144 25 re f\n")
        pdf.append(f"0.145 0.506 0.420 rg {bx} {520-y-25} {bw:.2f} 25 re f\n0 0 0 rg ")
        pdf.append(pdf_text(824, y+18, f"{benign:.1f}", 11))
    for i, (stage, (red, green, blue)) in enumerate(zip(STAGES, COLORS)):
        x = 250 + i * 100
        color = f'#{round(255*red):02x}{round(255*green):02x}{round(255*blue):02x}'
        svg.append(f'<rect x="{x}" y="474" width="16" height="12" fill="{color}"/>')
        svg.append(f'<text x="{x+22}" y="485" font-family="Arial" font-size="11">{stage}</text>')
        pdf.append(f"{red:.3f} {green:.3f} {blue:.3f} rg {x} {520-486} 16 12 re f\n0 0 0 rg ")
        pdf.append(pdf_text(x+22, 485, stage, 10))
    svg.append('<text x="45" y="505" font-family="Arial" font-size="11">Means of seeds 17, 42, 2026; all observed coverage keys are unique, so coverage preference is nonbinding.</text>')
    pdf.append(pdf_text(45, 505, "Means of seeds 17, 42, 2026; coverage preference is nonbinding.", 10))
    svg.append('</svg>')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "memory_composition.svg").write_text("\n".join(svg), encoding="utf-8")
    pdf_write(OUT / "memory_composition.pdf", "".join(pdf))
    print(OUT / "memory_composition.pdf")


if __name__ == "__main__":
    main()
