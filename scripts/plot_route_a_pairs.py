"""Paired Route A contrast plot from seed-matched aggregate CSVs."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, r"E:\ContinualGuard\scripts")
from plot_route_a import pdf_text, pdf_write

ROOT = Path(r"E:\ContinualGuard")
SOURCE = ROOT / "results/extended/analysis/paired_aggregate.csv"
OUT = ROOT / "figures/extended"
PAIRS = (("a0_reservoir", "a1_class"), ("a1_class", "a2_class_stage"),
         ("a2_class_stage", "a3_coverage"), ("a3_coverage", "a4_ce"),
         ("a4_ce", "a5_security"))


def main() -> None:
    rows = list(csv.DictReader(SOURCE.open(encoding="utf-8", newline="")))
    by = {(r["before"], r["after"], r["arm"]): r for r in rows}
    assert all((a, b, arm) in by for a, b in PAIRS for arm in ("short", "hard"))
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="520" viewBox="0 0 900 520">',
           '<rect width="900" height="520" fill="white"/>',
           '<text x="35" y="34" font-family="Arial" font-size="20" font-weight="bold">Adjacent selector contrasts (three matched seeds)</text>',
           '<text x="245" y="77" font-family="Arial" font-size="13">B-SHORT attack-TPR BWT change</text>',
           '<text x="587" y="77" font-family="Arial" font-size="13">B-HARD benign-FPR change</text>']
    pdf = ["1 1 1 rg 0 0 900 520 re f\n0 0 0 rg ",
           pdf_text(35, 34, "Adjacent selector contrasts (three matched seeds)", 18),
           pdf_text(245, 77, "B-SHORT attack-TPR BWT change", 12),
           pdf_text(587, 77, "B-HARD benign-FPR change", 12)]
    for left, right in ((245, 520), (587, 862)):
        zero = (left + right) / 2
        svg.append(f'<line x1="{zero:.1f}" y1="102" x2="{zero:.1f}" y2="436" stroke="#aaaaaa" stroke-dasharray="4,4"/>')
        pdf.append(f"0.66 0.66 0.66 RG [4 4] 0 d {zero:.2f} {520-102} m {zero:.2f} {520-436} l S [] 0 d\n")
        for tick in (-6, -3, 0, 3, 6):
            x = left + (tick + 6) / 12 * (right - left)
            svg.append(f'<text x="{x:.2f}" y="455" text-anchor="middle" font-family="Arial" font-size="11">{tick:+d}</text>')
            pdf.append("0 0 0 rg " + pdf_text(x-9, 455, f"{tick:+d}", 10))
    for i, (before, after) in enumerate(PAIRS):
        y = 120 + i * 67
        label = f"A{i} to A{i+1}"
        svg.append(f'<text x="35" y="{y+5}" font-family="Arial" font-size="13">{label}</text>')
        pdf.append("0 0 0 rg " + pdf_text(35, y+5, label, 12))
        for arm, metric, left, right, color in (
                ("short", "delta_tpr_bwt_pp", 245, 520, (0.05, 0.42, 0.67)),
                ("hard", "delta_benign_fpr_increase_pp", 587, 862, (0.77, 0.34, 0.12))):
            row = by[(before, after, arm)]
            mean = float(row[f"{metric}_mean"])
            sd = float(row[f"{metric}_sample_sd"])
            assert abs(mean) + sd <= 6.1
            xx = lambda value: left + (value + 6) / 12 * (right - left)
            lo, mid, hi = xx(mean - sd), xx(mean), xx(mean + sd)
            rgb = f'#{round(255*color[0]):02x}{round(255*color[1]):02x}{round(255*color[2]):02x}'
            svg.append(f'<line x1="{lo:.2f}" y1="{y}" x2="{hi:.2f}" y2="{y}" stroke="{rgb}" stroke-width="3"/>')
            svg.append(f'<line x1="{lo:.2f}" y1="{y-7}" x2="{lo:.2f}" y2="{y+7}" stroke="{rgb}"/>')
            svg.append(f'<line x1="{hi:.2f}" y1="{y-7}" x2="{hi:.2f}" y2="{y+7}" stroke="{rgb}"/>')
            svg.append(f'<circle cx="{mid:.2f}" cy="{y}" r="5" fill="{rgb}"/>')
            r, g, b = color
            pdf.append(f"{r} {g} {b} RG 3 w {lo:.2f} {520-y} m {hi:.2f} {520-y} l S\n")
            pdf.append(f"1 w {lo:.2f} {520-y+7} m {lo:.2f} {520-y-7} l S\n")
            pdf.append(f"{hi:.2f} {520-y+7} m {hi:.2f} {520-y-7} l S\n")
            pdf.append(f"{r} {g} {b} rg {mid-4:.2f} {520-y-4} 8 8 re f\n")
    svg.append('<text x="35" y="493" font-family="Arial" font-size="11">After minus before, percentage points; whiskers are sample SD, not confidence intervals. A2/A3 coverage is nonbinding.</text>')
    pdf.append("0 0 0 rg " + pdf_text(35, 493, "After minus before, pp; whiskers = sample SD, not CI. A2/A3 coverage is nonbinding.", 10))
    svg.append('</svg>')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "paired_selector_contrasts.svg").write_text("\n".join(svg), encoding="utf-8")
    pdf_write(OUT / "paired_selector_contrasts.pdf", "".join(pdf))
    print(OUT / "paired_selector_contrasts.pdf")


if __name__ == "__main__":
    main()
