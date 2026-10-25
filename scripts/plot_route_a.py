"""Dependency-free vector plots from Route A CSV results (SVG and PDF)."""
from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
DATA = ROOT / "results/extended/analysis/per_stage_curves.csv"
OUT = ROOT / "figures/extended"
COLORS = {
    "m0_no_replay": (0.08, 0.08, 0.08),
    "a0_reservoir": (0.0, 0.45, 0.70),
    "a1_class": (0.90, 0.62, 0.0),
    "a2_class_stage": (0.0, 0.62, 0.45),
    "a3_coverage": (0.80, 0.47, 0.65),
    "a4_ce": (0.84, 0.37, 0.0),
    "a5_security": (0.34, 0.71, 0.91),
}
LABELS = {"m0_no_replay": "No Replay", "a0_reservoir": "A0 Reservoir",
          "a1_class": "A1 Class", "a2_class_stage": "A2 Class+Stage",
          "a3_coverage": "A3 +Coverage", "a4_ce": "A4 +CE", "a5_security": "A5 Security"}
ORDER = tuple(COLORS)
W, H = 900, 520
LEFT, RIGHT, TOP, BOTTOM = 85, 680, 64, 420


def escape(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def pdf_write(path: Path, operations: str):
    stream = operations.encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
    ]
    chunks = [b"%PDF-1.4\n"]
    offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(sum(len(c) for c in chunks))
        chunks.append(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = sum(len(c) for c in chunks)
    chunks.append(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
    chunks.extend(f"{x:010d} 00000 n \n".encode() for x in offsets[1:])
    chunks.append(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(b"".join(chunks))


def pdf_text(x, y, txt, size=12):
    safe = str(txt).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return f"BT /F1 {size} Tf {x:.2f} {H-y:.2f} Td ({safe}) Tj ET\n"


def plot(rows, arm, task, metric, title, filename):
    subset = [r for r in rows if r["arm"] == arm and r["eval_stage"] == task]
    stages = ["AF2", "AF3", "AF4", "AF5"]
    xs = {stage: LEFT + i*(RIGHT-LEFT)/3 for i, stage in enumerate(stages)}
    # FPR is a percentage and individual frozen No Replay seeds reach 100%.
    # A 70% ceiling clipped those observations and understated the failure.
    y_max = 100
    y_min = 0
    def yy(v): return BOTTOM - (v-y_min)*(BOTTOM-TOP)/(y_max-y_min)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{LEFT}" y="32" font-family="Arial" font-size="20" font-weight="bold">{escape(title)}</text>']
    pdf = ["1 1 1 rg 0 0 900 520 re f\n0 0 0 rg ", pdf_text(LEFT, 32, title, 18)]
    for tick in range(0, 101, 20):
        y = yy(tick)
        svg.append(f'<line x1="{LEFT}" y1="{y:.2f}" x2="{RIGHT}" y2="{y:.2f}" stroke="#e0e0e0"/>')
        svg.append(f'<text x="{LEFT-15}" y="{y+4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{tick}</text>')
        pdf.append(f"0.88 0.88 0.88 RG {LEFT} {H-y:.2f} m {RIGHT} {H-y:.2f} l S\n0 0 0 rg ")
        pdf.append(pdf_text(LEFT-45, y+4, tick, 11))
    for stage in stages:
        x = xs[stage]
        svg.append(f'<text x="{x:.2f}" y="{BOTTOM+30}" text-anchor="middle" font-family="Arial" font-size="13">{stage}</text>')
        pdf.append(pdf_text(x-13, BOTTOM+30, stage, 11))
    svg.append(f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{BOTTOM}" stroke="black" stroke-width="1.5"/>')
    svg.append(f'<line x1="{LEFT}" y1="{BOTTOM}" x2="{RIGHT}" y2="{BOTTOM}" stroke="black" stroke-width="1.5"/>')
    pdf.append(f"0 0 0 RG 1.5 w {LEFT} {H-TOP} m {LEFT} {H-BOTTOM} l {RIGHT} {H-BOTTOM} l S\n")
    svg.append(f'<text x="{LEFT}" y="{H-24}" font-family="Arial" font-size="12">Bold: mean; whiskers: sample SD; pale lines: individual seeds. Frozen tests.</text>')
    pdf.append(pdf_text(LEFT, H-24, "Bold: mean; whiskers: sample SD; pale lines: individual seeds. Frozen tests.", 10))
    for idx, method in enumerate(ORDER):
        group = [r for r in subset if r["method"] == method]
        if not group:
            continue
        vals = defaultdict(list)
        for r in group:
            vals[r["trained_stage"]].append(float(r[metric]))
        series = [(xs[s], yy(statistics.mean(vals[s])), statistics.stdev(vals[s]) if len(vals[s]) > 1 else 0,
                   statistics.mean(vals[s])) for s in stages if s in vals]
        r,g,b = COLORS[method]
        color = f"#{round(r*255):02x}{round(g*255):02x}{round(b*255):02x}"
        # Show each seed so a 100% FPR trajectory cannot disappear under a mean.
        for seed in sorted({row["seed"] for row in group}):
            seed_rows = [row for row in group if row["seed"] == seed]
            seed_points = sorted(((xs[row["trained_stage"]], yy(float(row[metric])))
                                  for row in seed_rows), key=lambda value: value[0])
            if not seed_points:
                continue
            svg.append('<polyline points="' + " ".join(f"{x:.2f},{y:.2f}" for x, y in seed_points)
                       + f'" fill="none" stroke="{color}" stroke-width="1" opacity="0.25"/>')
            pale = tuple(0.75 + 0.25 * value for value in (r, g, b))
            pdf.append(f"{pale[0]:.4f} {pale[1]:.4f} {pale[2]:.4f} RG 0.75 w ")
            pdf.append(f"{seed_points[0][0]:.2f} {H-seed_points[0][1]:.2f} m ")
            pdf.extend(f"{x:.2f} {H-y:.2f} l " for x, y in seed_points[1:])
            pdf.append("S\n")
        points = " ".join(f"{x:.2f},{y:.2f}" for x,y,_,_ in series)
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        pdf.append(f"{r} {g} {b} RG 2.5 w ")
        if series:
            pdf.append(f"{series[0][0]:.2f} {H-series[0][1]:.2f} m ")
            pdf.extend(f"{x:.2f} {H-y:.2f} l " for x,y,_,_ in series[1:])
            pdf.append("S\n")
        for x,y,sd,mean in series:
            low, high = yy(min(y_max, mean+sd)), yy(max(y_min, mean-sd))
            svg.append(f'<line x1="{x:.2f}" y1="{low:.2f}" x2="{x:.2f}" y2="{high:.2f}" stroke="{color}" stroke-width="1"/>')
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
            pdf.append(f"1 w {x:.2f} {H-low:.2f} m {x:.2f} {H-high:.2f} l S\n")
            pdf.append(f"{r} {g} {b} rg {x-3:.2f} {H-y-3:.2f} 6 6 re f\n")
        legend_y = 78 + idx*28
        svg.append(f'<line x1="{RIGHT+22}" y1="{legend_y}" x2="{RIGHT+50}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{RIGHT+56}" y="{legend_y+4}" font-family="Arial" font-size="12">{escape(LABELS[method])}</text>')
        pdf.append(f"{r} {g} {b} RG 3 w {RIGHT+22} {H-legend_y} m {RIGHT+50} {H-legend_y} l S\n0 0 0 rg ")
        pdf.append(pdf_text(RIGHT+56, legend_y+4, LABELS[method], 10))
    svg.append("</svg>")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{filename}.svg").write_text("\n".join(svg), encoding="utf-8")
    pdf_write(OUT / f"{filename}.pdf", "".join(pdf))


def main():
    with DATA.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    plot(rows, "short", "AF3", "attack_tpr_pct",
         "B-SHORT: AF3 attack recall after later stages", "b_short_af3_ablation")
    plot(rows, "hard", "AF2", "benign_fpr_pct",
         "B-HARD: AF2 benign false-positive drift", "b_hard_af2_ablation")
    print(OUT)


if __name__ == "__main__":
    main()
