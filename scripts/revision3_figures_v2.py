"""Generate corrected primary figures from validated A0-v2 analysis."""
from pathlib import Path

from scripts import revision3_figures as base

base.DATA = Path(r"E:\ContinualGuard\results\revision3\analysis_v2")
base.SELECTOR = Path(r"E:\ContinualGuard\results\revision3\selector_audit_v2\selector_cells.csv")

if __name__ == "__main__":
    base.main()
