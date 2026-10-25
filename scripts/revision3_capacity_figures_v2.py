"""Generate corrected capacity figures from validated A0-v2 analysis."""
from pathlib import Path

from scripts import revision3_capacity_figures as base

base.DATA = Path(r"E:\ContinualGuard\results\revision3\analysis_v2")

if __name__ == "__main__":
    base.main()
