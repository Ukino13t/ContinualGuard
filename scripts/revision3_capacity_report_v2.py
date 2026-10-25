"""Generate a corrected capacity report while retaining the v1 report."""
from pathlib import Path

from scripts import revision3_capacity_report as base

base.DATA = Path(r"E:\ContinualGuard\results\revision3\analysis_v2")
base.OUT = Path(r"E:\ContinualGuard\reports\revision3\04_memory_capacity_v2.md")

if __name__ == "__main__":
    base.main()
    text = base.OUT.read_text(encoding="utf-8")
    text = text.replace(
        "22 new Revision 3 runs and 14 compatible frozen/reused cells",
        "18 corrected A0 v2 runs and 18 compatible frozen A1 cells",
    )
    text = text.replace(
        "Status: complete,",
        "Status: corrected v2, complete,",
        1,
    )
    base.OUT.write_text(text, encoding="utf-8")
