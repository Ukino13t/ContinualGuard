"""Aggregate the corrected A0-v2 versus frozen A1 capacity grid."""
from __future__ import annotations

from pathlib import Path

from scripts import revision3_capacity_aggregate as base
from scripts.revision3_capacity_validate_v2 import source

base.OUT = Path(r"E:\ContinualGuard\results\revision3\analysis_v2")
base.source = source

if __name__ == "__main__":
    base.main()
