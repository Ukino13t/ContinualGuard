"""Render corrected frozen Route A plots into versioned Revision 2 assets."""
from pathlib import Path

import plot_route_a

plot_route_a.OUT = Path(r"E:\ContinualGuard\figures\revision2")

if __name__ == "__main__":
    plot_route_a.main()
