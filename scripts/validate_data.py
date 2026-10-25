"""Thin wrapper; use the installed continualguard package."""
import sys
from continualguard.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "validate")
    main()
