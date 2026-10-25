"""Run the approved frozen H2-D1 checkpoint evaluation; no training path exists."""
from pathlib import Path
from continualguard.notinject_evaluation import run

if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])
