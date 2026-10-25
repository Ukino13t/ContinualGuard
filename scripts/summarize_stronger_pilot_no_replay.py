"""Validate and aggregate the completed Candidate B no-replay experiment."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from continualguard.stronger_pilot_summary import summarize


if __name__ == "__main__":
    result = summarize(ROOT)
    print(json.dumps({
        "status": result["aggregate"]["status"],
        "readiness": result["readiness"]["classification"],
        "runs": result["validation"]["runs"],
        "checkpoints": result["validation"]["checkpoints"],
    }, ensure_ascii=False))
