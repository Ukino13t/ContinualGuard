"""Independently re-aggregate and validate saved H2-D1 artifacts."""
import json
from pathlib import Path
from continualguard.notinject_evaluation import verify_saved

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    validation = verify_saved(root)
    print(json.dumps(validation["checks"], indent=2))
