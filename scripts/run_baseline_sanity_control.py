"""Run one approved baseline sanity/optimization-control trajectory."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from continualguard.config import load_config
from continualguard.controls import run_control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("control", choices=("s2_only", "sequential_s2_epochs", "reverse_order", "joint_epochs"))
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config = load_config(ROOT / "configs/real_pilot.yaml")
    config["seed"] = args.seed
    output = run_control(config, ROOT, args.control, f"configs/real_pilot.yaml#{args.control}-seed={args.seed}")
    print(output)


if __name__ == "__main__":
    main()
