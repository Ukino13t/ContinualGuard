"""Run one approved pooled joint/offline-reference seed."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from continualguard.config import load_config
from continualguard.train import run_joint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=(17, 42, 2026))
    args = parser.parse_args()
    config = load_config(ROOT / "configs/real_pilot.yaml")
    config["seed"] = args.seed
    print(run_joint(config, ROOT, f"configs/real_pilot.yaml#joint-seed={args.seed}"))


if __name__ == "__main__":
    main()
