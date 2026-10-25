"""Run Candidate B no-replay technical preflight or fixed sequential baselines."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from continualguard.stronger_pilot_no_replay import ARMS, SEEDS, run_baseline, run_preflight


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--all", action="store_true")
    group.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.preflight:
        print(run_preflight(ROOT))
        return
    if args.all:
        if args.seed is not None:
            parser.error("--seed cannot be combined with --all")
        run_preflight(ROOT)
        for seed in SEEDS:
            for arm in ARMS:
                print(run_baseline(ROOT, arm, seed))
        return
    if args.seed is None:
        parser.error("--arm requires --seed")
    print(run_baseline(ROOT, args.arm, args.seed))


if __name__ == "__main__":
    main()
