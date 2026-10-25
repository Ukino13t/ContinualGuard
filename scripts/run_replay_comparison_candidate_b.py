"""CLI for the frozen Candidate B replay comparison."""
from __future__ import annotations

import argparse
from pathlib import Path

from continualguard.replay_candidate_b import METHODS, SEEDS, run_preflight, run_replay
from continualguard.stronger_pilot_no_replay import ARMS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"E:\ContinualGuard"))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.preflight:
        run_preflight(args.root)
    elif args.all:
        for method in METHODS:
            for seed in SEEDS:
                for arm in ARMS:
                    run_replay(args.root, method, arm, seed)
    elif args.method and args.arm and args.seed:
        run_replay(args.root, args.method, args.arm, args.seed)
    else:
        parser.error("select --preflight, --all, or --method/--arm/--seed")


if __name__ == "__main__":
    main()
