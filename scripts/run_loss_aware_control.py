from __future__ import annotations

import argparse
from pathlib import Path

from continualguard.loss_aware_control import run_control, run_preflight
from continualguard.stronger_pilot_no_replay import ARMS, SEEDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"E:\ContinualGuard"))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.preflight:
        run_preflight(args.root)
    elif args.all:
        for seed in SEEDS:
            for arm in ARMS:
                run_control(args.root, arm, seed)
    elif args.arm and args.seed:
        run_control(args.root, args.arm, args.seed)
    else:
        parser.error("select --preflight, --all, or --arm/--seed")


if __name__ == "__main__":
    main()
