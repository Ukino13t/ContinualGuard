"""Run one approved AF1-AF4 secondary-order no-replay seed or validate its gates."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from continualguard.config import load_config
from continualguard.family_baseline import SECONDARY_ORDER, run_family_baseline, validate_contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(17, 42, 2026), default=42)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(ROOT / "configs/attack_family_order_control.yaml")
    config["seed"] = args.seed
    if args.validate_only:
        data = validate_contract(config, ROOT, SECONDARY_ORDER)
        print({"status": "passed", "records": len(data.rows), "stage_order": list(SECONDARY_ORDER), "AF5_included": data.audit["AF5_included"]})
        return
    print(run_family_baseline(
        config, ROOT, f"configs/attack_family_order_control.yaml#seed={args.seed}",
        experiment_regime="attack_family_secondary_order_no_replay",
        run_prefix="attack-family-order-control",
    ))


if __name__ == "__main__":
    main()
