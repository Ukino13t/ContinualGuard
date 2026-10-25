"""Run the remaining approved baselines serially on the single local GPU."""
import gc
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from continualguard.config import load_config
from continualguard.train import run, run_joint


def main():
    for regime, seeds in (("sequential", (42, 2026)), ("joint", (17, 42, 2026))):
        for seed in seeds:
            config = load_config(ROOT / "configs/real_pilot.yaml")
            config["seed"] = seed
            if regime == "sequential":
                output = run(config, ROOT, f"configs/real_pilot.yaml#seed={seed}")
            else:
                output = run_joint(config, ROOT, f"configs/real_pilot.yaml#joint-seed={seed}")
            print(output, flush=True)
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
