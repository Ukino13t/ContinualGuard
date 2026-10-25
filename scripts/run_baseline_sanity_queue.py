"""Run the approved controls serially on the single local GPU."""
import gc
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from continualguard.config import load_config
from continualguard.controls import run_control


def main():
    schedule = [
        *(('s2_only', seed) for seed in (17, 42, 2026)),
        *(('sequential_s2_epochs', seed) for seed in (17, 42, 2026)),
        ('reverse_order', 42),
        ('joint_epochs', 42),
    ]
    for control, seed in schedule:
        config = load_config(ROOT / "configs/real_pilot.yaml")
        config["seed"] = seed
        output = run_control(config, ROOT, control, f"configs/real_pilot.yaml#{control}-seed={seed}")
        print(output, flush=True)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
