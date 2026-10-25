"""Run missing frozen A0 M=200 cells sequentially, preserving completed work."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
OUTPUT = Path(r"Z:\ContinualGuard_Revision3\results\revision3\matched_ablation_amp256\M200\a0_matched")
LOG = ROOT / "results/revision3/p0_queue.jsonl"


def record(**fields):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), **fields}) + "\n")


def main() -> None:
    assert Path(sys.executable).resolve() == (ROOT / ".venv/Scripts/python.exe").resolve()
    assert (ROOT / "results/revision3/protocol_v1.json").is_file()
    assert (ROOT / "data/processed/stronger_pilot_candidate_b/materialization_manifest.json").is_file()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for arm in ("doc", "short", "hard"):
        for seed in (17, 42, 2026):
            metadata = OUTPUT / arm / f"seed{seed}/run_metadata.json"
            if metadata.is_file() and json.loads(metadata.read_text(encoding="utf-8")).get("status") == "COMPLETE":
                record(event="reuse_complete", arm=arm, seed=seed)
                continue
            e_free, z_free = shutil.disk_usage(ROOT).free, shutil.disk_usage(OUTPUT).free
            if e_free < 5_000_000_000 or z_free < 15_000_000_000:
                raise RuntimeError(f"Safety stop: E free={e_free}, Z free={z_free}")
            record(event="start", arm=arm, seed=seed, e_free=e_free, z_free=z_free)
            command = [sys.executable, "-m", "scripts.revision3_train", "--root", str(ROOT),
                       "--variant", "a0_matched", "--arm", arm, "--seed", str(seed),
                       "--capacity", "200"]
            result = subprocess.run(command, cwd=ROOT, check=False)
            record(event="complete" if result.returncode == 0 else "failed",
                   arm=arm, seed=seed, exit_code=result.returncode)
            if result.returncode:
                raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
