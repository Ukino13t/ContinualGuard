"""Run only the predeclared missing A0/A1 capacity cells, sequentially."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
ZROOT = Path(r"Z:\ContinualGuard_Revision3")
BASE = ZROOT / "results/revision3/matched_ablation_amp256"
PROTOCOL = ROOT / "results/revision3/capacity_protocol_v1.json"
LOG = ROOT / "results/revision3/capacity_queue.jsonl"


def log(**fields):
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), **fields}) + "\n")


def main() -> None:
    assert Path(sys.executable).resolve() == (ROOT / ".venv/Scripts/python.exe").resolve()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == "FROZEN_BEFORE_CAPACITY_TRAINING"
    assert len(protocol["planned_new_cells"]) == 22
    LOG.parent.mkdir(parents=True, exist_ok=True)
    # Seed-by-seed paired order keeps the A0/A1 comparison easy to audit.
    for arm in ("short", "hard"):
        for capacity in (50, 400):
            for seed in (17, 42, 2026):
                for method in ("a0_matched", "a1_matched"):
                    key = f"{method}/{arm}/seed{seed}/M{capacity}"
                    if key not in protocol["planned_new_cells"]:
                        continue
                    run = BASE / f"M{capacity}" / method / arm / f"seed{seed}"
                    meta = run / "run_metadata.json"
                    if meta.is_file() and json.loads(meta.read_text(encoding="utf-8")).get("status") == "COMPLETE":
                        log(event="reuse_complete", key=key)
                        continue
                    e_free = shutil.disk_usage(ROOT).free
                    z_free = shutil.disk_usage(ZROOT).free
                    if e_free < 5_000_000_000 or z_free < 15_000_000_000:
                        raise RuntimeError(f"Free-space safety stop: E={e_free}; Z={z_free}")
                    log(event="start", key=key, e_free=e_free, z_free=z_free)
                    if method == "a0_matched":
                        cmd = [sys.executable, "-m", "scripts.revision3_train", "--root", str(ROOT),
                               "--variant", method, "--arm", arm, "--seed", str(seed),
                               "--capacity", str(capacity)]
                    else:
                        cmd = [sys.executable, "-m", "scripts.revision3_capacity_a1", "--arm", arm,
                               "--seed", str(seed), "--capacity", str(capacity)]
                    result = subprocess.run(cmd, cwd=ROOT, check=False)
                    log(event="complete" if result.returncode == 0 else "failed",
                        key=key, exit_code=result.returncode)
                    if result.returncode:
                        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
