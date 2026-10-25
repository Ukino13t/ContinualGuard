"""Sequentially run frozen corrected A0 and missing frozen A1 capacity cells."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
ZROOT = Path(r"Z:\ContinualGuard_Revision3")
NEW = ZROOT / "results/revision3/matched_ablation_amp256_v2"
OLD = ZROOT / "results/revision3/matched_ablation_amp256"
PROTOCOL = ROOT / "results/revision3/protocol_v2.json"
LOG = ROOT / "results/revision3/v2_queue.jsonl"


def log(**fields: object) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), **fields}) + "\n")


def complete(path: Path) -> bool:
    meta = path / "run_metadata.json"
    return meta.is_file() and json.loads(meta.read_text(encoding="utf-8")).get("status") == "COMPLETE"


def run_one(method: str, arm: str, seed: int, capacity: int) -> None:
    base = NEW if method == "a0_matched" else OLD
    output = base / f"M{capacity}" / method / arm / f"seed{seed}"
    key = f"{method}/{arm}/seed{seed}/M{capacity}"
    if complete(output):
        log(event="reuse_complete", key=key, output=str(output))
        return
    e_free, z_free = shutil.disk_usage(ROOT).free, shutil.disk_usage(ZROOT).free
    if e_free < 5_000_000_000 or z_free < 15_000_000_000:
        raise RuntimeError(f"Free-space safety stop: E={e_free}; Z={z_free}")
    if method == "a0_matched":
        command = [sys.executable, "-m", "scripts.revision3_train_v2", "--root", str(ROOT),
                   "--variant", method, "--arm", arm, "--seed", str(seed),
                   "--capacity", str(capacity)]
    else:
        command = [sys.executable, "-m", "scripts.revision3_capacity_a1", "--arm", arm,
                   "--seed", str(seed), "--capacity", str(capacity)]
    log(event="start", key=key, output=str(output), e_free=e_free, z_free=z_free)
    result = subprocess.run(command, cwd=ROOT, check=False)
    log(event="complete" if result.returncode == 0 else "failed", key=key,
        exit_code=result.returncode)
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> None:
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise ValueError("Wrong interpreter")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_V2_TRAINING":
        raise ValueError("V2 protocol is not frozen")
    # Re-establish the primary M=200 paired comparison before capacity analysis.
    for arm in ("doc", "short", "hard"):
        for seed in (17, 42, 2026):
            run_one("a0_matched", arm, seed, 200)
    # Capacity only on the two stressed streams. A1 outputs are frozen/reused.
    for arm in ("short", "hard"):
        for capacity in (50, 400):
            for seed in (17, 42, 2026):
                run_one("a0_matched", arm, seed, capacity)
                run_one("a1_matched", arm, seed, capacity)
    log(event="queue_complete")


if __name__ == "__main__":
    main()
