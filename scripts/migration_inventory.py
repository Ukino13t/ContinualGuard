"""Create or compare a reproducible project migration inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


IMPORTANT_DIRS = (
    "configs",
    "data",
    "docs",
    "history",
    "papers",
    "results",
    "scripts",
    "src",
    "tests",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_map(root: Path, paths: list[Path]) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(paths)
        if path.is_file()
    }


def files_under(root: Path, relative: str, patterns: tuple[str, ...]) -> list[Path]:
    base = root / relative
    if not base.exists():
        return []
    found: set[Path] = set()
    for pattern in patterns:
        found.update(base.rglob(pattern))
    return sorted(path for path in found if path.is_file())


def inventory(
    root: Path,
    source_root: str,
    destination_root: str,
    source_filesystem: str,
    destination_filesystem: str,
    destination_free_bytes: int,
) -> dict:
    files = [path for path in root.rglob("*") if path.is_file()]
    metadata = sorted((root / "results").glob("*/metadata.json"))
    run_ids = []
    for path in metadata:
        try:
            run_ids.append(json.loads(path.read_text(encoding="utf-8"))["run_id"])
        except (KeyError, json.JSONDecodeError, UnicodeDecodeError):
            run_ids.append(path.parent.name)

    checkpoints = sorted(
        path.parent.relative_to(root).as_posix()
        for path in (root / "results").glob("*/checkpoints/*/model.safetensors")
    )
    core_paths = [
        root / "requirements-lock.txt",
        root / "pyproject.toml",
        root / "README.md",
        root / "docs/decision_log.md",
        root / "docs/experiment_history.md",
        root / "docs/change_log.md",
        root / "docs/attack_family_baseline_results.md",
        root / "results/attack_family_baseline_aggregate.json",
    ]
    source_paths = files_under(root, "src", ("*.py",))
    config_paths = files_under(root, "configs", ("*.yaml", "*.yml"))
    dataset_manifest_paths = files_under(root, "data/processed", ("*",)) + files_under(
        root, "data/manifests", ("*",)
    )
    pdf_paths = [
        path for path in files
        if path.suffix.lower() == ".pdf" and ".venv" not in path.parts
    ]

    return {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "inventory_root": str(root),
        "source_root": source_root,
        "destination_root": destination_root,
        "source_drive_filesystem": source_filesystem,
        "destination_drive_filesystem": destination_filesystem,
        "destination_free_bytes": destination_free_bytes,
        "active_project_process_check": {
            "python_processes": 0,
            "gpu_compute_processes_named_python": 0,
            "result": "no active training, evaluation, construction, cleanup, or checkpoint writer detected",
            "method": "Get-Process plus nvidia-smi before snapshot",
        },
        "migration_risks": [
            "Cross-volume copies expand source hard links into independent destination files",
            "The copied Windows virtual environment contains source-root absolute paths and must be rebuilt",
            "The active Codex workspace may keep the source root open and prevent safe same-session deletion",
            "External-drive availability and performance differ from the original internal volume",
        ],
        "total_file_count": len(files),
        "important_directories": {
            name: (root / name).is_dir() for name in IMPORTANT_DIRS
        },
        "python_version": platform.python_version(),
        "critical_hashes": hash_map(root, core_paths),
        "source_code_hashes": hash_map(root, source_paths),
        "yaml_config_hashes": hash_map(root, config_paths),
        "dataset_manifest_hashes": hash_map(root, dataset_manifest_paths),
        "supplied_pdf_hashes": hash_map(root, pdf_paths),
        "result_run_ids": sorted(run_ids),
        "stage_end_checkpoints": checkpoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--destination-root", required=True)
    parser.add_argument("--source-filesystem", required=True)
    parser.add_argument("--destination-filesystem", required=True)
    parser.add_argument("--destination-free-bytes", required=True, type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = inventory(
        args.root.resolve(),
        args.source_root,
        args.destination_root,
        args.source_filesystem,
        args.destination_filesystem,
        args.destination_free_bytes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "total_file_count": payload["total_file_count"],
        "run_count": len(payload["result_run_ids"]),
        "checkpoint_count": len(payload["stage_end_checkpoints"]),
        "dataset_manifest_hash_count": len(payload["dataset_manifest_hashes"]),
        "source_hash_count": len(payload["source_code_hashes"]),
        "pdf_hash_count": len(payload["supplied_pdf_hashes"]),
    }, indent=2))


if __name__ == "__main__":
    main()
