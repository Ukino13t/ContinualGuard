"""Apply the human-approved exact-duplicate NTFS hard-link cleanup.

This script deliberately contains an allow-list. It does not discover or remove
other duplicates, optimizer states, datasets, predictions, or unique artifacts.
"""

from __future__ import annotations

import ctypes
import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORAGE = ROOT / "history" / "storage"


MODEL_GROUPS = [
    {
        "sha256": "002187905f7816c0e15f569708622dc45c6c69c5a168c485f2cfef70839cf088",
        "source": "results/real-joint-seed42-20260909T170035-118348d838/checkpoints/JOINT/model.safetensors",
        "destinations": ["results/control-joint-epochs-seed42-20260911T044931-dc10425abc/checkpoints/JOINT-E1/model.safetensors"],
    },
    {
        "sha256": "5eb6a74e4143909a9529d758795ce73b19476bff9edb9cbba4a929d849a55dc8",
        "source": "results/control-s2-only-seed42-20260910T073224-342d69b25e/checkpoints/S2ONLY-E1/model.safetensors",
        "destinations": ["results/control-reverse-order-seed42-20260911T024900-9d04f60624/checkpoints/S2/model.safetensors"],
    },
    {
        "sha256": "e264b65418bcf93ffdfb264ba39d20affdc9077dfdb5e9a7ec534b57c8ee770a",
        "source": "results/real-sequential-seed17-20260909T045629-a72f49fa2a/checkpoints/S2/model.safetensors",
        "destinations": ["results/control-sequential-s2-epochs-seed17-20260910T151813-874c497d9c/checkpoints/S2-E1/model.safetensors"],
    },
    {
        "sha256": "0642246777f3d72743a5dcefa2242dc22b9ea2f8429552c6c9c769c251b4bffa",
        "source": "results/real-sequential-seed2026-20260909T090614-6ad5f5bec4/checkpoints/S2/model.safetensors",
        "destinations": ["results/control-sequential-s2-epochs-seed2026-20260910T225730-6a2f26de0e/checkpoints/S2-E1/model.safetensors"],
    },
    {
        "sha256": "4e49ecb2a7946750a8eda10d541d964db2b449fb82a9ffbc610a4a512b4d01f8",
        "source": "results/real-sequential-seed42-20260909T064952-2175ec20e4/checkpoints/S2/model.safetensors",
        "destinations": ["results/control-sequential-s2-epochs-seed42-20260910T190746-432fd7e38f/checkpoints/S2-E1/model.safetensors"],
    },
]

MANIFEST_PATHS = [
    "results/control-joint-epochs-seed42-20260911T044931-dc10425abc/data_manifest.json",
    "results/control-reverse-order-seed42-20260911T024900-9d04f60624/data_manifest.json",
    "results/control-s2-only-seed17-20260910T034321-d707574771/data_manifest.json",
    "results/control-s2-only-seed2026-20260910T113307-28ff67b7b9/data_manifest.json",
    "results/control-s2-only-seed42-20260910T073224-342d69b25e/data_manifest.json",
    "results/control-sequential-s2-epochs-seed17-20260910T151813-874c497d9c/data_manifest.json",
    "results/control-sequential-s2-epochs-seed2026-20260910T225730-6a2f26de0e/data_manifest.json",
    "results/control-sequential-s2-epochs-seed42-20260910T190746-432fd7e38f/data_manifest.json",
    "results/real-joint-seed17-20260909T111402-bda3d21c9d/data_manifest.json",
    "results/real-joint-seed2026-20260909T231824-990b686344/data_manifest.json",
    "results/real-joint-seed42-20260909T170035-118348d838/data_manifest.json",
    "results/real-sequential-seed17-20260909T045629-a72f49fa2a/data_manifest.json",
    "results/real-sequential-seed2026-20260909T090614-6ad5f5bec4/data_manifest.json",
    "results/real-sequential-seed42-20260909T064952-2175ec20e4/data_manifest.json",
]

SPM_PATHS = [
    "data/raw/huggingface/microsoft_deberta-v3-base/8ccc9b6f36199bec6961081d44eb72fb3f7353f3/spm.model",
    "results/control-joint-epochs-seed42-20260911T044931-dc10425abc/checkpoints/JOINT-E1/spm.model",
    "results/control-joint-epochs-seed42-20260911T044931-dc10425abc/checkpoints/JOINT-E2/spm.model",
    "results/control-joint-epochs-seed42-20260911T044931-dc10425abc/checkpoints/JOINT-E3/spm.model",
    "results/control-reverse-order-seed42-20260911T024900-9d04f60624/checkpoints/S1/spm.model",
    "results/control-reverse-order-seed42-20260911T024900-9d04f60624/checkpoints/S2/spm.model",
    "results/control-s2-only-seed17-20260910T034321-d707574771/checkpoints/S2ONLY-E1/spm.model",
    "results/control-s2-only-seed17-20260910T034321-d707574771/checkpoints/S2ONLY-E2/spm.model",
    "results/control-s2-only-seed17-20260910T034321-d707574771/checkpoints/S2ONLY-E3/spm.model",
    "results/control-s2-only-seed2026-20260910T113307-28ff67b7b9/checkpoints/S2ONLY-E1/spm.model",
    "results/control-s2-only-seed2026-20260910T113307-28ff67b7b9/checkpoints/S2ONLY-E2/spm.model",
    "results/control-s2-only-seed2026-20260910T113307-28ff67b7b9/checkpoints/S2ONLY-E3/spm.model",
    "results/control-s2-only-seed42-20260910T073224-342d69b25e/checkpoints/S2ONLY-E1/spm.model",
    "results/control-s2-only-seed42-20260910T073224-342d69b25e/checkpoints/S2ONLY-E2/spm.model",
    "results/control-s2-only-seed42-20260910T073224-342d69b25e/checkpoints/S2ONLY-E3/spm.model",
    "results/control-sequential-s2-epochs-seed17-20260910T151813-874c497d9c/checkpoints/S2-E1/spm.model",
    "results/control-sequential-s2-epochs-seed17-20260910T151813-874c497d9c/checkpoints/S2-E2/spm.model",
    "results/control-sequential-s2-epochs-seed17-20260910T151813-874c497d9c/checkpoints/S2-E3/spm.model",
    "results/control-sequential-s2-epochs-seed2026-20260910T225730-6a2f26de0e/checkpoints/S2-E1/spm.model",
    "results/control-sequential-s2-epochs-seed2026-20260910T225730-6a2f26de0e/checkpoints/S2-E2/spm.model",
    "results/control-sequential-s2-epochs-seed2026-20260910T225730-6a2f26de0e/checkpoints/S2-E3/spm.model",
    "results/control-sequential-s2-epochs-seed42-20260910T190746-432fd7e38f/checkpoints/S2-E1/spm.model",
    "results/control-sequential-s2-epochs-seed42-20260910T190746-432fd7e38f/checkpoints/S2-E2/spm.model",
    "results/control-sequential-s2-epochs-seed42-20260910T190746-432fd7e38f/checkpoints/S2-E3/spm.model",
    "results/real-joint-seed17-20260909T111402-bda3d21c9d/checkpoints/JOINT/spm.model",
    "results/real-joint-seed2026-20260909T231824-990b686344/checkpoints/JOINT/spm.model",
    "results/real-joint-seed42-20260909T170035-118348d838/checkpoints/JOINT/spm.model",
    "results/real-sequential-seed17-20260909T045629-a72f49fa2a/checkpoints/S1/spm.model",
    "results/real-sequential-seed17-20260909T045629-a72f49fa2a/checkpoints/S2/spm.model",
    "results/real-sequential-seed2026-20260909T090614-6ad5f5bec4/checkpoints/S1/spm.model",
    "results/real-sequential-seed2026-20260909T090614-6ad5f5bec4/checkpoints/S2/spm.model",
    "results/real-sequential-seed42-20260909T064952-2175ec20e4/checkpoints/S1/spm.model",
    "results/real-sequential-seed42-20260909T064952-2175ec20e4/checkpoints/S2/spm.model",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve(relative: str) -> Path:
    path = (ROOT / relative).resolve(strict=True)
    if path != ROOT and ROOT not in path.parents:
        raise RuntimeError(f"Path escapes workspace: {path}")
    if ".venv" in path.relative_to(ROOT).parts:
        raise RuntimeError(f"Refusing .venv path: {path}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    stat = path.stat()
    return {"device": stat.st_dev, "file_id": stat.st_ino, "link_count": stat.st_nlink}


def volume_info() -> dict:
    drive = ctypes.create_unicode_buffer(261)
    fs_name = ctypes.create_unicode_buffer(261)
    serial = ctypes.c_uint32()
    max_component = ctypes.c_uint32()
    flags = ctypes.c_uint32()
    ok = ctypes.windll.kernel32.GetVolumeInformationW(
        str(ROOT.drive + "\\"), drive, len(drive), ctypes.byref(serial),
        ctypes.byref(max_component), ctypes.byref(flags), fs_name, len(fs_name),
    )
    if not ok:
        raise ctypes.WinError()
    sectors_per_cluster = ctypes.c_uint32()
    bytes_per_sector = ctypes.c_uint32()
    free_clusters = ctypes.c_uint32()
    total_clusters = ctypes.c_uint32()
    ok = ctypes.windll.kernel32.GetDiskFreeSpaceW(
        str(ROOT.drive + "\\"), ctypes.byref(sectors_per_cluster), ctypes.byref(bytes_per_sector),
        ctypes.byref(free_clusters), ctypes.byref(total_clusters),
    )
    if not ok:
        raise ctypes.WinError()
    return {
        "filesystem": fs_name.value,
        "volume_serial": serial.value,
        "allocation_unit_bytes": sectors_per_cluster.value * bytes_per_sector.value,
    }


def allocated_size(path: Path) -> int:
    high = ctypes.c_uint32()
    low = ctypes.windll.kernel32.GetCompressedFileSizeW(str(path), ctypes.byref(high))
    if low == 0xFFFFFFFF and ctypes.get_last_error():
        raise ctypes.WinError()
    return (high.value << 32) | low


def exclusive_open_check(path: Path) -> None:
    GENERIC_READ = 0x80000000
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_NORMAL = 0x80
    invalid = ctypes.c_void_p(-1).value
    handle = ctypes.windll.kernel32.CreateFileW(
        str(path), GENERIC_READ, 0, None, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None
    )
    if handle == invalid:
        raise RuntimeError(f"Cannot acquire exclusive read handle: {path}: {ctypes.WinError()}")
    ctypes.windll.kernel32.CloseHandle(handle)


def hardlink_probe() -> dict:
    token = uuid.uuid4().hex
    source = STORAGE / f".hardlink_probe_{token}"
    target = STORAGE / f".hardlink_probe_{token}.link"
    try:
        source.write_bytes(b"continualguard hard-link probe\n")
        os.link(source, target)
        supported = source.stat().st_ino == target.stat().st_ino and source.stat().st_nlink >= 2
        return {"supported": supported, "source_identity": identity(source), "target_identity": identity(target)}
    finally:
        if target.exists():
            target.unlink()
        if source.exists():
            source.unlink()


def active_process_check() -> dict:
    result = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], text=True, capture_output=True, check=False)
    suspects = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) < 2:
            continue
        image_name, pid_text = row[0], row[1]
        try:
            pid = int(pid_text)
        except ValueError:
            pid = -1
        lowered = image_name.lower()
        if pid != os.getpid() and any(name in lowered for name in ("python.exe", "pythonw.exe", "torchrun", "deepspeed")):
            suspects.append({"image_name": image_name, "pid": pid})
    return {
        "method": "tasklist image-name scan plus per-file exclusive-open checks",
        "system_command_line_scan": "unavailable: Win32_Process CIM access denied",
        "tasklist_exit_code": result.returncode,
        "suspect_rows": suspects,
        "no_suspect_processes": not suspects,
    }


def validate_model(checkpoint_file: Path) -> dict:
    started = time.perf_counter()
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_file.parent, local_files_only=True
    )
    record = {
        "path": checkpoint_file.parent.relative_to(ROOT).as_posix(),
        "class": type(model).__name__,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "num_labels": model.config.num_labels,
        "seconds": time.perf_counter() - started,
        "status": "passed",
    }
    del model
    gc.collect()
    return record


def validate_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {"path": path.relative_to(ROOT).as_posix(), "top_level_type": type(payload).__name__, "status": "passed"}


def validate_tokenizer(spm_path: Path) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spm_path.parent, local_files_only=True)
    encoded = tokenizer("hard-link validation", add_special_tokens=True)
    return {
        "path": spm_path.parent.relative_to(ROOT).as_posix(),
        "class": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
        "probe_token_count": len(encoded["input_ids"]),
        "status": "passed",
    }


def replace_with_hardlink(source: Path, destination: Path, expected_sha: str) -> dict:
    exclusive_open_check(source)
    exclusive_open_check(destination)
    source_stat = source.stat()
    destination_stat = destination.stat()
    if source_stat.st_size != destination_stat.st_size:
        raise RuntimeError(f"Size mismatch: {source} vs {destination}")
    source_sha = sha256(source)
    destination_sha = sha256(destination)
    if source_sha != expected_sha or destination_sha != expected_sha:
        raise RuntimeError(f"Immediate SHA-256 verification failed: {source} vs {destination}")
    before_source = identity(source)
    before_destination = identity(destination)
    if before_source["file_id"] == before_destination["file_id"]:
        return {
            "source": source.relative_to(ROOT).as_posix(),
            "destination": destination.relative_to(ROOT).as_posix(),
            "action": "already_hardlinked",
            "sha256": expected_sha,
            "bytes": source_stat.st_size,
            "allocated_bytes_reclaimed": 0,
            "before": {"source": before_source, "destination": before_destination},
            "after": {"source": before_source, "destination": before_destination},
        }
    before_allocated = allocated_size(destination)
    backup = destination.with_name(destination.name + f".dedup_backup_{uuid.uuid4().hex}")
    os.link(destination, backup)
    try:
        destination.unlink()
        try:
            os.link(source, destination)
        except Exception:
            os.link(backup, destination)
            raise
        if source.stat().st_ino != destination.stat().st_ino:
            destination.unlink()
            os.link(backup, destination)
            raise RuntimeError(f"Hard-link identity verification failed: {destination}")
    finally:
        if backup.exists():
            backup.unlink()
    if sha256(destination) != expected_sha:
        raise RuntimeError(f"Post-link SHA-256 verification failed: {destination}")
    return {
        "source": source.relative_to(ROOT).as_posix(),
        "destination": destination.relative_to(ROOT).as_posix(),
        "action": "replaced_exact_duplicate_with_hardlink",
        "sha256": expected_sha,
        "bytes": source_stat.st_size,
        "allocated_bytes_reclaimed": before_allocated,
        "before": {"source": before_source, "destination": before_destination},
        "after": {"source": identity(source), "destination": identity(destination)},
    }


def save(snapshot_path: Path, report: dict) -> None:
    snapshot_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    STORAGE.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_path = STORAGE / f"exact_dedup_{timestamp}.json"
    volume = volume_info()
    if volume["filesystem"].upper() != "NTFS":
        raise RuntimeError(f"Hard-link cleanup requires NTFS, found {volume['filesystem']}")
    probe = hardlink_probe()
    if not probe["supported"]:
        raise RuntimeError("Hard-link probe failed; no fallback is permitted")

    manifest_group = {
        "sha256": "106ead5a6918e7d8a9a1137364e95a599212d3ed3d1359ee631edae47fcfca18",
        "source": MANIFEST_PATHS[0],
        "destinations": MANIFEST_PATHS[1:],
    }
    spm_group = {
        "sha256": "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd",
        "source": SPM_PATHS[0],
        "destinations": SPM_PATHS[1:],
    }
    groups = [*(dict(group, kind="model_safetensors") for group in MODEL_GROUPS), dict(manifest_group, kind="data_manifest"), dict(spm_group, kind="spm_model")]
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "pre_cleanup_snapshot",
        "root": str(ROOT),
        "filesystem": volume,
        "hardlink_probe": probe,
        "active_process_check": active_process_check(),
        "approved_scope": {
            "groups": len(groups),
            "model_groups": len(MODEL_GROUPS),
            "manifest_paths": len(MANIFEST_PATHS),
            "spm_paths": len(SPM_PATHS),
            "explicit_exclusions": [".venv", "state.pt", "unique weights", "predictions", "metrics", "configs", "metadata", "logs", "datasets", "unique manifests", "papers"],
        },
        "planned_groups": groups,
        "group_results": [],
    }
    save(snapshot_path, report)
    print(f"PRE_CLEANUP_SNAPSHOT={snapshot_path}", flush=True)

    try:
        for index, group in enumerate(groups, start=1):
            print(f"GROUP {index}/{len(groups)} {group['kind']} START", flush=True)
            source = resolve(group["source"])
            destinations = [resolve(path) for path in group["destinations"]]
            all_paths = [source, *destinations]
            sizes = {path.stat().st_size for path in all_paths}
            hashes = {sha256(path) for path in all_paths}
            if len(sizes) != 1 or hashes != {group["sha256"]}:
                raise RuntimeError(f"Group precheck failed for {group['kind']}: sizes={sizes}, hashes={hashes}")
            for path in all_paths:
                exclusive_open_check(path)
            result = {
                "index": index,
                "kind": group["kind"],
                "started_at": utc_now(),
                "source": group["source"],
                "destinations": group["destinations"],
                "sha256": group["sha256"],
                "bytes_each": next(iter(sizes)),
                "precheck": "size, SHA-256, and exclusive-open passed",
                "pre_validation": [],
                "actions": [],
                "post_validation": [],
            }
            if group["kind"] == "model_safetensors":
                result["pre_validation"] = [validate_model(path) for path in all_paths]
            elif group["kind"] == "data_manifest":
                result["pre_validation"] = [validate_manifest(path) for path in all_paths]
            elif group["kind"] == "spm_model":
                result["pre_validation"] = [validate_tokenizer(path) for path in all_paths]

            for destination in destinations:
                result["actions"].append(replace_with_hardlink(source, destination, group["sha256"]))

            if group["kind"] == "model_safetensors":
                result["post_validation"] = [validate_model(path) for path in all_paths]
            elif group["kind"] == "data_manifest":
                result["post_validation"] = [validate_manifest(path) for path in all_paths]
            elif group["kind"] == "spm_model":
                result["post_validation"] = [validate_tokenizer(path) for path in all_paths]
            result["finished_at"] = utc_now()
            result["status"] = "passed"
            report["group_results"].append(result)
            save(snapshot_path, report)
            print(f"GROUP {index}/{len(groups)} {group['kind']} PASSED", flush=True)
    except Exception as exc:
        report["status"] = "failed"
        report["failed_at"] = utc_now()
        report["error"] = f"{type(exc).__name__}: {exc}"
        save(snapshot_path, report)
        raise

    report["status"] = "hardlink_cleanup_complete"
    report["hardlink_completed_at"] = utc_now()
    report["reclaimed_content_bytes"] = sum(action["bytes"] for group in report["group_results"] for action in group["actions"] if action["action"] == "replaced_exact_duplicate_with_hardlink")
    report["reclaimed_allocated_bytes"] = sum(action["allocated_bytes_reclaimed"] for group in report["group_results"] for action in group["actions"])
    allocation_unit = volume["allocation_unit_bytes"]
    report["reclaimed_ntfs_allocation_unit_bytes"] = sum(
        ((action["bytes"] + allocation_unit - 1) // allocation_unit) * allocation_unit
        for group in report["group_results"] for action in group["actions"]
        if action["action"] == "replaced_exact_duplicate_with_hardlink"
    )
    save(snapshot_path, report)
    print(f"HARDLINK_CLEANUP_COMPLETE reclaimed_allocated_bytes={report['reclaimed_allocated_bytes']}", flush=True)
    print(snapshot_path, flush=True)


if __name__ == "__main__":
    main()
