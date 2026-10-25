"""Read-only project storage inventory and cleanup-candidate report."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIN_DUPLICATE_HASH_BYTES = 1_000_000
HASH_CACHE = {}


def sha256(path):
    key = str(path.resolve())
    if key in HASH_CACHE:
        return HASH_CACHE[key]
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    HASH_CACHE[key] = h.hexdigest()
    return HASH_CACHE[key]


def category(rel):
    parts, name = rel.parts, rel.name.lower()
    suffix = rel.suffix.lower()
    if parts[0] == ".venv": return "virtual_environment"
    if suffix == ".pdf": return "papers"
    if "__pycache__" in parts or parts[0] == ".pytest_cache" or suffix in {".pyc", ".pyo"}: return "transient_caches"
    if parts[0] == "docs": return "documentation"
    if parts[0] == "history": return "history"
    if parts[0] == "tests" and "fixtures" in parts: return "synthetic_fixtures"
    if parts[0] == "data":
        if len(parts) > 2 and parts[1] == "raw" and parts[2] == "huggingface":
            if name in {"model.safetensors", "pytorch_model.bin"}: return "base_model_weights"
            return "huggingface_model_tokenizer_cache"
        if len(parts) > 1 and parts[1] == "raw": return "raw_data"
        if len(parts) > 1 and parts[1] == "processed": return "processed_data"
        if len(parts) > 1 and parts[1] == "manifests": return "manifests"
        if len(parts) > 1 and parts[1] == "audits": return "data_audits"
    if parts[0] == "results":
        if name == "model.safetensors": return "trained_checkpoint_weights"
        if name == "state.pt": return "optimizer_scheduler_rng_states"
        if name in {"spm.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"}: return "checkpoint_tokenizer_copies"
        if name == "config.json" and "checkpoints" in parts: return "checkpoint_model_configs"
        if name.startswith("matrix_"): return "metric_matrices"
        if name == "confusion_counts.jsonl": return "confusion_counts"
        if name.endswith("log.jsonl") or name in {"stage_events.jsonl", "metrics.jsonl"}: return "run_logs"
        if suffix == ".jsonl": return "raw_predictions"
        if name in {"summary.json", "metadata.json"}: return "run_summaries_metadata"
        if name in {"resolved_config.yaml", "data_manifest.json", "data_audit.json", "length_audit.json", "exclusions.json"}: return "run_reproducibility_records"
        return "other_results"
    if parts[0] in {"tmp", ".tmp", "temp"}: return "temporary_files"
    if parts[0] == "src" or parts[0] == "scripts": return "code"
    if parts[0] == "tests": return "tests"
    if parts[0] == "configs": return "configuration"
    return "other_project_files"


def file_record(path):
    rel = path.relative_to(ROOT)
    return {"path": rel.as_posix(), "bytes": path.stat().st_size, "category": category(rel)}


def run_is_complete(path):
    run = next((ROOT / "results" / part for part in path.relative_to(ROOT).parts[1:2]), None)
    if not run:
        return False
    return (run / "summary.json").exists() and (run / "metadata.json").exists()


def candidate(path, action, reason, risk, recoverability, tier):
    rec = file_record(path)
    rec.update({"action": action, "reason": reason, "risk": risk,
                "recoverability": recoverability, "retention_tier": tier,
                "sha256": sha256(path) if path.stat().st_size else hashlib.sha256(b"").hexdigest(),
                "status": "proposed_only_no_action"})
    return rec


def audit():
    files = []
    for path in ROOT.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                files.append(file_record(path))
        except OSError:
            continue
    by_category = defaultdict(lambda: {"bytes": 0, "files": 0})
    top_dirs = defaultdict(lambda: {"bytes": 0, "files": 0})
    run_dirs = defaultdict(lambda: {"bytes": 0, "files": 0})
    for item in files:
        by_category[item["category"]]["bytes"] += item["bytes"]
        by_category[item["category"]]["files"] += 1
        parts = Path(item["path"]).parts
        top_dirs[parts[0]]["bytes"] += item["bytes"]
        top_dirs[parts[0]]["files"] += 1
        if parts[0] == "results" and len(parts) > 1:
            run_dirs[parts[1]]["bytes"] += item["bytes"]
            run_dirs[parts[1]]["files"] += 1

    large = [x for x in files if x["bytes"] >= MIN_DUPLICATE_HASH_BYTES]
    hashed = defaultdict(list)
    for index, item in enumerate(sorted(large, key=lambda x: x["path"]), 1):
        path = ROOT / Path(item["path"])
        hashed[(item["bytes"], sha256(path))].append(item["path"])
    duplicates = []
    for (size, file_hash), paths in hashed.items():
        if len(paths) > 1:
            duplicates.append({"sha256": file_hash, "bytes_each": size, "copies": len(paths),
                               "reclaimable_bytes_beyond_one": size * (len(paths) - 1), "paths": paths})
    duplicates.sort(key=lambda x: x["reclaimable_bytes_beyond_one"], reverse=True)

    conservative, aggressive = [], []
    for item in files:
        path = ROOT / Path(item["path"])
        if item["category"] in {"transient_caches", "temporary_files"}:
            conservative.append(candidate(path, "delete after approval", "regenerable interpreter/test cache or temporary file",
                                          "low", "regenerated by Python/tests", 3))
        if item["category"] == "optimizer_scheduler_rng_states" and run_is_complete(path):
            conservative.append(candidate(path, "delete after approval",
                                          "completed run has summary and metadata; optimizer state is unnecessary unless exact resume is required",
                                          "medium: exact optimizer resume becomes impossible",
                                          "model checkpoint, predictions, metrics, configuration, and logs remain", 3))
    # Aggressive-only model candidates: smoke checkpoints and non-selected E1/E3 endpoints.
    for item in files:
        path_text = item["path"].lower()
        if item["category"] == "trained_checkpoint_weights" and (
            "/smoke-" in "/" + path_text or
            ("control-s2-only" in path_text and any(x in path_text for x in ("s2only-e1", "s2only-e3")))
        ):
            aggressive.append(candidate(ROOT / Path(item["path"]), "archive or delete after approval",
                                        "smoke weight or non-selected S2-only endpoint; raw predictions and summaries remain",
                                        "high: future re-evaluation requires retraining",
                                        "retrain from pinned base model, data hash, seed, and resolved config", 3))
    # Exact duplicate tokenizer copies are candidates only after loaders point to the canonical HF snapshot.
    for group in duplicates:
        paths = [p for p in group["paths"] if category(Path(p)) == "checkpoint_tokenizer_copies"]
        for p in paths:
            aggressive.append(candidate(ROOT / Path(p), "deduplicate after loader validation",
                                        "identical tokenizer artifact is already present in the canonical pinned Hugging Face snapshot",
                                        "medium: checkpoint directory is no longer standalone",
                                        "restore from canonical tokenizer snapshot using the recorded hash", 3))
        weight_paths = [p for p in group["paths"] if category(Path(p)) == "trained_checkpoint_weights"]
        if len(weight_paths) > 1:
            def keep_priority(value):
                return (0 if "/real-" in "/" + value else 1, value)
            canonical = sorted(weight_paths, key=keep_priority)[0]
            for p in weight_paths:
                if p == canonical:
                    continue
                aggressive.append(candidate(ROOT / Path(p), "deduplicate after checkpoint-loader validation",
                                            f"byte-identical trained weights already retained at {canonical}",
                                            "medium: run directory loses a standalone weight copy",
                                            f"restore from {canonical}; SHA-256 is identical", 3))
        manifest_paths = [p for p in group["paths"] if Path(p).name == "data_manifest.json"]
        if len(manifest_paths) > 1:
            canonical = sorted(manifest_paths)[0]
            for p in sorted(manifest_paths)[1:]:
                aggressive.append(candidate(ROOT / Path(p), "replace with content-addressed reference after tool validation",
                                            f"byte-identical 38 MB manifest already retained at {canonical}",
                                            "medium: old tools may expect an inline per-run manifest",
                                            f"restore from {canonical}; SHA-256 is identical", 3))

    def unique_candidates(items):
        result = {}
        for item in items:
            result[item["path"]] = item
        return [result[k] for k in sorted(result)]
    conservative = unique_candidates(conservative)
    aggressive = unique_candidates([*conservative, *aggressive])
    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    result = {
        "generated_at_utc": generated,
        "audit_mode": "read_only_no_files_deleted_or_archived",
        "root": str(ROOT), "total_bytes": sum(x["bytes"] for x in files), "total_files": len(files),
        "category_breakdown": dict(sorted(by_category.items())),
        "top_level_directory_sizes": dict(sorted(top_dirs.items(), key=lambda x: x[1]["bytes"], reverse=True)),
        "result_run_directory_sizes": dict(sorted(run_dirs.items(), key=lambda x: x[1]["bytes"], reverse=True)),
        "largest_files": sorted(files, key=lambda x: x["bytes"], reverse=True)[:30],
        "large_duplicate_hash_groups": duplicates,
        "large_duplicate_threshold_bytes": MIN_DUPLICATE_HASH_BYTES,
        "cleanup_candidates": {"conservative": conservative, "aggressive": aggressive},
        "estimated_reclaimable_bytes": {
            "conservative": sum(x["bytes"] for x in conservative),
            "aggressive": sum(x["bytes"] for x in aggressive),
        },
        "actions_taken": [],
        "important_artifact_hashes": {
            "requirements-lock.txt": sha256(ROOT / "requirements-lock.txt"),
            "data/manifests/attack_family_shift/construction.json": sha256(ROOT / "data/manifests/attack_family_shift/construction.json"),
        },
    }
    out_dir = ROOT / "history" / "storage"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = generated.replace(":", "").replace("-", "")
    output = out_dir / f"storage_audit_{stamp}.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    latest = out_dir / "latest.json"
    latest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": output.relative_to(ROOT).as_posix(), "total_bytes": result["total_bytes"],
                      "files": result["total_files"], "duplicates": len(duplicates),
                      "conservative_bytes": result["estimated_reclaimable_bytes"]["conservative"],
                      "aggressive_bytes": result["estimated_reclaimable_bytes"]["aggressive"]}, indent=2))


if __name__ == "__main__":
    audit()
