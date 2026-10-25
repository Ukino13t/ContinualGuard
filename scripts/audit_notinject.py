"""Pin and audit the official NotInject dataset without constructing H2 training data."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "leolee99/NotInject"
REVISION = "847ae76cf8fea5ed325429e569ae8cfef022d2e0"
RAW = ROOT / "data/raw/notinject" / REVISION
AUDIT = ROOT / "data/audits/hard_benign/notinject"
MANIFEST = ROOT / "data/manifests/hard_benign/notinject"
SPLITS = {
    "NotInject_one": 1,
    "NotInject_two": 2,
    "NotInject_three": 3,
}
FILES = {
    "README.md": None,
    "data/NotInject_one-00000-of-00001.parquet": "5b1d3a8ec341205305efd8c6d2cabeb5ba0f594dd8726b08ee7fde1a46ddc6ec",
    "data/NotInject_two-00000-of-00001.parquet": "a5a8c1b3b851b0b5acc3b6caf8bfee26d7da631b2967dd31ca90358df8883feb",
    "data/NotInject_three-00000-of-00001.parquet": "d5c685b123570c092a0f374ba6bdf324b2026d905a2751a243eb2bb360d411a8",
}
EXPECTED_ALIASES = {
    "data/train-00000-of-00001.parquet": "data/NotInject_one-00000-of-00001.parquet",
    "data/validation-00000-of-00001.parquet": "data/NotInject_two-00000-of-00001.parquet",
    "data/test-00000-of-00001.parquet": "data/NotInject_three-00000-of-00001.parquet",
}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def token_set(text: str) -> set[str]:
    return set(re.findall(r"\w+", normalized(text), flags=re.UNICODE))


def percentile(values, fraction):
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def get_json(session, url, **kwargs):
    response = session.get(url, timeout=60, **kwargs)
    response.raise_for_status()
    return response.json()


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    MANIFEST.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    meta = get_json(session, f"https://huggingface.co/api/datasets/{DATASET_ID}")
    if meta["sha"] != REVISION:
        raise RuntimeError(f"Official HEAD changed: expected {REVISION}, got {meta['sha']}")
    tree = get_json(
        session,
        f"https://huggingface.co/api/datasets/{DATASET_ID}/tree/{REVISION}",
        params={"recursive": "true", "expand": "true"},
    )
    save_json(RAW / "source_api.json", meta)
    save_json(RAW / "source_tree.json", tree)
    tree_by_path = {item["path"]: item for item in tree if item["type"] == "file"}

    file_manifest = []
    for relative, expected_lfs_sha in FILES.items():
        url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{REVISION}/{relative}?download=true"
        response = session.get(url, timeout=60)
        response.raise_for_status()
        destination = RAW / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
        actual_sha = sha_file(destination)
        if expected_lfs_sha and actual_sha != expected_lfs_sha:
            raise RuntimeError(f"Hash mismatch for {relative}: {actual_sha}")
        file_manifest.append({
            "path": relative,
            "local_path": str(destination.relative_to(ROOT)),
            "bytes": destination.stat().st_size,
            "sha256": actual_sha,
            "git_oid": tree_by_path[relative]["oid"],
            "lfs_oid": tree_by_path[relative].get("lfs", {}).get("oid"),
            "url": url,
        })

    rows = []
    response_files = []
    features = None
    for split, trigger_count in SPLITS.items():
        split_rows = []
        for offset in (0, 100):
            payload = get_json(
                session,
                "https://datasets-server.huggingface.co/rows",
                params={"dataset": DATASET_ID, "config": "default", "split": split, "offset": offset, "length": 100},
            )
            if features is None:
                features = payload["features"]
            if payload["num_rows_total"] != 113 or payload["partial"]:
                raise RuntimeError(f"Unexpected dataset-server status for {split}")
            raw_path = RAW / "dataset_server" / f"{split}_{offset:03d}.json"
            save_json(raw_path, payload)
            response_files.append({
                "path": str(raw_path.relative_to(ROOT)),
                "sha256": sha_file(raw_path),
                "offset": offset,
                "returned": len(payload["rows"]),
            })
            split_rows.extend(payload["rows"])
        split_rows = {item["row_idx"]: item for item in split_rows}
        if sorted(split_rows) != list(range(113)):
            raise RuntimeError(f"Incomplete rows for {split}")
        for row_idx, wrapped in sorted(split_rows.items()):
            row = wrapped["row"]
            if set(row) != {"prompt", "word_list", "category"}:
                raise RuntimeError(f"Unexpected fields at {split}/{row_idx}: {set(row)}")
            rows.append({
                "source_split": split,
                "trigger_count_group": trigger_count,
                "row_idx": row_idx,
                **row,
            })

    prompts = [row["prompt"] for row in rows]
    norm_prompts = [normalized(text) for text in prompts]
    exact_groups = defaultdict(list)
    normalized_groups = defaultdict(list)
    for row, norm in zip(rows, norm_prompts):
        exact_groups[row["prompt"]].append((row["source_split"], row["row_idx"]))
        normalized_groups[norm].append((row["source_split"], row["row_idx"]))
    exact_dupes = [group for group in exact_groups.values() if len(group) > 1]
    normalized_dupes = [group for group in normalized_groups.values() if len(group) > 1]

    near_internal = []
    sets = [token_set(text) for text in prompts]
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            union = sets[left] | sets[right]
            score = len(sets[left] & sets[right]) / len(union) if union else 1.0
            if score >= 0.9:
                near_internal.append({
                    "left": f"{rows[left]['source_split']}:{rows[left]['row_idx']}",
                    "right": f"{rows[right]['source_split']}:{rows[right]['row_idx']}",
                    "token_set_jaccard": score,
                })

    current_sources = [
        ROOT / "data/processed/chen_attack_family_shift.jsonl",
        ROOT / "data/processed/chen_domain_shift.jsonl",
    ]
    current = []
    for path in current_sources:
        for row in load_jsonl(path):
            text = row.get("text", "")
            current.append({
                "dataset": str(path.relative_to(ROOT)),
                "example_id": row.get("example_id"),
                "normalized": normalized(text),
                "tokens": token_set(text),
            })
    exact_current = defaultdict(list)
    for item in current:
        exact_current[item["normalized"]].append(item)
    exact_cross = []
    substring_cross = []
    near_cross = []
    by_length = defaultdict(list)
    for item in current:
        by_length[len(item["tokens"])].append(item)
    for row, norm, query_tokens in zip(rows, norm_prompts, sets):
        rid = f"{row['source_split']}:{row['row_idx']}"
        for match in exact_current.get(norm, []):
            exact_cross.append({"notinject": rid, "dataset": match["dataset"], "example_id": match["example_id"]})
        for item in current:
            if norm and norm in item["normalized"]:
                substring_cross.append({"notinject": rid, "dataset": item["dataset"], "example_id": item["example_id"]})
        n = len(query_tokens)
        if n:
            low = math.ceil(0.9 * n)
            high = math.floor(n / 0.9)
            for length in range(low, high + 1):
                for item in by_length.get(length, []):
                    union = query_tokens | item["tokens"]
                    score = len(query_tokens & item["tokens"]) / len(union)
                    if score >= 0.9:
                        near_cross.append({
                            "notinject": rid,
                            "dataset": item["dataset"],
                            "example_id": item["example_id"],
                            "token_set_jaccard": score,
                        })

    model_config = yaml.safe_load((ROOT / "configs/attack_family_order_control.yaml").read_text(encoding="utf-8"))
    tokenizer_path = (ROOT / model_config["model"]["local_artifact"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_fast=False)
    token_lengths = [len(tokenizer.encode(text, add_special_tokens=True, truncation=False)) for text in prompts]

    aliases = []
    for alias, canonical in EXPECTED_ALIASES.items():
        a, c = tree_by_path[alias], tree_by_path[canonical]
        aliases.append({
            "alias_path": alias,
            "canonical_path": canonical,
            "same_git_oid": a["oid"] == c["oid"],
            "same_lfs_oid": a["lfs"]["oid"] == c["lfs"]["oid"],
            "lfs_sha256": a["lfs"]["oid"],
        })
    readme = (RAW / "README.md").read_text(encoding="utf-8")
    license_declared = bool(re.search(r"(?mi)^license:\s*mit\s*$", readme))

    record_manifest = []
    for row, norm, length in zip(rows, norm_prompts, token_lengths):
        record_manifest.append({
            "record_id": hashlib.sha256(f"{DATASET_ID}:{REVISION}:{row['source_split']}:{row['row_idx']}".encode()).hexdigest(),
            "source_split": row["source_split"],
            "source_row_idx": row["row_idx"],
            "trigger_count_group": row["trigger_count_group"],
            "word_list_count": len(row["word_list"]),
            "category": row["category"],
            "label": 0,
            "prompt_sha256": hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest(),
            "normalized_prompt_sha256": hashlib.sha256(norm.encode("utf-8")).hexdigest(),
            "deberta_tokens": length,
        })
    manifest_path = MANIFEST / "records.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in record_manifest),
        encoding="utf-8",
    )

    category_counts = Counter(row["category"] for row in rows)
    group_category_counts = {
        str(group): dict(sorted(Counter(row["category"] for row in rows if row["trigger_count_group"] == group).items()))
        for group in (1, 2, 3)
    }
    problems = {
        "empty_prompt": sum(not row["prompt"].strip() for row in rows),
        "empty_word_list": sum(not row["word_list"] for row in rows),
        "word_count_group_mismatch": sum(len(row["word_list"]) != row["trigger_count_group"] for row in rows),
        "unknown_category": sum(row["category"] not in {"Common Queries", "Technique Queries", "Virtual Creation", "Multilingual"} for row in rows),
        "truncated_dataset_server_cells": 0,
    }
    audit = {
        "status": "audit_complete_no_training",
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "active_project_root": str(ROOT),
        "dataset_id": DATASET_ID,
        "repository_revision": REVISION,
        "last_modified": meta.get("lastModified"),
        "availability": {"public": not meta.get("private"), "gated": meta.get("gated"), "disabled": meta.get("disabled")},
        "license": {
            "dataset_card_spdx": "MIT",
            "dataset_card_front_matter_verified": license_declared,
            "repository_tag": "license:mit",
            "scope_caveat": "The dataset card declares MIT. Preserve paper/dataset attribution; this audit is not legal advice.",
        },
        "provenance": {
            "official_publisher": "leolee99",
            "paper": "PIGuard / InjecGuard, arXiv:2410.22770 and ACL 2025",
            "intended_role": "over-defense evaluation on benign prompts containing injection-associated trigger words",
            "generation_disclosure": "Dataset card/paper describe constructed benign sentences; per-record generator/model provenance is not supplied.",
        },
        "schema": features,
        "records": len(rows),
        "label_distribution": {"0_benign": len(rows), "1_injection": 0},
        "trigger_group_counts": dict(sorted(Counter(row["trigger_count_group"] for row in rows).items())),
        "category_counts": dict(sorted(category_counts.items())),
        "trigger_group_by_category": group_category_counts,
        "non_ascii_prompt_count": sum(any(ord(character) > 127 for character in prompt) for prompt in prompts),
        "quality_checks": problems,
        "official_file_aliases": aliases,
        "raw_file_manifest": file_manifest,
        "dataset_server_response_manifest": response_files,
        "duplicates": {
            "internal_exact_prompt_groups": exact_dupes,
            "internal_normalized_prompt_groups": normalized_dupes,
            "internal_token_set_jaccard_ge_0_9": near_internal,
        },
        "overlap_with_existing_processed_data": {
            "datasets": [str(path.relative_to(ROOT)) for path in current_sources],
            "existing_records_scanned": len(current),
            "normalized_exact_matches": exact_cross,
            "normalized_prompt_substring_matches": substring_cross,
            "token_set_jaccard_ge_0_9": near_cross,
            "method": "Unicode NFKC + casefold + whitespace collapse; exact/substring; exact token-set Jaccard with cardinality filtering",
        },
        "length": {
            "tokenizer": model_config["model"]["checkpoint"],
            "tokenizer_revision": model_config["model"]["revision"],
            "max_length": 1280,
            "min": min(token_lengths),
            "median": percentile(token_lengths, 0.5),
            "p95": percentile(token_lengths, 0.95),
            "p99": percentile(token_lengths, 0.99),
            "max": max(token_lengths),
            "over_1280": sum(value > 1280 for value in token_lengths),
        },
        "usability": {
            "usable_as_frozen_benign_evaluation": len(rows) if not any(problems.values()) and not exact_dupes and not normalized_dupes else None,
            "usable_as_training_records": 0,
            "reason_training_zero": "NotInject is an evaluation benchmark; preserve all 339 records as a frozen hard-benign diagnostic until a human approves a training-source design.",
            "raw_content_git_ignored": True,
            "processed_H2_constructed": False,
            "training_executed": False,
            "replay": False,
        },
    }
    save_json(AUDIT / "audit.json", audit)
    save_json(AUDIT / "overlap_findings.json", {
        "internal_exact_prompt_groups": exact_dupes,
        "internal_normalized_prompt_groups": normalized_dupes,
        "internal_token_set_jaccard_ge_0_9": near_internal,
        "normalized_exact_matches": exact_cross,
        "normalized_prompt_substring_matches": substring_cross,
        "token_set_jaccard_ge_0_9": near_cross,
    })
    source_manifest = {
        "dataset_id": DATASET_ID,
        "repository_revision": REVISION,
        "source_url": f"https://huggingface.co/datasets/{DATASET_ID}",
        "revision_url": f"https://huggingface.co/datasets/{DATASET_ID}/tree/{REVISION}",
        "license": "MIT (dataset card)",
        "records": 339,
        "fields": ["prompt", "word_list", "category"],
        "raw_files": file_manifest,
        "official_aliases": aliases,
        "records_manifest": str(manifest_path.relative_to(ROOT)),
        "records_manifest_sha256": sha_file(manifest_path),
        "raw_text_committed": False,
    }
    save_json(MANIFEST / "source.json", source_manifest)
    print(json.dumps({
        "status": audit["status"],
        "revision": REVISION,
        "records": len(rows),
        "categories": audit["category_counts"],
        "quality_checks": problems,
        "duplicates": {key: len(value) for key, value in audit["duplicates"].items()},
        "cross_overlap": {
            key: len(value)
            for key, value in audit["overlap_with_existing_processed_data"].items()
            if isinstance(value, list) and key not in {"datasets"}
        },
        "length": audit["length"],
        "usable_frozen_benign": audit["usability"]["usable_as_frozen_benign_evaluation"],
        "usable_training": audit["usability"]["usable_as_training_records"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
