"""Fail-closed source, split, and Candidate B overlap gate for a future stream.

This checks a frozen manifest only; it never trains or picks a stream.  The
current project has no qualifying manifest, so confirmation remains unrun.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
ALLOWED = (ROOT.resolve(), Path(r"Z:\ContinualGuard_Revision3").resolve())
FIELDS = ("example_id", "text_sha256", "normalized_text_sha256", "model_view_sha256",
          "source_native_id", "duplicate_group_id", "base_document_id", "template_id")


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def safe_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if not any(path == base or base in path.parents for base in ALLOWED):
        raise ValueError("Confirmation input is outside the authorized E/Z project folders")
    return path


def sets(records: list[dict]) -> dict[str, set[str]]:
    return {field: {str(r[field]) for r in records if r.get(field) is not None}
            for field in FIELDS}


def disjoint(a: list[dict], b: list[dict], context: str) -> None:
    left, right = sets(a), sets(b)
    overlaps = {field: len(left[field] & right[field]) for field in FIELDS}
    if any(overlaps.values()):
        raise ValueError(f"Identity/hash/parent overlap in {context}: {overlaps}")


def candidate_b_tests():
    result = []
    for arm in ("b_doc", "b_short", "b_hard"):
        for stage in range(1, 5):
            result.extend(rows(ROOT / "data/processed/stronger_pilot_candidate_b" /
                               arm / f"stage{stage}/test.jsonl"))
    return result


def validate(manifest_path: Path) -> dict:
    manifest = json.loads(safe_path(str(manifest_path)).read_text(encoding="utf-8"))
    if manifest.get("previously_explored") is not False:
        raise ValueError("Confirmation source is not genuinely prospective")
    if not all(manifest.get(key) for key in ("source", "revision", "license", "design_frozen_at_utc")):
        raise ValueError("Source, revision, license, and freeze time are required")
    stages = manifest.get("stages", [])
    if len(stages) < 3 or len({s["name"] for s in stages}) != len(stages):
        raise ValueError("At least three distinct sequential stages are required")
    frozen_tests = candidate_b_tests()
    split_rows = {split: [] for split in ("train", "val", "test")}
    file_hashes = {}
    for stage in stages:
        if not stage.get("attack_shift") or not stage.get("benign_shift"):
            raise ValueError("Each stage needs a predefined malicious and benign shift")
        for split in split_rows:
            specification = stage["splits"][split]
            path = safe_path(specification["path"])
            digest = sha(path)
            if digest != specification["sha256"]:
                raise ValueError(f"Frozen file hash mismatch: {path}")
            batch = rows(path)
            labels = Counter(row.get("label") for row in batch)
            if labels != Counter({int(k): v for k, v in specification["label_counts"].items()}):
                raise ValueError(f"Frozen label/count mismatch: {path}")
            for row in batch:
                if row.get("split") != split or row.get("task_id") != stage["name"]:
                    raise ValueError("Stage/split identity mismatch")
                if row.get("label") not in (0, 1) or any(not row.get(f) for f in FIELDS[:4]):
                    raise ValueError("Binary label and four stable hash fields required")
            disjoint(batch, frozen_tests, f"Candidate B TEST vs {stage['name']}/{split}")
            split_rows[split].extend(batch)
            file_hashes[str(path)] = digest
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        disjoint(split_rows[a], split_rows[b], f"confirmation {a}/{b}")
    # No repeated ID or exact/model-view text across stages within a split.
    for split, batch in split_rows.items():
        for field in FIELDS[:4]:
            values = [str(row[field]) for row in batch]
            if len(set(values)) != len(values):
                raise ValueError(f"Repeated {field} within confirmation {split}")
    return {"status": "PASS", "manifest_sha256": sha(safe_path(str(manifest_path))),
            "stage_count": len(stages), "counts": {k: len(v) for k, v in split_rows.items()},
            "input_sha256": file_hashes,
            "interpretation": "Pass is an identity/hash gate, not proof against semantic-neighbor overlap"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    arguments = parser.parse_args()
    result = validate(arguments.manifest)
    output = ROOT / "reports/revision3/confirmation_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
