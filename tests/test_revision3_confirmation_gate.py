"""Synthetic identity-gate checks; no real confirmation stream is asserted."""
from __future__ import annotations

import hashlib
import json

import pytest

from scripts import revision3_validate_confirmation as gate


def h(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def make_manifest(tmp_path):
    stages = []
    for index in range(3):
        name = f"S{index + 1}"
        splits = {}
        for split in ("train", "val", "test"):
            path = tmp_path / f"{name}_{split}.jsonl"
            records = []
            for label in (0, 1):
                token = f"synthetic-{name}-{split}-{label}"
                records.append({"example_id": token, "task_id": name,
                                "split": split, "label": label,
                                "text_sha256": h(token),
                                "normalized_text_sha256": h(token + "-norm"),
                                "model_view_sha256": h(token + "-view"),
                                "source_native_id": token + "-native",
                                "duplicate_group_id": token + "-group",
                                "base_document_id": token + "-parent",
                                "template_id": token + "-template"})
            path.write_text("".join(json.dumps(x) + "\n" for x in records), encoding="utf-8")
            splits[split] = {"path": str(path), "sha256": gate.sha(path),
                             "label_counts": {"0": 1, "1": 1}}
        stages.append({"name": name, "attack_shift": f"synthetic-attack-{index}",
                       "benign_shift": f"synthetic-benign-{index}", "splits": splits})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"source": "synthetic-test-only", "revision": "v1",
                                    "license": "synthetic", "previously_explored": False,
                                    "design_frozen_at_utc": "2026-01-01T00:00:00Z",
                                    "stages": stages}), encoding="utf-8")
    return manifest


def test_disjoint_synthetic_manifest_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "candidate_b_tests", lambda: [])
    result = gate.validate(make_manifest(tmp_path))
    assert result["status"] == "PASS"
    assert result["counts"] == {"train": 6, "val": 6, "test": 6}


def test_candidate_test_hash_overlap_fails(tmp_path, monkeypatch):
    manifest = make_manifest(tmp_path)
    first = json.loads((tmp_path / "S1_test.jsonl").read_text(encoding="utf-8").splitlines()[0])
    monkeypatch.setattr(gate, "candidate_b_tests", lambda: [first])
    with pytest.raises(ValueError, match="overlap"):
        gate.validate(manifest)


def test_unfrozen_source_fails(tmp_path, monkeypatch):
    manifest = make_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["previously_explored"] = True
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(gate, "candidate_b_tests", lambda: [])
    with pytest.raises(ValueError, match="prospective"):
        gate.validate(manifest)


def test_changed_input_hash_fails(tmp_path, monkeypatch):
    manifest = make_manifest(tmp_path)
    path = tmp_path / "S2_val.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    monkeypatch.setattr(gate, "candidate_b_tests", lambda: [])
    with pytest.raises(ValueError, match="hash mismatch"):
        gate.validate(manifest)
