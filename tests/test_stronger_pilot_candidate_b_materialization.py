import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import materialize_stronger_pilot_candidate_b as m


def row(stage, split, label, index, arm="B-DOC", benign_group="B0"):
    example_id = f"e-{arm}-{stage}-{split}-{label}-{index}"
    result = {
        "text": "fixture text",
        "label": label,
        "task_id": stage,
        "split": split,
        "source": {"dataset": "fixture", "revision": "r", "upstream_split": split, "record_id": example_id, "construction": "fixture"},
        "example_id": example_id,
        "attack_family": "none" if label == 0 else m.FAMILY_NAMES[stage],
        "domain": "fixture",
        "base_document_id": "base-" + example_id,
        "injection_id": None if label == 0 else "inj-" + example_id,
        "injection_position": "none" if label == 0 else "head",
        "text_sha256": "text-" + example_id,
        "normalized_text_sha256": "norm-" + example_id,
        "model_view_sha256": "model-" + example_id,
        "source_native_id": "native-" + example_id,
        "source_parent_id": "parent-" + example_id,
        "exact_group": "exact-" + example_id,
        "normalized_group": "ng-" + example_id,
        "model_view_group": "mg-" + example_id,
        "project_semantic_parent_group": "semantic-" + example_id,
        "project_template_group": "template-" + example_id,
        "arm": arm,
    }
    if label == 0:
        result.update(benign_group=benign_group, benign_partition=f"{benign_group}-1")
    return result


def fixture_arm(arm="B-DOC", quotas=None):
    quotas = quotas or {"train": 2, "val": 1, "test": 1}
    return [row(stage, split, label, i, arm) for stage in m.FAMILIES.values() for split, count in quotas.items() for label in (0, 1) for i in range(count)]


def matched_fixture_arms():
    arms = {name: fixture_arm(name) for name in ("B-DOC", "B-SHORT", "B-HARD")}
    for arm in arms.values():
        for item in arm:
            if item["label"] == 1:
                item["example_id"] = item["example_id"].replace(item["arm"], "SHARED")
                item["text_sha256"] = "text-" + item["example_id"]
                item["base_document_id"] = "base-" + item["example_id"]
                item["injection_id"] = "inj-" + item["example_id"]
                item["source_native_id"] = "native-" + item["example_id"]
                item["template_id"] = "tmpl-" + item["example_id"]
                item["normalized_template_hash"] = "nth-" + item["example_id"]
    for name in ("B-SHORT", "B-HARD"):
        for item in arms[name]:
            if item["label"] == 0 and item["task_id"] == "AF2":
                item["example_id"] = item["example_id"].replace(name, "SHARED-D0")
                item["text_sha256"] = "text-" + item["example_id"]
                item["benign_partition"] = "D0-1"
    for item in arms["B-DOC"]:
        if item["label"] == 0 and item["task_id"] == "AF2":
            item["example_id"] = item["example_id"].replace("B-DOC", "SHARED-D0")
            item["text_sha256"] = "text-" + item["example_id"]
            item["benign_partition"] = "D0-1"
    for item in arms["B-HARD"]:
        if item["label"] == 0 and item["task_id"] == "AF3":
            item["example_id"] = item["example_id"].replace("B-HARD", "B-SHORT")
            item["text_sha256"] = "text-" + item["example_id"]
            item["benign_partition"] = "B0-2"
    return arms


def test_deterministic_materialization(tmp_path):
    rows = [{"example_id": "b", "v": 2}, {"example_id": "a", "v": 1}]
    path = tmp_path / "x.jsonl"
    m.write_jsonl(path, rows)
    first = path.read_bytes()
    m.write_jsonl(path, list(reversed(rows)))
    assert path.read_bytes() == first


def test_exact_count_contract():
    quotas = {"train": 2, "val": 1, "test": 1}
    counts = m.validate_arm_contract(fixture_arm(quotas=quotas), quotas)
    assert len(counts) == 24


def test_source_hash_mismatch_rejected(tmp_path):
    path = tmp_path / "source"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash mismatch"):
        m.verify_file_hash(path, "0" * 64)


def test_invalid_label_rejected():
    rows = fixture_arm()
    rows[0]["label"] = 2
    with pytest.raises(ValueError, match="invalid labels"):
        m.validate_labels(rows)


def test_malicious_cross_arm_mismatch_rejected(monkeypatch):
    arms = matched_fixture_arms()
    monkeypatch.setattr(m, "verify_matched_arms", m.verify_matched_arms)
    # The production-sized overlap check is reached only after malicious equality.
    next(r for r in arms["B-HARD"] if r["label"] == 1)["injection_position"] = "tail"
    with pytest.raises(ValueError, match="malicious cross-arm mismatch"):
        m.verify_matched_arms(arms)


def test_d0_stage1_anchor_mismatch_rejected():
    arms = matched_fixture_arms()
    target = next(r for r in arms["B-HARD"] if r["label"] == 0 and r["task_id"] == "AF2")
    target["example_id"] = "changed-D0"
    with pytest.raises(ValueError, match="D0 Stage-1 anchor mismatch"):
        m.verify_matched_arms(arms)


def test_b0_stage2_anchor_mismatch_rejected():
    arms = matched_fixture_arms()
    target = next(r for r in arms["B-HARD"] if r["label"] == 0 and r["task_id"] == "AF3")
    target["example_id"] = "changed-B0"
    with pytest.raises(ValueError, match="B0 Stage-2 anchor mismatch"):
        m.verify_matched_arms(arms)


def test_prohibited_cross_stage_duplicate_rejected():
    rows = fixture_arm()
    first = next(r for r in rows if r["task_id"] == "AF2")
    later = next(r for r in rows if r["task_id"] == "AF3")
    later["normalized_text_sha256"] = first["normalized_text_sha256"]
    with pytest.raises(ValueError, match="cross-stage/split duplicate"):
        m.validate_isolation(rows)


def test_max_length_validation():
    class Tokenizer:
        def __call__(self, texts, **kwargs):
            return {"input_ids": [list(range(len(text.split()) + 2)) for text in texts]}

    text = "x " * 1300
    token_ids = list(range(1302))
    rows = [{"text": text, "text_sha256": m.sha_text(text), "normalized_text_sha256": m.sha_text(m.source_tools.norm(text)), "model_view_sha256": m.sha_text(",".join(map(str, token_ids))), "deberta_tokens": 1302, "example_id": "long"}]
    with pytest.raises(ValueError, match="max_length"):
        m.audit_token_lengths(rows, Tokenizer())


def test_af5_compositional_overlap_policy():
    rows = []
    counts = {"train": 200, "val": 30, "test": 60}
    for split, count in counts.items():
        for i in range(count):
            rows.append({
                "task_id": "AF5", "label": 1, "split": split,
                "attack_template_components": ["ignore", "escape_separation", "fake_completion"],
                "template_id": f"af5-{split}-{i}", "text_sha256": f"af5-text-{split}-{i}",
                "base_document_id": f"af5-base-{split}-{i}", "injection_id": f"af5-inj-{split}-{i}",
                "normalized_template_hash": f"af5-norm-template-{split}-{i}",
            })
    for stage in ("AF2", "AF3", "AF4"):
        rows.append({
            "task_id": stage, "label": 1, "split": "train", "template_id": f"{stage}-template",
            "text_sha256": f"{stage}-text", "base_document_id": f"{stage}-base",
            "injection_id": f"{stage}-inj", "normalized_template_hash": f"{stage}-norm-template",
        })
    result = m.validate_af5_policy(rows)
    assert result["intrinsic_component_reuse_allowed"] is True
    assert result["interpretation"] == "compositional/combined; not family-disjoint"


def test_no_detector_guided_selection_dependency():
    authority = {"selection_inputs": "source metadata only; no detector predictions or losses", "records": [{"example_id": "a"}]}
    assert m.verify_no_detector_guided_selection([authority]) is True
    bad = copy.deepcopy(authority)
    bad["records"][0]["prediction"] = 1
    with pytest.raises(ValueError, match="detector-guided"):
        m.verify_no_detector_guided_selection([authority, bad])


def test_materialized_artifacts_when_present():
    target = ROOT / "data/processed/stronger_pilot_candidate_b"
    if not target.exists():
        pytest.skip("materialization has not run yet")
    manifest = json.loads((target / "materialization_manifest.json").read_text(encoding="utf-8"))
    report_path = ROOT / manifest["validation_artifact"]
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == manifest["validation_artifact_sha256"]
    assert len(manifest["files"]) == 36
    assert manifest["totals"]["arm_memberships"] == 6960
    for item in manifest["files"]:
        path = ROOT / item["relative_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        assert sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) == item["row_count"]
