"""Materialize the frozen Candidate B hash assignments locally; never train."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import audit_h2_mini_sources as source_tools

EXPECTED_ROOT = Path(r"E:\ContinualGuard")
TOKENIZER_NAME = "microsoft/deberta-v3-base"
TOKENIZER_REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
MAX_LENGTH = 1280
CHEN_REVISION = "1e9e0f00179adc154e9ca991b9cd3206c99f9588"
OASST_REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
OR_REVISION = "e36d8b80e81837c8a8f264bbb2a49f1b32c7e272"
NOTINJECT_REVISION = "847ae76cf8fea5ed325429e569ae8cfef022d2e0"
POLICY = "candidate-b-hash-assignment-v1-20261005"
SALT = "stronger-pilot-candidate-b-v1-20261005"

MROOT = ROOT / "data/manifests/stronger_pilot"
MAL = MROOT / "stronger_shared_malicious_assignment.json"
BDOC = MROOT / "stronger_b_doc_assignment_proposal.json"
BSHORT = MROOT / "stronger_b_short_assignment_proposal.json"
BHARD = MROOT / "stronger_b_hard_assignment_proposal.json"
ASSIGN_VALID = MROOT / "stronger_assignment_validation.json"
CAPACITY = ROOT / "results/stronger_pilot_capacity_audit.json"
RESULT_ASSIGN_VALID = ROOT / "results/stronger_pilot_assignment_validation.json"
B0_AUTHORITY = ROOT / "data/manifests/hard_benign/B0_final_accepted.json"
BH_AUTHORITY = ROOT / "data/manifests/hard_benign/BH_final_eligible.json"
BSHORT_AUTHORITY = ROOT / "data/manifests/hard_benign/b_short_final_hash_manifest.json"
NOTINJECT_SOURCE = ROOT / "data/manifests/hard_benign/notinject/source.json"

TARGET = ROOT / "data/processed/stronger_pilot_candidate_b"
TMP_NAME = "_tmp_materialization"
PUBLISH_CONTAINER = ROOT / "data/processed/_publish_stronger_pilot_candidate_b"
RESULT = ROOT / "results/stronger_pilot_materialization_validation.json"
RESULT_TMP = ROOT / "results/_tmp_stronger_pilot_materialization_validation.json"
SPLIT_FILE = {"train": "train", "validation": "val", "test": "test"}
FAMILIES = {1: "AF2", 2: "AF3", 3: "AF4", 4: "AF5"}
FAMILY_NAMES = {"AF2": "ignore", "AF3": "escape_separation", "AF4": "fake_completion", "AF5": "combined"}

APPROVED_AUTHORITY_HASHES = {
    "data/manifests/stronger_pilot/stronger_shared_malicious_assignment.json": "211f986ef0d2c479d4198c65f6b4ab59789a77aa70d860af2ab0f02dd2f6d6ad",
    "data/manifests/stronger_pilot/stronger_b_doc_assignment_proposal.json": "b2aae59c201c7302bb85b9da79bc5b841fe7de94490fc6660b30d2aa9ba05f6b",
    "data/manifests/stronger_pilot/stronger_b_short_assignment_proposal.json": "e4ecda4fa312976de0692287444da138b011cac93f9e833a4de6cd4e7625ade0",
    "data/manifests/stronger_pilot/stronger_b_hard_assignment_proposal.json": "43a10c9de0144ec2f8f6fd5e5bf33eff7390a9f573f4217ce24957acf192c841",
    "data/manifests/stronger_pilot/stronger_assignment_validation.json": "a69f3deaa0bee1dfa4004c65ff348b7bd74d4223e4dd082578783445afeb2019",
    "results/stronger_pilot_assignment_validation.json": "a69f3deaa0bee1dfa4004c65ff348b7bd74d4223e4dd082578783445afeb2019",
    "results/stronger_pilot_capacity_audit.json": "22ae1edad72c910501f37693436f1aa57c7def820fa0467dbac74714014e32d4",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file_hash(path: Path, expected: str) -> None:
    actual = file_sha(path)
    if actual != expected:
        try:
            display = rel(path)
        except ValueError:
            display = str(path)
        raise ValueError(f"source hash mismatch: {display} expected={expected} actual={actual}")


def canonical_line(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(canonical_line(row) for row in sorted(rows, key=lambda r: r["example_id"]))
    path.write_text(content, encoding="utf-8", newline="\n")


def percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def length_stats(values: list[int]) -> dict:
    return {
        "count": len(values),
        "min": min(values),
        "median": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "count_gt_1280": sum(x > MAX_LENGTH for x in values),
        "truncation_required": sum(x > MAX_LENGTH for x in values),
    }


def verify_no_detector_guided_selection(authorities: list[dict]) -> bool:
    forbidden = {"prediction", "predictions", "margin", "loss", "score", "logit", "checkpoint"}

    def keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield str(key).casefold()
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    shared = authorities[0]
    statement = shared.get("selection_inputs", "").casefold()
    if "no detector predictions or losses" not in statement:
        raise ValueError("missing prediction-independent selection declaration")
    for authority in authorities:
        bad = sorted(k for k in keys(authority.get("records", [])) if k in forbidden)
        if bad:
            raise ValueError(f"detector-guided selection metadata found: {bad[:5]}")
    return True


def verify_authority():
    if ROOT.resolve() != EXPECTED_ROOT.resolve():
        raise RuntimeError(f"active root must be {EXPECTED_ROOT}; got {ROOT}")
    expected_python = ROOT / ".venv/Scripts/python.exe"
    if Path(sys.executable).resolve() != expected_python.resolve():
        raise RuntimeError(f"active interpreter must be {expected_python}; got {sys.executable}")
    for path, digest in APPROVED_AUTHORITY_HASHES.items():
        verify_file_hash(ROOT / path, digest)

    validation = load(ASSIGN_VALID)
    result_validation = load(RESULT_ASSIGN_VALID)
    capacity = load(CAPACITY)
    if validation != result_validation:
        raise ValueError("assignment validation copies differ")
    if validation.get("status") != "READY_FOR_HUMAN_MATERIALIZATION_APPROVAL" or not validation.get("all_checks_pass"):
        raise ValueError("assignment validation is not approved")
    if validation.get("policy") != POLICY or validation.get("deterministic_salt") != SALT:
        raise ValueError("assignment policy or salt changed")
    for path, digest in validation["proposal_hashes"].items():
        verify_file_hash(ROOT / path, digest)
    for path, digest in capacity["input_manifest_hashes"].items():
        verify_file_hash(ROOT / path, digest)
    for source in capacity["source_inputs"].values():
        verify_file_hash(ROOT / source["path"], source["sha256"])
        for item in source.get("raw_files", []):
            verify_file_hash(ROOT / item["path"], item["sha256"])

    expected_revisions = {
        "Chen_processed": CHEN_REVISION,
        "OASST1": OASST_REVISION,
        "OR-Bench": OR_REVISION,
        "NotInject": NOTINJECT_REVISION,
    }
    for name, revision in expected_revisions.items():
        if capacity["source_inputs"][name]["revision"] != revision:
            raise ValueError(f"pinned revision mismatch: {name}")
    return validation, capacity


def authority_equivalent(meta: dict, authority: dict, fields: list[str], kind: str) -> None:
    for field in fields:
        if meta.get(field) != authority.get(field):
            raise ValueError(f"{kind} authority mismatch {field}: {meta.get('project_record_id')}")


def load_source_rows(all_meta: list[dict], capacity: dict):
    wanted_chen = {r["source_native_id"] for r in all_meta if r.get("source") not in {"OASST1", "OR-Bench"}}
    chen_path = ROOT / capacity["source_inputs"]["Chen_processed"]["path"]
    chen = {}
    with chen_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("example_id") in wanted_chen:
                chen[row["example_id"]] = row
    if set(chen) != wanted_chen:
        raise ValueError(f"missing Chen rows: {len(wanted_chen - set(chen))}")

    wanted_oa = {r["source_row_index"] for r in all_meta if r.get("source") == "OASST1"}
    oa_path = ROOT / capacity["source_inputs"]["OASST1"]["path"]
    oasst = {}
    with gzip.open(oa_path, "rt", encoding="utf-8", errors="strict") as stream:
        for index, line in enumerate(stream):
            if index in wanted_oa:
                oasst[index] = json.loads(line)
    if set(oasst) != wanted_oa:
        raise ValueError(f"missing OASST rows: {len(wanted_oa - set(oasst))}")

    wanted_or = {r["source_row_index"] for r in all_meta if r.get("source") == "OR-Bench"}
    or_path = ROOT / capacity["source_inputs"]["OR-Bench"]["path"]
    or_rows = {}
    with or_path.open(encoding="utf-8-sig", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            if index in wanted_or:
                or_rows[index] = row
    if set(or_rows) != wanted_or:
        raise ValueError(f"missing OR-Bench rows: {len(wanted_or - set(or_rows))}")
    return chen, oasst, or_rows


def verify_malicious_meta(meta: dict, src: dict) -> None:
    expected = {
        "example_id": meta["source_native_id"],
        "base_document_id": meta["base_document_id"],
        "injection_id": meta["injection_id"],
        "task_id": meta["attack_family_id"],
        "attack_family": meta["attack_family"],
        "injection_position": meta["injection_position"],
        "split": SPLIT_FILE[meta["split"]],
        "text_sha256": meta["raw_text_sha256"],
    }
    for field, value in expected.items():
        if src.get(field) != value:
            raise ValueError(f"malicious source mismatch {field}: {meta['project_record_id']}")
    if src.get("label") != 1 or meta.get("label") != 1:
        raise ValueError("malicious label mismatch")
    if src["source"].get("record_id") != meta.get("source_record_id"):
        raise ValueError("malicious source-record mismatch")
    if src["source"].get("revision") != meta.get("source_revision") or meta["source_revision"] != CHEN_REVISION:
        raise ValueError("malicious revision mismatch")
    if src["payload_source"].get("record_id") != meta.get("payload_source_record_id"):
        raise ValueError("malicious payload identity mismatch")
    if src["payload_source"].get("revision") != meta.get("payload_source_revision"):
        raise ValueError("malicious payload revision mismatch")
    if src.get("template_id") != meta.get("attack_template_id"):
        raise ValueError("malicious complete-template mismatch")
    if src.get("normalized_template_hash") != meta.get("normalized_template_hash"):
        raise ValueError("malicious normalized-template mismatch")
    if sorted(src.get("attack_template_components", [])) != sorted(meta.get("attack_template_components", [])):
        raise ValueError("malicious component mismatch")
    if sha_text(src["base_document_text"]) != meta.get("base_document_sha256"):
        raise ValueError("malicious base-document text hash mismatch")
    if sha_text(src["injection_text"]) != meta.get("injection_sha256"):
        raise ValueError("malicious injection text hash mismatch")


def malicious_record(meta: dict, src: dict, arm: str) -> dict:
    verify_malicious_meta(meta, src)
    row = dict(src)
    row.update(
        example_id=meta["project_record_id"],
        assignment_project_record_id=meta["project_record_id"],
        source_native_id=meta["source_native_id"],
        source_record_id=meta["source_record_id"],
        stage=meta["stage"],
        task_id=meta["attack_family_id"],
        split=SPLIT_FILE[meta["split"]],
        arm=arm,
        text_sha256=meta["raw_text_sha256"],
        normalized_text_sha256=meta["normalized_text_sha256"],
        model_view_sha256=meta["model_view_sha256"],
        deberta_tokens=meta["deberta_tokens"],
        template_id=meta["attack_template_id"],
        attack_template_id=meta["attack_template_id"],
        normalized_template_hash=meta["normalized_template_hash"],
    )
    return row


def verify_d0_meta(meta: dict, src: dict) -> None:
    expected = {
        "example_id": meta["source_native_id"],
        "base_document_id": meta["base_document_id"],
        "label": 0,
        "attack_family": "none",
        "injection_id": None,
        "injection_position": "none",
        "split": SPLIT_FILE[meta["split"]],
        "text_sha256": meta["raw_text_sha256"],
    }
    for field, value in expected.items():
        if src.get(field) != value:
            raise ValueError(f"D0 source mismatch {field}: {meta['project_record_id']}")
    if src["source"].get("record_id") != meta.get("source_document_record_id"):
        raise ValueError("D0 source-document mismatch")
    if src["source"].get("revision") != meta.get("source_revision") or meta["source_revision"] != CHEN_REVISION:
        raise ValueError("D0 revision mismatch")
    if src.get("text") != src.get("base_document_text") or src.get("injection_text") is not None:
        raise ValueError("D0 contains injected or rewritten text")
    if meta.get("annotation_status") != "chen_source_deterministic_clean_companion_previously_audited":
        raise ValueError("D0 audit status mismatch")


def d0_record(meta: dict, src: dict, arm: str) -> dict:
    verify_d0_meta(meta, src)
    row = dict(src)
    row.update(
        example_id=meta["project_record_id"],
        assignment_project_record_id=meta["project_record_id"],
        source_native_id=meta["source_native_id"],
        source_document_record_id=meta["source_document_record_id"],
        stage=meta["stage"],
        task_id=FAMILIES[meta["stage"]],
        split=SPLIT_FILE[meta["split"]],
        arm=arm,
        text_sha256=meta["raw_text_sha256"],
        normalized_text_sha256=meta["normalized_text_sha256"],
        model_view_sha256=meta["model_view_sha256"],
        deberta_tokens=meta["deberta_tokens"],
        benign_group="D0",
        benign_partition=meta["benign_partition"],
        paired_malicious_project_record_id=meta["paired_malicious_project_record_id"],
        paired_malicious_source_native_id=meta["paired_malicious_source_native_id"],
    )
    return row


BENIGN_AUTHORITY_FIELDS = [
    "project_record_id", "review_id", "candidate_id", "source", "source_revision",
    "source_native_id", "source_parent_id", "source_immediate_parent_id", "source_row_index",
    "language", "raw_text_sha256", "normalized_text_sha256", "model_view_sha256",
    "exact_group", "normalized_group", "model_view_group", "project_semantic_parent_group",
    "project_template_group", "label",
]


def benign_reviewed_record(meta: dict, authority: dict, src: dict, arm: str, group: str) -> dict:
    authority_equivalent(meta, authority, BENIGN_AUTHORITY_FIELDS, group)
    if authority.get("label") != 0 or not authority.get("human_review_status"):
        raise ValueError(f"{group} lacks accepted review provenance")
    if group == "B0":
        if authority.get("accepted_stratum") != "B0":
            raise ValueError("B0 authority is not accepted B0")
        if meta.get("source_revision") != OASST_REVISION:
            raise ValueError("B0 revision mismatch")
        identity = (src.get("message_id"), src.get("message_tree_id"), src.get("parent_id"))
        expected = (meta.get("source_native_id"), meta.get("source_parent_id"), meta.get("source_immediate_parent_id"))
        if identity != expected:
            raise ValueError(f"B0 OASST identity mismatch: {meta['project_record_id']}")
        text = src["text"]
        source_record_id = meta["source_native_id"]
        domain = "oasst1_ordinary_benign"
        base_document_id = meta["source_parent_id"] or meta["project_record_id"]
    elif group == "BH":
        if meta.get("source_revision") != OR_REVISION:
            raise ValueError("BH revision mismatch")
        if not authority.get("post_review_audit_status", "").startswith("passed"):
            raise ValueError("BH authority did not pass review audit")
        text = src["prompt"]
        source_record_id = meta["candidate_id"]
        domain = "or_bench_hard_benign"
        base_document_id = meta["project_record_id"]
    else:
        raise ValueError(f"unsupported benign group {group}")
    if sha_text(text) != meta["raw_text_sha256"]:
        raise ValueError(f"{group} raw hash mismatch: {meta['project_record_id']}")
    row = {
        "text": text,
        "label": 0,
        "task_id": FAMILIES[meta["stage"]],
        "split": SPLIT_FILE[meta["split"]],
        "source": {
            "dataset": meta["source"],
            "revision": meta["source_revision"],
            "upstream_split": "reviewed_local_source",
            "record_id": source_record_id,
            "construction": "Candidate B frozen reviewed benign materialization",
        },
        "example_id": meta["project_record_id"],
        "assignment_project_record_id": meta["project_record_id"],
        "stage": meta["stage"],
        "arm": arm,
        "attack_family": "none",
        "domain": domain,
        "base_document_id": base_document_id,
        "injection_id": None,
        "injection_position": "none",
        "text_sha256": meta["raw_text_sha256"],
        "normalized_text_sha256": meta["normalized_text_sha256"],
        "model_view_sha256": meta["model_view_sha256"],
        "deberta_tokens": meta["deberta_tokens"],
        "language": meta["language"],
        "benign_group": group,
        "benign_partition": meta["benign_partition"],
        "annotation_status": "human_reviewed_final_hash_assignment",
        "review_id": meta["review_id"],
        "review_coverage": meta.get("review_coverage"),
        "review_status": authority["human_review_status"],
        "review_evidence_sha256": sha_text(json.dumps(authority.get("reviewer_evidence", []), ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
        "source_native_id": meta.get("source_native_id"),
        "source_parent_id": meta.get("source_parent_id"),
        "source_immediate_parent_id": meta.get("source_immediate_parent_id"),
        "source_row_index": meta["source_row_index"],
        "duplicate_group_id": meta["exact_group"],
        "exact_group": meta["exact_group"],
        "normalized_group": meta["normalized_group"],
        "model_view_group": meta["model_view_group"],
        "project_semantic_parent_group": meta["project_semantic_parent_group"],
        "project_template_group": meta["project_template_group"],
    }
    if group == "BH":
        row.update(
            source_category=meta.get("source_category"),
            historical_descriptive_stratum=meta.get("historical_proposed_stratum"),
            cue_family_ids=meta.get("cue_family_ids", []),
        )
    return row


def build_arm(proposal: dict, chen: dict, oasst: dict, or_rows: dict, b0_auth: dict, bh_auth: dict) -> list[dict]:
    arm = proposal["arm"]
    rows = []
    for meta in proposal["records"]:
        if meta["label"] == 1:
            rows.append(malicious_record(meta, chen[meta["source_native_id"]], arm))
        elif meta.get("benign_partition", "").startswith("D0-"):
            rows.append(d0_record(meta, chen[meta["source_native_id"]], arm))
        elif meta.get("benign_partition", "").startswith("B0-"):
            rows.append(benign_reviewed_record(meta, b0_auth[meta["project_record_id"]], oasst[meta["source_row_index"]], arm, "B0"))
        elif meta.get("benign_partition", "").startswith("BH-"):
            rows.append(benign_reviewed_record(meta, bh_auth[meta["project_record_id"]], or_rows[meta["source_row_index"]], arm, "BH"))
        else:
            raise ValueError(f"unknown assignment record: {meta.get('project_record_id')}")
    return rows


def validate_labels(rows: list[dict]) -> bool:
    bad = [r.get("example_id") for r in rows if type(r.get("label")) is not int or r["label"] not in (0, 1)]
    if bad:
        raise ValueError(f"invalid labels: {bad[:5]}")
    for row in rows:
        if row["label"] == 0 and (row["attack_family"] != "none" or row["injection_id"] is not None or row["injection_position"] != "none"):
            raise ValueError(f"invalid benign label contract: {row['example_id']}")
        if row["label"] == 1 and (row["attack_family"] == "none" or not row["injection_id"] or row["injection_position"] == "none"):
            raise ValueError(f"invalid malicious label contract: {row['example_id']}")
    return True


def validate_arm_contract(rows: list[dict], quotas=None) -> Counter:
    quotas = quotas or {"train": 200, "val": 30, "test": 60}
    validate_labels(rows)
    ids = [r["example_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate example_id within arm")
    counts = Counter((r["task_id"], r["split"], r["label"]) for r in rows)
    for stage in FAMILIES.values():
        for split, count in quotas.items():
            for label in (0, 1):
                if counts[(stage, split, label)] != count:
                    raise ValueError(f"exact count mismatch {stage}/{split}/{label}")
    if len(rows) != 2320 and quotas == {"train": 200, "val": 30, "test": 60}:
        raise ValueError("arm total mismatch")
    return counts


def location(row: dict):
    return row["task_id"], row["split"]


def validate_isolation(rows: list[dict]) -> dict:
    checks = [
        "text_sha256", "normalized_text_sha256", "model_view_sha256", "source_native_id",
        "source_parent_id", "exact_group", "normalized_group", "model_view_group",
        "project_semantic_parent_group", "project_template_group", "injection_id",
    ]
    for field in checks:
        seen = {}
        for row in rows:
            value = row.get(field)
            if value is None:
                continue
            scoped = (row["source"]["dataset"], row["source"]["revision"], value) if field.startswith("source_") or field.endswith("group") else value
            old = seen.get(scoped)
            if old and old != location(row):
                raise ValueError(f"prohibited cross-stage/split duplicate {field}: {scoped}")
            seen[scoped] = location(row)

    by_base = defaultdict(list)
    for row in rows:
        by_base[row["base_document_id"]].append(row)
    intentional_pairs = 0
    for base_id, members in by_base.items():
        locs = {location(r) for r in members}
        if len(locs) > 1:
            raise ValueError(f"base document crosses stage/split: {base_id}")
        if len(members) > 1:
            if len(members) != 2 or {r["label"] for r in members} != {0, 1}:
                raise ValueError(f"undeclared base-document reuse: {base_id}")
            benign = next(r for r in members if r["label"] == 0)
            malicious = next(r for r in members if r["label"] == 1)
            if benign.get("benign_group") != "D0" or benign.get("paired_malicious_project_record_id") != malicious["example_id"]:
                raise ValueError(f"invalid D0 malicious pairing: {base_id}")
            intentional_pairs += 1
    return {"cross_stage_or_split_collisions": 0, "intentional_D0_base_pairs": intentional_pairs}


def malicious_signature(row: dict):
    return (
        row["example_id"], row["text_sha256"], row["attack_family"], row["injection_position"],
        row["label"], row["task_id"], row["split"], row["base_document_id"], row["injection_id"],
        row.get("template_id"), row.get("normalized_template_hash"),
    )


def benign_signature(row: dict):
    return row["example_id"], row["text_sha256"], row["label"], row["task_id"], row["split"], row.get("benign_partition")


def verify_matched_arms(arms: dict[str, list[dict]]) -> dict:
    malicious = {name: {malicious_signature(r) for r in rows if r["label"] == 1} for name, rows in arms.items()}
    if len({frozenset(v) for v in malicious.values()}) != 1:
        raise ValueError("malicious cross-arm mismatch")
    stage1 = {name: {benign_signature(r) for r in rows if r["label"] == 0 and r["task_id"] == "AF2"} for name, rows in arms.items()}
    if len({frozenset(v) for v in stage1.values()}) != 1:
        raise ValueError("D0 Stage-1 anchor mismatch")
    stage2_short = {benign_signature(r) for r in arms["B-SHORT"] if r["label"] == 0 and r["task_id"] == "AF3"}
    stage2_hard = {benign_signature(r) for r in arms["B-HARD"] if r["label"] == 0 and r["task_id"] == "AF3"}
    if stage2_short != stage2_hard:
        raise ValueError("B0 Stage-2 anchor mismatch")

    ids = {name: {r["example_id"] for r in rows if r["label"] == 0} for name, rows in arms.items()}
    overlaps = {
        "B-DOC__B-SHORT": len(ids["B-DOC"] & ids["B-SHORT"]),
        "B-DOC__B-HARD": len(ids["B-DOC"] & ids["B-HARD"]),
        "B-SHORT__B-HARD": len(ids["B-SHORT"] & ids["B-HARD"]),
    }
    if overlaps != {"B-DOC__B-SHORT": 290, "B-DOC__B-HARD": 290, "B-SHORT__B-HARD": 580}:
        raise ValueError(f"undeclared cross-arm benign sharing: {overlaps}")
    return {
        "malicious_records_per_arm": 1160,
        "malicious_exact_match": True,
        "shared_stage1_D0_records": 290,
        "shared_stage2_B0_records": 290,
        "benign_cross_arm_identity_overlap": overlaps,
        "expected_overlap_only": True,
    }


def validate_af5_policy(rows: list[dict]) -> dict:
    af5 = [r for r in rows if r["task_id"] == "AF5" and r["label"] == 1]
    if len(af5) != 290:
        raise ValueError("AF5 count mismatch")
    required = {"ignore", "escape_separation", "fake_completion"}
    if any(set(r.get("attack_template_components", [])) != required for r in af5):
        raise ValueError("AF5 compositional component mismatch")
    owners = defaultdict(set)
    for row in af5:
        owners[row["template_id"]].add(row["split"])
    if any(len(splits) > 1 for splits in owners.values()):
        raise ValueError("AF5 complete-template cross-split leakage")
    prior = [r for r in rows if r["task_id"] in {"AF2", "AF3", "AF4"} and r["label"] == 1]
    for field in ["text_sha256", "base_document_id", "injection_id", "template_id", "normalized_template_hash"]:
        if {r.get(field) for r in af5} & {r.get(field) for r in prior}:
            raise ValueError(f"AF5 unintended identity overlap: {field}")
    return {
        "interpretation": "compositional/combined; not family-disjoint",
        "selected_records": 290,
        "records_with_all_three_components": 290,
        "complete_template_cross_split_overlap": 0,
        "exact_identity_overlap_with_AF2_AF4": 0,
        "intrinsic_component_reuse_allowed": True,
    }


def attach_and_verify_model_views(rows: list[dict], meta_by_id: dict[str, dict], tokenizer) -> list[int]:
    hashes, lengths = source_tools.mviews([r["text"] for r in rows], tokenizer)
    for row, model_hash, length in zip(rows, hashes, lengths):
        meta = meta_by_id[row["assignment_project_record_id"]]
        raw_hash = sha_text(row["text"])
        normalized_hash = sha_text(source_tools.norm(row["text"]))
        if raw_hash != meta["raw_text_sha256"]:
            raise ValueError(f"raw text hash mismatch: {row['example_id']}")
        if normalized_hash != meta["normalized_text_sha256"]:
            raise ValueError(f"normalized text hash mismatch: {row['example_id']}")
        if model_hash != meta["model_view_sha256"]:
            raise ValueError(f"model-view hash mismatch: {row['example_id']}")
        if length != meta["deberta_tokens"]:
            raise ValueError(f"token length mismatch: {row['example_id']}")
        row["text_sha256"] = raw_hash
        row["normalized_text_sha256"] = normalized_hash
        row["model_view_sha256"] = model_hash
        row["deberta_tokens"] = length
    return lengths


def validate_schema(rows: list[dict]) -> bool:
    from continualguard.data import validate_record
    for row in rows:
        validate_record(row)
    return True


def audit_token_lengths(rows: list[dict], tokenizer) -> dict:
    hashes, lengths = source_tools.mviews([r["text"] for r in rows], tokenizer)
    for row, model_hash, length in zip(rows, hashes, lengths):
        if sha_text(row["text"]) != row["text_sha256"]:
            raise ValueError(f"post-file raw hash mismatch: {row['example_id']}")
        if sha_text(source_tools.norm(row["text"])) != row["normalized_text_sha256"]:
            raise ValueError(f"post-file normalized hash mismatch: {row['example_id']}")
        if model_hash != row["model_view_sha256"] or length != row["deberta_tokens"]:
            raise ValueError(f"post-file model-view mismatch: {row['example_id']}")
    stats = length_stats(lengths)
    if stats["truncation_required"]:
        raise ValueError("max_length validation failed")
    return stats


def validate_external_independence(arms: dict[str, list[dict]], capacity: dict) -> tuple[dict, dict]:
    ni_source = load(NOTINJECT_SOURCE)
    ni_path = ROOT / ni_source["records_manifest"]
    verify_file_hash(ni_path, ni_source["records_manifest_sha256"])
    ni = read_jsonl(ni_path)
    if len(ni) != 339 or ni_source["repository_revision"] != NOTINJECT_REVISION:
        raise ValueError("NotInject authority mismatch")

    bshort_manifest = load(BSHORT_AUTHORITY)
    bshort_path = ROOT / "data/processed/h2_mini/external/b_short.jsonl"
    bshort = read_jsonl(bshort_path)
    if len(bshort_manifest["records"]) != 120 or len(bshort) != 120:
        raise ValueError("B-short authority mismatch")
    materialized_by_id = {r["assignment_project_record_id"]: r for r in bshort}
    for meta in bshort_manifest["records"]:
        row = materialized_by_id.get(meta["project_record_id"])
        if not row or row["text_sha256"] != meta["raw_text_sha256"] or row["normalized_text_sha256"] != meta["normalized_text_sha256"] or row["model_view_sha256"] != meta["model_view_sha256"]:
            raise ValueError("B-short frozen materialization mismatch")

    unique = {}
    for rows in arms.values():
        for row in rows:
            old = unique.get(row["example_id"])
            sig = (row["text_sha256"], row["normalized_text_sha256"], row["model_view_sha256"])
            if old and old != sig:
                raise ValueError("cross-arm shared identity content mismatch")
            unique[row["example_id"]] = sig
    benchmark = list(unique.values())
    for index, field in enumerate(["text_sha256", "normalized_text_sha256", "model_view_sha256"]):
        bench_values = {sig[index] for sig in benchmark}
        bshort_values = {r[field] for r in bshort}
        if bench_values & bshort_values:
            raise ValueError(f"B-short overlap: {field}")
    ni_raw = {r["prompt_sha256"] for r in ni}
    ni_norm = {r["normalized_prompt_sha256"] for r in ni}
    if ni_raw & {x[0] for x in benchmark} or ni_norm & {x[1] for x in benchmark}:
        raise ValueError("NotInject overlap")

    refs = {
        "status": "FROZEN_EXTERNAL_REFERENCES_ONLY",
        "NotInject": {
            "records": 339,
            "revision": NOTINJECT_REVISION,
            "records_manifest": rel(ni_path),
            "records_manifest_sha256": file_sha(ni_path),
            "role": "frozen external hard-benign diagnostic",
        },
        "B-short": {
            "records": 120,
            "manifest": rel(BSHORT_AUTHORITY),
            "manifest_sha256": file_sha(BSHORT_AUTHORITY),
            "materialized_path": rel(bshort_path),
            "materialized_sha256": file_sha(bshort_path),
            "role": "unmatched descriptive short-form ordinary-benign external control",
        },
        "prohibited_uses": ["training", "validation", "threshold selection", "epoch selection", "hyperparameter selection", "benchmark construction", "replay selection"],
        "raw_text_in_reference_file": False,
    }
    return refs, {"NotInject_records": 339, "B_short_records": 120, "raw_overlap": 0, "normalized_overlap": 0, "model_view_overlap_B_short": 0}


def write_tree(tmp: Path, arms: dict[str, list[dict]]) -> list[dict]:
    directory = {"B-DOC": "b_doc", "B-SHORT": "b_short", "B-HARD": "b_hard"}
    files = []
    for arm, rows in arms.items():
        for stage in range(1, 5):
            task = FAMILIES[stage]
            for split in ("train", "val", "test"):
                subset = [r for r in rows if r["task_id"] == task and r["split"] == split]
                path = tmp / directory[arm] / f"stage{stage}" / f"{split}.jsonl"
                write_jsonl(path, subset)
                counts = Counter(r["label"] for r in subset)
                files.append({
                    "relative_path": f"data/processed/stronger_pilot_candidate_b/{directory[arm]}/stage{stage}/{split}.jsonl",
                    "row_count": len(subset),
                    "byte_size": path.stat().st_size,
                    "sha256": file_sha(path),
                    "arm": arm,
                    "stage": stage,
                    "split": split,
                    "benign_count": counts[0],
                    "malicious_count": counts[1],
                })
    return files


def read_tree(root: Path) -> dict[str, list[dict]]:
    directory = {"B-DOC": "b_doc", "B-SHORT": "b_short", "B-HARD": "b_hard"}
    arms = {}
    for arm, dirname in directory.items():
        rows = []
        for stage in range(1, 5):
            for split in ("train", "val", "test"):
                rows.extend(read_jsonl(root / dirname / f"stage{stage}" / f"{split}.jsonl"))
        arms[arm] = rows
    return arms


def verify_deterministic_files(root: Path, files: list[dict]) -> bool:
    for item in files:
        path = root / Path(item["relative_path"]).relative_to("data/processed/stronger_pilot_candidate_b")
        rows = read_jsonl(path)
        expected = "".join(canonical_line(r) for r in sorted(rows, key=lambda r: r["example_id"])).encode("utf-8")
        if path.read_bytes() != expected:
            raise ValueError(f"nondeterministic serialization: {path}")
        if len(rows) != item["row_count"] or file_sha(path) != item["sha256"]:
            raise ValueError(f"written file integrity mismatch: {path}")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=TARGET)
    args = parser.parse_args(argv)
    if args.output_root.resolve() != TARGET.resolve():
        raise ValueError("only the approved Candidate B output root is allowed")
    if TARGET.exists() or PUBLISH_CONTAINER.exists() or RESULT.exists() or RESULT_TMP.exists():
        raise RuntimeError("materialization target, publish container, or result already exists; refusing overwrite")

    target_created = False
    try:
        assignment_validation, capacity = verify_authority()
        shared = load(MAL)
        proposals = {"B-DOC": load(BDOC), "B-SHORT": load(BSHORT), "B-HARD": load(BHARD)}
        if len(shared["records"]) != 1160 or any(len(p["records"]) != 2320 for p in proposals.values()):
            raise ValueError("approved assignment counts changed")
        for name, proposal in proposals.items():
            if proposal.get("status") != "READY_FOR_HUMAN_MATERIALIZATION_APPROVAL" or proposal.get("arm") != name:
                raise ValueError(f"proposal not approved: {name}")
            if proposal.get("policy") != POLICY or proposal.get("deterministic_salt") != SALT:
                raise ValueError(f"proposal policy changed: {name}")
        verify_no_detector_guided_selection([shared, *proposals.values()])

        b0_records = load(B0_AUTHORITY)["records"]
        bh_records = load(BH_AUTHORITY)["records"]
        b0_auth = {r["project_record_id"]: r for r in b0_records}
        bh_auth = {r["project_record_id"]: r for r in bh_records}
        all_meta = [r for p in proposals.values() for r in p["records"]]
        if any(r.get("benign_partition", "").startswith("B0-") and r["project_record_id"] not in b0_auth for r in all_meta):
            raise ValueError("unapproved B0 assignment detected")
        if any(r.get("benign_partition", "").startswith("BH-") and r["project_record_id"] not in bh_auth for r in all_meta):
            raise ValueError("unapproved BH assignment detected")

        chen, oasst, or_rows = load_source_rows(all_meta, capacity)
        arms = {name: build_arm(proposal, chen, oasst, or_rows, b0_auth, bh_auth) for name, proposal in proposals.items()}
        meta_by_id = {}
        for meta in all_meta:
            old = meta_by_id.get(meta["project_record_id"])
            comparison = {k: v for k, v in meta.items() if k != "arm"}
            if old and {k: v for k, v in old.items() if k != "arm"} != comparison:
                raise ValueError(f"cross-arm assignment metadata mismatch: {meta['project_record_id']}")
            meta_by_id[meta["project_record_id"]] = meta

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoTokenizer
        tokenizer_path = ROOT / "data/raw/huggingface/microsoft_deberta-v3-base" / TOKENIZER_REVISION
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_fast=False)

        all_rows = [row for rows in arms.values() for row in rows]
        attach_and_verify_model_views(all_rows, meta_by_id, tokenizer)
        validate_schema(all_rows)
        for rows in arms.values():
            validate_arm_contract(rows)
            validate_isolation(rows)
        matched = verify_matched_arms(arms)
        af5 = validate_af5_policy(arms["B-DOC"])
        external_refs, external_checks = validate_external_independence(arms, capacity)

        TARGET.mkdir(parents=True)
        target_created = True
        tmp = TARGET / TMP_NAME
        tmp.mkdir()
        files = write_tree(tmp, arms)
        save_json(tmp / "external/references.json", external_refs)
        verify_deterministic_files(tmp, files)

        actual = read_tree(tmp)
        for rows in actual.values():
            validate_schema(rows)
            validate_arm_contract(rows)
            validate_isolation(rows)
        matched_post = verify_matched_arms(actual)
        if matched_post != matched:
            raise ValueError("post-file matched-control result changed")
        af5_post = validate_af5_policy(actual["B-DOC"])
        if af5_post != af5:
            raise ValueError("post-file AF5 policy result changed")

        unique = {}
        for rows in actual.values():
            for row in rows:
                sig = (row["text_sha256"], row["normalized_text_sha256"], row["model_view_sha256"])
                old = unique.get(row["example_id"])
                if old and old[:3] != sig:
                    raise ValueError("shared record changed across arms")
                unique[row["example_id"]] = (*sig, row)
        if len(unique) != 3770:
            raise ValueError(f"unique record count mismatch: {len(unique)}")
        unique_rows = [value[3] for value in unique.values()]
        token_stats = audit_token_lengths(unique_rows, tokenizer)
        expected_tokens = capacity["token_lengths"]["unique_selected"]
        for key in ("count", "min", "median", "p95", "p99", "max", "count_gt_1280"):
            if token_stats[key] != expected_tokens[key]:
                raise ValueError(f"token statistic mismatch {key}")

        assignment_hashes = {path: file_sha(ROOT / path) for path in APPROVED_AUTHORITY_HASHES}
        source_hashes = {path: digest for path, digest in capacity["input_manifest_hashes"].items()}
        for name, source in capacity["source_inputs"].items():
            source_hashes[source["path"]] = source["sha256"]
            for item in source.get("raw_files", []):
                source_hashes[item["path"]] = item["sha256"]

        checks = {
            "authoritative_manifest_integrity": True,
            "source_revision_integrity": True,
            "source_file_integrity": True,
            "text_hash_integrity": True,
            "normalized_hash_integrity": True,
            "model_view_hash_integrity": True,
            "exact_counts": True,
            "binary_labels": True,
            "malicious_equality_across_arms": True,
            "shared_stage1_D0": True,
            "shared_stage2_B0": True,
            "split_isolation": True,
            "stage_isolation": True,
            "grouping_isolation": True,
            "contamination_checks": True,
            "AF5_compositional_overlap_policy": True,
            "token_length": token_stats["truncation_required"] == 0,
            "external_diagnostic_independence": True,
            "deterministic_materialization": True,
            "detector_guided_selection_dependency_absent": True,
            "atomic_publish_preconditions": True,
        }
        report = {
            "status": "PASS_READY_FOR_HUMAN_NO_REPLAY_TRAINING_APPROVAL",
            "created_at": now(),
            "checks": checks,
            "all_checks_pass": all(checks.values()),
            "assignment_manifest_hashes": assignment_hashes,
            "source_revision_hashes": source_hashes,
            "materialization_manifest": "data/processed/stronger_pilot_candidate_b/materialization_manifest.json",
            "output_files": files,
            "counts": {
                "B-DOC": 2320,
                "B-SHORT": 2320,
                "B-HARD": 2320,
                "arm_memberships": 6960,
                "unique_selected_records": 3770,
                "jsonl_files": 36,
                "per_stage": 580,
                "per_stage_split": {
                    "train": {"total": 400, "benign": 200, "malicious": 200},
                    "val": {"total": 60, "benign": 30, "malicious": 30},
                    "test": {"total": 120, "benign": 60, "malicious": 60},
                },
            },
            "matched_control": matched,
            "AF5": af5,
            "isolation": {
                "cross_stage_or_split_collisions": 0,
                "undeclared_cross_arm_sharing": 0,
                "intentional_D0_base_pairs_per_B_DOC": 1160,
            },
            "external_diagnostics": external_checks,
            "token_length_audit": token_stats,
            "atomic_publish": {
                "temporary_path": "data/processed/stronger_pilot_candidate_b/_tmp_materialization",
                "final_path": "data/processed/stronger_pilot_candidate_b",
                "published": True,
            },
            "training": False,
            "evaluation": False,
            "checkpoints_created": False,
            "optimizer_created": False,
            "replay": False,
            "selective_adaptation": False,
            "numerical_memory_budget": None,
            "replay_readiness": {"R0": "active", "R1": "not entered", "R2": "not entered"},
        }
        save_json(RESULT_TMP, report)
        validation_hash = file_sha(RESULT_TMP)
        materializer_hash = file_sha(Path(__file__))
        references_path = tmp / "external/references.json"
        manifest = {
            "status": "MATERIALIZATION_PASS",
            "created_at": report["created_at"],
            "materializer": rel(Path(__file__)),
            "materializer_script_sha256": materializer_hash,
            "active_root": str(ROOT),
            "interpreter": sys.executable,
            "assignment_manifest_hashes": assignment_hashes,
            "source_revision_hashes": source_hashes,
            "validation_artifact": rel(RESULT),
            "validation_artifact_sha256": validation_hash,
            "tokenizer": TOKENIZER_NAME,
            "tokenizer_revision": TOKENIZER_REVISION,
            "max_length": MAX_LENGTH,
            "files": files,
            "external_reference_metadata": {
                "relative_path": "data/processed/stronger_pilot_candidate_b/external/references.json",
                "byte_size": references_path.stat().st_size,
                "sha256": file_sha(references_path),
                "raw_text_present": False,
            },
            "totals": {"B-DOC": 2320, "B-SHORT": 2320, "B-HARD": 2320, "arm_memberships": 6960, "unique_selected_records": 3770, "jsonl_files": 36},
            "raw_text_local_git_ignored": True,
            "training": False,
            "evaluation": False,
            "replay": False,
            "selective_adaptation": False,
            "numerical_memory_budget": None,
        }
        save_json(tmp / "materialization_manifest.json", manifest)
        for item in files:
            path = tmp / Path(item["relative_path"]).relative_to("data/processed/stronger_pilot_candidate_b")
            verify_file_hash(path, item["sha256"])
        verify_file_hash(RESULT_TMP, manifest["validation_artifact_sha256"])

        # Publish the fully validated temporary tree as a single directory rename.
        os.replace(TARGET, PUBLISH_CONTAINER)
        os.replace(PUBLISH_CONTAINER / TMP_NAME, TARGET)
        PUBLISH_CONTAINER.rmdir()
        os.replace(RESULT_TMP, RESULT)
        for item in files:
            verify_file_hash(ROOT / item["relative_path"], item["sha256"])
        verify_file_hash(RESULT, manifest["validation_artifact_sha256"])
        verify_file_hash(TARGET / "external/references.json", manifest["external_reference_metadata"]["sha256"])

        print(json.dumps({
            "status": report["status"],
            "jsonl_files": len(files),
            "counts": report["counts"],
            "token_lengths": token_stats,
            "manifest": rel(TARGET / "materialization_manifest.json"),
            "manifest_sha256": file_sha(TARGET / "materialization_manifest.json"),
            "validation": rel(RESULT),
            "validation_sha256": file_sha(RESULT),
        }, indent=2))
        return 0
    except Exception as exc:
        if RESULT_TMP.exists():
            RESULT_TMP.unlink()
        quarantine = None
        if TARGET.exists():
            quarantine = ROOT / "data/processed" / ("_quarantine_stronger_pilot_candidate_b_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
            os.replace(TARGET, quarantine)
        elif PUBLISH_CONTAINER.exists():
            quarantine = ROOT / "data/processed" / ("_quarantine_stronger_pilot_candidate_b_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
            os.replace(PUBLISH_CONTAINER, quarantine)
        save_json(RESULT, {
            "status": "FAIL_QUARANTINED_NOT_APPROVED",
            "created_at": now(),
            "error": str(exc),
            "quarantine": rel(quarantine) if quarantine else None,
            "published": False,
            "training": False,
            "evaluation": False,
            "replay": False,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
