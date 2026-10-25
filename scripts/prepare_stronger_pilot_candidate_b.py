"""Audit and prepare Candidate-B hash-only assignments; never materialize raw JSONL."""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path(r"E:\ContinualGuard")
MANIFEST_DIR = ROOT / "data/manifests/stronger_pilot"
RESULT_CAPACITY = ROOT / "results/stronger_pilot_capacity_audit.json"
RESULT_VALIDATION = ROOT / "results/stronger_pilot_assignment_validation.json"
CHEN_FILE = ROOT / "data/processed/chen_attack_family_shift.jsonl"
H2_MAL = ROOT / "data/manifests/h2_mini/h2m_shared_malicious_assignment.json"
H2_C0 = ROOT / "data/manifests/h2_mini/h2m_c0_final_assignment_proposal.json"
H2_F = ROOT / "data/manifests/h2_mini/h2m_f_final_assignment_proposal.json"
H2_VALIDATION = ROOT / "data/manifests/h2_mini/h2m_final_assignment_validation.json"
CHEN_CONSTRUCTION = ROOT / "data/manifests/attack_family_shift/construction.json"
CHEN_SOURCES = ROOT / "data/manifests/attack_family_shift/sources.json"
OVERLAP_AUDIT = ROOT / "data/audits/attack_family_shift/overlap_audit.json"
OASST_AUDIT = ROOT / "data/audits/hard_benign/oasst1_full_audit.json"
OR_AUDIT = ROOT / "data/audits/hard_benign/or_bench_full_audit.json"
NOTINJECT_SOURCE = ROOT / "data/manifests/hard_benign/notinject/source.json"
BSHORT_MANIFEST = ROOT / "data/manifests/hard_benign/b_short_final_hash_manifest.json"

TOKENIZER_NAME = "microsoft/deberta-v3-base"
TOKENIZER_REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
CHEN_REVISION = "1e9e0f00179adc154e9ca991b9cd3206c99f9588"
OASST_REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
OR_REVISION = "e36d8b80e81837c8a8f264bbb2a49f1b32c7e272"
NOTINJECT_REVISION = "847ae76cf8fea5ed325429e569ae8cfef022d2e0"
MAX_LENGTH = 1280
SALT = "stronger-pilot-candidate-b-v1-20261005"
POLICY = "candidate-b-hash-assignment-v1-20261005"
FAMILIES = {1: "AF2", 2: "AF3", 3: "AF4", 4: "AF5"}
FAMILY_NAMES = {"AF2": "ignore", "AF3": "escape_separation", "AF4": "fake_completion", "AF5": "combined"}
SPLIT_QUOTAS = {"train": 200, "validation": 30, "test": 60}
SOURCE_SPLIT = {"train": "train", "validation": "val", "test": "test"}
POSITION_QUOTAS = {
    "train": {"head": 67, "middle": 66, "tail": 67},
    "validation": {"head": 10, "middle": 10, "tail": 10},
    "test": {"head": 20, "middle": 20, "tail": 20},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def percentile(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)] if ordered else None


def length_stats(values):
    values = list(values)
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "median": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
        "count_gt_1280": sum(x > MAX_LENGTH for x in values),
    }


def hashed_key(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def metadata_only(record: dict) -> dict:
    forbidden = {"text", "base_document_text", "payload_text", "injection_text", "prompt"}
    return {key: value for key, value in record.items() if key not in forbidden}


def load_h2_text_index():
    index = {}
    for arm in ("h2m_c0", "h2m_f"):
        for stage in range(1, 5):
            for split in ("train", "val", "test"):
                for row in read_jsonl(ROOT / f"data/processed/h2_mini/{arm}/stage{stage}/{split}.jsonl"):
                    key = row["assignment_project_record_id"]
                    if key in index and index[key] != row["text"]:
                        raise ValueError(f"H2 text mismatch for {key}")
                    index[key] = row["text"]
    return index


def tokenize_records(text_by_id):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    tokenizer_path = ROOT / "data/raw/huggingface/microsoft_deberta-v3-base" / TOKENIZER_REVISION
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_fast=False)
    ids = sorted(text_by_id)
    lengths = {}
    model_hashes = {}
    for start in range(0, len(ids), 500):
        batch_ids = ids[start:start + 500]
        encoded = tokenizer(
            [text_by_id[x] for x in batch_ids], add_special_tokens=True, truncation=False, padding=False
        )["input_ids"]
        for key, token_ids in zip(batch_ids, encoded):
            lengths[key] = len(token_ids)
            model_hashes[key] = sha_text(",".join(map(str, token_ids)))
        print(f"tokenized {min(start + 500, len(ids))}/{len(ids)}", flush=True)
    return lengths, model_hashes


def choose_af5(rows):
    candidates_all = [row for row in rows if row["task_id"] == "AF5" and row["label"] == 1]
    templates = sorted({row["template_id"] for row in candidates_all})
    counts = {
        template: {
            split: Counter(
                row["injection_position"] for row in candidates_all
                if row["template_id"] == template and row["split"] == SOURCE_SPLIT[split]
            ) for split in SPLIT_QUOTAS
        } for template in templates
    }
    # Freeze complete-template ownership across source splits. Validation is the
    # scarcest source split, then test; train receives every remaining template.
    remaining = set(templates)
    owner = {}
    ownership_diagnostics = {}
    for split in ("validation", "test"):
        deficits = dict(POSITION_QUOTAS[split])
        chosen_templates = []
        while any(value > 0 for value in deficits.values()):
            ranked = []
            for template in remaining:
                gain = sum(min(deficits[pos], counts[template][split][pos]) for pos in deficits)
                if gain:
                    ranked.append((-gain, hashed_key(SALT, "template_owner", split, template), template))
            if not ranked:
                raise ValueError(f"AF5 split-exclusive template capacity failure {split}: {deficits}")
            template = min(ranked)[2]
            remaining.remove(template)
            owner[template] = split
            chosen_templates.append(template)
            for pos in deficits:
                deficits[pos] = max(0, deficits[pos] - counts[template][split][pos])
        ownership_diagnostics[split] = {
            "owned_templates": len(chosen_templates),
            "capacity_by_position": dict(sum((counts[t][split] for t in chosen_templates), Counter())),
        }
    for template in remaining:
        owner[template] = "train"
    ownership_diagnostics["train"] = {
        "owned_templates": len(remaining),
        "capacity_by_position": dict(sum((counts[t]["train"] for t in remaining), Counter())),
    }

    selected = []
    diagnostics = {"template_ownership": ownership_diagnostics}
    for split, position_targets in POSITION_QUOTAS.items():
        source_split = SOURCE_SPLIT[split]
        for position, target in position_targets.items():
            candidates = [
                row for row in candidates_all
                if row["split"] == source_split and row["injection_position"] == position
                and owner[row["template_id"]] == split
            ]
            groups = defaultdict(list)
            for row in candidates:
                groups[row["template_id"]].append(row)
            for template_id in groups:
                groups[template_id].sort(key=lambda r: hashed_key(SALT, split, position, template_id, r["example_id"]))
            template_order = sorted(groups, key=lambda t: hashed_key(SALT, split, position, t))
            chosen = []
            round_index = 0
            while len(chosen) < target:
                advanced = False
                for template_id in template_order:
                    if round_index < len(groups[template_id]):
                        chosen.append(groups[template_id][round_index])
                        advanced = True
                        if len(chosen) == target:
                            break
                if not advanced:
                    raise ValueError(f"AF5 capacity failure {split}/{position}")
                round_index += 1
            selected.extend(chosen)
            diagnostics[f"{split}:{position}"] = {
                "available": len(candidates),
                "selected": len(chosen),
                "available_templates": len(groups),
                "selected_templates": len({r["template_id"] for r in chosen}),
            }
    if len(selected) != 290 or len({r["example_id"] for r in selected}) != 290:
        raise ValueError("AF5 exact count/identity failure")
    split_templates = {
        split: {r["template_id"] for r in selected if r["split"] == SOURCE_SPLIT[split]}
        for split in SPLIT_QUOTAS
    }
    if any(split_templates[a] & split_templates[b] for a in SPLIT_QUOTAS for b in SPLIT_QUOTAS if a < b):
        raise ValueError("AF5 selected complete-template cross-split leakage")
    diagnostics["selected_complete_template_cross_split_overlap"] = 0
    diagnostics["selected_unique_complete_templates"] = len(set().union(*split_templates.values()))
    return selected, diagnostics


def malicious_meta(row, stage, prior_h2, existing=None):
    rid = existing["project_record_id"] if existing else "cgsp-mal-" + row["example_id"][:24]
    result = {
        "project_record_id": rid,
        "source_native_id": row["example_id"],
        "source_record_id": row["source"]["record_id"],
        "source_revision": row["source"]["revision"],
        "source_upstream_split": row["source"].get("upstream_split"),
        "base_document_id": row["base_document_id"],
        "injection_id": row["injection_id"],
        "payload_source_record_id": row["payload_source"]["record_id"],
        "payload_source_revision": row["payload_source"]["revision"],
        "attack_family_id": row["task_id"],
        "attack_family": row["attack_family"],
        "attack_template_id": row["attack_template_id"],
        "normalized_template_hash": row["normalized_template_hash"],
        "attack_template_components": row["attack_template_components"],
        "injection_position": row["injection_position"],
        "split": {v: k for k, v in SOURCE_SPLIT.items()}[row["split"]],
        "stage": stage,
        "label": 1,
        "raw_text_sha256": row["text_sha256"],
        "base_document_sha256": sha_text(row["base_document_text"]),
        "injection_sha256": sha_text(row["injection_text"]),
        "domain": row["domain"],
        "language": row["language"],
        "prior_h2_mini_record": prior_h2,
        "record_role": "malicious",
    }
    return result


def d0_meta(row, malicious, stage):
    result = {
        "project_record_id": "cgsp-d0-" + row["example_id"][:24],
        "source_native_id": row["example_id"],
        "source_document_record_id": row["source"]["record_id"],
        "source_revision": row["source"]["revision"],
        "source_upstream_split": row["source"].get("upstream_split"),
        "base_document_id": row["base_document_id"],
        "paired_malicious_project_record_id": malicious["project_record_id"],
        "paired_malicious_source_native_id": malicious["source_native_id"],
        "split": malicious["split"],
        "stage": stage,
        "label": 0,
        "benign_partition": f"D0-{stage}",
        "raw_text_sha256": row["text_sha256"],
        "domain": row["domain"],
        "language": row["language"],
        "attack_family": "none",
        "injection_id": None,
        "injection_position": "none",
        "annotation_status": "chen_source_deterministic_clean_companion_previously_audited",
        "prior_h2_mini_record": False,
        "prior_attack_family_baseline_record": True,
        "within_stage_cross_label_base_pairing": True,
        "record_role": "benign",
    }
    return result


def reused_benign_meta(row, arm, new_stage, partition):
    result = dict(row)
    result["stage"] = new_stage
    result["benign_partition"] = partition
    result["arm"] = arm
    result["prior_h2_mini_record"] = True
    result["prior_h2_mini_partition"] = row["benign_partition"]
    result["record_role"] = "benign"
    return metadata_only(result)


def attach_token_evidence(records, text_by_id, lengths, model_hashes):
    for record in records:
        rid = record["project_record_id"]
        text = text_by_id[rid]
        raw_hash = sha_text(text)
        if raw_hash != record["raw_text_sha256"]:
            raise ValueError(f"raw hash mismatch {rid}")
        norm_hash = sha_text(normalized(text))
        if record.get("normalized_text_sha256") and record["normalized_text_sha256"] != norm_hash:
            raise ValueError(f"normalized hash mismatch {rid}")
        if record.get("model_view_sha256") and record["model_view_sha256"] != model_hashes[rid]:
            raise ValueError(f"model-view hash mismatch {rid}")
        if record.get("deberta_tokens") is not None and record["deberta_tokens"] != lengths[rid]:
            raise ValueError(f"token-length mismatch {rid}")
        record["normalized_text_sha256"] = norm_hash
        record["model_view_sha256"] = model_hashes[rid]
        record["deberta_tokens"] = lengths[rid]


def exact_counts(records):
    return Counter((r["stage"], r["split"], r["label"]) for r in records)


def validate_arm(records, arm):
    counts = exact_counts(records)
    for stage in range(1, 5):
        for split, target in SPLIT_QUOTAS.items():
            for label in (0, 1):
                if counts[(stage, split, label)] != target:
                    raise ValueError(f"{arm} count mismatch {stage}/{split}/{label}")
    if len(records) != 2320:
        raise ValueError(f"{arm} total mismatch")
    if any("text" in r or "base_document_text" in r or "payload_text" in r or "injection_text" in r for r in records):
        raise ValueError(f"raw text in {arm} proposal")
    return counts


def sig(record):
    return tuple(record.get(k) for k in (
        "project_record_id", "source_native_id", "raw_text_sha256", "model_view_sha256",
        "attack_family_id", "stage", "split", "injection_position", "label"
    ))


def benign_sig(record):
    return tuple(record.get(k) for k in (
        "project_record_id", "source_native_id", "raw_text_sha256", "model_view_sha256",
        "stage", "split", "label", "benign_partition"
    ))


def group_leakage(records, fields, scope_name):
    findings = {}
    for field in fields:
        seen = {}
        collisions = []
        for row in records:
            value = row.get(field)
            if value is None:
                continue
            location = (row["stage"], row["split"])
            if value in seen and seen[value] != location:
                collisions.append({"value": value, "locations": [seen[value], location]})
            else:
                seen[value] = location
        findings[field] = {"cross_stage_or_split_collisions": len(collisions), "sample": collisions[:5]}
        if collisions:
            raise ValueError(f"{scope_name} leakage in {field}")
    return findings


def distribution(records, field):
    return dict(sorted(Counter(str(r.get(field, "unknown")) for r in records).items()))


def token_bin(value):
    if value <= 16:
        return "le16"
    if value <= 32:
        return "17_32"
    if value <= 64:
        return "33_64"
    if value <= 128:
        return "65_128"
    return "gt128"


def proposal(arm, benign_design, shared_malicious, benign):
    records = []
    for record in shared_malicious:
        item = dict(record)
        item["arm"] = arm
        records.append(item)
    for record in benign:
        item = dict(record)
        item["arm"] = arm
        records.append(item)
    records.sort(key=lambda r: (r["stage"], r["split"], r["label"], r["project_record_id"]))
    return {
        "status": "READY_FOR_HUMAN_MATERIALIZATION_APPROVAL",
        "created_at": now(),
        "policy": POLICY,
        "deterministic_salt": SALT,
        "arm": arm,
        "benign_design": benign_design,
        "records": records,
        "counts": {"total": len(records), "malicious": 1160, "benign": 1160},
        "per_stage": {str(stage): {"total": 580, "malicious": 290, "benign": 290} for stage in range(1, 5)},
        "raw_text_present": False,
        "materialized": False,
        "training": False,
        "replay": False,
        "numerical_memory_budget": None,
    }


def main():
    if ROOT.resolve() != EXPECTED_ROOT.resolve():
        raise RuntimeError(f"Active root must be {EXPECTED_ROOT}; got {ROOT}")
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise RuntimeError(f"Active interpreter mismatch: {sys.executable}")

    construction = read_json(CHEN_CONSTRUCTION)
    if sha_file(CHEN_FILE) != construction["processed_sha256"]:
        raise ValueError("Chen processed hash mismatch")
    h2_validation = read_json(H2_VALIDATION)
    if not h2_validation["all_checks_pass"]:
        raise ValueError("H2 final assignment authority is not PASS")
    for path in (H2_MAL, H2_C0, H2_F):
        expected = h2_validation["manifest_hashes"].get(rel(path))
        if not expected or sha_file(path) != expected:
            raise ValueError(f"H2 authority hash mismatch: {rel(path)}")
    chen_rows = read_jsonl(CHEN_FILE)
    chen_by_id = {row["example_id"]: row for row in chen_rows}
    h2_mal_obj = read_json(H2_MAL)
    h2_mal_by_family = defaultdict(list)
    for row in h2_mal_obj["records"]:
        h2_mal_by_family[row["attack_family_id"]].append(row)

    af5_selected, af5_selection = choose_af5(chen_rows)
    shared_malicious = []
    raw_text_by_id = {}
    for stage, family in FAMILIES.items():
        if family != "AF5":
            source_meta = h2_mal_by_family[family]
            if len(source_meta) != 290:
                raise ValueError(f"H2 malicious capacity failure {family}")
            source_rows = [chen_by_id[item["source_native_id"]] for item in source_meta]
            existing_by_source = {item["source_native_id"]: item for item in source_meta}
            stage_records = [malicious_meta(row, stage, True, existing_by_source[row["example_id"]]) for row in source_rows]
        else:
            source_rows = af5_selected
            stage_records = [malicious_meta(row, stage, False) for row in source_rows]
        for meta, source in zip(stage_records, source_rows):
            shared_malicious.append(meta)
            raw_text_by_id[meta["project_record_id"]] = source["text"]

    if Counter((r["stage"], r["split"]) for r in shared_malicious) != Counter(
        {(stage, split): count for stage in range(1, 5) for split, count in SPLIT_QUOTAS.items()}
    ):
        raise ValueError("shared malicious exact count failure")
    for stage in range(1, 5):
        actual = Counter((r["split"], r["injection_position"]) for r in shared_malicious if r["stage"] == stage)
        expected = Counter({(split, pos): n for split, p in POSITION_QUOTAS.items() for pos, n in p.items()})
        if actual != expected:
            raise ValueError(f"position contract failure stage {stage}")

    clean_index = defaultdict(list)
    for row in chen_rows:
        if row["label"] == 0 and row["task_id"] in FAMILY_NAMES:
            clean_index[(row["task_id"], row["split"], row["base_document_id"])].append(row)
    d0 = []
    d0_logic_failures = []
    for malicious in shared_malicious:
        source_split = SOURCE_SPLIT[malicious["split"]]
        companions = clean_index[(malicious["attack_family_id"], source_split, malicious["base_document_id"])]
        if len(companions) != 1:
            raise ValueError(f"clean companion multiplicity failure {malicious['project_record_id']}")
        clean = companions[0]
        checks = {
            "label_zero": clean["label"] == 0,
            "attack_family_none": clean["attack_family"] == "none",
            "injection_id_none": clean["injection_id"] is None,
            "position_none": clean["injection_position"] == "none",
            "text_equals_base_document": clean["text"] == clean["base_document_text"],
            "revision_pinned": clean["source"]["revision"] == CHEN_REVISION,
            "example_identity_differs_from_malicious": clean["example_id"] != malicious["source_native_id"],
            "raw_text_differs_from_malicious": clean["text_sha256"] != malicious["raw_text_sha256"],
            "base_document_intentionally_matches": clean["base_document_id"] == malicious["base_document_id"],
        }
        if not all(checks.values()):
            d0_logic_failures.append({"record": clean["example_id"], "checks": checks})
        meta = d0_meta(clean, malicious, malicious["stage"])
        d0.append(meta)
        raw_text_by_id[meta["project_record_id"]] = clean["text"]
    if d0_logic_failures:
        raise ValueError(f"D0 label logic failure count={len(d0_logic_failures)}")

    c0 = read_json(H2_C0)
    f = read_json(H2_F)
    h2_text = load_h2_text_index()
    b0 = []
    for old_partition, new_stage in (("B0-2", 2), ("B0-3", 3), ("B0-4", 4)):
        rows = [r for r in c0["records"] if r.get("benign_partition") == old_partition]
        if len(rows) != 290:
            raise ValueError(f"B0 capacity failure {old_partition}")
        for row in rows:
            meta = reused_benign_meta(row, "shared", new_stage, old_partition)
            b0.append(meta)
            raw_text_by_id[meta["project_record_id"]] = h2_text[meta["project_record_id"]]
    bh = []
    for old_partition, new_stage in (("BH-1", 3), ("BH-2", 4)):
        rows = [r for r in f["records"] if r.get("benign_partition") == old_partition]
        if len(rows) != 290:
            raise ValueError(f"BH capacity failure {old_partition}")
        for row in rows:
            meta = reused_benign_meta(row, "B-HARD", new_stage, old_partition)
            bh.append(meta)
            raw_text_by_id[meta["project_record_id"]] = h2_text[meta["project_record_id"]]

    lengths, model_hashes = tokenize_records(raw_text_by_id)
    unique_meta = {r["project_record_id"]: r for r in shared_malicious + d0 + b0 + bh}
    attach_token_evidence(list(unique_meta.values()), raw_text_by_id, lengths, model_hashes)
    if any(value > MAX_LENGTH for value in lengths.values()):
        raise ValueError("selected record requires truncation")

    d0_by_stage = {stage: [r for r in d0 if r["stage"] == stage] for stage in range(1, 5)}
    b0_by_stage = {stage: [r for r in b0 if r["stage"] == stage] for stage in (2, 3, 4)}
    bh_by_stage = {stage: [r for r in bh if r["stage"] == stage] for stage in (3, 4)}
    b_doc_benign = list(d0)
    b_short_benign = d0_by_stage[1] + b0_by_stage[2] + b0_by_stage[3] + b0_by_stage[4]
    b_hard_benign = d0_by_stage[1] + b0_by_stage[2] + bh_by_stage[3] + bh_by_stage[4]

    proposals = {
        "B-DOC": proposal("B-DOC", "D0-1/D0-2/D0-3/D0-4", shared_malicious, b_doc_benign),
        "B-SHORT": proposal("B-SHORT", "D0-1/B0-2/B0-3/B0-4", shared_malicious, b_short_benign),
        "B-HARD": proposal("B-HARD", "D0-1/B0-2/BH-1/BH-2", shared_malicious, b_hard_benign),
    }
    for arm, obj in proposals.items():
        validate_arm(obj["records"], arm)

    malicious_sets = {
        arm: {sig(r) for r in obj["records"] if r["label"] == 1}
        for arm, obj in proposals.items()
    }
    if len({frozenset(value) for value in malicious_sets.values()}) != 1:
        raise ValueError("matched malicious arms failure")
    stage1_sets = {
        arm: {benign_sig(r) for r in obj["records"] if r["label"] == 0 and r["stage"] == 1}
        for arm, obj in proposals.items()
    }
    if len({frozenset(value) for value in stage1_sets.values()}) != 1:
        raise ValueError("Stage-1 D0 anchor mismatch")
    stage2_short = {benign_sig(r) for r in proposals["B-SHORT"]["records"] if r["label"] == 0 and r["stage"] == 2}
    stage2_hard = {benign_sig(r) for r in proposals["B-HARD"]["records"] if r["label"] == 0 and r["stage"] == 2}
    if stage2_short != stage2_hard:
        raise ValueError("Stage-2 B0 anchor mismatch")

    d0_leakage = group_leakage(d0, [
        "source_native_id", "source_document_record_id", "base_document_id", "raw_text_sha256",
        "normalized_text_sha256", "model_view_sha256"
    ], "D0")
    b0_leakage = group_leakage(b0, [
        "source_native_id", "source_parent_id", "exact_group", "normalized_group", "model_view_group",
        "project_semantic_parent_group", "project_template_group", "raw_text_sha256",
        "normalized_text_sha256", "model_view_sha256"
    ], "B0")
    bh_leakage = group_leakage(bh, [
        "candidate_id", "exact_group", "normalized_group", "model_view_group",
        "project_semantic_parent_group", "project_template_group", "raw_text_sha256",
        "normalized_text_sha256", "model_view_sha256"
    ], "BH")
    malicious_leakage = group_leakage(shared_malicious, [
        "source_native_id", "base_document_id", "injection_id", "raw_text_sha256",
        "normalized_text_sha256", "model_view_sha256"
    ], "malicious")

    benign_sets = {
        arm: {r["project_record_id"] for r in obj["records"] if r["label"] == 0}
        for arm, obj in proposals.items()
    }
    pair_overlaps = {
        "B-DOC__B-SHORT": len(benign_sets["B-DOC"] & benign_sets["B-SHORT"]),
        "B-DOC__B-HARD": len(benign_sets["B-DOC"] & benign_sets["B-HARD"]),
        "B-SHORT__B-HARD": len(benign_sets["B-SHORT"] & benign_sets["B-HARD"]),
    }
    if pair_overlaps != {"B-DOC__B-SHORT": 290, "B-DOC__B-HARD": 290, "B-SHORT__B-HARD": 580}:
        raise ValueError(f"unexpected benign cross-arm overlap {pair_overlaps}")

    af5_all = [r for r in chen_rows if r["task_id"] == "AF5"]
    af5_mal_all = [r for r in af5_all if r["label"] == 1]
    prior_mal_source = {r["source_native_id"] for r in shared_malicious if r["attack_family_id"] != "AF5"}
    prior_rows = [chen_by_id[x] for x in prior_mal_source]
    af5_overlap = {
        "selected_vs_selected_AF2_AF4": {
            "exact_text": len({r["text_sha256"] for r in af5_selected} & {r["text_sha256"] for r in prior_rows}),
            "base_document_id": len({r["base_document_id"] for r in af5_selected} & {r["base_document_id"] for r in prior_rows}),
            "payload_id": len({r["injection_id"] for r in af5_selected} & {r["injection_id"] for r in prior_rows}),
            "complete_template_id": len({r["template_id"] for r in af5_selected} & {r["template_id"] for r in prior_rows}),
            "normalized_template_hash": len({r["normalized_template_hash"] for r in af5_selected} & {r["normalized_template_hash"] for r in prior_rows}),
        },
        "intrinsic_component_overlap": {
            "AF2": "ignore",
            "AF3": "escape_separation",
            "AF4": "fake_completion",
            "AF5_selected_records_with_all_three_components": sum(
                set(r["attack_template_components"]) == {"ignore", "escape_separation", "fake_completion"}
                for r in af5_selected
            ),
        },
        "interpretation": "AF5 is a compositional/combined attack shift, not a family-disjoint attack family. Component reuse is intended; exact example/base/payload/complete-template identity reuse is prohibited.",
    }

    source_inputs = {
        "Chen_processed": {"path": rel(CHEN_FILE), "sha256": sha_file(CHEN_FILE), "revision": CHEN_REVISION, "license": "Chen repository MIT; SQuAD CC BY-SA 4.0; Alpaca-derived payloads CC BY-NC 4.0"},
        "OASST1": {}, "OR-Bench": {}, "NotInject": {},
    }
    oa = read_json(OASST_AUDIT)["source"]
    oa_file = next(x for x in oa["files"] if x["file_name"] == "2023-04-12_oasst_ready.messages.jsonl.gz")
    source_inputs["OASST1"] = {"path": oa_file["local_path"].replace("\\", "/"), "sha256": oa_file["sha256"], "revision": oa["revision"], "license": oa["license"]}
    ora = read_json(OR_AUDIT)["source"]
    or_file = next(x for x in ora["files"] if x["file_name"] == "or-bench-80k.csv")
    source_inputs["OR-Bench"] = {"path": or_file["local_path"].replace("\\", "/"), "sha256": or_file["sha256"], "revision": ora["revision"], "license": ora["license"]}
    ni = read_json(NOTINJECT_SOURCE)
    source_inputs["NotInject"] = {"path": ni["records_manifest"].replace("\\", "/"), "sha256": ni["records_manifest_sha256"], "revision": ni["repository_revision"], "license": ni["license"], "raw_files": [{"path": x["local_path"].replace("\\", "/"), "sha256": x["sha256"]} for x in ni["raw_files"]]}
    for name in ("OASST1", "OR-Bench"):
        item = source_inputs[name]
        if sha_file(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"{name} raw source hash mismatch")
    if (oa["revision"], ora["revision"], ni["repository_revision"]) != (OASST_REVISION, OR_REVISION, NOTINJECT_REVISION):
        raise ValueError("pinned source revision mismatch")
    if sha_file(ROOT / source_inputs["NotInject"]["path"]) != source_inputs["NotInject"]["sha256"]:
        raise ValueError("NotInject record manifest hash mismatch")
    for item in source_inputs["NotInject"]["raw_files"]:
        if sha_file(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"NotInject raw source hash mismatch: {item['path']}")

    group_stats = {}
    for name, records in (("D0", d0), ("B0", b0), ("BH", bh), ("malicious", shared_malicious)):
        group_stats[name] = {
            "records": len(records),
            "token_lengths": length_stats(r["deberta_tokens"] for r in records),
            "stages": distribution(records, "stage"),
            "splits": distribution(records, "split"),
            "languages": distribution(records, "language"),
            "token_length_bins": dict(sorted(Counter(token_bin(r["deberta_tokens"]) for r in records).items())),
        }
    group_stats["BH"]["source_categories"] = distribution(bh, "source_category")
    group_stats["BH"]["historical_descriptive_strata"] = distribution(bh, "historical_proposed_stratum")
    group_stats["B0"]["source_categories"] = {"OASST1_root_prompter": len(b0)}
    group_stats["D0"]["domains"] = distribution(d0, "domain")
    group_stats["B0"]["per_partition"] = {
        partition: {
            "records": len([r for r in b0 if r["benign_partition"] == partition]),
            "languages": distribution([r for r in b0 if r["benign_partition"] == partition], "language"),
            "token_length_bins": dict(sorted(Counter(token_bin(r["deberta_tokens"]) for r in b0 if r["benign_partition"] == partition).items())),
        } for partition in ("B0-2", "B0-3", "B0-4")
    }
    group_stats["BH"]["per_partition"] = {
        partition: {
            "records": len([r for r in bh if r["benign_partition"] == partition]),
            "source_categories": distribution([r for r in bh if r["benign_partition"] == partition], "source_category"),
            "historical_descriptive_strata": distribution([r for r in bh if r["benign_partition"] == partition], "historical_proposed_stratum"),
            "token_length_bins": dict(sorted(Counter(token_bin(r["deberta_tokens"]) for r in bh if r["benign_partition"] == partition).items())),
        } for partition in ("BH-1", "BH-2")
    }

    selected_raw = {r["raw_text_sha256"] for r in unique_meta.values()}
    selected_norm = {r["normalized_text_sha256"] for r in unique_meta.values()}
    bshort = read_json(BSHORT_MANIFEST)["records"]
    ni_records = read_jsonl(ROOT / source_inputs["NotInject"]["path"])
    external_overlap = {
        "B_short_raw": len(selected_raw & {r["raw_text_sha256"] for r in bshort}),
        "B_short_normalized": len(selected_norm & {r["normalized_text_sha256"] for r in bshort}),
        "NotInject_raw": len(selected_raw & {r["prompt_sha256"] for r in ni_records}),
        "NotInject_normalized": len(selected_norm & {r["normalized_prompt_sha256"] for r in ni_records}),
    }
    if any(external_overlap.values()):
        raise ValueError(f"external diagnostic contamination: {external_overlap}")

    h2_unique_ids = set()
    for arm in ("h2m_c0", "h2m_f"):
        for stage in range(1, 5):
            for split in ("train", "val", "test"):
                for row in read_jsonl(ROOT / f"data/processed/h2_mini/{arm}/stage{stage}/{split}.jsonl"):
                    h2_unique_ids.add(row["assignment_project_record_id"])
    stronger_unique = set(unique_meta)
    reuse_ids = stronger_unique & h2_unique_ids
    reuse_by_group = {
        "malicious_AF2_AF4": sum(r["project_record_id"] in reuse_ids for r in shared_malicious),
        "D0": sum(r["project_record_id"] in reuse_ids for r in d0),
        "B0": sum(r["project_record_id"] in reuse_ids for r in b0),
        "BH": sum(r["project_record_id"] in reuse_ids for r in bh),
    }

    shared_manifest = {
        "status": "READY_FOR_HUMAN_MATERIALIZATION_APPROVAL",
        "created_at": now(),
        "policy": POLICY,
        "deterministic_salt": SALT,
        "stage_order": ["AF2", "AF3", "AF4", "AF5"],
        "records": sorted(shared_malicious, key=lambda r: (r["stage"], r["split"], r["injection_position"], r["project_record_id"])),
        "counts": {"total": 1160, "per_stage": 290, "train_per_stage": 200, "validation_per_stage": 30, "test_per_stage": 60},
        "AF5_role": "compositional_combined_attack_shift_not_family_disjoint",
        "selection_inputs": "source metadata, split, position, template ID, and frozen hash salt only; no detector predictions or losses",
        "raw_text_present": False,
        "materialized": False,
        "training": False,
    }

    input_hashes = {rel(path): sha_file(path) for path in [
        CHEN_FILE, H2_MAL, H2_C0, H2_F, H2_VALIDATION, CHEN_CONSTRUCTION,
        CHEN_SOURCES, OVERLAP_AUDIT, OASST_AUDIT, OR_AUDIT, NOTINJECT_SOURCE, BSHORT_MANIFEST,
    ]}
    capacity = {
        "status": "PASS",
        "created_at": now(),
        "active_root": str(ROOT),
        "interpreter": sys.executable,
        "scope": "Candidate-B source/capacity audit and hash-only assignment preparation; no materialization or training",
        "detector": {"name": TOKENIZER_NAME, "revision": TOKENIZER_REVISION, "max_length": MAX_LENGTH},
        "policy": POLICY,
        "deterministic_salt": SALT,
        "source_inputs": source_inputs,
        "input_manifest_hashes": input_hashes,
        "AF5": {
            "all_records": len(af5_all),
            "split_label_counts": {f"{split}:{label}": count for (split, label), count in sorted(Counter((r["split"], r["label"]) for r in af5_all).items())},
            "malicious_records": len(af5_mal_all),
            "unique_payload_ids": len({r["injection_id"] for r in af5_mal_all}),
            "unique_complete_templates": len({r["template_id"] for r in af5_mal_all}),
            "unique_base_documents": len({r["base_document_id"] for r in af5_mal_all}),
            "positions": dict(sorted(Counter(r["injection_position"] for r in af5_mal_all).items())),
            "source_revisions": sorted({r["source"]["revision"] for r in af5_mal_all} | {r["payload_source"]["revision"] for r in af5_mal_all}),
            "selected": 290,
            "selection_diagnostics": af5_selection,
            "overlap": af5_overlap,
        },
        "capacity": {
            "D0": {"available_clean_AF2_AF5": 7320, "eligible_exact_companions": len(d0), "required": 1160, "selected": len(d0), "new_content_review_required": False, "evidence": "Previously audited Chen clean companions; label-0/no-injection/text-equals-base contract reverified for every selected row."},
            "B0": {"finalized_available": 1160, "required": 870, "selected": len(b0), "partitions": ["B0-2", "B0-3", "B0-4"], "new_content_review_required": False},
            "BH": {"finalized_available": 870, "required": 580, "selected": len(bh), "partitions": ["BH-1", "BH-2"], "new_content_review_required": False},
        },
        "D0_label_logic": {"records_checked": len(d0), "failures": 0, "within_stage_base_document_pairing": len(d0), "exact_record_identity_overlap_with_malicious": 0, "raw_text_overlap_with_paired_malicious": 0},
        "group_statistics": group_stats,
        "token_lengths": {**group_stats, "unique_selected": length_stats(lengths.values())},
        "H2_mini_reuse": {"stronger_unique_records": len(stronger_unique), "reused_unique_records": len(reuse_ids), "new_to_H2_mini_unique_records": len(stronger_unique - h2_unique_ids), "by_group": reuse_by_group, "external_banks_reused": {"NotInject": 339, "B-short": 120}},
        "human_review": {"new_content_labeling_required": False, "estimated_new_content_review_hours": 0.0, "estimated_manifest_signoff_hours": "0.5-1.0", "reason": "D0 uses reverified deterministic clean companions and B0/BH reuse finalized human-reviewed pools."},
        "external_diagnostics": {"NotInject": {"records": 339, "role": "frozen evaluation only"}, "B-short": {"records": 120, "role": "frozen unmatched descriptive evaluation only"}, "selected_hash_overlap": external_overlap},
        "materialized": False,
        "training": False,
        "replay": False,
        "selective_adaptation": False,
        "numerical_memory_budget": None,
    }

    validation = {
        "status": "READY_FOR_HUMAN_MATERIALIZATION_APPROVAL",
        "created_at": now(),
        "policy": POLICY,
        "deterministic_salt": SALT,
        "checks": {
            "source_hash_and_revision_integrity": True,
            "AF5_exact_audit": True,
            "AF5_intrinsic_component_reuse_declared": True,
            "prediction_independent_selection": True,
            "exact_counts": True,
            "malicious_position_targets": True,
            "D0_clean_label_contract": True,
            "D0_B0_BH_capacity": True,
            "benign_group_isolation": True,
            "malicious_identity_isolation": True,
            "matched_malicious_all_arms": True,
            "shared_stage1_D0_anchor": True,
            "shared_stage2_B0_anchor": True,
            "no_unintended_benign_cross_arm_sharing": True,
            "NotInject_and_Bshort_excluded_from_assignment": True,
            "token_length_integrity": True,
            "metadata_only_no_raw_text": True,
        },
        "all_checks_pass": True,
        "counts": {"per_arm": 2320, "three_arm_memberships": 6960, "unique_selected_records": len(stronger_unique), "shared_malicious": 1160, "D0": 1160, "B0": 870, "BH": 580},
        "matched_arm_verification": {
            "malicious_signatures_per_arm": {arm: len(value) for arm, value in malicious_sets.items()},
            "malicious_sets_identical": True,
            "stage1_D0_signatures_per_arm": {arm: len(value) for arm, value in stage1_sets.items()},
            "stage1_D0_sets_identical": True,
            "stage2_B0_BSHORT_BHARD_sets_identical": True,
            "benign_cross_arm_identity_overlap": pair_overlaps,
            "expected_overlap_only": True,
        },
        "leakage": {"D0": d0_leakage, "B0": b0_leakage, "BH": bh_leakage, "malicious": malicious_leakage, "AF5_overlap": af5_overlap},
        "token_lengths": {
            "unique_selected": length_stats(lengths.values()),
            "B-DOC": length_stats(r["deberta_tokens"] for r in proposals["B-DOC"]["records"]),
            "B-SHORT": length_stats(r["deberta_tokens"] for r in proposals["B-SHORT"]["records"]),
            "B-HARD": length_stats(r["deberta_tokens"] for r in proposals["B-HARD"]["records"]),
        },
        "template_policy": {
            "construction_pattern_reuse": "Allowed and intrinsic across splits within reused AF2/AF3/AF4 under the frozen project policy; AF4 has one family wrapper. AF5 was newly assigned with complete-template IDs split-exclusive.",
            "prohibited": "Exact example, base-document, and payload identity across stages/splits; complete-template identity across stages; and AF5 complete-template identity across splits. All are zero in the selected stream.",
        },
        "H2_mini_reuse": capacity["H2_mini_reuse"],
        "proposed_comparisons": {
            "B-SHORT_minus_B-DOC": "effect associated with document-context to ordinary short-form source/format shift",
            "B-HARD_minus_B-SHORT": "incremental effect associated with reviewed hard-benign pressure after common Stage 2, with source/language caveats",
            "within_arm": ["attack TPR retention", "benign FPR increase", "macro-F1", "accuracy", "TNR", "peak forgetting", "BWT"],
        },
        "acquisition_gates": {"attack_TPR_min": 0.85, "benign_TNR_min": 0.85, "paired_macro_F1_min": 0.85, "passing_seeds_required": 2, "seeds": 3},
        "retention_primary": {"malicious": "attack TPR retention from first acquisition", "benign": "FPR increase from first acquisition"},
        "readiness": {"R0": "active", "R1": "not entered", "R2": "not entered", "criteria_changed": False},
        "future_execution": {"arms": 3, "seeds": [17, 42, 2026], "runs": 9, "stages_per_run": 4, "stage_end_checkpoints": 36, "estimated_GPU_runtime_minutes": "39-58 based on H2-mini"},
        "materialized": False,
        "training": False,
        "replay": False,
        "selective_adaptation": False,
        "numerical_memory_budget": None,
    }

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    write_json(MANIFEST_DIR / "stronger_shared_malicious_assignment.json", shared_manifest)
    write_json(MANIFEST_DIR / "stronger_b_doc_assignment_proposal.json", proposals["B-DOC"])
    write_json(MANIFEST_DIR / "stronger_b_short_assignment_proposal.json", proposals["B-SHORT"])
    write_json(MANIFEST_DIR / "stronger_b_hard_assignment_proposal.json", proposals["B-HARD"])
    proposal_hashes = {
        rel(path): sha_file(path) for path in [
            MANIFEST_DIR / "stronger_shared_malicious_assignment.json",
            MANIFEST_DIR / "stronger_b_doc_assignment_proposal.json",
            MANIFEST_DIR / "stronger_b_short_assignment_proposal.json",
            MANIFEST_DIR / "stronger_b_hard_assignment_proposal.json",
        ]
    }
    validation["proposal_hashes"] = proposal_hashes
    write_json(MANIFEST_DIR / "stronger_assignment_validation.json", validation)
    write_json(RESULT_CAPACITY, capacity)
    write_json(RESULT_VALIDATION, validation)
    print(json.dumps({
        "status": validation["status"],
        "capacity": capacity["capacity"],
        "counts": validation["counts"],
        "matched": validation["matched_arm_verification"],
        "token_lengths": validation["token_lengths"],
        "proposal_hashes": proposal_hashes,
    }, indent=2))


if __name__ == "__main__":
    main()
