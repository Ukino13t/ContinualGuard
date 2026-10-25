"""Deterministic Chen real-data preparation and audit; no training entry point."""
from collections import Counter, defaultdict
from datetime import date
import json
import math
from pathlib import Path
import re

from transformers import AutoTokenizer

from .artifacts import digest, write_json
from .data import length_audit, normalized, validate_record

CHEN_REVISION = "1e9e0f00179adc154e9ca991b9cd3206c99f9588"
TOKENIZER_REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
SEED = 42
RAW_FILES = {
    "crafted_instruction_data_alpaca.json": "ae8a03ce27ea5d220da9dacf52fb04a38463a78182de287bffc181a51e253acf",
    "crafted_instruction_data_context_squad.json": "ecf659f231584141e14f3d9c9db79dbd6627031e2f61d94b18cbd7f2381be141",
    "crafted_instruction_data_context_tri.json": "dd1ca57b4062ddefe282618792a2af02c3737f72d888b3da62fc91e8da16b7e7",
    "crafted_instruction_data_squad_injection_qa.json": "985b232aae2cb91cd5709d95e4d83c6a00d5fd807bd86d06e89ec22c9bba42d5",
    "crafted_instruction_data_tri_injection_qa.json": "6f52a36c8a3627ea297a833481d3446688753271d90b267447ef65c82be42835",
}
STAGES = {
    "S1": {"domain": "squad_qa_documents", "train_file": "crafted_instruction_data_context_squad.json", "test_file": "crafted_instruction_data_squad_injection_qa.json", "dataset": "Chen et al. released SQuAD-derived data"},
    "S2": {"domain": "triviaqa_web_documents", "train_file": "crafted_instruction_data_context_tri.json", "test_file": "crafted_instruction_data_tri_injection_qa.json", "dataset": "Chen et al. released TriviaQA-derived data"},
}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _stable(*parts):
    return digest("\x1f".join(map(str, parts)).encode("utf-8"))


def _record_hash(row):
    value = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return digest(value.encode("utf-8"))


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _split_unique(rows, val_fraction=.1):
    """Deduplicate and split base groups before variants are constructed."""
    groups = defaultdict(list)
    excluded = []
    for row in rows:
        content = normalized(row.get("_model_context", row["context"]))
        if not content:
            excluded.append({"reason": "empty base document", "duplicate_group_id": _stable("base", ""),
                             "excluded_raw_index": row["_raw_index"], "canonical_raw_index": None})
        else:
            groups[_stable("base", content)].append(row)
    ordered = sorted(groups, key=lambda group: _stable(SEED, "base-split", group))
    val_groups = set(ordered[:round(len(ordered) * val_fraction)])
    kept = []
    for group in ordered:
        members = sorted(groups[group], key=lambda row: row["_raw_index"])
        canonical = dict(members[0], _duplicate_group_id=group,
                         _split="val" if group in val_groups else "train")
        kept.append(canonical)
        for duplicate in members[1:]:
            same_original = normalized(duplicate["context"]) == normalized(canonical["context"])
            excluded.append({"reason": "duplicate normalized base document" if same_original else
                             "duplicate model input after 384-piece preprocessing", "duplicate_group_id": group,
                             "excluded_raw_index": duplicate["_raw_index"], "canonical_raw_index": canonical["_raw_index"]})
    return kept, excluded


def _allocation(total):
    weights = {"none": .40, "head": .15, "middle": .30, "tail": .15}
    raw = {key: total * value for key, value in weights.items()}
    result = {key: math.floor(value) for key, value in raw.items()}
    order = sorted(weights, key=lambda key: (-(raw[key] - result[key]), key))
    for key in order[:total - sum(result.values())]:
        result[key] += 1
    return result


def _partition_payloads(payloads):
    groups, duplicates = {}, []
    for row in payloads:
        payload_id = _stable("payload", normalized(row["instruction"]))
        if payload_id in groups:
            duplicates.append({"reason": "duplicate normalized Alpaca instruction", "excluded_raw_index": row["_raw_index"],
                               "canonical_raw_index": groups[payload_id]["_raw_index"], "injection_id": payload_id})
        else:
            groups[payload_id] = dict(row, _injection_id=payload_id)
    ordered = sorted(groups.values(), key=lambda row: _stable(SEED, "payload-split", row["_injection_id"]))
    val_count = round(len(ordered) * .1)
    return {"val": ordered[:val_count], "train": ordered[val_count:]}, duplicates


def _insert(base, payload, position, placement_key):
    if position == "head":
        return payload + " " + base, [0, len(payload)]
    if position == "tail":
        start = len(base) + 1
        return base + " " + payload, [start, start + len(payload)]
    if position != "middle":
        raise ValueError(f"Unknown injection position: {position}")
    parts = base.split(" ")
    index = int(_stable(SEED, "middle", placement_key), 16) % len(parts)
    prefix, suffix = " ".join(parts[:index]), " ".join(parts[index:])
    text = (prefix + " " if prefix else "") + payload + (" " + suffix if suffix else "")
    start = len(prefix) + bool(prefix)
    return text, [start, start + len(payload)]


def _source(stage, split, raw_index, raw_row, file_name, construction):
    return {"dataset": STAGES[stage]["dataset"], "revision": CHEN_REVISION,
            "upstream_split": "train" if split in {"train", "val"} else "validation",
            "record_id": str(raw_row.get("id") or f"index:{raw_index:06d}"), "construction": construction,
            "source_file": file_name, "source_file_sha256": RAW_FILES[file_name],
            "raw_record_sha256": _record_hash(raw_row), "raw_index": raw_index}


def _finish(row):
    row["text_sha256"] = digest(row["text"].encode("utf-8"))
    validate_record(row)
    return row


def construct_training_rows(stage, source_rows, payload_rows, tokenizer):
    """Construct 40/15/30/15 train/val mixtures after group splitting."""
    source_rows = [dict(row, _model_context=tokenizer.convert_tokens_to_string(
        tokenizer.tokenize(row["context"])[:384])) for row in source_rows]
    unique, base_exclusions = _split_unique(source_rows)
    payload_pools, payload_exclusions = _partition_payloads(payload_rows)
    output = []
    for split in ("train", "val"):
        members = sorted((row for row in unique if row["_split"] == split),
                         key=lambda row: _stable(SEED, stage, split, row["_duplicate_group_id"]))
        allocation = _allocation(len(members))
        positions = [position for position in ("none", "head", "middle", "tail")
                     for _ in range(allocation[position])]
        pool, injected_index = payload_pools[split], 0
        for raw, position in zip(members, positions, strict=True):
            raw_index = raw["_raw_index"]
            original = {key: value for key, value in raw.items() if not key.startswith("_")}
            # This mirrors the released HeadDataset's 384-piece preprocessing.
            base = raw["_model_context"]
            base_id = _stable(stage, "train-base", raw["_duplicate_group_id"])
            common = {
                "task_id": stage, "domain": STAGES[stage]["domain"], "split": split,
                "base_document_id": base_id, "base_document_text": base,
                "duplicate_group_id": raw["_duplicate_group_id"], "language": "en",
                "annotation_status": "chen_source_project_constructed_variant",
                "source": _source(stage, split, raw_index, original, STAGES[stage]["train_file"],
                                  "Chen HeadDataset 384-piece preprocessing; deterministic group split and mixture"),
                "upstream_context_sha256": digest(raw["context"].encode("utf-8")),
                "chen_fields": {key: value for key, value in original.items() if key != "context"},
            }
            if position == "none":
                row = dict(common, text=base, label=0, attack_family="none", injection_id=None,
                           injection_position="none", example_id=_stable(stage, split, base_id, "none"))
            else:
                payload = pool[injected_index % len(pool)]
                injected_index += 1
                text, span = _insert(base, payload["instruction"], position, base_id)
                payload_original = {key: value for key, value in payload.items() if not key.startswith("_")}
                row = dict(common, text=text, label=1, attack_family="naive",
                           injection_id=payload["_injection_id"], injection_position=position,
                           injection_text=payload["instruction"], injection_span=span,
                           example_id=_stable(stage, split, base_id, position), template_id="naive_plain_insertion_v1",
                           payload_source={"dataset": "Chen et al. released Stanford Alpaca subset",
                                           "revision": CHEN_REVISION, "upstream_split": "train",
                                           "record_id": f"index:{payload['_raw_index']:06d}",
                                           "source_file": "crafted_instruction_data_alpaca.json",
                                           "source_file_sha256": RAW_FILES["crafted_instruction_data_alpaca.json"],
                                           "raw_record_sha256": _record_hash(payload_original)})
            output.append(_finish(row))
    return output, base_exclusions, payload_exclusions


def construct_test_rows(stage, source_rows):
    """Create clean/head/middle/tail same-split variants for released benchmarks."""
    output = []
    for raw in source_rows:
        raw_index = raw["_raw_index"]
        original = {key: value for key, value in raw.items() if not key.startswith("_")}
        base = raw["input"]
        base_group = _stable("base", normalized(base))
        base_id = _stable(stage, "test-base", base_group)
        common = {"task_id": stage, "domain": STAGES[stage]["domain"], "split": "test",
                  "base_document_id": base_id, "base_document_text": base, "duplicate_group_id": base_group,
                  "language": "en", "annotation_status": "chen_benchmark_project_constructed_variant",
                  "source": _source(stage, "test", raw_index, original, STAGES[stage]["test_file"],
                                    "Four same-split variants from Chen benchmark; no pre-truncation"),
                  "chen_fields": {key: value for key, value in original.items() if key not in {"input", "injection"}}}
        output.append(_finish(dict(common, text=base, label=0, attack_family="none", injection_id=None,
                                   injection_position="none", example_id=_stable(stage, "test", base_id, "none"))))
        payload = raw["injection"]
        injection_id = _stable("payload", normalized(payload))
        for position in ("head", "middle", "tail"):
            text, span = _insert(base, payload, position, base_id)
            output.append(_finish(dict(common, text=text, label=1, attack_family="naive",
                                       injection_id=injection_id, injection_position=position,
                                       injection_text=payload, injection_span=span,
                                       example_id=_stable(stage, "test", base_id, position),
                                       template_id="naive_plain_insertion_v1",
                                       payload_source={"dataset": "Chen Inj-SQuAD/Inj-TriviaQA released payload",
                                                       "revision": CHEN_REVISION, "upstream_split": "validation",
                                                       "record_id": f"index:{raw_index:06d}",
                                                       "source_file": STAGES[stage]["test_file"],
                                                       "source_file_sha256": RAW_FILES[STAGES[stage]["test_file"]],
                                                       "raw_record_sha256": _record_hash(original)})))
    return output


def _groups(rows, field, transform=None):
    result = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None:
            result[transform(value) if transform else value].append(row)
    return result


def overlap_audit(rows):
    audits = {}
    specs = {
        "exact_text": ("text_sha256", None),
        "normalized_near_duplicate": ("text", lambda value: _stable("normalized-text", re.sub(r"[^\w]+", " ", value.casefold()).strip())),
        "base_document": ("duplicate_group_id", None),
        "injection_payload": ("injection_id", None),
        "attack_template": ("template_id", None),
    }
    for name, (field, transform) in specs.items():
        repeats = []
        for value, members in _groups(rows, field, transform).items():
            if len(members) < 2:
                continue
            stages, splits = sorted({r["task_id"] for r in members}), sorted({r["split"] for r in members})
            repeats.append({"group": value, "records": len(members), "stages": stages, "splits": splits,
                            "cross_stage": len(stages) > 1, "cross_split": len(splits) > 1,
                            "example_ids": [r["example_id"] for r in members[:10]]})
        audits[name] = {"repeated_groups": len(repeats),
                        "repeated_records_beyond_first": sum(x["records"] - 1 for x in repeats),
                        "cross_stage_groups": sum(x["cross_stage"] for x in repeats),
                        "cross_split_groups": sum(x["cross_split"] for x in repeats),
                        "sample_groups": repeats[:25]}
    parts = {split: [r for r in rows if r["split"] == split] for split in ("train", "val", "test")}
    def values(part, field):
        return {r[field] for r in part if r.get(field) is not None}
    leakage = {}
    for field in ("text_sha256", "duplicate_group_id", "injection_id"):
        leakage[field] = {"train_val": len(values(parts["train"], field) & values(parts["val"], field)),
                          "train_test": len(values(parts["train"], field) & values(parts["test"], field)),
                          "val_test": len(values(parts["val"], field) & values(parts["test"], field))}
    s2_test = values([r for r in parts["test"] if r["task_id"] == "S2"], "duplicate_group_id")
    s1_train = values([r for r in parts["train"] if r["task_id"] == "S1"], "duplicate_group_id")
    leakage["future_stage_test_in_stage1_train"] = len(s2_test & s1_train)
    blocking = [f"cross-split {field} overlap" for field, value in leakage.items()
                if isinstance(value, dict) and any(value.values())]
    if leakage["future_stage_test_in_stage1_train"]:
        blocking.append("future-stage test base appears in stage-1 training")
    return {"audits": audits, "split_leakage": leakage, "blocking_findings": sorted(set(blocking)),
            "interpretation": "Cross-stage test payload sharing is expected and prevents unseen-attack claims."}


def dataset_statistics(rows, token_lengths):
    result = []
    for stage in ("S1", "S2"):
        for split in ("train", "val", "test"):
            part = [r for r in rows if r["task_id"] == stage and r["split"] == split]
            labels, positions = Counter(r["label"] for r in part), Counter(r["injection_position"] for r in part)
            lengths = sorted(token_lengths[r["example_id"]] for r in part)
            def percentile(q):
                return lengths[max(0, math.ceil(q * len(lengths)) - 1)] if lengths else None
            result.append({"stage": stage, "split": split, "total_examples": len(part),
                           "benign_count": labels[0], "injected_count": labels[1],
                           "benign_proportion": labels[0] / len(part), "injected_proportion": labels[1] / len(part),
                           "head_count": positions["head"], "middle_count": positions["middle"], "tail_count": positions["tail"],
                           "unique_base_documents": len({r["base_document_id"] for r in part}),
                           "unique_injection_payloads": len({r["injection_id"] for r in part if r["injection_id"]}),
                           "exact_duplicate_examples": len(part) - len({r["text_sha256"] for r in part}),
                           "median_token_length": percentile(.5), "p90_token_length": percentile(.9),
                           "p95_token_length": percentile(.95), "p99_token_length": percentile(.99),
                           "max_token_length": max(lengths) if lengths else None})
    return result


def _raw_manifests(data_dir, output_dir):
    meta = {
        "crafted_instruction_data_alpaca.json": ("Stanford Alpaca-derived injection pool", "train"),
        "crafted_instruction_data_context_squad.json": ("SQuAD-derived detector contexts", "train"),
        "crafted_instruction_data_context_tri.json": ("TriviaQA-derived detector contexts", "train"),
        "crafted_instruction_data_squad_injection_qa.json": ("Inj-SQuAD benchmark", "validation"),
        "crafted_instruction_data_tri_injection_qa.json": ("Inj-TriviaQA benchmark", "validation"),
    }
    sources, records = [], []
    for file_name, expected in RAW_FILES.items():
        path, actual = Path(data_dir) / file_name, digest((Path(data_dir) / file_name).read_bytes())
        if actual != expected:
            raise ValueError(f"Raw source hash mismatch: {file_name}")
        rows, (dataset, split) = _json(path), meta[file_name]
        sources.append({"dataset": dataset, "revision": CHEN_REVISION,
                        "upstream_url": f"https://github.com/LukeChen-go/indirect-pia-detection/blob/{CHEN_REVISION}/data/{file_name}",
                        "repository": "LukeChen-go/indirect-pia-detection",
                        "license": "MIT repository license; underlying source terms are recorded in docs/real_data_audit.md",
                        "upstream_split": split, "records_obtained": len(rows),
                        "provenance_layer": "paper and public code release",
                        "local_file": f"data/raw/chen_et_al_2025/{CHEN_REVISION}/repository/data/{file_name}",
                        "file_sha256": actual, "retrieval_date": date.today().isoformat()})
        records.extend({"source": dataset, "revision": CHEN_REVISION,
                        "original_record_id": str(row.get("id") or f"index:{index:06d}"),
                        "file": file_name, "file_sha256": actual, "record_sha256": _record_hash(row),
                        "retrieval_date": date.today().isoformat(), "upstream_split": split}
                       for index, row in enumerate(rows))
    write_json(Path(output_dir) / "sources.json", sources)
    _write_jsonl(Path(output_dir) / "raw_records.jsonl", records)


def prepare(root):
    root = Path(root).resolve()
    repo = root / "data" / "raw" / "chen_et_al_2025" / CHEN_REVISION / "repository"
    data_dir = repo / "data"
    tokenizer_dir = root / "data" / "raw" / "huggingface" / "microsoft_deberta-v3-base" / TOKENIZER_REVISION
    manifest_dir, audit_dir = root / "data" / "manifests" / "real_data", root / "data" / "audits" / "real_data"
    _raw_manifests(data_dir, manifest_dir)
    tokenizer_files = []
    for path in sorted(tokenizer_dir.iterdir()):
        if path.is_file():
            tokenizer_files.append({"file": path.name, "bytes": path.stat().st_size,
                                    "sha256": digest(path.read_bytes())})
    write_json(manifest_dir / "tokenizer.json", {
        "model": "microsoft/deberta-v3-base", "revision": TOKENIZER_REVISION,
        "upstream_url": f"https://huggingface.co/microsoft/deberta-v3-base/tree/{TOKENIZER_REVISION}",
        "license": "MIT", "retrieval_date": date.today().isoformat(), "files": tokenizer_files,
    })
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=False)
    payloads = [dict(row, _raw_index=i) for i, row in enumerate(_json(data_dir / "crafted_instruction_data_alpaca.json"))]
    all_rows, exclusions, payload_exclusions = [], [], []
    for stage, spec in STAGES.items():
        source = [dict(row, _raw_index=i) for i, row in enumerate(_json(data_dir / spec["train_file"]))]
        training, base_exclusions, payload_exclusions = construct_training_rows(stage, source, payloads, tokenizer)
        for item in base_exclusions:
            item.update(stage=stage, source_file=spec["train_file"])
        exclusions.extend(base_exclusions)
        benchmark = [dict(row, _raw_index=i) for i, row in enumerate(_json(data_dir / spec["test_file"]))]
        all_rows.extend(training)
        all_rows.extend(construct_test_rows(stage, benchmark))
    exclusions.extend(payload_exclusions)
    split_order = {"train": 0, "val": 1, "test": 2}
    all_rows.sort(key=lambda row: (row["task_id"], split_order[row["split"]], row["example_id"]))
    if len({row["example_id"] for row in all_rows}) != len(all_rows):
        raise ValueError("Duplicate constructed example_id")
    processed = root / "data" / "processed" / "chen_domain_shift.jsonl"
    _write_jsonl(processed, all_rows)
    _write_jsonl(audit_dir / "source_exclusions.jsonl", exclusions)
    overlap = overlap_audit(all_rows)
    write_json(audit_dir / "overlap_audit.json", overlap)
    lengths = length_audit(all_rows, tokenizer)
    per_example = lengths.pop("per_example")
    _write_jsonl(audit_dir / "token_lengths.jsonl", per_example)
    write_json(audit_dir / "length_audit.json", lengths)
    for limit in (512, 768, 1024, 1280):
        _write_jsonl(audit_dir / f"length_exclusions_{limit}.jsonl",
                     (item for item in lengths["exclusions"] if item["max_length"] == limit))
    token_lengths = {item["example_id"]: item["token_length"] for item in per_example}
    write_json(audit_dir / "dataset_statistics.json", dataset_statistics(all_rows, token_lengths))
    construction = {
        "status": "audit_complete_training_not_run", "seed": SEED,
        "chen_revision": CHEN_REVISION, "tokenizer": "microsoft/deberta-v3-base",
        "tokenizer_revision": TOKENIZER_REVISION, "processed_file": processed.relative_to(root).as_posix(),
        "processed_sha256": digest(processed.read_bytes()), "records": len(all_rows),
        "algorithm": {
            "train_validation_split": "deduplicate normalized 384-piece model-view groups, then stable-hash order 90/10",
            "training_composition": {"benign": .40, "head": .15, "middle": .30, "tail": .15},
            "test_variants": "clean plus naive head/middle/tail for every released benchmark base",
            "payload_isolation": "Alpaca normalized payload identities partitioned 90/10; official test payloads separate",
            "training_context_preprocessing": "first 384 slow-tokenizer pieces, matching public HeadDataset",
            "model_input": "text only",
        },
        "blocking_overlap_findings": overlap["blocking_findings"], "training_executed": False,
    }
    write_json(audit_dir / "construction_manifest.json", construction)
    return construction
