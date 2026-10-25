"""Strict schema, conservative leakage audit, and stage-scoped access."""
from collections import defaultdict
import re
from .artifacts import digest, read_jsonl

REQUIRED = {"text", "label", "task_id", "attack_family", "domain", "split", "source", "example_id", "base_document_id", "injection_id", "injection_position", "text_sha256"}


def normalized(text):
    return " ".join(text.casefold().split())


def validate_record(row):
    if not isinstance(row, dict) or not REQUIRED <= row.keys():
        raise ValueError("Missing required schema fields")
    for field in ("text", "task_id", "attack_family", "domain", "example_id", "base_document_id"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise ValueError(f"Invalid {field}")
    if type(row["label"]) is not int or row["label"] not in (0, 1):
        raise ValueError("label must be integer 0 or 1")
    if row["split"] not in {"train", "val", "test"}:
        raise ValueError("Invalid split")
    if row["injection_position"] not in {"none", "head", "middle", "tail", "unknown"}:
        raise ValueError("Invalid injection position")
    source = row["source"]
    if not isinstance(source, dict) or not all(isinstance(source.get(k), str) and source[k].strip() for k in ("dataset", "revision", "upstream_split", "record_id", "construction")):
        raise ValueError("Source must preserve dataset/revision/upstream_split/record_id/construction")
    if row["text_sha256"] != digest(row["text"].encode("utf-8")):
        raise ValueError("Text hash mismatch")
    if row["label"] == 0:
        if row["attack_family"] != "none" or row["injection_position"] != "none" or row["injection_id"] is not None:
            raise ValueError("Inconsistent benign metadata")
    elif row["attack_family"] == "none" or row["injection_position"] == "none" or not isinstance(row["injection_id"], str) or not row["injection_id"]:
        raise ValueError("Injection requires nonempty identity and attack metadata")
    span = row.get("injection_span")
    for field in ("injection_text", "base_document_text", "duplicate_group_id", "attack_goal", "template_id",
                  "attack_template_id", "normalized_template_hash", "language", "benign_group", "annotation_status"):
        if row.get(field) is not None and (not isinstance(row[field], str) or not row[field].strip()):
            raise ValueError(f"Invalid optional field {field}")
    if span is not None:
        if row["label"] != 1 or not isinstance(span, list) or len(span) != 2 or not all(type(x) is int for x in span) or not 0 <= span[0] < span[1] <= len(row["text"]):
            raise ValueError("Invalid character injection_span")
        if row.get("injection_text") and row["text"][span[0]:span[1]] != row["injection_text"]:
            raise ValueError("injection_span does not match original injection_text")
    if row.get("injection_text") and (row["label"] != 1 or row["injection_text"] not in row["text"]):
        raise ValueError("Injection text is missing from positively labeled document")


def audit_records(rows, stage_order):
    errors, warnings, near_duplicates, duplicates = [], [], [], []
    groups = {key: defaultdict(list) for key in ("example_id", "base_document_id", "text_sha256", "normalized", "duplicate_group_id", "injection_id", "base_document_text", "injection_text")}
    for row in rows:
        validate_record(row)
        if row["task_id"] not in stage_order:
            errors.append(f"Unknown stage: {row['task_id']}")
        for key, table in groups.items():
            value = normalized(row["text"]) if key == "normalized" else row.get(key)
            if key in {"base_document_text", "injection_text"} and isinstance(value, str):
                value = digest(normalized(value).encode())
            if value is not None:
                table[value].append(row)
    for key, table in groups.items():
        for value, members in table.items():
            splits = {r["split"] for r in members}
            stages = {r["task_id"] for r in members}
            if len(members) > 1:
                duplicates.append({"field": key, "group": value, "example_ids": [r["example_id"] for r in members]})
            if key in {"text_sha256", "normalized"} and len({r["label"] for r in members}) > 1:
                errors.append(f"Conflicting labels for identical {key}: {value}")
            if key == "example_id" and len(members) > 1:
                errors.append(f"Duplicate example_id: {value}")
            if len(splits) > 1:
                errors.append(f"Cross-split {key}: {value}")
            if key not in {"injection_id", "injection_text", "example_id"} and len(stages) > 1 and any(r["split"] == "train" for r in members):
                errors.append(f"Repeated historical training group {key}: {value}")
            if key == "injection_id" and len(stages) > 1:
                warnings.append(f"Shared payload {value}; stages are not unseen-payload tests")
    # Token-set Jaccard candidate audit. Conservative and deterministic; no model statistics.
    tokens = [set(re.findall(r"\w+", normalized(r["text"]))) for r in rows]
    inverted = defaultdict(list)
    for i, token_set in enumerate(tokens):
        candidates = set(j for token in token_set for j in inverted[token])
        for j in sorted(candidates):
            union = token_set | tokens[j]
            score = len(token_set & tokens[j]) / len(union) if union else 0
            if score >= 0.9 and rows[i]["text_sha256"] != rows[j]["text_sha256"]:
                pair = {"a": rows[j]["example_id"], "b": rows[i]["example_id"], "jaccard": score}
                near_duplicates.append(pair)
                if rows[i]["split"] != rows[j]["split"]:
                    errors.append(f"Cross-split near duplicate: {pair['a']} / {pair['b']}")
        for token in token_set:
            inverted[token].append(i)
    for stage in stage_order:
        for split in ("train", "val", "test"):
            labels = {r["label"] for r in rows if r["task_id"] == stage and r["split"] == split}
            if labels != {0, 1}:
                errors.append(f"Primary stage {stage}/{split} must contain both labels")
    missing_original = [r["example_id"] for r in rows if r["source"]["dataset"] != "synthetic_software_fixture" and (not r.get("base_document_text") or (r["label"] == 1 and not r.get("injection_text")))]
    if missing_original:
        warnings.append("Original base/payload text unavailable for some records; ID-only lineage cannot establish content independence")
    return {"errors": sorted(set(errors)), "warnings": sorted(set(warnings)), "duplicates": duplicates, "near_duplicates": near_duplicates, "missing_original_content_ids": missing_original, "records": len(rows)}


class StageData:
    def __init__(self, rows, stage_order):
        report = audit_records(rows, stage_order)
        if report["errors"]:
            raise ValueError("; ".join(report["errors"]))
        self.rows, self.stage_order, self.audit = rows, list(stage_order), report

    @classmethod
    def from_path(cls, path, stage_order):
        return cls(read_jsonl(path), stage_order)

    def records(self, stage, split):
        if stage not in self.stage_order or split not in {"train", "val", "test"}:
            raise ValueError("Invalid stage or split")
        return [r for r in self.rows if r["task_id"] == stage and r["split"] == split]


def length_audit(rows, tokenizer, lengths=(512, 768, 1024, 1280)):
    counts = defaultdict(lambda: {"total": 0, "retained": 0, "excluded": 0})
    exclusions, per_example = [], []
    for row in rows:
        size = len(tokenizer.encode(row["text"], add_special_tokens=True, truncation=False))
        span = row.get("injection_span")
        span_end_length = None
        if row["label"] == 1:
            if span is None:
                raise ValueError(f"Injected example lacks injection_span: {row['example_id']}")
            # The detector uses right padding.  Tokenizing the prefix through the
            # injected span gives a conservative boundary for right truncation.
            span_end_length = len(tokenizer.encode(row["text"][:span[1]], add_special_tokens=True, truncation=False))
        example = {
            "example_id": row["example_id"], "stage": row["task_id"],
            "split": row["split"], "label": row["label"],
            "injection_position": row["injection_position"],
            "token_length": size, "injection_span_end_token_length": span_end_length,
            "candidates": {},
        }
        for limit in lengths:
            retained = size <= limit
            span_fully_present = None if span_end_length is None else span_end_length <= limit
            example["candidates"][str(limit)] = {
                "retained_without_truncation": retained,
                "injection_span_fully_present_if_right_truncated": span_fully_present,
            }
            if retained and span_fully_present is False:
                raise ValueError(f"Retained positive would lose its injection span: {row['example_id']}")
            for dimensions in (
                (row["task_id"], "all", "all", "all"),
                (row["task_id"], row["split"], "all", "all"),
                (row["task_id"], row["split"], str(row["label"]), "all"),
                (row["task_id"], row["split"], "all", row["injection_position"]),
                (row["task_id"], row["split"], str(row["label"]), row["injection_position"]),
            ):
                item = counts[(limit, *dimensions)]
                item["total"] += 1
                item["retained" if retained else "excluded"] += 1
            if not retained:
                exclusions.append({
                    "example_id": row["example_id"], "max_length": limit,
                    "token_length": size, "stage": row["task_id"],
                    "split": row["split"], "label": row["label"],
                    "injection_position": row["injection_position"],
                    "injection_span_fully_present_if_right_truncated": span_fully_present,
                    "reason": "complete input exceeds limit; no truncation",
                })
        per_example.append(example)
    coverage = []
    for key, value in sorted(counts.items()):
        coverage.append({
            "max_length": key[0], "stage": key[1], "split": key[2],
            "label": key[3], "injection_position": key[4], **value,
            "retained_fraction": value["retained"] / value["total"],
            "excluded_fraction": value["excluded"] / value["total"],
        })
    return {"coverage": coverage, "exclusions": exclusions, "per_example": per_example}
