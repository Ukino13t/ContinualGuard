import json
from pathlib import Path

from continualguard.artifacts import digest
from continualguard.data import length_audit, validate_record
from continualguard.real_data import (
    CHEN_REVISION, RAW_FILES, _partition_payloads, _split_unique,
    construct_training_rows, overlap_audit,
)

ROOT = Path(__file__).resolve().parents[1]


class WordTokenizer:
    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

    def encode(self, text, add_special_tokens=True, truncation=False):
        assert truncation is False
        ids = list(range(len(text.split())))
        return [98, *ids, 99] if add_special_tokens else ids


def contexts():
    return [{"context": f"document {i} with stable content", "id": str(i), "_raw_index": i}
            for i in range(40)]


def payloads():
    return [{"instruction": f"payload {i}", "input": "", "output": "x", "_raw_index": i}
            for i in range(20)]


def test_real_schema_and_deterministic_construction():
    one, exclusions_one, _ = construct_training_rows("S1", contexts(), payloads(), WordTokenizer())
    two, exclusions_two, _ = construct_training_rows("S1", contexts(), payloads(), WordTokenizer())
    assert one == two and exclusions_one == exclusions_two
    assert all(validate_record(row) is None for row in one)
    for split in ("train", "val"):
        part = [row for row in one if row["split"] == split]
        benign = sum(row["injection_position"] == "none" for row in part)
        assert abs(benign / len(part) - .4) <= 1 / len(part)


def test_base_group_split_isolation_and_deduplication():
    rows = contexts()
    rows.append({"context": rows[0]["context"].upper(), "id": "duplicate", "_raw_index": 40})
    kept, excluded = _split_unique(rows)
    assert len(kept) == 40 and len(excluded) == 1
    assert len({row["_duplicate_group_id"] for row in kept}) == len(kept)


def test_payload_overlap_reporting_and_no_future_contamination():
    base = {"text": "doc", "label": 1, "attack_family": "naive", "domain": "d", "split": "test",
            "source": {}, "base_document_id": "b", "injection_id": "p", "injection_position": "head",
            "text_sha256": digest(b"doc"), "duplicate_group_id": "g", "template_id": "t"}
    rows = [dict(base, task_id="S1", example_id="a"),
            dict(base, task_id="S2", example_id="b", base_document_id="c", duplicate_group_id="h")]
    report = overlap_audit(rows)
    assert report["audits"]["injection_payload"]["cross_stage_groups"] == 1
    assert report["split_leakage"]["future_stage_test_in_stage1_train"] == 0


def test_length_span_integrity():
    row = {"text": "one two three PAY LOAD tail", "label": 1, "task_id": "S1", "split": "test",
           "injection_position": "middle", "example_id": "x", "injection_span": [14, 22]}
    report = length_audit([row], WordTokenizer(), lengths=(5, 7, 8))
    per = report["per_example"][0]["candidates"]
    assert per["5"]["injection_span_fully_present_if_right_truncated"] is False
    assert per["7"]["injection_span_fully_present_if_right_truncated"] is True
    assert per["8"]["retained_without_truncation"] is True


def test_source_manifest_hash_contract():
    data = ROOT / "data" / "raw" / "chen_et_al_2025" / CHEN_REVISION / "repository" / "data"
    assert len(CHEN_REVISION) == 40 and all(len(value) == 64 for value in RAW_FILES.values())
    for name, expected in RAW_FILES.items():
        assert digest((data / name).read_bytes()) == expected


def test_payload_partition_disjoint():
    pools, _ = _partition_payloads(payloads())
    train = {row["_injection_id"] for row in pools["train"]}
    val = {row["_injection_id"] for row in pools["val"]}
    assert train.isdisjoint(val)


def test_materialized_real_schema_split_isolation_and_no_future_contamination():
    path = ROOT / "data" / "processed" / "chen_domain_shift.jsonl"
    split_bases = {split: set() for split in ("train", "val", "test")}
    split_payloads = {split: set() for split in ("train", "val", "test")}
    s1_train, s2_test, count = set(), set(), 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            validate_record(row)
            count += 1
            split_bases[row["split"]].add(row["duplicate_group_id"])
            if row["injection_id"]:
                split_payloads[row["split"]].add(row["injection_id"])
            if row["task_id"] == "S1" and row["split"] == "train":
                s1_train.add(row["duplicate_group_id"])
            if row["task_id"] == "S2" and row["split"] == "test":
                s2_test.add(row["duplicate_group_id"])
    assert count == 43542
    assert split_bases["train"].isdisjoint(split_bases["val"] | split_bases["test"])
    assert split_bases["val"].isdisjoint(split_bases["test"])
    assert split_payloads["train"].isdisjoint(split_payloads["val"] | split_payloads["test"])
    assert split_payloads["val"].isdisjoint(split_payloads["test"])
    assert s1_train.isdisjoint(s2_test)
