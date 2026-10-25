from continualguard.attack_family_data import (
    IGNORE_TEMPLATES, SPLIT_BASE_COUNTS, STAGES, _template, construct_rows, overlap_report,
)


def _bases(n, offset=0):
    return [{"raw": {"context": f"document {i + offset}", "id": str(i + offset)}, "raw_index": i + offset,
             "base": f"document {i + offset}", "duplicate_group_id": f"group-{i + offset}"} for i in range(n)]


def _payloads(n):
    return [{"raw": {"instruction": f"payload {i}"}, "raw_index": i,
             "payload": f"payload {i}", "injection_id": f"payload-{i}"} for i in range(n)]


def test_templates_are_deterministic_and_escape_does_not_collapse_to_naive():
    assert len(IGNORE_TEMPLATES) == 10
    for stage, family, _ in STAGES:
        one = _template(stage, family, "key")
        assert one == _template(stage, family, "key")
        if family == "escape_separation":
            assert one[0].startswith(("\n", "\t"))


def test_stage_stream_is_balanced_and_cross_stage_disjoint(monkeypatch):
    monkeypatch.setattr("continualguard.attack_family_data.SPLIT_BASE_COUNTS", {"train": 6, "val": 3, "test": 3})
    rows = construct_rows(_bases(45), _bases(15, offset=100), _payloads(60))
    for stage, family, _ in STAGES:
        for split in ("train", "val", "test"):
            part = [r for r in rows if r["task_id"] == stage and r["split"] == split]
            assert len(part) == 2 * {"train": 6, "val": 3, "test": 3}[split]
            assert sum(r["label"] == 1 for r in part) == len(part) // 2
            assert {r["attack_family"] for r in part if r["label"] == 1} == {family}
    report = overlap_report(rows)
    assert report["blocking_findings"] == []
