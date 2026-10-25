import copy
import json
from pathlib import Path
import pytest
import yaml
from continualguard.artifacts import digest, read_jsonl
from continualguard.cli import audit_run, main
from continualguard.config import load_config
from continualguard.data import audit_records, validate_record
from continualguard.model import make_tokenizer

ROOT = Path(__file__).resolve().parents[1]


def test_unknown_config_and_unsafe_stage(tmp_path):
    original = load_config(ROOT / "configs/smoke.yaml")
    for update in [{"replay_capacity": 1}, {"stage_order": ["../S1"]}, {"seed": -1}]:
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump({**original, **update}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(path)


def test_original_payload_overlap_and_span():
    rows = read_jsonl(ROOT / "tests/fixtures/synthetic.jsonl")
    attacks = [r for r in rows if r["label"] == 1]
    for row in attacks:
        row["injection_text"] = row["text"][row["injection_span"][0]:]
    attacks[0]["injection_text"] = "absent attack"
    with pytest.raises(ValueError, match="does not match"):
        validate_record(attacks[0])


def test_exact_and_near_duplicates():
    rows = read_jsonl(ROOT / "tests/fixtures/synthetic.jsonl")
    a = next(r for r in rows if r["split"] == "train" and r["label"] == 0)
    b = next(r for r in rows if r["split"] == "test" and r["label"] == 0)
    b["text"] = a["text"] + " ."
    b["text_sha256"] = digest(b["text"].encode())
    report = audit_records(rows, ["S1", "S2"])
    assert report["near_duplicates"]
    assert any("Cross-split near duplicate" in e for e in report["errors"])
    b["text"] = a["text"]
    b["text_sha256"] = a["text_sha256"]
    report = audit_records(rows, ["S1", "S2"])
    assert any(d["field"] == "text_sha256" for d in report["duplicates"])


def test_unpinned_real_tokenizer_rejected():
    with pytest.raises(ValueError, match="pinned"):
        make_tokenizer({"kind": "pretrained_deberta", "checkpoint": "microsoft/deberta-v3-base", "revision": "main"})


def test_audit_artifacts_and_saved_summary(tmp_path, monkeypatch, capsys):
    config = load_config(ROOT / "configs/smoke.yaml")
    config["data_path"] = "fixture.jsonl"
    (tmp_path / "fixture.jsonl").write_bytes((ROOT / "tests/fixtures/synthetic.jsonl").read_bytes())
    output = audit_run(config, tmp_path, "test-config")
    audit = json.loads((output / "length_audit.json").read_text())
    assert {c["max_length"] for c in audit["coverage"]} == {512, 768, 1024, 1280}
    for limit in (512, 768, 1024, 1280):
        assert json.loads((output / f"exclusions_{limit}.json").read_text()) == []
    assert len(read_jsonl(tmp_path / "history/experiments.jsonl")) == 1
    monkeypatch.setattr("sys.argv", ["continualguard", "summary", "--root", str(tmp_path), "--run-id", output.name])
    main()
    assert json.loads(capsys.readouterr().out)["records"] == 56


def test_failed_audit_is_logged(tmp_path):
    config = load_config(ROOT / "configs/smoke.yaml")
    with pytest.raises(FileNotFoundError):
        audit_run(config, tmp_path, "missing-fixture")
    entry = read_jsonl(tmp_path / "history/experiments.jsonl")[0]
    assert entry["status"] == "failed" and "error" in entry
