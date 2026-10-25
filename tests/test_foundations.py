import copy
import pytest
from continualguard.artifacts import read_jsonl
from continualguard.config import load_config
from continualguard.data import StageData, validate_record, audit_records, length_audit
from continualguard.metrics import classification_metrics, summarize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def rows():
    return read_jsonl(ROOT / "tests/fixtures/synthetic.jsonl")


def test_schema(rows):
    validate_record(rows[0])
    for key, value in [("text", ""), ("label", True), ("text_sha256", "bad"), ("split", "future")]:
        broken = dict(rows[0], **{key: value})
        with pytest.raises(ValueError):
            validate_record(broken)


def test_order(rows):
    data = StageData(rows, ["S2", "S1"])
    assert data.stage_order == ["S2", "S1"]
    assert all(r["task_id"] == "S2" for r in data.records("S2", "train"))


def test_leakage(rows):
    broken = copy.deepcopy(rows)
    broken[-1]["base_document_id"] = broken[0]["base_document_id"]
    assert any("Cross-split base_document_id" in e for e in audit_records(broken, ["S1", "S2"])["errors"])
    broken = copy.deepcopy(rows)
    attack = next(r for r in broken if r["label"] == 1 and r["split"] == "train")
    next(r for r in broken if r["label"] == 1 and r["split"] == "test")["injection_id"] = attack["injection_id"]
    assert any("injection_id" in e for e in audit_records(broken, ["S1", "S2"])["errors"])


def test_metrics():
    m = classification_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert all(m[k] == .5 for k in ["accuracy", "macro_f1", "attack_recall", "benign_fpr"])
    m = classification_metrics([0, 0], [0, 1])
    assert m["attack_recall"] is None and m["macro_f1"] is None
    assert classification_metrics([], [])["accuracy"] is None
    assert classification_metrics([0, 1], [0, 0])["macro_f1"] == pytest.approx(1/3)


def test_cl_signs():
    cells = []
    for trained, ev, value in [("S1", "S1", .8), ("S2", "S1", .6), ("S2", "S2", .9)]:
        cells.append(dict(trained_stage=trained, eval_stage=ev, phase="post", accuracy=value, macro_f1=value, attack_recall=value, benign_fpr=1-value))
    result = summarize(cells, ["S1", "S2"])
    assert result[0]["accuracy"]["bwt"] is None
    assert result[1]["accuracy"]["bwt"] == pytest.approx(-.2)
    assert result[1]["accuracy"]["peak_forgetting"] == pytest.approx(.2)
    assert result[1]["benign_fpr"]["increase_from_acquisition"] == pytest.approx(.2)
    assert "bwt" not in result[1]["benign_fpr"]


def test_peak_forgetting_is_zero_when_prior_task_improves():
    cells = []
    for trained, ev, value in [("S1", "S1", .8), ("S2", "S1", .9), ("S2", "S2", .7)]:
        cells.append(dict(trained_stage=trained, eval_stage=ev, phase="post", accuracy=value, macro_f1=value, attack_recall=value, benign_fpr=1-value))
    result = summarize(cells, ["S1", "S2"])
    assert result[1]["accuracy"]["bwt"] == pytest.approx(.1)
    assert result[1]["accuracy"]["peak_forgetting"] == pytest.approx(0.0)


def test_config():
    assert load_config(ROOT / "configs/smoke.yaml")["training"]["optimizer_transition"] == "reset"
    assert load_config(ROOT / "configs/real_pilot.yaml")["model"]["max_length"] == 1280


def test_length_no_truncation(rows):
    class Tokens:
        def encode(self, text, **kwargs):
            assert kwargs["truncation"] is False
            return list(range(600))
    report = length_audit(rows[:1], Tokens())
    assert report["exclusions"][0]["max_length"] == 512
    assert all(c["excluded"] == 0 for c in report["coverage"] if c["max_length"] >= 768)
