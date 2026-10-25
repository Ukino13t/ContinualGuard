import copy
import json
import shutil
from pathlib import Path
import pytest
import torch
from continualguard.artifacts import read_jsonl
from continualguard.config import load_config
from continualguard.model import make_model, make_tokenizer, batch_inputs
from continualguard.evaluate import predict
from continualguard.reproducibility import seed_everything, rng_state, load_checkpoint
from continualguard.train import run, train_stage, make_optimizer

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def config():
    return load_config(ROOT / "configs/smoke.yaml")

@pytest.fixture
def work(tmp_path):
    (tmp_path / "tests/fixtures").mkdir(parents=True)
    shutil.copy(ROOT / "tests/fixtures/synthetic.jsonl", tmp_path / "tests/fixtures/synthetic.jsonl")
    return tmp_path

def same_weights(a, b):
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)

def test_metadata_isolation(config):
    tokenizer = make_tokenizer(config["model"])
    rows = read_jsonl(ROOT / config["data_path"])[:2]
    altered = [dict(r, task_id="FUTURE", domain="changed", source={}, attack_family="changed") for r in rows]
    original = batch_inputs(rows, tokenizer, 512, "cpu")
    changed = batch_inputs(altered, tokenizer, 512, "cpu")
    assert all(torch.equal(original[k], changed[k]) for k in original)
    with pytest.raises(ValueError, match="overlength"):
        batch_inputs([dict(rows[0], text="x" * 600)], tokenizer, 512, "cpu")

def test_evaluation_isolation(config):
    seed_everything(42)
    model = make_model(config["model"])
    tokenizer = make_tokenizer(config["model"])
    rows = read_jsonl(ROOT / config["data_path"])[:4]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    random_before = rng_state()
    optimizer, scheduler = make_optimizer(model, config)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    a = predict(model, tokenizer, rows, config)
    b = predict(model, tokenizer, rows, config)
    assert a == b and model.training
    same_weights(before, model.state_dict())
    assert torch.equal(random_before["torch"], rng_state()["torch"])
    assert optimizer_before == optimizer.state_dict()

def test_train_rejects_noncurrent(config, tmp_path):
    seed_everything(42)
    model = make_model(config["model"])
    optimizer, scheduler = make_optimizer(model, config)
    rows = read_jsonl(ROOT / config["data_path"])
    for forbidden in [next(r for r in rows if r["task_id"] == "S2"), next(r for r in rows if r["split"] == "test")]:
        with pytest.raises(ValueError, match="historical/future"):
            train_stage(model, make_tokenizer(config["model"]), [forbidden], config, optimizer, scheduler, tmp_path, "S1", 0)

def test_smoke_reproducibility_checkpoint_and_resume(config, work):
    first = run(config, work)
    second = run(config, work)
    assert first.name != second.name
    assert json.loads((first / "summary.json").read_text()) == json.loads((second / "summary.json").read_text())
    assert (first / "train_log.jsonl").read_bytes() == (second / "train_log.jsonl").read_bytes()
    for metric in ["accuracy", "macro_f1", "attack_recall", "benign_fpr"]:
        assert (first / f"matrix_{metric}.csv").exists()
    assert len([r for r in read_jsonl(first / "metrics.jsonl") if r["phase"] == "post"]) == 3
    confusion = read_jsonl(first / "confusion_counts.jsonl")
    assert len(confusion) == len(read_jsonl(first / "metrics.jsonl"))
    assert all({"tp", "tn", "fp", "fn", "attack_support", "benign_support"} <= row.keys() for row in confusion)
    events = [e for e in read_jsonl(first / "stage_events.jsonl") if e["event"] == "stage_start"]
    assert [e["optimizer_state_entries_before_training"] for e in events] == [0, 0]
    all_rows = read_jsonl(work / config["data_path"])
    ids = {r["example_id"]: r for r in all_rows}
    for entry in read_jsonl(first / "train_log.jsonl"):
        assert all(ids[i]["split"] == "train" and ids[i]["task_id"] == entry["stage"] for i in entry["example_ids"])
    model, state = load_checkpoint(first / "checkpoints/S2")
    expected = read_jsonl(first / "predictions/post_S2_S1.jsonl")
    actual = predict(model, make_tokenizer(config["model"]), [r for r in all_rows if r["task_id"] == "S1" and r["split"] == "test"], config)
    assert [x["attack_probability"] for x in actual] == [x["attack_probability"] for x in expected]
    resumed = run(config, work, resume=first / "checkpoints/S1")
    resumed_model, _ = load_checkpoint(resumed / "checkpoints/S2")
    same_weights(model.state_dict(), resumed_model.state_dict())
    assert json.loads((first / "summary.json").read_text()) == json.loads((resumed / "summary.json").read_text())
    assert (first / "train_log.jsonl").read_bytes() == (resumed / "train_log.jsonl").read_bytes()
    assert len(read_jsonl(work / "history/experiments.jsonl")) == 3
    assert all(r["run_type"] == "software smoke test" for r in read_jsonl(work / "history/experiments.jsonl"))

def test_future_stage_does_not_change_earlier_weights(config, work):
    original = run(config, work, stop_after=1)
    rows = read_jsonl(work / config["data_path"])
    from continualguard.artifacts import digest
    for row in rows:
        if row["task_id"] == "S2":
            row["text"] += " Future material only."
            row["text_sha256"] = digest(row["text"].encode())
    (work / config["data_path"]).write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    changed = run(config, work, stop_after=1)
    a, _ = load_checkpoint(original / "checkpoints/S1")
    b, _ = load_checkpoint(changed / "checkpoints/S1")
    same_weights(a.state_dict(), b.state_dict())
    assert (original / "train_log.jsonl").read_bytes() == (changed / "train_log.jsonl").read_bytes()

def test_optional_carry_boundary_resume(config, work):
    config["training"]["optimizer_transition"] = "carry"
    original = run(config, work)
    events = [e for e in read_jsonl(original / "stage_events.jsonl") if e["event"] == "stage_start"]
    assert events[1]["optimizer_state_entries_before_training"] > 0
    resumed = run(config, work, resume=original / "checkpoints/S1")
    a, _ = load_checkpoint(original / "checkpoints/S2")
    b, _ = load_checkpoint(resumed / "checkpoints/S2")
    same_weights(a.state_dict(), b.state_dict())

def test_unapproved_real_training_contract_is_blocked(config, work):
    config["run_type"] = "pilot experiment"
    with pytest.raises(ValueError, match="approved smoke or real sequential-baseline contract"):
        run(config, work)
