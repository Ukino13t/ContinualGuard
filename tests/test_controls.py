import copy
import csv
import json

import pytest

from continualguard.config import load_config
from continualguard.controls import _approved_config, _epoch_summary


def metric_cell(row_id, stage, accuracy, macro_f1, attack_recall, benign_fpr):
    return {
        "trained_stage": row_id,
        "eval_stage": stage,
        "phase": "post",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "attack_recall": attack_recall,
        "benign_fpr": benign_fpr,
        "tp": 1,
        "tn": 1,
        "fp": 0,
        "fn": 0,
        "attack_support": 1,
        "benign_support": 1,
    }


def test_epoch_summary_separates_retention_and_fpr(tmp_path):
    acquisition = {"accuracy": 0.9, "macro_f1": 0.8, "attack_recall": 0.7, "benign_fpr": 0.1, "tp": 1, "tn": 1, "fp": 0, "fn": 0, "attack_support": 1, "benign_support": 1}
    cells = [metric_cell("S2-E1", "S1", 0.8, 0.6, 0.65, 0.2), metric_cell("S2-E1", "S2", 0.7, 0.5, 0.55, 0.3)]
    summary = _epoch_summary(tmp_path, "sequential_s2_epochs", cells, ["S2-E1"], ("S1", "S2"), acquisition)
    epoch = summary["epoch_results"][0]
    assert epoch["s1_bwt"] == pytest.approx({"accuracy": -0.1, "macro_f1": -0.2, "attack_recall": -0.05})
    assert epoch["s1_benign_fpr_change"] == pytest.approx(0.1)
    with (tmp_path / "matrix_attack_recall.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows == [["trained_stage", "S1", "S2"], ["S1", "0.7", ""], ["S2-E1", "0.65", "0.55"]]
    assert json.loads((tmp_path / "summary.json").read_text())["replay"] == "not implemented or run"


def test_control_contract_rejects_scope_expansion():
    config = load_config("configs/real_pilot.yaml")
    config["seed"] = 17
    _approved_config(config, "s2_only")
    bad = copy.deepcopy(config)
    bad["model"]["max_length"] = 1024
    with pytest.raises(ValueError, match="baseline-sanity contract"):
        _approved_config(bad, "s2_only")
    with pytest.raises(ValueError, match="baseline-sanity contract"):
        _approved_config(config, "reverse_order")
