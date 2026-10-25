import torch
from continualguard.notinject_evaluation import labels_and_probabilities, model_input_records, summarize_checkpoint


def synthetic_rows():
    groups = ["HB1"] * 113 + ["HB2"] * 113 + ["HB3"] * 113
    categories = ["Common"] * 126 + ["Technique"] * 87 + ["Virtual Creation"] * 42 + ["Multilingual"] * 84
    return [
        {
            "record_id": f"r{index}",
            "prompt": f"prompt {index}",
            "true_label": 0,
            "hb_group": group,
            "category": category,
        }
        for index, (group, category) in enumerate(zip(groups, categories))
    ]


def test_model_input_contains_prompt_only():
    records = synthetic_rows()[:2]
    isolated = model_input_records(records)
    assert isolated == [{"text": "prompt 0"}, {"text": "prompt 1"}]
    assert all(set(row) == {"text"} for row in isolated)


def test_frozen_threshold_uses_original_argmax():
    labels, probabilities = labels_and_probabilities(torch.tensor([[0.0, 0.0], [0.0, 0.1], [0.1, 0.0]]))
    assert labels == [0, 1, 0]
    assert probabilities[0] == 0.5
    assert probabilities[1] > 0.5
    assert probabilities[2] < 0.5


def test_exact_339_and_group_category_aggregation():
    records = synthetic_rows()
    predictions = []
    for index, row in enumerate(records):
        predictions.append({**row, "predicted_label": int(index % 10 == 0), "p_attack": index / 1000})
    result = summarize_checkpoint(predictions)
    assert result["records"] == 339
    assert sum(item["count"] for item in result["group_fpr"].values()) == 339
    assert [result["group_fpr"][key]["count"] for key in ("HB1", "HB2", "HB3")] == [113, 113, 113]
    assert {key: value["count"] for key, value in result["category_fpr"].items()} == {
        "Common": 126, "Technique": 87, "Virtual Creation": 42, "Multilingual": 84
    }
    assert result["false_positives"] == 34
    assert result["benign_accuracy"] == 1 - result["overall_fpr"]


def test_summary_rejects_non_339_bank():
    try:
        summarize_checkpoint(synthetic_rows()[:-1])
    except ValueError as error:
        assert "339" in str(error)
    else:
        raise AssertionError("Expected frozen-bank count validation")
