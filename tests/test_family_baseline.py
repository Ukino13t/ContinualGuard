from pathlib import Path

import pytest

from continualguard.family_baseline import ApprovedFamilyData, EXPECTED, FAMILIES, STAGES


def row(stage, split, label, index, position="none"):
    family = FAMILIES[stage] if label else "none"
    return {
        "text": f"text-{stage}-{split}-{label}-{index}-{position}", "label": label,
        "task_id": stage, "attack_family": family, "domain": "test", "split": split,
        "source": {}, "example_id": f"{stage}-{split}-{label}-{index}-{position}",
        "base_document_id": f"b-{stage}-{split}-{index}",
        "injection_id": f"i-{stage}-{split}-{index}" if label else None,
        "injection_position": position, "text_sha256": "0" * 64,
    }


def fixture_rows():
    # The production gate fixes 18,300 records. Patch counts to keep this unit test small.
    rows = []
    for stage in (*STAGES, "AF5"):
        family = FAMILIES.get(stage, "combined")
        for split in EXPECTED:
            rows.append({**row(STAGES[0], split, 0, 0), "task_id": stage})
            for position in ("head", "middle", "tail"):
                item = row(STAGES[0], split, 1, 0, position)
                item.update(task_id=stage, attack_family=family)
                rows.append(item)
    return rows


def test_contract_constants_are_the_approved_primary_stream():
    assert STAGES == ("AF1", "AF2", "AF3", "AF4")
    assert "AF5" not in STAGES
    assert FAMILIES == {"AF1": "naive", "AF2": "ignore", "AF3": "escape_separation", "AF4": "fake_completion"}


def test_data_gate_rejects_non_audited_record_count():
    with pytest.raises(ValueError, match="18,300"):
        ApprovedFamilyData(fixture_rows())

def test_secondary_order_contract_uses_same_audited_data():
    from continualguard.config import load_config
    from continualguard.family_baseline import SECONDARY_ORDER, validate_contract

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/attack_family_order_control.yaml")
    data = validate_contract(config, root, SECONDARY_ORDER)
    assert len(data.rows) == 14_640
    assert config["stage_order"] == ["AF1", "AF4", "AF3", "AF2"]
    assert data.audit["AF5_included"] is False

