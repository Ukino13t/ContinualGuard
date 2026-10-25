import copy
import json
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from continualguard.h2_mini_no_replay import H2MiniData, approved_config, metric_block, state_fingerprint, validate_config


def test_h2_config_is_frozen():
    for seed in (17,42,2026):
        cfg=approved_config(ROOT,seed)
        validate_config(cfg)
        assert cfg['training']['learning_rate']==1e-5
        assert cfg['training']['batch_size']==4
        assert cfg['model']['max_length']==1280


def test_h2_config_rejects_replay_like_change():
    cfg=approved_config(ROOT,42)
    bad=copy.deepcopy(cfg);bad['training']['epochs']=2
    with pytest.raises(ValueError,match='approved H2-mini'):validate_config(bad)


def test_materialized_arm_contract_and_external_banks():
    for arm in ('c0','f'):
        data=H2MiniData(ROOT,arm)
        assert sum(len(data.records(stage,split)) for stage in ('AF1','AF2','AF3','AF4') for split in ('train','val','test'))==2320
        assert len(data.notinject)==339
        assert len(data.bshort)==120


def test_matched_malicious_and_shared_stage1_b0():
    c0=H2MiniData(ROOT,'c0');f=H2MiniData(ROOT,'f')
    for stage in ('AF1','AF2','AF3','AF4'):
        for split in ('train','val','test'):
            assert [r['example_id'] for r in c0.records(stage,split) if r['label']==1]==[r['example_id'] for r in f.records(stage,split) if r['label']==1]
    for split in ('train','val','test'):
        assert [r['example_id'] for r in c0.records('AF1',split) if r['label']==0]==[r['example_id'] for r in f.records('AF1',split) if r['label']==0]


def test_metric_signs_and_one_class_macro_f1():
    rows=[{'true_label':1,'predicted_label':1,'p_attack':.9},{'true_label':1,'predicted_label':0,'p_attack':.4}]
    m=metric_block(rows)
    assert m['attack_tpr']==.5 and m['attack_fnr']==.5
    assert m['benign_fpr'] is None and m['macro_f1'] is None


def test_checkpoint_payload_excludes_optimizer_source():
    source=(ROOT/'src/continualguard/h2_mini_no_replay.py').read_text(encoding='utf-8')
    function=source[source.index('def save_boundary_checkpoint'):source.index('def load_boundary_checkpoint')]
    assert 'optimizer.state_dict' not in function
    assert 'optimizer_state_saved' in function
