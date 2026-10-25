import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def load(name): return json.loads((ROOT/'results'/name).read_text(encoding='utf-8'))

def test_h2_no_replay_artifacts_complete_and_validated():
    a=load('h2_mini_no_replay_aggregate.json')
    assert a['status']=='COMPLETE_VALIDATED'
    assert (a['runs'],a['stage_end_checkpoints'],a['failed_runs'],a['resumed_runs'])==(6,24,0,0)
    assert a['artifact_validation']['checkpoint_files_hashed']==168
    assert a['artifact_validation']['prediction_files']==108

def test_h2_paired_initialization_and_stage1_equivalence():
    v=load('h2_mini_no_replay_artifact_validation.json')
    for seed in ('17','42','2026'):
        assert v['initialization_hashes']['c0'][seed]==v['initialization_hashes']['f'][seed]
        assert all(v['stage1_paired_equivalence'][seed].values())

def test_h2_all_acquisition_gates_pass():
    r=load('h2_mini_no_replay_readiness.json')
    assert len(r['acquisition_gates'])==8
    assert all(x['gate_pass'] and x['passing_seeds']==3 for x in r['acquisition_gates'])

def test_h2_readiness_remains_r0_without_replay_budget():
    r=load('h2_mini_no_replay_readiness.json')
    assert r['classification']=='R0'
    assert not r['R1_numeric_candidates'] and not r['R2_numeric_candidates']
    assert r['replay_scientifically_justified'] is False
    assert r['numerical_memory_budget'] is None

def test_h2_figure_tables_and_position_fields_exist():
    base=ROOT/'results/h2_mini_no_replay/tables'
    expected={'attack_tpr_trajectory.csv','benign_fpr_trajectory.csv','external_fpr_trajectory.csv','acquisition.csv','retention_by_task.csv','retention_bwt_by_run.csv','paired_F_minus_C0.csv','paired_external_F_minus_C0.csv'}
    assert expected=={p.name for p in base.glob('*.csv')}
    header=(base/'attack_tpr_trajectory.csv').read_text(encoding='utf-8').splitlines()[0]
    assert all(x in header for x in ('attack_tpr_head','attack_tpr_middle','attack_tpr_tail'))
