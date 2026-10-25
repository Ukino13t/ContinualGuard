"""Record the completed H2-D1 evaluation in project documentation and history."""
from __future__ import annotations
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs'
HISTORY=ROOT/'history'
A=json.loads((ROOT/'results/notinject_checkpoint_evaluation_aggregate.json').read_text(encoding='utf-8'))
V=json.loads((ROOT/'results/notinject_checkpoint_evaluation_artifact_validation.json').read_text(encoding='utf-8'))
NOW=datetime.now().astimezone().isoformat()


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def marker(path,name,block):
    text=path.read_text(encoding='utf-8-sig')
    start=f'<!-- {name}:BEGIN -->'; end=f'<!-- {name}:END -->'
    text=re.sub(r'\n?'+re.escape(start)+r'.*?'+re.escape(end)+r'\n?', '\n', text, flags=re.S)
    path.write_text(text.rstrip()+f'\n\n{start}\n{block.rstrip()}\n{end}\n',encoding='utf-8')


def replace_row(path,qid,row):
    text=path.read_text(encoding='utf-8-sig')
    text=re.sub(rf'^\| {re.escape(qid)} \|.*$',row,text,flags=re.M)
    path.write_text(text,encoding='utf-8')


def upsert_jsonl(path,record,key):
    rows=[json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    rows=[x for x in rows if x.get(key)!=record[key]]+[record]
    path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')

readiness='''# Replay readiness review

## Current classification

**R0 — replay remains unjustified.** R1 and R2 are not entered. Replay and selective adaptation remain blocked.

The completed family-stream controls still show near-ceiling attack acquisition and small, seed/order-dependent final retention changes: primary final attack-TPR BWT is `-0.185 ± 0.185` percentage points and secondary BWT is `-0.062 ± 0.283` points.

## H2-D1 result

All 24 existing stage-end checkpoints were evaluated on the frozen 339-record NotInject bank, producing 8,136 predictions. AF1 checkpoint FPR is `96.067% ± 1.802 pp` across seeds. This is severe hard-benign over-defense despite 100% AF1 attack TPR and macro-F1.

The FPR does not worsen consistently across the sequence. Primary final FPR is `84.464% ± 8.123 pp`, a `-11.603 ± 9.339 pp` change from AF1 with decreases in all seeds. Secondary final FPR is `95.772% ± 4.506 pp`, a `-0.295 ± 5.628 pp` mean change; two seeds rise and one falls. AF4 immediately after AF1 reduces FPR in all seeds (`-27.729 ± 28.871 pp`) and AF3 often reverses that improvement.

These results establish over-defense, order sensitivity, seed sensitivity, and benign-boundary movement. They do not establish persistent continual forgetting. NotInject is evaluation-only and cannot show whether a hard-benign stage can be acquired or retained.

## Readiness implication

H2-D1 alone does not satisfy R1 or R2. The high absolute FPR strengthens the case for constructing H2-F with a separately licensed hard-benign training source and a matched short-form ordinary-benign control. That construction requires human approval and a materialized data audit before any training. It does not justify replay.

See `docs/notinject_checkpoint_evaluation.md` and `docs/replay_entry_criteria.md`.'''
(DOCS/'replay_readiness_review.md').write_text(readiness+'\n',encoding='utf-8')

marker(DOCS/'experiment_history.md','NOTINJECT_H2D1_20260913','''## 2026-09-13 — H2-D1 frozen NotInject checkpoint evaluation

Evaluated 24 existing stage-end checkpoints from the primary and secondary AF1-AF4 no-replay streams on all 339 pinned NotInject records, for 8,136 predictions. No training or selection occurred. AF1 mean FPR was 96.067% ± 1.802 pp. Primary final change was -11.603 ± 9.339 pp; secondary final change was -0.295 ± 5.628 pp. Over-defense is severe, but sequential worsening is not persistent or three-seed consistent. All prediction, aggregate, reproducibility, and checkpoint/optimizer immutability validations passed. Replay readiness remains R0.''')

marker(DOCS/'decision_log.md','DECISION_D025','''## D025

- **date:** 2026-09-13
- **decision:** Record severe NotInject over-defense but retain R0 because H2-D1 does not show persistent, seed-consistent sequential worsening.
- **motivation:** AF1 FPR is 96.067% ± 1.802 pp, while primary final FPR falls in every seed and secondary final changes have mixed signs. AF4 immediately after AF1 lowers FPR in every seed before later rebound.
- **consequences:** H2-F construction is more strongly motivated, with a matched short-form ordinary-benign control. H2-F training, replay, and selective adaptation remain blocked.
- **status:** H2-D1 complete and human review required for the next data-construction step.''')

marker(DOCS/'change_log.md','CHANGE_C014','''## C014

- **date:** 2026-09-13
- **reason:** Execute the human-approved H2-D1 evaluation-only NotInject diagnostic.
- **effect:** Adds prompt-only inference for 24 frozen checkpoints, 8,136 raw predictions, aggregate/group/category/confidence/trajectory/trade-off outputs, and independent artifact validation.
- **verification:** 24×339 counts, source slices, fixed threshold, no training API, aggregate recomputation, 16-record repeat inference, and pre/post checkpoint plus optimizer-state hashes passed.
- **scientific_behavior_changed:** false; existing models were evaluated only.
- **limitations:** Short prompt input may add context/domain shift; checkpoints are correlated states; no H2-F data, training, replay, or selective adaptation.''')

marker(DOCS/'assumptions_and_risks.md','RISKS_R051_R054','''| R051 | Existing ordinary-benign tests hide a severe hard-benign boundary failure: AF1 NotInject FPR is 96.067% ± 1.802 pp | high | Keep NotInject frozen and report it separately from attack metrics | confirmed |
| R052 | NotInject prompt-only inputs are shorter and differently framed than Chen-derived document contexts | high | Add a matched short-form ordinary-benign control to H2-F; avoid attributing all absolute FPR to trigger words | open design requirement |
| R053 | Group/category composition is confounded and checkpoint states are correlated | medium | Report every cell and seed; treat across-checkpoint means as descriptive | accepted limitation |
| R054 | FP16 repeated probabilities can differ at the few-e-6 level while labels remain identical | low | Record a 1e-5 tolerance and require identical labels | measured and mitigated |''')

replace_row(DOCS/'open_questions.md','Q021','| Q021 | May H2-D1 evaluate existing checkpoints on all 339 frozen NotInject records? | Measures over-defense without retraining | Completed with 24 checkpoints and 8,136 predictions | Passed artifact validation | resolved 2026-09-13 |')
replace_row(DOCS/'open_questions.md','Q022','| Q022 | Are the four H2-F stages, exact counts, and matched H2-C0 control accepted? | Fixes interpretation before construction | Accepted as design; add a matched short-form ordinary-benign control | H2-D1 context-shift finding | resolved at design level 2026-09-13 |')
marker(DOCS/'open_questions.md','QUESTIONS_POST_H2D1','''| Q026 | How much of the absolute NotInject FPR is trigger bias versus short-prompt/context shift? | Affects H2-F attribution | Construct a matched short-form ordinary-benign diagnostic before training | Licensed source and matched design | open human decision |
| Q027 | May a separately licensed B1-B3 source be acquired and materialized for H2-F audit? | H2-D1 strongly motivates a bidirectional benchmark | Approve data construction/audit only, then stop before training | Source, license, provenance, labels, overlap, and exact-count manifest | open human decision |''')

agg=ROOT/'results/notinject_checkpoint_evaluation_aggregate.json'
pred=ROOT/'results/notinject_checkpoint_evaluation_predictions.jsonl'
val=ROOT/'results/notinject_checkpoint_evaluation_artifact_validation.json'
experiment={
    'run_id':'notinject-checkpoint-evaluation-20260913', 'date':NOW,
    'experiment_regime':'H2-D1_frozen_hard_benign_checkpoint_evaluation',
    'dataset':{'id':'leolee99/NotInject','revision':'847ae76cf8fea5ed325429e569ae8cfef022d2e0','records':339,'label':0},
    'orders':A['orders'],'seeds':[17,42,2026],'checkpoint_count':24,'prediction_count':8136,
    'model_input':'prompt text only','threshold_rule':A['threshold_rule'],
    'result_summary':{'AF1_fpr_mean':0.9606686332350048,'AF1_fpr_sample_sd':0.01802389653866606,'primary_final_change_pp_mean':-11.602753195673547,'primary_final_change_pp_sample_sd':9.33912930532964,'secondary_final_change_pp_mean':-0.2949852507374599,'secondary_final_change_pp_sample_sd':5.62795989036546,'interpretation':'severe over-defense; no persistent seed-consistent sequential worsening; R0'},
    'artifacts':{'aggregate':str(agg.relative_to(ROOT)),'aggregate_sha256':sha(agg),'predictions':str(pred.relative_to(ROOT)),'predictions_sha256':sha(pred),'validation':str(val.relative_to(ROOT)),'validation_sha256':sha(val)},
    'validation_status':V['status'],'training':False,'optimizer_loaded':False,'threshold_changed':False,'model_selection':False,'H2_F_constructed':False,'replay':False,'selective_adaptation':False,'status':'complete'
}
decision={'date':NOW,'id':'D025','decision':'Record severe NotInject over-defense but retain R0 because H2-D1 does not show persistent seed-consistent sequential worsening','motivation':'AF1 FPR is 96.067% +/- 1.802 pp; primary final FPR falls in every seed and secondary final changes have mixed signs','alternatives_considered':'Advance to replay from absolute FPR alone; call the result catastrophic forgetting; train on NotInject','source_evidence':'Validated 24-checkpoint/8136-prediction H2-D1 aggregate and pre/post checkpoint hashes','consequences':'H2-F construction with a matched short-form benign control is more strongly motivated; training and replay remain blocked','status':'accepted analytical stop; human review required'}
change={'date':NOW,'id':'C014','files_changed':['src/continualguard/notinject_evaluation.py','scripts/run_notinject_checkpoint_evaluation.py','scripts/verify_notinject_checkpoint_evaluation.py','scripts/finalize_notinject_checkpoint_evaluation.py','scripts/summarize_notinject_checkpoint_evaluation.py','scripts/render_notinject_checkpoint_evaluation_report.py','tests/test_notinject_evaluation.py','results/notinject_checkpoint_evaluation_*','docs/notinject_checkpoint_evaluation.md','docs/replay_readiness_review.md','docs/experiment_history.md','docs/decision_log.md','docs/change_log.md','docs/assumptions_and_risks.md','docs/open_questions.md','history/experiments.jsonl','history/decisions.jsonl','history/changes.jsonl'],'reason':'Execute approved H2-D1 frozen checkpoint evaluation','effect':'Adds validated prompt-only predictions and over-defense/order/seed analysis while retaining R0','tests_run':['focused pytest: 4 passed','artifact validation: all checks passed','full pytest/compileall/pip check pending final verification'],'scientific_behavior_changed':False,'limitations':['short-prompt context shift may contribute','no H2-F data or training','no replay or selective adaptation']}
upsert_jsonl(HISTORY/'experiments.jsonl',experiment,'run_id')
upsert_jsonl(HISTORY/'decisions.jsonl',decision,'id')
upsert_jsonl(HISTORY/'changes.jsonl',change,'id')
print('recorded H2-D1 documentation and history')
