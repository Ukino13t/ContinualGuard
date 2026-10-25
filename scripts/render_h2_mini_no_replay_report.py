"""Render the completed H2-mini no-replay report and synchronize decision history."""
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
A=json.loads((ROOT/'results/h2_mini_no_replay_aggregate.json').read_text(encoding='utf-8'))
P=json.loads((ROOT/'results/h2_mini_no_replay_pairwise.json').read_text(encoding='utf-8'))
R=json.loads((ROOT/'results/h2_mini_no_replay_readiness.json').read_text(encoding='utf-8'))
V=json.loads((ROOT/'results/h2_mini_no_replay_artifact_validation.json').read_text(encoding='utf-8'))
MARK='H2_MINI_NO_REPLAY_COMPLETE_20261005'
SEEDS=(17,42,2026)
STAGES=('AF1','AF2','AF3','AF4')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pp(x): return f'{100*x:.3f}'
def ms(values):
 values=list(values); return f'{100*statistics.mean(values):.3f} +/- {100*statistics.stdev(values):.3f}'
def vec(values): return ' / '.join(pp(x) for x in values)
def prepend(rel,body):
 p=ROOT/rel;text=p.read_text(encoding='utf-8-sig')
 if MARK in text:return
 lines=text.splitlines();i=1 if lines and lines[0].startswith('#') else 0
 lines[i:i]=['',f'<!-- {MARK}:BEGIN -->',body.strip(),f'<!-- {MARK}:END -->','']
 p.write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8')

def acq_rows():
 out=[]
 for arm in ('H2m-C0','H2m-F'):
  for stage in STAGES:
   rows=sorted((x for x in A['acquisition'] if x['arm']==arm and x['stage']==stage),key=lambda x:x['seed'])
   out.append('| %s | %s | %s | %s | %s | %s | %d/3 |'%(
    arm,stage,vec(x['attack_tpr'] for x in rows),vec(x['benign_tnr'] for x in rows),
    vec(x['benign_fpr'] for x in rows),vec(x['paired_macro_f1'] for x in rows),sum(x['gate_pass'] for x in rows)))
 return '\n'.join(out)

def retention_rows():
 out=[]
 for arm in ('H2m-C0','H2m-F'):
  rows=sorted((x for x in A['retention_by_run'] if x['arm']==arm),key=lambda x:x['seed'])
  out.append('| %s | %s | %s | %s | %s | %s |'%(
   arm,vec(x['attack_tpr_bwt'] for x in rows),vec(x['paired_macro_f1_bwt'] for x in rows),
   vec(x['paired_accuracy_bwt'] for x in rows),vec(x['benign_fpr_increase'] for x in rows),
   ms(x['benign_fpr_increase'] for x in rows)))
 return '\n'.join(out)

def task_rows():
 out=[]
 for arm in ('H2m-C0','H2m-F'):
  for task in STAGES[:-1]:
   rows=sorted((x for x in A['retention_by_task'] if x['arm']==arm and x['task']==task),key=lambda x:x['seed'])
   out.append('| %s | %s | %s | %s | %s | %s |'%(
    arm,task,vec(x['attack_tpr_change'] for x in rows),vec(x['paired_macro_f1_change'] for x in rows),
    vec(x['benign_fpr_change'] for x in rows),vec(x['benign_peak_fpr_increase'] for x in rows)))
 return '\n'.join(out)

def external_rows():
 out=[]
 for arm in ('H2m-C0','H2m-F'):
  for stage in STAGES:
   rows=sorted((x for x in A['external_trajectory'] if x['arm']==arm and x['checkpoint_stage']==stage),key=lambda x:x['seed'])
   out.append('| %s | %s | %s | %s | %s | %s |'%(
    arm,stage,ms(x['notinject_fpr'] for x in rows),ms(x['bshort_fpr'] for x in rows),
    ms(x['bshort_english_fpr'] for x in rows),ms(x['bshort_multilingual_fpr'] for x in rows)))
 return '\n'.join(out)

def pair_rows():
 out=[]
 for stage in STAGES:
  x=next(y for y in P['stage_aggregate'] if y['checkpoint_stage']=='AF4' and y['evaluated_stage']==stage)
  out.append('| %s | %s | %s | %s | %s |'%(
   stage,pp(x['attack_tpr_difference_F_minus_C0']['mean']),
   f"{pp(x['paired_macro_f1_difference_F_minus_C0']['mean'])} +/- {pp(x['paired_macro_f1_difference_F_minus_C0']['sample_sd'])}",
   f"{pp(x['benign_fpr_difference_F_minus_C0']['mean'])} +/- {pp(x['benign_fpr_difference_F_minus_C0']['sample_sd'])}",
   f"{pp(x['benign_tnr_difference_F_minus_C0']['mean'])} +/- {pp(x['benign_tnr_difference_F_minus_C0']['sample_sd'])}"))
 return '\n'.join(out)

def runtime_rows():
 out=[]
 for arm_key,arm in (('c0','H2m-C0'),('f','H2m-F')):
  for seed in SEEDS:
   m=json.loads((ROOT/f'results/h2_mini_no_replay/{arm_key}/seed{seed}/run_metadata.json').read_text(encoding='utf-8'))
   peak=max(json.loads((ROOT/f'results/h2_mini_no_replay/{arm_key}/seed{seed}/stage{i}/metrics.json').read_text(encoding='utf-8'))['peak_gpu_memory_bytes'] for i in range(1,5))
   start=datetime.fromisoformat(m['started_at']);finish=datetime.fromisoformat(m['finished_at'])
   out.append('| %s | %d | %.3f | %.3f | %.3f |'%(arm,seed,m['runtime_seconds'],(finish-start).total_seconds(),peak/1e9))
 return '\n'.join(out)

final_bshort_f=[x for x in A['external_trajectory'] if x['arm']=='H2m-F' and x['checkpoint_stage']=='AF4']
final_ni=[x for x in A['external_trajectory'] if x['checkpoint_stage']=='AF4']
max_position_failure=min(min(x['attack_tpr_head'],x['attack_tpr_middle'],x['attack_tpr_tail']) for x in A['attack_trajectory'])
next_prompt=('Accept the completed H2-mini no-replay results and retain R0. Keep replay, selective adaptation, and numerical memory-budget selection blocked. '
'Record that all acquisition gates passed, attack-TPR BWT was 0 in both arms, H2m-F benign-FPR increase BWT was +0.185 +/- 0.321 percentage points, no R1 threshold was reached, and final NotInject FPR was 0 in both arms while H2m-F final B-short FPR was 0.833 +/- 0.833 percentage points and confined to the English slice. '
'Proceed with benchmark-design analysis only: diagnose ceiling saturation and propose an independently audited stronger security-bidirectional stream with predeclared acquisition/retention gates, provenance, licenses, leakage controls, and matched controls. '
'Do not materialize or train a new benchmark, implement replay or selective adaptation, or choose a replay memory budget until the design is reviewed by a human.')

report=f'''# H2-mini no-replay baseline results

Status: **COMPLETE, VALIDATED, R0 ACTIVE**  
Date: 2026-10-05  
Detector: `microsoft/deberta-v3-base` at `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`  
Arms: H2m-C0 and H2m-F; seeds 17/42/2026; order AF1 -> AF2 -> AF3 -> AF4

## Scope and execution

The approved current-stage-only no-replay experiment completed all 6 runs and all 24 stage-end checkpoints. There were no failed or resumed runs. Each stage used one epoch, batch size 4, AdamW at 1e-5, constant scheduling, max gradient norm 1, float16 CUDA, optimizer reset, and sequentially carried model weights. No historical sample, replay, selective adaptation, test-guided selection, or numerical replay memory budget was used.

Technical preflight: **PASS**. It exercised the actual tokenizer, largest selected batch, forward/backward pass, optimizer step, max_length=1280, checkpoint save/load, and evaluation path without CUDA OOM. Preflight peak allocated GPU memory was {json.loads((ROOT/'results/h2_mini_no_replay/preflight/preflight.json').read_text())['peak_gpu_memory_bytes']/1e9:.3f} GB.

Independent artifact validation: **PASS**. It recomputed 168 checkpoint-file hashes across 24 checkpoints, verified 108 prediction files containing 18,216 rows, checked checkpoint lineage and the absence of optimizer state, and found no unpublished stage attempt. Same-seed C0/F initialization hashes match, and all three Stage-1 paired model hashes, evaluations, external diagnostics, training losses, and validation losses are identical.

| Arm | Seed | Recorded stage runtime (s) | Recorded run wall time (s) | Peak allocated GPU GB |
|---|---:|---:|---:|---:|
{runtime_rows()}

Total recorded stage runtime: **{A['total_stage_runtime_seconds']:.3f} s**. Summed recorded run wall time: **{A['total_run_wall_seconds']:.3f} s**. Including the {A['preflight_runtime_seconds']:.3f} s preflight gives **{A['total_run_wall_seconds']+A['preflight_runtime_seconds']:.3f} s (26.655 min)**.

## Acquisition

Values are seed 17 / 42 / 2026 in percent. Every arm-stage passes the frozen criterion in all 3 seeds.

| Arm | Stage | Attack TPR | Benign TNR | Benign FPR | Paired macro-F1 | Passing seeds |
|---|---|---:|---:|---:|---:|---:|
{acq_rows()}

H2m-F acquired every attack and benign stage adequately. All head, middle, and tail attack-position TPR values at every acquired checkpoint were at least **{100*max_position_failure:.3f}%**.

## Retention and BWT

Signed changes are later minus first acquisition. Positive FPR change is harmful; positive TPR/macro-F1/accuracy BWT is beneficial. Values are seed 17 / 42 / 2026 percentage points.

| Arm | Attack-TPR BWT | Paired macro-F1 BWT | Paired accuracy BWT | Benign-FPR increase BWT | Benign-FPR mean +/- sample SD |
|---|---:|---:|---:|---:|---:|
{retention_rows()}

| Arm | Earlier task | Final attack-TPR change | Final macro-F1 change | Final benign-FPR change | Peak benign-FPR increase |
|---|---|---:|---:|---:|---:|
{task_rows()}

Attack-TPR peak forgetting is **0.000 points for every family, arm, and seed**. The maximum paired macro-F1 peak forgetting is **0.834 points**. The maximum benign peak-FPR increase is **1.667 points**. These are small/transient effects, not catastrophic forgetting.

H2m-C0 mean attack-TPR BWT is 0.000 +/- 0.000 points and macro-F1 BWT is +0.185 +/- 0.160 points. H2m-F mean attack-TPR BWT is 0.000 +/- 0.000 points and macro-F1 BWT is -0.093 +/- 0.160 points. H2m-F benign-FPR increase BWT is +0.185 +/- 0.321 points. Its only harmful final acquisition-relative benign change is AF1 for seed 2026 (+1.667 points); seeds 17 and 42 are unchanged.

## External diagnostics

Each cell is mean +/- sample SD across seeds in percentage points.

| Arm | Checkpoint | NotInject FPR | B-short FPR | B-short English FPR | B-short multilingual FPR |
|---|---|---:|---:|---:|---:|
{external_rows()}

Final NotInject FPR is 0 in every arm/seed. Intermediate NotInject FPR is at most 0.885 points and does not worsen persistently. H2m-C0 B-short FPR is 0 at every checkpoint. Final H2m-F B-short FPR is {ms(x['bshort_fpr'] for x in final_bshort_f)} points: seed values {vec(x['bshort_fpr'] for x in sorted(final_bshort_f,key=lambda x:x['seed']))}. The final errors are confined to English ({ms(x['bshort_english_fpr'] for x in final_bshort_f)} points); multilingual FPR is 0. B-short is language-unmatched, so this difference is descriptive and not a pure trigger-word causal effect. NotInject does not remain worse than B-short after H2-mini training.

## Matched arm comparison

Final-checkpoint differences are H2m-F minus H2m-C0, mean +/- sample SD in percentage points.

| Evaluated stage | Attack TPR | Paired macro-F1 | Benign FPR | Benign TNR |
|---|---:|---:|---:|---:|
{pair_rows()}

Persistent hard-benign pressure does not change malicious retention: every paired attack-TPR difference is zero. At the final checkpoint it moves AF1 benign FPR by +1.667 points in all seeds and AF2 benign FPR by -1.667 points in all seeds; the corresponding paired macro-F1 differences are -0.833 and +0.833 points. These are cross-arm endpoint differences, while acquisition-relative AF1 degradation occurs in only one seed. They remain below the frozen R1 threshold.

## Scientific questions

1. **Acquisition:** yes. H2m-F passes all attack-TPR, benign-TNR, and paired macro-F1 gates in 3/3 seeds at all four stages.
2. **Malicious retention:** hard-benign pressure does not increase attack-family forgetting; attack-TPR BWT and peak forgetting are exactly zero.
3. **Benign retention:** no consistent acquisition-relative degradation is established. The largest mean harmful candidate is H2m-F AF1 benign FPR at +0.556 +/- 0.962 points, harmful in 1/3 seeds.
4. **Over-defense:** final NotInject FPR is 0 in both arms. There is no persistent F-arm improvement to attribute to hard-benign pressure.
5. **B-short:** C0 remains at 0; F ends at 0.833 +/- 0.833 points, English-only and seed-sensitive. NotInject is not substantially worse than B-short.
6. **Stability/order:** malicious behavior is fully stable across seeds. Small benign/external changes are seed- or partition-sensitive. This experiment does not include an H2 reverse-order control, so general order symmetry is unresolved.

## Replay readiness

**R0 remains active; R1 and R2 are not entered.** All evidence and acquisition gates pass, but no predeclared primary retention metric has harmful direction in at least 2/3 seeds together with mean degradation of at least 2.0 points. There are zero R1 numeric candidates and zero R2 numeric candidates. Replay is not scientifically justified; neither a narrow replay sensitivity study nor a full replay comparison is eligible.

The result is better described as ceiling saturation with small benign transfer effects. Severe external over-defense observed in earlier checkpoints is not persistent here and is not forgetting evidence.

## Artifacts and tests

- `results/h2_mini_no_replay/`: 6 run directories, 24 checkpoints, raw predictions, confusion counts, metrics, configs, hashes, and matrices.
- `results/h2_mini_no_replay_aggregate.json`: full trajectories and aggregate retention.
- `results/h2_mini_no_replay_pairwise.json`: paired F-minus-C0 comparisons.
- `results/h2_mini_no_replay_readiness.json`: frozen gate application.
- `results/h2_mini_no_replay_artifact_validation.json`: independent artifact validation.
- `results/h2_mini_no_replay/tables/`: eight machine-readable figure/table CSVs.
- Complete project test suite: **102 passed, 0 failed, 1 warning**. The warning is the known PyTorch/Python 3.14 future-compatibility warning.

## Remaining limitations

The 60-per-class stage tests give coarse 1.667-point increments. Malicious performance is ceiling-saturated, limiting sensitivity to retention pressure. H2m-F and C0 intentionally differ in benign source/composition after Stage 1; this supports the operational comparison but leaves residual source/domain confounding. B-short has unmatched language composition and only 42 English examples. The malicious stream is Chen-derived project construction rather than an exact paper reproduction. With three seeds, paired differences are descriptive and do not support strong significance claims.

## Exact recommended next prompt

> {next_prompt}
'''
(ROOT/'docs/h2_mini_no_replay_results.md').write_text(report,encoding='utf-8')

prepend('docs/h2_mini_no_replay_protocol.md',f'''## Completed no-replay execution - 2026-10-05

The fixed protocol completed all six H2m-C0/H2m-F runs and 24 stage checkpoints without failure or resume. All acquisition gates passed. Attack-TPR BWT was 0 in both arms; no frozen R1/R2 retention threshold was reached. Full results: `docs/h2_mini_no_replay_results.md`. Replay, selective adaptation, and numerical memory-budget selection remain blocked.''')
prepend('docs/replay_readiness_review.md',f'''## H2-mini no-replay readiness decision - 2026-10-05

**R0 active; R1/R2 not entered.** The complete three-seed matched experiment passes data, initialization, execution, and acquisition gates, but has zero R1/R2 numeric retention candidates. Attack-TPR BWT and peak forgetting are 0 in both arms. H2m-F macro-F1 BWT is -0.093 +/- 0.160 points and benign-FPR increase BWT is +0.185 +/- 0.321 points, both below the 2-point gate. Replay is not scientifically justified.''')
prepend('docs/replay_comparison_protocol.md','''## Comparison remains blocked after H2-mini baseline - 2026-10-05

The no-replay result remains R0. Neither a narrow replay sensitivity study nor a full replay-method comparison is eligible. The detector, future teacher controls, and method-isolation rules remain fixed, but no replay method and no numerical memory budget may be selected until a new human decision follows stronger evidence.''')
prepend('docs/decision_log.md','''## D045 - Retain R0 after complete H2-mini no-replay baseline (2026-10-05)

All six runs and 24 checkpoints completed and passed independent validation. Acquisition passed in every arm/stage/seed. Attack-TPR BWT and peak forgetting were zero; small benign/macro-F1 changes did not reach frozen R1 criteria. Replay and selective adaptation remain blocked, and no numerical memory budget was chosen.''')
prepend('docs/change_log.md','''## C033 - Execute and validate H2-mini no-replay comparison (2026-10-05)

Added the fixed H2-mini runner, preflight, aggregator, report renderer, result-level tests, 24 optimizer-independent stage checkpoints, raw predictions/confusions/matrices, aggregate/pairwise/readiness JSON, and eight figure-ready CSV tables. Validation: 24/24 checkpoints, 168 checkpoint files, 108 prediction files and 18,216 prediction rows passed; full suite 102 passed, 0 failed, 1 warning.''')
prepend('docs/assumptions_and_risks.md','''## Risks after H2-mini no-replay execution - 2026-10-05

The observed near-ceiling malicious acquisition and retention reduce sensitivity to a continual-learning effect. Stage tests have 60 examples per class, so one error changes a class rate by 1.667 points. H2m-F benign source/composition, B-short language mismatch, Chen-derived construction, three-seed uncertainty, and the absence of an H2 reverse-order control limit causal and general claims. Small external or benign differences must not be called catastrophic forgetting or replay evidence.''')
prepend('docs/open_questions.md','''## Open questions after H2-mini no-replay execution - 2026-10-05

1. Should the replay branch be closed unless a stronger independently audited stream produces retention loss above the frozen gate?
2. Can a future security-bidirectional benchmark avoid malicious ceiling saturation while preserving adequate acquisition and matched controls?
3. Is a reverse-order or independently replicated hard condition worth designing before any replay proposal?
4. What source and language balance should replace the unmatched B-short diagnostic in a future design?

No replay method, selective adaptation, or numerical memory budget is currently eligible.''')

now=datetime.now(timezone.utc).isoformat()
decision={
 'date':now,'id':'D045','decision':'Retain R0 after the complete validated H2-mini no-replay comparison',
 'execution':{'runs':6,'stage_end_checkpoints':24,'failed':0,'resumed':0},
 'acquisition':'all 8 arm-stage gates pass in 3/3 seeds',
 'retention':{'H2m-C0_attack_TPR_BWT_pp':0.0,'H2m-F_attack_TPR_BWT_pp':0.0,'H2m-C0_macro_F1_BWT_pp':0.185,'H2m-F_macro_F1_BWT_pp':-0.093,'H2m-F_benign_FPR_increase_BWT_pp':0.185},
 'readiness':{'R0':'active','R1':'not entered','R2':'not entered','R1_numeric_candidates':0,'R2_numeric_candidates':0},
 'replay_scientifically_justified':False,'narrow_replay_sensitivity_eligible':False,'full_replay_comparison_eligible':False,
 'replay':False,'selective_adaptation':False,'numerical_memory_budget':None,'next_prompt':next_prompt}
change={'date':now,'id':'C033','decision_id':'D045','reason':'Execute, validate, aggregate, and document the approved H2-mini no-replay baseline',
 'files_changed':['configs/h2_mini_no_replay.yaml','src/continualguard/h2_mini_no_replay.py','src/continualguard/h2_mini_summary.py','scripts/run_h2_mini_no_replay.py','scripts/summarize_h2_mini_no_replay.py','scripts/render_h2_mini_no_replay_report.py','tests/test_h2_mini_no_replay.py','tests/test_h2_mini_no_replay_results.py','results/h2_mini_no_replay/**','results/h2_mini_no_replay_aggregate.json','results/h2_mini_no_replay_pairwise.json','results/h2_mini_no_replay_readiness.json','results/h2_mini_no_replay_artifact_validation.json','docs/h2_mini_no_replay_results.md','docs/h2_mini_no_replay_protocol.md','docs/replay_readiness_review.md','docs/replay_comparison_protocol.md','docs/decision_log.md','docs/change_log.md','docs/assumptions_and_risks.md','docs/open_questions.md','history/decisions.jsonl','history/changes.jsonl'],
 'validation':['technical preflight PASS','6/6 runs and 24/24 checkpoints complete','168 checkpoint file hashes PASS','108 prediction files and 18,216 rows PASS','paired initialization and Stage-1 equivalence PASS','102 tests passed, 0 failed, 1 warning'],
 'scope':'No-replay baseline only; replay and selective adaptation not implemented; numerical memory budget not selected'}
for rel,obj,identifier in [('history/decisions.jsonl',decision,'D045'),('history/changes.jsonl',change,'C033')]:
 p=ROOT/rel;text=p.read_text(encoding='utf-8')
 if identifier not in text:
  with p.open('a',encoding='utf-8',newline='\n') as h:
   if text and not text.endswith('\n'):h.write('\n')
   h.write(json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n')

print(json.dumps({'status':'REPORTS_UPDATED','decision':'D045','change':'C033','report':'docs/h2_mini_no_replay_results.md','aggregate_sha256':sha(ROOT/'results/h2_mini_no_replay_aggregate.json'),'pairwise_sha256':sha(ROOT/'results/h2_mini_no_replay_pairwise.json'),'readiness_sha256':sha(ROOT/'results/h2_mini_no_replay_readiness.json')},indent=2))
