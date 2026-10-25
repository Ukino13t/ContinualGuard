"""Prepare the approved frozen-reserve H2-mini B0 final top-up review package."""
from __future__ import annotations
import csv,gzip,json,os,sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from h2_mini_b0_review_common import load,save,sha,file_sha,key,choose_diverse
import audit_h2_mini_sources as source_tools

PACKAGE='h2mini-b0-final-topup-v1-20261004'
SEED=2026100403
OA_REV='fdf72ae0827c1cda404aff25b6603abec9e3399b'
TOK_REV='8ccc9b6f36199bec6961081d44eb72fb3f7353f3'
TOPDIR=ROOT/'review/b0_final_topup'
MANIFEST=ROOT/'review/b0_final_topup_manifest.json'
RESERVE=ROOT/'data/manifests/hard_benign/b0_small_topup_reserve.json'
RESERVE_AUDIT=ROOT/'data/audits/hard_benign/b0_small_topup_reserve_audit.json'
CAPACITY=ROOT/'results/h2_mini_b0_expanded_capacity.json'
SOURCE_AUDIT=ROOT/'data/audits/hard_benign/oasst1_full_audit.json'
ANALYZER=ROOT/'scripts/analyze_h2_mini_b0_final_topup.py'
FIELDS=['review_id','text','source','language','token_length','review_role','reviewer_decision','b0_confirmed','reason_code','confidence','optional_note']
HUMAN=['reviewer_decision','b0_confirmed','reason_code','confidence','optional_note']
GROUP_FIELDS=['source_native_id','source_parent_id','source_immediate_parent_id','raw_text_sha256','normalized_text_sha256','model_view_sha256','exact_group','normalized_group','model_view_group','project_semantic_parent_group','project_template_group']
MARKER='H2_MINI_B0_FINAL_TOPUP_PREPARED_20261004'


def utcnow(): return datetime.now(timezone.utc).isoformat()
def rel(p): return p.resolve().relative_to(ROOT).as_posix()
def append_once(path,text):
    current=path.read_text(encoding='utf-8-sig') if path.exists() else ''
    if MARKER in current: raise RuntimeError(f'Existing marker prevents duplicate update: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(current.rstrip()+'\n\n'+text.strip()+'\n',encoding='utf-8')
def ids(records,field): return {r[field] for r in records if r.get(field)}
def all_records(path):
    obj=load(path)
    return obj.get('records',obj if isinstance(obj,list) else [])


def main():
    if TOPDIR.exists() or MANIFEST.exists(): raise RuntimeError('Top-up package already exists; refusing overwrite')
    if (ROOT/'history/decisions.jsonl').read_text(encoding='utf-8').find('"id": "D042"')>=0: raise RuntimeError('D042 already exists')
    if (ROOT/'history/changes.jsonl').read_text(encoding='utf-8').find('"id": "C029"')>=0: raise RuntimeError('C029 already exists')
    reserve=load(RESERVE); audit=load(RESERVE_AUDIT); capacity=load(CAPACITY)
    assert file_sha(RESERVE)=='e1defad9d4b307c1ba443a9b8cd7d09f310f685620019947a63f2ffe1e95826d'
    assert file_sha(RESERVE_AUDIT)=='8f5a3a6a4d2e9c5ccce31e8e90602751d9c3f59080879cb54c3a47f26ab99ea1'
    assert reserve['status']=='METADATA_ONLY_UNREVIEWED_RESERVE'
    assert audit['status']=='RESERVE_CANDIDATE_AUDIT_PASS_NOT_HUMAN_REVIEWED'
    assert reserve['source_revision']==audit['pinned_OASST_revision']==OA_REV
    assert reserve['count']==65 and reserve['training_candidates']==40 and reserve['evaluation_candidates']==25
    records=reserve['records']
    assert len(records)==len(ids(records,'review_id'))==len(ids(records,'source_native_id'))==65
    assert Counter(r['reserve_purpose'] for r in records)==Counter({'TRAINING_REVIEW_RESERVE':40,'EVALUATION_DOUBLE_REVIEW_RESERVE':25})
    assert all(r['human_review_status']=='NOT_REVIEWED' and r['review_assignments_created'] is False and r['final_assignment'] is None for r in records)
    assert reserve['human_review_worksheets_created'] is False and reserve['human_review_performed'] is False and reserve['accepted_capacity_added']==0
    assert audit['selected_reserve']==65 and audit['selected_overlap_findings']==0 and audit['thresholds_unchanged'] is True and audit['detector_scores_used'] is False
    assert audit['internal_comparison']['affected_queries']==0
    for name in ['all_prior_review_and_reserves','NotInject','AF1_AF4_all_labels']:
        assert audit['external_comparison'][name]['affected_queries']==0 and audit['external_comparison'][name]['pairs']==[]
    for relpath,digest in audit['source_hashes'].items():
        p=ROOT/Path(relpath.replace('\\','/'))
        assert p.exists() and file_sha(p)==digest,(relpath,'source hash mismatch')

    # Direct membership checks augment the frozen group/contamination audit.
    reserve_native=ids(records,'source_native_id'); reserve_review=ids(records,'review_id')
    checked={}
    for name,path in {
      'calibration':'review/calibration_manifest.json','production':'review/production_review_manifest.json',
      'replenishment':'review/b0_replenishment_manifest.json','B0_accepted':'data/manifests/hard_benign/B0_accepted.json',
      'B0_expanded_review_eligible':'data/manifests/hard_benign/B0_expanded_review_eligible.json',
      'B0_expanded_group_safe':'data/manifests/hard_benign/B0_expanded_group_safe.json',
      'B_short':'data/manifests/hard_benign/B_short_accepted.json','B1':'data/manifests/hard_benign/B1_accepted.json',
      'B2':'data/manifests/hard_benign/B2_accepted.json','B3':'data/manifests/hard_benign/B3_accepted.json'}.items():
        rr=all_records(ROOT/path)
        overlap=(reserve_native & ids(rr,'source_native_id')) | (reserve_review & ids(rr,'review_id'))
        assert not overlap,(name,sorted(overlap))
        checked[name]={'records_checked':len(rr),'id_overlap':0}

    source_audit=load(SOURCE_AUDIT); srcmeta=source_audit['source']
    srcentry=next(x for x in srcmeta['files'] if x['file_name']=='2023-04-12_oasst_ready.messages.jsonl.gz')
    source_path=ROOT/srcentry['local_path']
    assert srcmeta['revision']==OA_REV and file_sha(source_path)==srcentry['sha256']==audit['source_hashes'][rel(source_path)]
    wanted={r['source_row_index']:r for r in records}; texts={}
    with gzip.open(source_path,'rt',encoding='utf-8',errors='strict') as stream:
        for index,line in enumerate(stream):
            if index not in wanted: continue
            row=json.loads(line); meta=wanted[index]
            assert row['message_id']==meta['source_native_id']==meta['candidate_id']
            assert row['message_tree_id']==meta['source_parent_id']
            assert row.get('parent_id')==meta['source_immediate_parent_id']
            assert row['lang']==meta['language']
            assert sha(row['text'])==meta['raw_text_sha256']
            texts[meta['review_id']]=row['text']
    assert len(texts)==65
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from transformers import AutoTokenizer
    tokenizer_path=ROOT/'data/raw/huggingface/microsoft_deberta-v3-base'/TOK_REV
    tok=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True,use_fast=False)
    ordered=[texts[r['review_id']] for r in records]
    model_hashes,token_counts=source_tools.mviews(ordered,tok)
    for r,mh,n,text in zip(records,model_hashes,token_counts,ordered):
        assert n==r['deberta_tokens']<=1280
        assert mh==r['model_view_sha256']
        assert sha(source_tools.norm(text))==r['normalized_text_sha256']
        for field in GROUP_FIELDS: assert field in r
        assert r['source_immediate_parent_id'] is None  # all selected OASST1 records are root prompts

    # Selected records must remain isolated under every frozen exact/group identifier.
    for field in GROUP_FIELDS:
        if field=='source_immediate_parent_id': continue
        assert len(ids(records,field))==65,(field,'non-unique within reserve')

    training=[r for r in records if r['reserve_purpose']=='TRAINING_REVIEW_RESERVE']
    evaluation=[r for r in records if r['reserve_purpose']=='EVALUATION_DOUBLE_REVIEW_RESERVE']
    primary_a=choose_diverse(training,20,f'{PACKAGE}|{SEED}|training-primary-A')
    a_ids=ids(primary_a,'review_id')
    assert len(a_ids)==20
    manifest_records=[]
    for source in records:
        r=dict(source)
        if r['reserve_purpose']=='TRAINING_REVIEW_RESERVE':
            who='A' if r['review_id'] in a_ids else 'B'
            assigns=[{'reviewer':who,'review_role':'PRIMARY','worksheet':f'reviewer_{who}_batch_01.csv'}]
        else:
            assigns=[{'reviewer':who,'review_role':'DOUBLE_REVIEW','worksheet':f'reviewer_{who}_batch_01.csv'} for who in ['A','B']]
        r['reviewer_assignments']=assigns;r['human_review_status']='NOT_STARTED';r['final_assignment']=None
        manifest_records.append(r)
    assert Counter((a['reviewer'],a['review_role']) for r in manifest_records for a in r['reviewer_assignments'])==Counter({('A','PRIMARY'):20,('B','PRIMARY'):20,('A','DOUBLE_REVIEW'):25,('B','DOUBLE_REVIEW'):25})

    TOPDIR.mkdir(parents=True)
    batches={}
    for who in ['A','B']:
        assigned=[]
        for r in manifest_records:
            aa=[a for a in r['reviewer_assignments'] if a['reviewer']==who]
            if aa: assigned.append((r,aa[0]))
        assigned.sort(key=lambda x:key(PACKAGE,SEED,'worksheet-order',who,x[0]['review_id']))
        path=TOPDIR/f'reviewer_{who}_batch_01.csv'
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\r\n');writer.writeheader()
            for r,a in assigned:
                writer.writerow({'review_id':r['review_id'],'text':texts[r['review_id']],'source':'OpenAssistant/oasst1',
                  'language':r['language'],'token_length':r['deberta_tokens'],'review_role':a['review_role'],**{x:'' for x in HUMAN}})
        batches[path.name]={'reviewer':who,'rows':len(assigned),'primary':sum(a['review_role']=='PRIMARY' for r,a in assigned),
          'double_review':sum(a['review_role']=='DOUBLE_REVIEW' for r,a in assigned),'review_ids':[r['review_id'] for r,a in assigned],
          'encoding':'UTF-8 with BOM','prepared_sha256':file_sha(path),'all_human_fields_blank':True}
    assert batches['reviewer_A_batch_01.csv']['rows']==batches['reviewer_B_batch_01.csv']['rows']==45

    frozen_paths=[RESERVE,RESERVE_AUDIT,CAPACITY,SOURCE_AUDIT,source_path,ROOT/'scripts/h2_mini_b0_review_common.py',ANALYZER]
    manifest={'package_id':PACKAGE,'status':'PREPARED_AWAITING_HUMAN_REVIEW','created_at':utcnow(),
      'active_project_root':str(ROOT),'active_interpreter':sys.executable,'source':'OpenAssistant/oasst1','source_license':'Apache-2.0',
      'source_revision':OA_REV,'tokenizer':'microsoft/deberta-v3-base','tokenizer_revision':TOK_REV,'max_length':1280,
      'reserve_provenance':{'path':rel(RESERVE),'sha256':file_sha(RESERVE),'audit_path':rel(RESERVE_AUDIT),'audit_sha256':file_sha(RESERVE_AUDIT),
        'unchanged_record_count':65,'all_unused_and_unreviewed':True,'substitutions':0},
      'verification':{'native_ids':65,'conversation_tree_ids':65,'raw_hashes':65,'normalized_hashes':65,'model_view_hashes':65,
        'semantic_parent_groups':65,'template_groups':65,'historical_membership_checks':checked,'frozen_group_audit_pass':True,
        'external_overlap':{'prior_review_and_reserves':0,'NotInject':0,'AF1_AF4':0},'detector_outputs_used':False},
      'sampling':{'seed':SEED,'algorithm':'frozen choose_diverse: language-proportional quotas, length-quartile coverage, deterministic SHA-256 ordering',
        'training_primary_balance':{'A':20,'B':20},'evaluation_assignment':'all 25 assigned independently to A and B'},
      'worksheet_columns':FIELDS,'human_fields':HUMAN,
      'allowed_values':{'review_role':['PRIMARY','DOUBLE_REVIEW'],'reviewer_decision':['ACCEPT_BENIGN','REJECT_TRUE_INJECTION','REJECT_AMBIGUOUS_INTENT','REJECT_HARMFUL_OR_UNSAFE','REJECT_CONTEXT_MISSING','REJECT_PRIVACY_OR_RIGHTS','REJECT_QUALITY'],'b0_confirmed':['YES','NO','UNSURE'],'confidence':['HIGH','MEDIUM','LOW']},
      'eligibility_rule':'reviewer_decision=ACCEPT_BENIGN and b0_confirmed=YES; evaluation requires this independently from both reviewers',
      'safety_rule':'ACCEPT_BENIGN versus REJECT_TRUE_INJECTION is automatically ineligible and is not automatically adjudicated',
      'workload':{'training_candidates':40,'training_primary_judgments':40,'evaluation_candidates':25,'evaluation_double_judgments':50,
        'total_judgments':90,'A_judgments':45,'B_judgments':45,'planning_seconds_per_judgment':25,'projected_seconds':2250,
        'projected_minutes':37.5,'projected_person_hours':0.625,'per_reviewer_projected_minutes':18.75,'timing_is_planning_estimate_only':True},
      'frozen_capacity':{'training':780,'evaluation':350,'unique':1130},'required_topup':{'training':20,'evaluation':10,'unique':30},
      'required_final_capacity':{'training':800,'validation':120,'test':240,'evaluation':360,'unique':1160},
      'batches':batches,'records':manifest_records,'frozen_dependencies':{rel(p):file_sha(p) for p in frozen_paths},
      'ordinary_B0_rubric_unchanged':True,'B0_disagreement_gate_unchanged':0.10,'existing_180_pair_audit_revisited':False,
      'human_judgments_performed':False,'existing_labels_changed':False,'final_assignment':False,'research_JSONL_materialized':False,
      'model_training_performed':False,'replay_implemented':False,'numerical_replay_memory_budget':None,'replay_readiness':'R0'}
    save(MANIFEST,manifest)

    gi=ROOT/'.gitignore';g=gi.read_text(encoding='utf-8') if gi.exists() else ''
    line='review/b0_final_topup/'
    if line not in g.splitlines(): gi.write_text(g.rstrip()+'\n'+line+'\n',encoding='utf-8')

    plan=f'''<!-- {MARKER}:BEGIN -->
# H2-mini final B0 top-up review plan

Status: **PREPARED_AWAITING_HUMAN_REVIEW**. This package uses only the frozen metadata-only OASST1 reserve. All 65 records were freshly checked against the pinned raw source and tokenizer, and the accepted frozen group/contamination audit remains PASS. No substitution was needed.

## Fixed evidence and need

- Current frozen capacity: train 780, double-reviewed evaluation 350, unique 1130.
- Required accepted top-up: train 20, evaluation 10, unique 30.
- Reserve used for review: 40 training candidates and 25 evaluation candidates.
- Historical 180-pair gate, ordinary-B0 rubric, and 10% threshold are unchanged.

## Assignments and workload

- Reviewer A: 20 PRIMARY + 25 DOUBLE_REVIEW = 45 judgments.
- Reviewer B: 20 PRIMARY + 25 DOUBLE_REVIEW = 45 judgments.
- Total: 90 judgments. At 25 seconds per judgment, the planning estimate is 2,250 seconds = 37.5 minutes = 0.625 person-hours. This is not a measured timing requirement.
- The 25 evaluation judgments must be completed independently; neither reviewer may see the other's answers.

## Rubric and eligibility

Review only whether the text is genuinely benign, ordinary short-form input, context-complete, usable, and free of prompt injection, ambiguous intent, disallowed harm, privacy/rights problems, or quality failures. Do not assign B1/B2/B3/BH difficulty. Training eligibility requires ACCEPT_BENIGN + YES from its primary reviewer. Evaluation eligibility requires ACCEPT_BENIGN + YES from both reviewers. An ACCEPT_BENIGN versus REJECT_TRUE_INJECTION conflict is ineligible and remains unadjudicated.

## Human procedure

1. Each reviewer opens only their assigned UTF-8 BOM CSV.
2. Preserve `review_id` through `review_role` exactly.
3. Fill `reviewer_decision`, `b0_confirmed`, `reason_code`, `confidence`, and optionally `optional_note`. Give a reason code for every rejection.
4. Keep reviewers independent and save the completed CSV at the same path.
5. After all 90 judgments are complete, run `E:\\ContinualGuard\\.venv\\Scripts\\python.exe scripts\\analyze_h2_mini_b0_final_topup.py`.

If capacity passes, the next run may continue only with the already approved hash-only finalization and must stop before JSONL materialization or training. If capacity fails, stop and request human approval. R0 remains active; R1 and R2 are not entered.
<!-- {MARKER}:END -->'''
    (ROOT/'docs/h2_mini_b0_final_topup_plan.md').write_text(plan+'\n',encoding='utf-8')
    updates={
      'h2_mini_b0_replenishment_results.md':'Final top-up package prepared from the frozen 40/25 reserve. No labels have been collected; accepted capacity remains 780/350/1130 until analysis.',
      'h2_mini_simplified_design.md':'The final B0 top-up uses 40 single-primary training candidates and 25 independently double-reviewed evaluation candidates under the unchanged ordinary-B0 rubric.',
      'h2_mini_design.md':'H2-mini remains at pre-materialization R0. The final B0 top-up is a human-review capacity step and does not constitute continual-learning evidence.',
      'h2_mini_final_materialization_plan.md':'Materialization remains blocked. A complete top-up PASS may trigger the approved hash-only assignment workflow in the next run, which must stop before JSONL.',
      'h2_mini_no_replay_protocol.md':'Replay remains blocked during top-up review, analysis, and hash-only finalization. No numerical replay memory budget has been selected.',
      'replay_comparison_protocol.md':'The teacher-guidance controls remain fixed. The top-up package changes no detector, optimization, evaluation, or replay setting.',
      'replay_readiness_review.md':'R0 active; R1 not entered; R2 not entered. Preparing 90 human judgments changes dataset readiness only after eligibility analysis.',
      'decision_log.md':'D042: Approve and prepare one final targeted B0 top-up using only the frozen 40/25 reserve; stop for human review before final assignments.',
      'change_log.md':'C029: Added two UTF-8 BOM review worksheets, frozen-reserve manifest, analyzer, documentation, and history entries. No human labels or model artifacts were created.',
      'assumptions_and_risks.md':'Top-up yield is uncertain: at least 20/40 training and 10/25 evaluation candidates must qualify. Double-review conflicts remain ineligible; no threshold may be weakened.',
      'open_questions.md':'Pending: completion of all 90 judgments and whether the accepted yield reaches train +20 and evaluation +10. Numerical replay memory budget remains undecided.'}
    for name,body in updates.items():
        append_once(ROOT/'docs'/name,f'''<!-- {MARKER}:BEGIN -->
## Final B0 top-up review package — 2026-10-04

{body}

Package: `review/b0_final_topup_manifest.json`. Human judgment is the next required action. No final assignments, JSONL, training, replay, or memory-budget choice is authorized in this run.
<!-- {MARKER}:END -->''')

    now=utcnow()
    decision={'date':now,'id':'D042','decision':'Prepare final targeted B0 top-up human-review package using only the frozen reserve',
      'reserve_verified':{'training':40,'evaluation':25,'unused_unreviewed_group_safe':True,'substitutions':0},
      'assignments':{'A':45,'B':45,'total':90},'frozen_capacity':{'training':780,'evaluation':350,'unique':1130},
      'required_accepted_topup':{'training':20,'evaluation':10,'unique':30},'existing_labels_changed':False,
      'final_assignment':False,'materialization':False,'training':False,'replay':False,'numerical_memory_budget':None,
      'replay_readiness':'R0','next_step':'Two reviewers independently complete their assigned worksheets; then run the analyzer. Continue hash-only finalization only after a complete capacity PASS.'}
    change={'date':now,'id':'C029','decision_id':'D042','reason':'Prepare the human-review package for the approved minimum final B0 capacity top-up',
      'files_changed':['scripts/analyze_h2_mini_b0_final_topup.py','scripts/prepare_h2_mini_b0_final_topup.py','review/b0_final_topup/reviewer_A_batch_01.csv','review/b0_final_topup/reviewer_B_batch_01.csv','review/b0_final_topup_manifest.json','.gitignore','docs/h2_mini_b0_final_topup_plan.md',*[f'docs/{x}' for x in updates],'history/decisions.jsonl','history/changes.jsonl'],
      'validation':['65 frozen records freshly verified against pinned OASST1 source and tokenizer','zero historical ID membership and accepted frozen group/contamination audit PASS','A45/B45 and exactly90 blank judgments','UTF-8 BOM worksheet schema/hash validation'],
      'scope':'Human-review package preparation only; no judgments, final assignments, JSONL, training, replay, or numerical memory budget'}
    with (ROOT/'history/decisions.jsonl').open('a',encoding='utf-8',newline='') as f:f.write(json.dumps(decision,ensure_ascii=False)+'\n')
    with (ROOT/'history/changes.jsonl').open('a',encoding='utf-8',newline='') as f:f.write(json.dumps(change,ensure_ascii=False)+'\n')
    print(json.dumps({'status':'PREPARED_AWAITING_HUMAN_REVIEW','reserve':{'training':40,'evaluation':25},'judgments':{'A':45,'B':45,'total':90},'worksheets':{k:v['prepared_sha256'] for k,v in batches.items()},'manifest':rel(MANIFEST),'projected_person_hours':0.625,'R0':'active','R1':'not entered','R2':'not entered'},indent=2))

if __name__=='__main__': main()