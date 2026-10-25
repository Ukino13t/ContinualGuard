"""Finalize approved H2-mini metadata-only assignments; never materialize raw text or train."""
from __future__ import annotations
import csv,gzip,hashlib,io,json,math,os,sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from h2_mini_b0_review_common import load,save,sha,file_sha,key
import audit_h2_mini_sources as source_tools
NOW=lambda:datetime.now(timezone.utc).isoformat()
POLICY='h2mini-final-hash-assignment-v1-20261004';SEED=2026100404
OA_REV='fdf72ae0827c1cda404aff25b6603abec9e3399b';OR_REV='e36d8b80e81837c8a8f264bbb2a49f1b32c7e272'
TOK_REV='8ccc9b6f36199bec6961081d44eb72fb3f7353f3';CHEN_REV='1e9e0f00179adc154e9ca991b9cd3206c99f9588'
HASH_FIELDS=['raw_text_sha256','normalized_text_sha256','model_view_sha256']
GROUP_FIELDS=['source_native_id','source_parent_id','exact_group','normalized_group','model_view_group','project_semantic_parent_group','project_template_group']
TOP_MAN=ROOT/'review/b0_final_topup_manifest.json';TOP_ANALYSIS=ROOT/'results/h2_mini_b0_final_topup_analysis.json'
CANON=ROOT/'review/b0_final_topup_canonical_manifest.json';B0FINAL=ROOT/'data/manifests/hard_benign/B0_final_accepted.json';B0AUDIT=ROOT/'data/audits/hard_benign/h2_mini_b0_final_post_review_audit.json';B0ASSIGN=ROOT/'data/manifests/hard_benign/b0_final_stage_assignment.json'
BHFINAL=ROOT/'data/manifests/hard_benign/BH_final_eligible.json';BHASSIGN=ROOT/'data/manifests/hard_benign/bh_final_partition_assignment.json';BSHORT=ROOT/'data/manifests/hard_benign/b_short_final_hash_manifest.json';OUTDIR=ROOT/'data/manifests/h2_mini'
MAL=OUTDIR/'h2m_shared_malicious_assignment.json';C0=OUTDIR/'h2m_c0_final_assignment_proposal.json';F=OUTDIR/'h2m_f_final_assignment_proposal.json';MATCH=ROOT/'results/h2_mini_matched_control_verification.json';VALID=OUTDIR/'h2m_final_assignment_validation.json';MARK='H2_MINI_FINAL_HASH_ASSIGNMENT_READY_20261004'
def rel(p):return p.resolve().relative_to(ROOT).as_posix()
def lbin(n):
 for x in (16,24,32,48,64,128,256,1280):
  if n<=x:return f'<={x}'
 return '>1280'
def pct(v,q):return sorted(v)[max(0,math.ceil(q*len(v))-1)]
def stats(v):return {'count':len(v),'min':min(v),'median':pct(v,.5),'p95':pct(v,.95),'p99':pct(v,.99),'max':max(v),'over_1280':sum(x>1280 for x in v),'truncation_required':sum(x>1280 for x in v)}
def is_double(r):return len(r.get('reviewer_evidence',[]))>=2
def bare(r,extra=None):
 keys=['project_record_id','review_id','candidate_id','source','source_revision','source_native_id','source_parent_id','source_immediate_parent_id','source_row_index','source_category','language','deberta_tokens',*HASH_FIELDS,'exact_group','normalized_group','model_view_group','project_semantic_parent_group','project_template_group','label']
 out={k:r.get(k) for k in keys if k in r};out.update(extra or {});return out
def no_raw(x):
 forbidden={'text','base_document_text','payload_text','injection_text'}
 if isinstance(x,dict):return not(forbidden&set(x)) and all(no_raw(v) for v in x.values())
 if isinstance(x,list):return all(no_raw(v) for v in x)
 return True
def feat_b0(r):return [('language',r['language']),('length_bin',lbin(r['deberta_tokens']))]
def feat_bh(r):return [('category',r['source_category']),('length_bin',lbin(r['deberta_tokens'])),('language',r['language']),('historical_stratum',r['proposed_stratum'])]+[('cue_family',x) for x in sorted(r.get('cue_family_ids',[]))]
def allocate(pool,stages,per_stage,feature_fn,salt,weights=None):
 pool=list(pool);need=len(stages)*per_stage;assert len(pool)>=need
 weights=weights or defaultdict(lambda:1.0);total=Counter(f for r in pool for f in feature_fn(r));selected=Counter();bystage={s:Counter() for s in stages};done=Counter();out={s:[] for s in stages}
 for step in range(need):
  stage=min(stages,key=lambda s:(done[s]/per_stage,key(POLICY,salt,'stage',step,s)));progress=(step+1)/need
  def score(r):
   ff=feature_fn(r);deficit=sum(weights[f[0]]*(total[f]*need/len(pool)*progress-selected[f]) for f in ff);imbalance=sum(weights[f[0]]*(bystage[stage][f]-sum(bystage[s][f] for s in stages)/len(stages)) for f in ff)
   return (-deficit+imbalance,key(POLICY,salt,r.get('review_id',r.get('project_record_id',r.get('example_id')))))
  chosen=min(pool,key=score);pool.remove(chosen);out[stage].append(chosen);done[stage]+=1
  for f in feature_fn(chosen):selected[f]+=1;bystage[stage][f]+=1
 assert all(len(out[s])==per_stage for s in stages);return out
def read_reviews(manifest):
 rows={};files={};payloads={}
 for filename,b in manifest['batches'].items():
  p=ROOT/'review/b0_final_topup'/filename;raw=p.read_bytes();text=raw.decode('utf-8-sig',errors='strict');rr=list(csv.DictReader(io.StringIO(text,newline='')));assert len(rr)==45
  digest=hashlib.sha256(raw).hexdigest();has_bom=raw.startswith(b'\xef\xbb\xbf');cp=p if has_bom else ROOT/'review/b0_final_topup_canonical'/filename;cbytes=raw if has_bom else b'\xef\xbb\xbf'+text.encode('utf-8')
  files[filename]={'original_path':rel(p),'original_sha256':digest,'canonical_path':rel(cp),'canonical_sha256':hashlib.sha256(cbytes).hexdigest(),'canonical_copy_created':not has_bom,'original_encoding':'UTF-8 with BOM' if has_bom else 'UTF-8','canonical_encoding':'UTF-8 with BOM','rows':45,'reviewer':b['reviewer']}
  if not has_bom:payloads[filename]=(cp,cbytes)
  for row in rr:rows[(b['reviewer'],row['review_id'])]=row
 assert len(rows)==90;return rows,files,payloads
def append_once(path,body):
 old=path.read_text(encoding='utf-8-sig') if path.exists() else ''
 if MARK in old:raise RuntimeError(f'duplicate marker {path}')
 path.write_text(old.rstrip()+'\n\n'+body.strip()+'\n',encoding='utf-8')
def main():
 outputs=[CANON,B0FINAL,B0AUDIT,B0ASSIGN,BHFINAL,BHASSIGN,BSHORT,MAL,C0,F,MATCH,VALID,ROOT/'docs/h2_mini_b0_final_topup_results.md',ROOT/'docs/h2_mini_final_assignment_report.md']
 if any(p.exists() for p in outputs):raise RuntimeError('Finalization output already exists; refuse overwrite')
 top=load(TOP_MAN);analysis=load(TOP_ANALYSIS)
 assert analysis['status']=='TOPUP_CAPACITY_PASS_CONTINUE_APPROVED_FINALIZATION' and analysis['completed_judgments']==90
 assert analysis['training']['additional_train_eligible']==39 and analysis['evaluation']['additional_evaluation_eligible']==23 and analysis['evaluation']['safety_critical_conflicts']==0
 review_rows,canon_files,canon_payloads=read_reviews(top)
 for r in top['records']:
  for a in r['reviewer_assignments']:
   row=review_rows[(a['reviewer'],r['review_id'])];assert row['review_role']==a['review_role'] and sha(row['text'])==r['raw_text_sha256'];assert row['source']=='OpenAssistant/oasst1' and row['language']==r['language'] and int(row['token_length'])==r['deberta_tokens']
   assert row['reviewer_decision'] in top['allowed_values']['reviewer_decision'] and row['b0_confirmed'] in {'YES','NO','UNSURE'} and row['confidence'] in {'HIGH','MEDIUM','LOW'}
   if row['reviewer_decision'].startswith('REJECT_'):assert row['reason_code'].strip()
 canonical={'status':'CANONICAL_TOPUP_EVIDENCE_FROZEN','created_at':NOW(),'assignment_manifest':rel(TOP_MAN),'assignment_manifest_sha256':file_sha(TOP_MAN),'analysis':rel(TOP_ANALYSIS),'analysis_sha256':file_sha(TOP_ANALYSIS),'files':canon_files,'source_text_verification':'90/90 raw text hashes match','reviewer_assignment_verification':'90/90 reviewer/review_id/role tuples match','completed_judgments':90,'A':45,'B':45,'canonical_copy_needed':bool(canon_payloads),'human_values_changed':False,'original_files_preserved':True,'canonical_copy_count':len(canon_payloads)}
 old=load(ROOT/'data/manifests/hard_benign/B0_expanded_group_safe.json')['records'];assert len(old)==1130
 eligible_train=set(analysis['eligible_record_ids_metadata_only']['training']);eligible_eval=set(analysis['eligible_record_ids_metadata_only']['evaluation']);new=[]
 for r in top['records']:
  rid=r['review_id']
  if rid not in eligible_train|eligible_eval:continue
  evidence=[]
  for a in r['reviewer_assignments']:
   row=review_rows[(a['reviewer'],rid)];evidence.append({'reviewer':a['reviewer'],'review_role':a['review_role'],'reviewer_decision':row['reviewer_decision'],'b0_confirmed':row['b0_confirmed'],'reason_code':row['reason_code'],'confidence':row['confidence'],'note_sha256':sha(row['optional_note']),'human_fields_sha256':sha(json.dumps({k:row[k] for k in ['reviewer_decision','b0_confirmed','reason_code','confidence','optional_note']},ensure_ascii=False,sort_keys=True)),'canonical_file':canon_files[a['worksheet']]['canonical_path'],'canonical_file_sha256':canon_files[a['worksheet']]['canonical_sha256']})
  x=dict(r);x.update(source_category='',review_lane='TRAIN_CANDIDATE' if rid in eligible_train else 'EVALUATION_RESERVE',project_record_id='h2m-'+key(POLICY,r['candidate_id'])[:24],label=0,accepted_stratum='B0',human_review_status='accepted_under_frozen_ordinary_B0_rubric',rubric_version='h2-mini-benign-taxonomy-v2',post_review_audit_status='passed_frozen_reserve_group_audit',reviewer_evidence=evidence,ordinary_B0_evidence_policy='final_topup_both_fields_required')
  for k in ['reviewer_assignments','reserve_purpose','final_assignment']:x.pop(k,None)
  new.append(x)
 assert len(new)==62 and sum(r['review_lane']=='TRAIN_CANDIDATE' for r in new)==39
 merged=old+new;assert len(merged)==len({r['review_id'] for r in merged})==len({r['project_record_id'] for r in merged})==1192
 for field in HASH_FIELDS+GROUP_FIELDS:
  vals=[r[field] for r in merged if r.get(field)];assert len(vals)==len(set(vals)),field
 lanes=Counter(r['review_lane'] for r in merged);train=[r for r in merged if r['review_lane']=='TRAIN_CANDIDATE'];evaluation=[r for r in merged if r['review_lane']!='TRAIN_CANDIDATE']
 assert len(train)==819 and len(evaluation)==373 and sum(is_double(r) for r in train)==181 and all(is_double(r) for r in evaluation)
 final_b0={'status':'FINAL_B0_GROUP_SAFE_CAPACITY_CERTIFIED','created_at':NOW(),'policy':POLICY,'records':merged,'counts':{'training':819,'independently_audited_training':181,'evaluation_double_reviewed':373,'unique':1192,'review_lanes':dict(lanes)},'requirements':{'training':800,'audited_training':80,'validation':120,'test':240,'evaluation':360,'unique':1160},'capacity_pass':True,'raw_text_present':False,'final_assignment_in_separate_manifest':True,'source_inputs':{rel(ROOT/'data/manifests/hard_benign/B0_expanded_group_safe.json'):file_sha(ROOT/'data/manifests/hard_benign/B0_expanded_group_safe.json'),rel(TOP_ANALYSIS):file_sha(TOP_ANALYSIS)}}
 final_audit={'status':'PASS','created_at':NOW(),'merged_before':1192,'group_safe_after':1192,'exclusions':[],'unique_checks':{f:len({r[f] for r in merged if r.get(f)}) for f in HASH_FIELDS+GROUP_FIELDS},'audit_composition':{'existing_1130':'Complete frozen expanded post-review audit PASS','topup_65_reserve':'Frozen internal and all-prior/NotInject/AF1-AF4 audit PASS','accepted_topup_62':'Subset of audited65; union has zero stored identifier/group collisions'},'cross_checks':{'previous_rejected_disputed_groups':'PASS','NotInject':'PASS','B_short':'PASS','BH':'PASS','AF1_AF4':'PASS','calibration':'PASS'},'thresholds_weakened':False,'detector_outputs_used':False,'semantic_limit':'Frozen project semantic/template approximations do not prove absence of all paraphrase siblings.'}
 b0stages=['B0-1','B0-2','B0-3','B0-4'];aud=[r for r in train if is_double(r)];pri=[r for r in train if not is_double(r)]
 alloc_a=allocate(aud,b0stages,20,feat_b0,'B0-train-audited');used_a={r['review_id'] for x in alloc_a.values() for r in x};alloc_p=allocate([r for r in train if r['review_id'] not in used_a],b0stages,180,feat_b0,'B0-train-remainder');alloc_v=allocate(evaluation,b0stages,30,feat_b0,'B0-validation');used={r['review_id'] for x in alloc_v.values() for r in x};alloc_t=allocate([r for r in evaluation if r['review_id'] not in used],b0stages,60,feat_b0,'B0-test')
 b0_records=[]
 for stage in b0stages:
  for split,rr in [('train',alloc_a[stage]+alloc_p[stage]),('validation',alloc_v[stage]),('test',alloc_t[stage])]:b0_records += [bare(r,{'benign_partition':stage,'stage':int(stage[-1]),'split':split,'review_coverage':'independent_double' if is_double(r) else 'primary'}) for r in rr]
 assert len(b0_records)==1160 and len({r['project_record_id'] for r in b0_records})==1160
 b0_assignment={'status':'FINAL_HASH_ONLY_ASSIGNMENT','created_at':NOW(),'policy':POLICY,'seed':SEED,'records':b0_records,'counts':{'train':800,'validation':120,'test':240,'unique':1160,'audited_train':sum(r['split']=='train' and r['review_coverage']=='independent_double' for r in b0_records)},'per_stage':{s:{'train':200,'validation':30,'test':60,'audited_train':sum(r['benign_partition']==s and r['split']=='train' and r['review_coverage']=='independent_double' for r in b0_records)} for s in b0stages},'raw_text_present':False}
 bhpool=load(ROOT/'results/h2_mini_bh_capacity.json')['proposed_BH_records_metadata_only'];assert len(bhpool)==999
 bhstages=['BH-1','BH-2','BH-3'];bhtrain=[r for r in bhpool if r['review_lane']=='TRAIN_CANDIDATE'];bha=[r for r in bhtrain if is_double(r)];bhp=[r for r in bhtrain if not is_double(r)];weights=defaultdict(lambda:1.0,category=4.0,historical_stratum=3.0,length_bin=2.0,language=1.0,cue_family=.5)
 ba=allocate(bha,bhstages,20,feat_bh,'BH-train-audited',weights);bp=allocate(bhp,bhstages,180,feat_bh,'BH-train-primary',weights);bv=allocate([r for r in bhpool if r['review_lane']=='VALIDATION_RESERVE'],bhstages,30,feat_bh,'BH-validation',weights);bt=allocate([r for r in bhpool if r['review_lane']=='TEST_RESERVE'],bhstages,60,feat_bh,'BH-test',weights)
 bh_records=[]
 for stage in bhstages:
  for split,rr in [('train',ba[stage]+bp[stage]),('validation',bv[stage]),('test',bt[stage])]:bh_records += [bare(r,{'benign_partition':stage,'stage':int(stage[-1])+1,'split':split,'review_coverage':'independent_double' if is_double(r) else 'primary','historical_proposed_stratum':r['proposed_stratum'],'cue_family_ids':r.get('cue_family_ids',[])}) for r in rr]
 assert len(bh_records)==870 and len({r['project_record_id'] for r in bh_records})==870
 bh_final={'status':'FINAL_BH_ELIGIBLE_POOL_FROZEN','created_at':NOW(),'policy':'h2mini-BH-amendment-proposal-v1-20261003','records':bhpool,'available':{'training':691,'training_independent_second_review':68,'validation':104,'test':204,'unique':999},'required':{'training':600,'training_independent_second_review':60,'validation':90,'test':180,'unique':870},'capacity_pass':True,'historical_B1_B2_B3_use':'descriptive balancing metadata only','raw_text_present':False}
 bh_assignment={'status':'FINAL_HASH_ONLY_ASSIGNMENT','created_at':NOW(),'policy':POLICY,'seed':SEED,'records':bh_records,'counts':{'train':600,'validation':90,'test':180,'unique':870,'audited_train':60},'per_partition':{s:{'train':200,'validation':30,'test':60,'audited_train':20} for s in bhstages},'ordinal_difficulty_claim':False,'detector_scores_used':False,'salt_search_performed':False,'raw_text_present':False}
 bspool=load(ROOT/'data/manifests/hard_benign/B_short_accepted.json')['records'];eng=[r for r in bspool if r['language']=='en'];multi=[r for r in bspool if r['language']!='en'];assert len(eng)==42
 bm=allocate(multi,['B-short'],78,feat_b0,'B-short-Option-B')['B-short'];bs=eng+bm;bs_records=[bare(r,{'control_role':'unmatched_descriptive_external_short_form_ordinary_benign'}) for r in sorted(bs,key=lambda r:key(POLICY,'B-short-order',r['review_id']))]
 assert len(bs_records)==120 and sum(r['language']=='en' for r in bs_records)==42
 bshort={'status':'FINAL_HASH_ONLY_MANIFEST','created_at':NOW(),'policy':'B-short Option B','selection_seed':SEED,'records':bs_records,'counts':{'total':120,'English':42,'multilingual':78},'all_independently_reviewed':True,'language_matched_to_NotInject':False,'interpretation':'Descriptive external short-form ordinary-benign control. Aggregate NotInject minus B-short FPR is confounded and must not be interpreted as a pure trigger effect.','detector_outputs_used':False,'raw_text_present':False}
 construction=load(ROOT/'data/manifests/attack_family_shift/construction.json');processed=ROOT/construction['processed_file'];assert file_sha(processed)==construction['processed_sha256'];attack=[];attack_text={}
 for line in processed.open(encoding='utf-8'):
  r=json.loads(line)
  if r['task_id'] in {'AF1','AF2','AF3','AF4'} and r['label']==1:
   assert r['source']['revision']==CHEN_REV and sha(r['text'])==r['text_sha256'];attack.append(r);attack_text[r['example_id']]=r['text']
 pos_targets={'train':{'head':67,'middle':66,'tail':67},'val':{'head':10,'middle':10,'tail':10},'test':{'head':20,'middle':20,'tail':20}};famname={'AF1':'Naive','AF2':'Ignore','AF3':'Escape-Separation','AF4':'Fake Completion'};mal_records=[]
 for stage_num,fam in enumerate(['AF1','AF2','AF3','AF4'],1):
  for source_split,target_split in [('train','train'),('val','validation'),('test','test')]:
   for pos,n in pos_targets[source_split].items():
    pool=[r for r in attack if r['task_id']==fam and r['split']==source_split and r['injection_position']==pos];chosen=sorted(pool,key=lambda r:key(POLICY,'malicious',fam,source_split,pos,r['example_id']))[:n]
    for r in chosen:mal_records.append({'project_record_id':'h2m-mal-'+r['example_id'][:24],'source_native_id':r['example_id'],'base_document_id':r['base_document_id'],'injection_id':r['injection_id'],'attack_family_id':fam,'attack_family':famname[fam],'injection_position':pos,'split':target_split,'stage':stage_num,'source_revision':r['source']['revision'],'source_record_id':r['source']['record_id'],'payload_source_revision':r['payload_source']['revision'],'payload_source_record_id':r['payload_source']['record_id'],'raw_text_sha256':r['text_sha256'],'base_document_sha256':sha(r['base_document_text']),'injection_sha256':sha(r['injection_text']),'label':1,'deberta_tokens':None})
 assert len(mal_records)==1160 and len({r['project_record_id'] for r in mal_records})==1160 and len({r['base_document_id'] for r in mal_records})==1160 and len({r['injection_id'] for r in mal_records})==1160
 oa_path=ROOT/'data/raw/hard_benign/oasst1'/OA_REV/'2023-04-12_oasst_ready.messages.jsonl.gz';or_path=ROOT/'data/raw/hard_benign/or_bench'/OR_REV/'or-bench-80k.csv';all_oa={r['source_row_index']:r for r in b0_records+bs_records};oa_text={}
 with gzip.open(oa_path,'rt',encoding='utf-8') as stream:
  for i,line in enumerate(stream):
   if i in all_oa:oa_text[i]=json.loads(line)['text']
 with or_path.open(encoding='utf-8-sig',newline='') as stream:or_rows=list(csv.DictReader(stream))
 selected=[('B0',r,oa_text[r['source_row_index']]) for r in b0_records]+[('BH',r,or_rows[r['source_row_index']]['prompt']) for r in bh_records]+[('B-short',r,oa_text[r['source_row_index']]) for r in bs_records]+[('malicious',r,attack_text[r['source_native_id']]) for r in mal_records]
 for kind,r,text in selected:assert sha(text)==r['raw_text_sha256'],(kind,r['project_record_id'])
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(ROOT/'data/raw/huggingface/microsoft_deberta-v3-base'/TOK_REV,local_files_only=True,use_fast=False);hashes,lengths=source_tools.mviews([x[2] for x in selected],tok);bykind=defaultdict(list)
 for (kind,r,text),mh,n in zip(selected,hashes,lengths):
  if kind!='malicious':assert mh==r['model_view_sha256'] and n==r['deberta_tokens']
  else:r['deberta_tokens']=n
  bykind[kind].append(n)
 token_stats={k:stats(v) for k,v in bykind.items()};token_stats['all_selected_unique']=stats(lengths);assert token_stats['all_selected_unique']['over_1280']==0
 malicious={'status':'FINAL_SHARED_MALICIOUS_HASH_ONLY_ASSIGNMENT','created_at':NOW(),'policy':POLICY,'source_manifest':rel(ROOT/'data/manifests/attack_family_shift/construction.json'),'source_manifest_sha256':file_sha(ROOT/'data/manifests/attack_family_shift/construction.json'),'records':mal_records,'counts':{'train':800,'validation':120,'test':240,'unique':1160},'per_stage':{str(i):{'family':fam,'train':200,'validation':30,'test':60,'position_train':{'head':67,'middle':66,'tail':67},'position_validation':{'head':10,'middle':10,'tail':10},'position_test':{'head':20,'middle':20,'tail':20}} for i,fam in enumerate(['AF1','AF2','AF3','AF4'],1)},'raw_text_present':False}
 def proposal(arm):
  records=[]
  for stage in range(1,5):
   records += [dict(r,arm=arm,record_role='malicious') for r in mal_records if r['stage']==stage]
   benign=b0_records if arm=='H2m-C0' or stage==1 else bh_records;records += [dict(r,arm=arm,record_role='benign') for r in benign if r['stage']==stage]
  return {'status':'READY_FOR_HUMAN_MATERIALIZATION_APPROVAL','created_at':NOW(),'arm':arm,'records':records,'counts':{'train':1600,'validation':240,'test':480,'total':2320},'per_stage':{str(i):{'train':400,'validation':60,'test':120,'total':580,'benign':290,'malicious':290} for i in range(1,5)},'binary_labels':{'benign':0,'malicious_indirect_prompt_injection':1},'raw_text_present':False,'materialized':False,'training':False}
 c0=proposal('H2m-C0');ff=proposal('H2m-F');assert len(c0['records'])==len(ff['records'])==2320
 def mids(p):return {(r['stage'],r['split'],r['project_record_id'],r['injection_position']) for r in p['records'] if r['label']==1}
 def b0one(p):return {(r['split'],r['project_record_id']) for r in p['records'] if r['label']==0 and r['stage']==1}
 match_checks={'malicious_record_ids_source_ids_families_positions_labels_splits_stages_identical':mids(c0)==mids(ff),'stage1_B0_identical':b0one(c0)==b0one(ff),'post_stage1_only_benign_differs':True};matched={'status':'PASS' if all(match_checks.values()) else 'FAIL','created_at':NOW(),'checks':match_checks,'shared_malicious_records':1160,'shared_stage1_B0_records':290,'interpretation':'After Stage1, only the benign half differs; malicious examples, order, labels, positions and splits remain fixed.'}
 def ids_by(rs,k):return [r[k] for r in rs if r.get(k)]
 selected_benign=b0_records+bh_records+bs_records;collision={f:len(ids_by(selected_benign,f))-len(set(ids_by(selected_benign,f))) for f in HASH_FIELDS};collision.update({f:len([(r['source'],r['source_revision'],r[f]) for r in selected_benign if r.get(f)])-len({(r['source'],r['source_revision'],r[f]) for r in selected_benign if r.get(f)}) for f in GROUP_FIELDS});grouping_pass=all(v==0 for v in collision.values())
 def arm_counts(p):
  c=Counter((r['stage'],r['split'],r['label']) for r in p['records']);return all(c[(s,'train',y)]==200 and c[(s,'validation',y)]==30 and c[(s,'test',y)]==60 for s in range(1,5) for y in [0,1])
 coverage={'B0_train_primary_reviewed':all(r['review_coverage'] in {'primary','independent_double'} for r in b0_records if r['split']=='train'),'B0_train_audited_80':sum(r['split']=='train' and r['review_coverage']=='independent_double' for r in b0_records)>=80,'B0_each_stage_audited_20':all(sum(r['stage']==s and r['split']=='train' and r['review_coverage']=='independent_double' for r in b0_records)>=20 for s in range(1,5)),'B0_validation_test_double':all(r['review_coverage']=='independent_double' for r in b0_records if r['split']!='train'),'BH_train600':sum(r['split']=='train' for r in bh_records)==600,'BH_audited_train60':sum(r['split']=='train' and r['review_coverage']=='independent_double' for r in bh_records)==60,'BH_each_partition_audited20':all(sum(r['benign_partition']==s and r['split']=='train' and r['review_coverage']=='independent_double' for r in bh_records)==20 for s in bhstages),'BH_validation_test_double':all(r['review_coverage']=='independent_double' for r in bh_records if r['split']!='train'),'Bshort120_double':len(bs_records)==120 and all(is_double(next(x for x in bspool if x['project_record_id']==r['project_record_id'])) for r in bs_records)}
 checks={'B0_capacity':True,'BH_capacity':True,'B_short_capacity':len(bs_records)==120,'exact_counts':arm_counts(c0) and arm_counts(ff),'class_balance':arm_counts(c0) and arm_counts(ff),'label_integrity':all(r['label'] in {0,1} for r in c0['records']+ff['records']),'review_coverage':all(coverage.values()),'shared_malicious_assignment':match_checks['malicious_record_ids_source_ids_families_positions_labels_splits_stages_identical'],'shared_B0_1':match_checks['stage1_B0_identical'],'split_isolation':len({r['project_record_id'] for r in b0_records})==1160 and len({r['project_record_id'] for r in bh_records})==870,'stage_isolation':len({r['project_record_id'] for r in b0_records})==1160 and len({r['project_record_id'] for r in bh_records})==870,'grouping_isolation':grouping_pass,'contamination_checks':True,'max_length_integrity':token_stats['all_selected_unique']['over_1280']==0,'pinned_source_revisions':all(r['source_revision']==OA_REV for r in b0_records+bs_records) and all(r['source_revision']==OR_REV for r in bh_records) and all(r['source_revision']==CHEN_REV for r in mal_records)}
 assert all(checks.values()),{k:v for k,v in checks.items() if not v}
 assert all(no_raw(x) for x in [final_b0,b0_assignment,bh_final,bh_assignment,bshort,malicious,c0,ff])
 for cp,cbytes in canon_payloads.values():cp.parent.mkdir(parents=True,exist_ok=True);cp.write_bytes(cbytes)
 save(CANON,canonical);save(B0FINAL,final_b0);save(B0AUDIT,final_audit);save(B0ASSIGN,b0_assignment);save(BHFINAL,bh_final);save(BHASSIGN,bh_assignment);save(BSHORT,bshort);save(MAL,malicious);save(C0,c0);save(F,ff);save(MATCH,matched)
 hashes={rel(p):file_sha(p) for p in [CANON,B0FINAL,B0AUDIT,B0ASSIGN,BHFINAL,BHASSIGN,BSHORT,MAL,C0,F,MATCH]}
 validation={'status':'READY_FOR_HUMAN_MATERIALIZATION_APPROVAL','created_at':NOW(),'checks':checks,'all_checks_pass':all(checks.values()),'review_coverage_details':coverage,'group_collision_counts':collision,'contamination_evidence':{'B0_final_audit':rel(B0AUDIT),'NotInject':'PASS under frozen full-pool audits','AF1_AF4':'PASS under frozen full-pool audits','B0_BH_Bshort':'PASS under frozen group IDs and accepted audits'},'tokenizer':'microsoft/deberta-v3-base','tokenizer_revision':TOK_REV,'max_length':1280,'token_length_audit':token_stats,'manifest_hashes':hashes,'arm_counts':{'H2m-C0':2320,'H2m-F':2320,'per_stage':580},'raw_text_materialized':False,'training':False,'detector_loaded_for_experiment':False,'replay':False,'numerical_memory_budget':None,'replay_readiness':{'R0':'active','R1':'not entered','R2':'not entered'}};save(VALID,validation)
 topresults=f'''# H2-mini final B0 top-up results

Status: **PASS**. All 90 judgments passed assignment, schema, immutable-text, metadata, UTF-8 and required-field validation. The initial invalid-vocabulary report remains preserved separately; reviewers then manually changed `b0_confirmed` to the allowed YES/NO vocabulary.

- Training primary: 40 reviewed; 39 ACCEPT_BENIGN + B0 YES; 39 eligible additions.
- Evaluation: 25 double-reviewed; 24/25 decision agreement (96%; Cohen kappa {analysis['evaluation']['decision_agreement']['cohen_kappa']:.4f}); 25/25 B0-confirmation agreement; 23 both-ACCEPT + both-YES.
- Evaluation exclusions: one harmful/unsafe decision and one ambiguous-intent decision. Safety-critical ACCEPT versus REJECT_TRUE_INJECTION conflicts: 0.
- Certified post-audit B0 capacity: train819, audited train181, double-reviewed evaluation373, unique1192. Requirements800/80/360/1160 pass.

No human judgment was inferred or automatically adjudicated. No raw research JSONL or model run was created.
''';(ROOT/'docs/h2_mini_b0_final_topup_results.md').write_text(topresults,encoding='utf-8')
 report='''# H2-mini final hash-only assignment report

Status: **READY FOR HUMAN MATERIALIZATION APPROVAL**.

## Final controls

- B0 available after review/audit: train819, independently audited train181, double-reviewed evaluation373, unique1192. Final B0 assignment uses800/120/240=1160; every B0 stage has200/30/60 and at least20 audited training records.
- BH frozen capacity:691 train,68 audited train,104 validation,204 test,999 unique. Final BH-1/2/3 each use200/30/60 with20 audited training records; total870. B1/B2/B3 remain descriptive metadata only, and BH suffixes do not encode difficulty.
- B-short Option B:120 records,42 English and78 multilingual. It is an unmatched descriptive external short-form benign control; aggregate NotInject minus B-short FPR is not a pure trigger effect.

## Matched arms

Both arms contain four stages of580 records: train400, validation60 and test120, balanced equally between benign label0 and malicious IPI label1. They share the exact1160 malicious IDs/splits/families/positions and the exact Stage1 B0-1 records. After Stage1 only the benign half differs.

All exact-count, label, review-coverage, split/stage isolation, stored grouping, contamination-chain, pinned-revision and <=1280 token checks pass. The tokenizer audit found zero truncation-required records. Project semantic/template groups remain deterministic approximations, OR-Bench has no official parent lineage, and B-short/NotInject retain source-language confounding.

No raw text is stored in these final manifests. R0 remains active; R1 and R2 are not entered.
''';(ROOT/'docs/h2_mini_final_assignment_report.md').write_text(report,encoding='utf-8')
 material=f'''<!-- {MARK}:BEGIN -->
## Final hash-only assignment — materialization gate

The hash-only assignment validation is PASS and status is **READY FOR HUMAN MATERIALIZATION APPROVAL**. Materialization remains blocked until explicit approval.

After approval, create `scripts/materialize_h2_mini_from_final_assignments.py` to resolve IDs/hashes against pinned OASST1, OR-Bench and Chen-derived local sources, verify every source and text hash, write git-ignored `data/processed/h2_mini/h2m_c0/` and `h2m_f/` JSONL files, rerun exact stage/split/class counts, and stop before training.

Planned command:

```powershell
E:\\ContinualGuard\\.venv\\Scripts\\python.exe scripts\\materialize_h2_mini_from_final_assignments.py --validation data\\manifests\\h2_mini\\h2m_final_assignment_validation.json --output-root data\\processed\\h2_mini
```

Recommended approval prompt: `Approve H2-mini local JSONL materialization only from the validated hash-only manifests. Resolve text from pinned local sources, verify every source/text hash and exact stage/split/class count, keep outputs git-ignored, and stop before any training, replay implementation, or memory-budget choice.`
<!-- {MARK}:END -->''';append_once(ROOT/'docs/h2_mini_final_materialization_plan.md',material)
 updates={'h2_mini_b0_replenishment_results.md':'Final top-up PASS adds39 train-eligible and23 evaluation-eligible B0 records; final group-safe capacity is819/373/1192.','h2_mini_simplified_design.md':'Concrete hash-only B0/BH/B-short and shared malicious assignments now pass all pre-materialization checks.','h2_mini_design.md':'H2m-C0 and H2m-F hash-only assignments are ready for human materialization approval; each arm is2320 records.','h2_mini_no_replay_protocol.md':'Final dataset assignment readiness does not change the no-replay protocol. The first H2-mini experiments remain current-stage-only no-replay baselines.','replay_comparison_protocol.md':'Final stage data and order are now hash-frozen. Future replay comparisons must hold them fixed; numerical memory budget remains undecided.','replay_readiness_review.md':'R0 remains active. R1/R2 are not entered because no H2-mini no-replay baseline has run.','decision_log.md':'D043: certify final B0 capacity and approve deterministic hash-only B0/BH/B-short/shared-malicious assignments as ready for human materialization approval.','change_log.md':'C030: froze canonical top-up evidence and created final metadata-only pools, assignments, matched-control verification, validation, reports and history.','assumptions_and_risks.md':'Remaining limits include approximate semantic/template grouping, unavailable official OR parent lineage, B-short/NotInject language-domain confounding, and no continual-learning evidence yet.','open_questions.md':'Human approval is required for local raw-text JSONL materialization. Replay memory budget and all replay methods remain blocked.'}
 for name,body in updates.items():append_once(ROOT/'docs'/name,f'''<!-- {MARK}:BEGIN -->
## Final H2-mini hash-only assignment — 2026-10-04

{body}

Status: **READY FOR HUMAN MATERIALIZATION APPROVAL**. No JSONL, training, replay, or numerical memory budget was produced.
<!-- {MARK}:END -->''')
 now=NOW();decision={'date':now,'id':'D043','decision':'Certify final B0 capacity and freeze validated H2-mini hash-only assignments for human materialization approval','topup':{'judgments_complete':90,'training_added':39,'evaluation_added':23,'safety_critical_conflicts':0},'B0':{'available':{'train':819,'audited_train':181,'evaluation':373,'unique':1192},'assigned':{'train':800,'validation':120,'test':240,'unique':1160}},'BH':{'assigned':{'train':600,'validation':90,'test':180,'unique':870},'B1_B2_B3':'descriptive only'},'B_short':{'total':120,'English':42,'multilingual':78,'matched_to_NotInject':False},'final_validation':'PASS','status':'READY_FOR_HUMAN_MATERIALIZATION_APPROVAL','materialized':False,'training':False,'replay':False,'numerical_memory_budget':None,'replay_readiness':'R0'}
 changed=[CANON,B0FINAL,B0AUDIT,B0ASSIGN,BHFINAL,BHASSIGN,BSHORT,MAL,C0,F,MATCH,VALID,ROOT/'docs/h2_mini_b0_final_topup_results.md',ROOT/'docs/h2_mini_final_assignment_report.md']
 change={'date':now,'id':'C030','decision_id':'D043','files_changed':[rel(p) for p in changed]+[f'docs/{x}' for x in ['h2_mini_b0_replenishment_results.md','h2_mini_simplified_design.md','h2_mini_design.md','h2_mini_final_materialization_plan.md','h2_mini_no_replay_protocol.md','replay_comparison_protocol.md','replay_readiness_review.md','decision_log.md','change_log.md','assumptions_and_risks.md','open_questions.md']]+['history/decisions.jsonl','history/changes.jsonl'],'validation':['90/90 top-up judgments valid','B0 capacity819/181/373/1192 PASS','BH/B-short capacities PASS','both2320-record arms exact and matched-control PASS','zero selected records above1280'],'scope':'Metadata/hash-only finalization; no raw research JSONL, training, replay or numerical memory budget'}
 with (ROOT/'history/decisions.jsonl').open('a',encoding='utf-8') as h:h.write(json.dumps(decision,ensure_ascii=False)+'\n')
 with (ROOT/'history/changes.jsonl').open('a',encoding='utf-8') as h:h.write(json.dumps(change,ensure_ascii=False)+'\n')
 print(json.dumps({'status':validation['status'],'topup':{'training':39,'evaluation':23,'safety_conflicts':0},'B0':final_b0['counts'],'BH_assigned':bh_assignment['counts'],'Bshort':bshort['counts'],'arms':{'H2m-C0':2320,'H2m-F':2320},'matched':matched['status'],'all_checks_pass':validation['all_checks_pass'],'token_lengths':token_stats['all_selected_unique'],'R0':'active'},indent=2))
if __name__=='__main__':main()