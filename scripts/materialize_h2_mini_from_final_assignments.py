"""Materialize validated H2-mini assignments locally and stop before training."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,math,os,shutil,sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from h2_mini_b0_review_common import sha,file_sha,load
import audit_h2_mini_sources as source_tools
EXPECTED_ROOT=Path(r'E:\ContinualGuard');TOK_REV='8ccc9b6f36199bec6961081d44eb72fb3f7353f3';MAX_LENGTH=1280
OA_REV='fdf72ae0827c1cda404aff25b6603abec9e3399b';OR_REV='e36d8b80e81837c8a8f264bbb2a49f1b32c7e272';CHEN_REV='1e9e0f00179adc154e9ca991b9cd3206c99f9588'
C0=ROOT/'data/manifests/h2_mini/h2m_c0_final_assignment_proposal.json';F=ROOT/'data/manifests/h2_mini/h2m_f_final_assignment_proposal.json';MAL=ROOT/'data/manifests/h2_mini/h2m_shared_malicious_assignment.json';FINAL_VALID=ROOT/'data/manifests/h2_mini/h2m_final_assignment_validation.json';BSHORT=ROOT/'data/manifests/hard_benign/b_short_final_hash_manifest.json'
RESULT=ROOT/'results/h2_mini_materialization_validation.json';TARGET=ROOT/'data/processed/h2_mini';TMP=ROOT/'data/processed/_tmp_h2_mini_materialization';SPLIT_FILE={'train':'train','validation':'val','test':'test'};FILE_SPLIT={v:k for k,v in SPLIT_FILE.items()}
def now():return datetime.now(timezone.utc).isoformat()
def rel(p):return p.resolve().relative_to(ROOT).as_posix()
def save_json(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def canonical_line(row):return json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n'
def write_jsonl(path,rows):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(canonical_line(r) for r in sorted(rows,key=lambda x:x['example_id'])),encoding='utf-8',newline='\n')
def read_jsonl(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
def verify_file_hash(path,expected):
 actual=file_sha(path)
 if actual!=expected:raise ValueError(f'source hash mismatch: {rel(path)} expected={expected} actual={actual}')
def pct(v,q):return sorted(v)[max(0,math.ceil(q*len(v))-1)]
def length_stats(v):return {'count':len(v),'min':min(v),'median':pct(v,.5),'p95':pct(v,.95),'p99':pct(v,.99),'max':max(v),'over_1280':sum(x>MAX_LENGTH for x in v),'truncation_required':sum(x>MAX_LENGTH for x in v)}
def validate_labels(rows):
 bad=[r.get('example_id') for r in rows if type(r.get('label')) is not int or r['label'] not in (0,1)]
 if bad:raise ValueError(f'invalid labels: {bad[:5]}')
def validate_unique_and_isolation(rows):
 ids=[r['example_id'] for r in rows]
 if len(ids)!=len(set(ids)):raise ValueError('duplicate example_id')
 for field in ['base_document_id','text_sha256','normalized_text_sha256','model_view_sha256','duplicate_group_id']:
  seen={}
  for r in rows:
   value=r.get(field)
   if not value:continue
   loc=(r['task_id'],r['split']);old=seen.get(value)
   if old and old!=loc:raise ValueError(f'cross-split/stage leakage {field}: {value}')
   seen[value]=loc
 for field in ['source_native_id','source_parent_id','exact_group','normalized_group','model_view_group','project_semantic_parent_group','project_template_group']:
  seen={}
  for r in rows:
   value=r.get(field)
   if not value:continue
   scoped=(r['source']['dataset'],r['source']['revision'],value);loc=(r['task_id'],r['split']);old=seen.get(scoped)
   if old and old!=loc:raise ValueError(f'cross-split/stage leakage {field}: {scoped}')
   seen[scoped]=loc
 return True
def validate_arm_contract(rows,quotas=None):
 quotas=quotas or {'train':200,'val':30,'test':60};validate_labels(rows);validate_unique_and_isolation(rows)
 c=Counter((r['task_id'],r['split'],r['label']) for r in rows)
 for stage in ['AF1','AF2','AF3','AF4']:
  for split,n in quotas.items():
   for label in [0,1]:
    if c[(stage,split,label)]!=n:raise ValueError(f'exact count mismatch {stage}/{split}/{label}: {c[(stage,split,label)]}!={n}')
 if len(rows)!=2*4*sum(quotas.values()):raise ValueError('arm total mismatch')
 return c
def matched_signature(r):return (r['example_id'],r['source_native_id'],r['attack_family'],r['injection_position'],r['label'],r['split'],r['task_id'])
def verify_matched_arms(c0,f):
 cm={matched_signature(r) for r in c0 if r['label']==1};fm={matched_signature(r) for r in f if r['label']==1}
 if cm!=fm:raise ValueError('matched-arm malicious mismatch')
 cb={(r['example_id'],r['split'],r['text_sha256']) for r in c0 if r['label']==0 and r['task_id']=='AF1'};fb={(r['example_id'],r['split'],r['text_sha256']) for r in f if r['label']==0 and r['task_id']=='AF1'}
 if cb!=fb:raise ValueError('Stage-1 B0 mismatch')
 c_later={r['example_id'] for r in c0 if r['label']==0 and r['task_id']!='AF1'};f_later={r['example_id'] for r in f if r['label']==0 and r['task_id']!='AF1'}
 if c_later&f_later:raise ValueError('post-Stage1 benign overlap')
 return {'malicious_records':len(cm),'stage1_B0_records':len(cb),'post_stage1_benign_overlap':0}
def audit_token_lengths(rows,tokenizer):
 lengths=[len(tokenizer.encode(r['text'],add_special_tokens=True,truncation=False)) for r in rows];d=length_stats(lengths)
 if d['truncation_required']:raise ValueError('max_length integrity failed')
 return d
def verify_authority():
 validation=load(FINAL_VALID)
 if validation['status']!='READY_FOR_HUMAN_MATERIALIZATION_APPROVAL' or not validation['all_checks_pass']:raise ValueError('final assignment validation is not PASS')
 for path,digest in validation['manifest_hashes'].items():verify_file_hash(ROOT/path,digest)
 sources={}
 for name,filename in [('oasst1_full_audit.json','2023-04-12_oasst_ready.messages.jsonl.gz'),('or_bench_full_audit.json','or-bench-80k.csv')]:
  audit=load(ROOT/'data/audits/hard_benign'/name);entry=next(x for x in audit['source']['files'] if x['file_name']==filename);path=ROOT/entry['local_path'];verify_file_hash(path,entry['sha256']);sources[name]={'path':path,'sha256':entry['sha256'],'revision':audit['source']['revision']}
 construction=load(ROOT/'data/manifests/attack_family_shift/construction.json');processed=ROOT/construction['processed_file'];verify_file_hash(processed,construction['processed_sha256']);sources['chen']={'path':processed,'sha256':construction['processed_sha256'],'revision':CHEN_REV}
 if sources['oasst1_full_audit.json']['revision']!=OA_REV or sources['or_bench_full_audit.json']['revision']!=OR_REV:raise ValueError('pinned source revision mismatch')
 return validation,sources
def load_source_rows(all_assignments,bshort,sources):
 oa_meta=[r for r in all_assignments+bshort if r.get('source')=='OASST1'];or_meta=[r for r in all_assignments if r.get('source')=='OR-Bench'];mal_meta=[r for r in all_assignments if r.get('label')==1]
 wanted={r['source_row_index'] for r in oa_meta};oa={}
 with gzip.open(sources['oasst1_full_audit.json']['path'],'rt',encoding='utf-8',errors='strict') as stream:
  for i,line in enumerate(stream):
   if i in wanted:oa[i]=json.loads(line)
 if len(oa)!=len(wanted):raise ValueError('missing OASST source rows')
 with sources['or_bench_full_audit.json']['path'].open(encoding='utf-8-sig',newline='') as stream:or_rows=list(csv.DictReader(stream))
 wanted_mal={r['source_native_id'] for r in mal_meta};chen={}
 with sources['chen']['path'].open(encoding='utf-8') as stream:
  for line in stream:
   row=json.loads(line)
   if row.get('example_id') in wanted_mal:chen[row['example_id']]=row
 if set(chen)!=wanted_mal:raise ValueError('missing Chen-derived source rows')
 return oa,or_rows,chen
def benign_record(meta,text,arm,external=False):
 if sha(text)!=meta['raw_text_sha256']:raise ValueError(f'raw hash mismatch: {meta["project_record_id"]}')
 dataset=meta['source'];revision=meta['source_revision']
 if dataset=='OASST1':
  domain='oasst1_ordinary_benign';record_id=meta['source_native_id'];base=meta['source_parent_id'] or meta['project_record_id']
 elif dataset=='OR-Bench':
  domain='or_bench_hard_benign';record_id=meta['candidate_id'];base=meta['project_record_id']
 else:raise ValueError(f'unauthorized benign source: {dataset}')
 stage=meta.get('stage');partition=meta.get('benign_partition');group='B0' if (partition or '').startswith('B0') else 'BH' if (partition or '').startswith('BH') else 'B-short'
 split='test' if external else SPLIT_FILE[meta['split']];task='B-short' if external else f'AF{stage}'
 row={'text':text,'label':0,'task_id':task,'split':split,'source':{'dataset':dataset,'revision':revision,'upstream_split':meta.get('split','external'),'record_id':record_id,'construction':'H2-mini final reviewed benign materialization'},'example_id':meta['project_record_id'],'attack_family':'none','domain':domain,'base_document_id':base,'injection_id':None,'injection_position':'none','text_sha256':meta['raw_text_sha256'],'language':meta['language'],'benign_group':group,'benign_partition':partition,'annotation_status':'human_reviewed_final_hash_assignment','assignment_project_record_id':meta['project_record_id'],'review_id':meta.get('review_id'),'source_native_id':meta.get('source_native_id'),'source_parent_id':meta.get('source_parent_id'),'source_immediate_parent_id':meta.get('source_immediate_parent_id'),'duplicate_group_id':meta.get('exact_group'),'exact_group':meta.get('exact_group'),'normalized_group':meta.get('normalized_group'),'model_view_group':meta.get('model_view_group'),'project_semantic_parent_group':meta.get('project_semantic_parent_group'),'project_template_group':meta.get('project_template_group')}
 if not external:row['stage']=stage;row['arm']=arm
 else:row['control_role']='unmatched_descriptive_external_short_form_ordinary_benign'
 return row
def malicious_record(meta,src,arm):
 expected_split=SPLIT_FILE[meta['split']]
 pairs=[(src['example_id'],meta['source_native_id'],'example_id'),(src['base_document_id'],meta['base_document_id'],'base_document_id'),(src['injection_id'],meta['injection_id'],'injection_id'),(src['task_id'],meta['attack_family_id'],'family'),(src['injection_position'],meta['injection_position'],'position'),(src['split'],expected_split,'split'),(src['text_sha256'],meta['raw_text_sha256'],'raw_hash')]
 for actual,expected,name in pairs:
  if actual!=expected:raise ValueError(f'malicious identity mismatch {name}: {meta["project_record_id"]}')
 if src['label']!=1 or meta['label']!=1 or src['source']['revision']!=CHEN_REV:raise ValueError('malicious label/revision mismatch')
 row=dict(src);row.update(stage=meta['stage'],arm=arm,assignment_project_record_id=meta['project_record_id'],source_native_id=src['example_id'])
 return row
def build_rows(proposal,bshort_meta,oa,or_rows,chen,arm):
 rows=[]
 for meta in proposal['records']:
  if meta['label']==1:rows.append(malicious_record(meta,chen[meta['source_native_id']],arm));continue
  if meta['source']=='OASST1':
   src=oa[meta['source_row_index']]
   if (src['message_id'],src['message_tree_id'],src.get('parent_id'))!=(meta['source_native_id'],meta['source_parent_id'],meta.get('source_immediate_parent_id')):raise ValueError('OASST identity mismatch')
   text=src['text']
  else:text=or_rows[meta['source_row_index']]['prompt']
  rows.append(benign_record(meta,text,arm))
 ext=[]
 for meta in bshort_meta:
  src=oa[meta['source_row_index']]
  if src['message_id']!=meta['source_native_id']:raise ValueError('B-short identity mismatch')
  ext.append(benign_record(meta,src['text'],'external',external=True))
 return rows,ext
def attach_and_verify_model_views(rows,assignment_by_pid,tokenizer):
 texts=[r['text'] for r in rows];model_hashes,lengths=source_tools.mviews(texts,tokenizer)
 for row,mh,n in zip(rows,model_hashes,lengths):
  meta=assignment_by_pid[row['assignment_project_record_id']]
  raw=sha(row['text']);norm=sha(source_tools.norm(row['text']))
  if raw!=meta['raw_text_sha256']:raise ValueError(f'raw text hash resolution failed: {row["example_id"]}')
  if meta.get('normalized_text_sha256') and norm!=meta['normalized_text_sha256']:raise ValueError(f'normalized hash resolution failed: {row["example_id"]}')
  if meta.get('model_view_sha256') and mh!=meta['model_view_sha256']:raise ValueError(f'model-view hash resolution failed: {row["example_id"]}')
  if meta.get('deberta_tokens') is not None and n!=meta['deberta_tokens']:raise ValueError(f'token length mismatch: {row["example_id"]}')
  row['normalized_text_sha256']=norm;row['model_view_sha256']=mh;row['deberta_tokens']=n
 return lengths
def validate_schema(rows):
 from continualguard.data import validate_record
 for row in rows:validate_record(row)
def strict_content_unique(rows):
 for field in ['text_sha256','normalized_text_sha256','model_view_sha256','base_document_id','duplicate_group_id']:
  values=[r[field] for r in rows if r.get(field)]
  if len(values)!=len(set(values)):raise ValueError(f'duplicate/leakage in {field}')
 for field in ['source_native_id','source_parent_id','exact_group','normalized_group','model_view_group','project_semantic_parent_group','project_template_group']:
  values=[(r['source']['dataset'],r['source']['revision'],r[field]) for r in rows if r.get(field)]
  if len(values)!=len(set(values)):raise ValueError(f'duplicate/leakage in scoped {field}')
def post_file_token_audit(rows,tokenizer):
 texts=[r['text'] for r in rows];hashes,lengths=source_tools.mviews(texts,tokenizer)
 for r,mh,n in zip(rows,hashes,lengths):
  if sha(r['text'])!=r['text_sha256'] or sha(source_tools.norm(r['text']))!=r['normalized_text_sha256'] or mh!=r['model_view_sha256'] or n!=r['deberta_tokens']:raise ValueError(f'post-file hash mismatch: {r["example_id"]}')
 d=length_stats(lengths)
 if d['truncation_required']:raise ValueError('post-file max_length failure')
 return d
def validate_materialized(c0,f,bshort,tokenizer,notinject):
 validate_schema(c0+f+bshort);validate_arm_contract(c0);validate_arm_contract(f);strict_content_unique(c0);strict_content_unique(f);strict_content_unique(bshort)
 matched=verify_matched_arms(c0,f)
 if len(bshort)!=120 or Counter('English' if r['language']=='en' else 'multilingual' for r in bshort)!=Counter({'English':42,'multilingual':78}):raise ValueError('B-short composition mismatch')
 if any(r['label']!=0 or r['split']!='test' or r['task_id']!='B-short' for r in bshort):raise ValueError('B-short independence/label failure')
 arm=c0+f;aid={r['example_id'] for r in arm};anative={r.get('source_native_id') for r in arm if r.get('source_native_id')};bid={r['example_id'] for r in bshort};bnative={r.get('source_native_id') for r in bshort if r.get('source_native_id')}
 if aid&bid or anative&bnative:raise ValueError('B-short identity contamination')
 for field in ['text_sha256','normalized_text_sha256','model_view_sha256']:
  if {r[field] for r in arm}&{r[field] for r in bshort}:raise ValueError(f'B-short content contamination {field}')
 ni_raw={r['prompt_sha256'] for r in notinject};ni_norm={r['normalized_prompt_sha256'] for r in notinject}
 if ni_raw&{r['text_sha256'] for r in c0+f+bshort} or ni_norm&{r['normalized_text_sha256'] for r in c0+f+bshort}:raise ValueError('NotInject contamination')
 expected={'AF1':'naive','AF2':'ignore','AF3':'escape_separation','AF4':'fake_completion'}
 for r in c0+f:
  if r['label']==1 and (r['task_id'] not in expected or r['attack_family']!=expected[r['task_id']]):raise ValueError('unauthorized AF-family assignment')
  if r['label']==0 and r['attack_family']!='none':raise ValueError('benign attack-family corruption')
 c0stats=post_file_token_audit(c0,tokenizer);fstats=post_file_token_audit(f,tokenizer);bstats=post_file_token_audit(bshort,tokenizer)
 unique={r['assignment_project_record_id']:r for r in c0+f+bshort};ustats=post_file_token_audit(list(unique.values()),tokenizer)
 return {'matched':matched,'token_lengths':{'H2m-C0':c0stats,'H2m-F':fstats,'B-short':bstats,'unique_selected':ustats},'notinject_records_checked':len(notinject),'Bshort_identity_overlap':0,'Bshort_content_overlap':0,'NotInject_hash_overlap':0}
def write_tree(tmp,c0,f,bshort):
 files=[]
 for arm,dirname,rows in [('H2m-C0','h2m_c0',c0),('H2m-F','h2m_f',f)]:
  for stage in range(1,5):
   for split in ['train','val','test']:
    subset=[r for r in rows if r['task_id']==f'AF{stage}' and r['split']==split];path=tmp/dirname/f'stage{stage}'/f'{split}.jsonl';write_jsonl(path,subset)
    files.append({'relative_file':f'data/processed/h2_mini/{dirname}/stage{stage}/{split}.jsonl','row_count':len(subset),'byte_size':path.stat().st_size,'sha256':file_sha(path),'arm':arm,'stage':stage,'split':split,'class_counts':{str(k):v for k,v in sorted(Counter(r['label'] for r in subset).items())}})
 path=tmp/'external/b_short.jsonl';write_jsonl(path,bshort);files.append({'relative_file':'data/processed/h2_mini/external/b_short.jsonl','row_count':len(bshort),'byte_size':path.stat().st_size,'sha256':file_sha(path),'arm':'external','stage':None,'split':'test','class_counts':{'0':len(bshort)}})
 return files
def read_tree(tmp):
 arms={}
 for dirname in ['h2m_c0','h2m_f']:
  rows=[]
  for stage in range(1,5):
   for split in ['train','val','test']:rows+=read_jsonl(tmp/dirname/f'stage{stage}'/f'{split}.jsonl')
  arms[dirname]=rows
 return arms['h2m_c0'],arms['h2m_f'],read_jsonl(tmp/'external/b_short.jsonl')
def deterministic_files(tmp,files):
 for item in files:
  p=tmp/Path(item['relative_file']).relative_to('data/processed/h2_mini');rows=read_jsonl(p);expected=''.join(canonical_line(r) for r in sorted(rows,key=lambda x:x['example_id'])).encode('utf-8')
  if p.read_bytes()!=expected:raise ValueError(f'nondeterministic serialization: {p}')
 return True
def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--validation',type=Path,default=FINAL_VALID);parser.add_argument('--output-root',type=Path,default=TARGET);args=parser.parse_args(argv)
 if ROOT.resolve()!=EXPECTED_ROOT.resolve():raise RuntimeError(f'active root must be {EXPECTED_ROOT}; got {ROOT}')
 if args.validation.resolve()!=FINAL_VALID.resolve() or args.output_root.resolve()!=TARGET.resolve():raise ValueError('Only approved validation and output paths are allowed')
 if TARGET.exists() or TMP.exists() or RESULT.exists():raise RuntimeError('materialization output/temp/result already exists; refuse overwrite')
 try:
  validation,sources=verify_authority();c0p=load(C0);fp=load(F);mal=load(MAL);bsp=load(BSHORT)
  if c0p['status']!='READY_FOR_HUMAN_MATERIALIZATION_APPROVAL' or fp['status']!='READY_FOR_HUMAN_MATERIALIZATION_APPROVAL':raise ValueError('proposal not approved')
  if len(c0p['records'])!=2320 or len(fp['records'])!=2320 or len(mal['records'])!=1160 or len(bsp['records'])!=120:raise ValueError('authority record counts changed')
  all_meta=c0p['records']+fp['records'];oa,or_rows,chen=load_source_rows(all_meta,bsp['records'],sources);c0rows,bshort1=build_rows(c0p,bsp['records'],oa,or_rows,chen,'H2m-C0');frows,bshort2=build_rows(fp,bsp['records'],oa,or_rows,chen,'H2m-F')
  if [canonical_line(r) for r in bshort1]!=[canonical_line(r) for r in bshort2]:raise ValueError('B-short resolution nondeterministic')
  amap={r['project_record_id']:r for r in all_meta+bsp['records']}
  os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';from transformers import AutoTokenizer
  tokenizer_path=ROOT/'data/raw/huggingface/microsoft_deberta-v3-base'/TOK_REV
  tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True,use_fast=False)
  pre_lengths=attach_and_verify_model_views(c0rows+frows+bshort1,amap,tokenizer)
  validate_schema(c0rows+frows+bshort1);validate_arm_contract(c0rows);validate_arm_contract(frows);verify_matched_arms(c0rows,frows)
  notinject_source=load(ROOT/'data/manifests/hard_benign/notinject/source.json');ni_path=ROOT/notinject_source['records_manifest'];verify_file_hash(ni_path,notinject_source['records_manifest_sha256']);notinject=read_jsonl(ni_path)
  TMP.mkdir(parents=True);file_entries=write_tree(TMP,c0rows,frows,bshort1);deterministic_files(TMP,file_entries)
  actual_c0,actual_f,actual_bs=read_tree(TMP);post=validate_materialized(actual_c0,actual_f,actual_bs,tokenizer,notinject)
  source_hashes={rel(C0):file_sha(C0),rel(F):file_sha(F),rel(MAL):file_sha(MAL),rel(FINAL_VALID):file_sha(FINAL_VALID),rel(BSHORT):file_sha(BSHORT),rel(sources['oasst1_full_audit.json']['path']):sources['oasst1_full_audit.json']['sha256'],rel(sources['or_bench_full_audit.json']['path']):sources['or_bench_full_audit.json']['sha256'],rel(sources['chen']['path']):sources['chen']['sha256'],rel(ni_path):notinject_source['records_manifest_sha256']}
  manifest={'status':'MATERIALIZATION_PASS','created_at':now(),'materializer':rel(Path(__file__)),'active_root':str(ROOT),'interpreter':sys.executable,'source_manifest_hashes':source_hashes,'tokenizer':'microsoft/deberta-v3-base','tokenizer_revision':TOK_REV,'max_length':MAX_LENGTH,'files':file_entries,'totals':{'H2m-C0':2320,'H2m-F':2320,'B-short':120,'physical_rows':4760},'raw_text_local_git_ignored':True,'training':False,'replay':False,'numerical_memory_budget':None}
  save_json(TMP/'materialization_manifest.json',manifest)
  # Manifest validates the already validated file entries and contains no raw prompts.
  for item in file_entries:
   p=TMP/Path(item['relative_file']).relative_to('data/processed/h2_mini');verify_file_hash(p,item['sha256']);assert len(read_jsonl(p))==item['row_count']
  os.replace(TMP,TARGET)
  for item in file_entries:verify_file_hash(ROOT/item['relative_file'],item['sha256'])
  checks={'source_revision_integrity':True,'source_file_hash_integrity':True,'raw_text_hash_resolution':True,'normalized_model_view_hash_resolution':True,'exact_counts':True,'label_integrity':True,'split_stage_isolation':True,'matched_malicious_halves':True,'shared_stage1_B0':True,'grouping_isolation':True,'contamination_checks':True,'token_length_integrity':post['token_lengths']['unique_selected']['truncation_required']==0,'B_short_independence':True,'deterministic_materialization':True,'checkpoint_independent_materialization':True}
  report={'status':'PASS_READY_FOR_HUMAN_NO_REPLAY_TRAINING_APPROVAL','created_at':now(),'checks':checks,'all_checks_pass':all(checks.values()),'source_hashes':source_hashes,'materialization_manifest':rel(TARGET/'materialization_manifest.json'),'materialization_manifest_sha256':file_sha(TARGET/'materialization_manifest.json'),'output_files':file_entries,'counts':{'H2m-C0':2320,'H2m-F':2320,'B-short':120,'per_arm_stage':580,'per_stage_split':{'train':{'total':400,'benign':200,'malicious':200},'val':{'total':60,'benign':30,'malicious':30},'test':{'total':120,'benign':60,'malicious':60}}},'matched_control':post['matched'],'token_length_audit':post['token_lengths'],'NotInject':{'records':339,'modified':False,'policy':'external evaluation only; no training/validation/tuning/method ranking'},'B_short':{'records':120,'English':42,'multilingual':78,'role':'unmatched descriptive external short-form ordinary-benign control','included_in_training_or_validation':False},'atomic_publish':{'temporary_path':rel(TMP),'final_path':rel(TARGET),'published':True},'training':False,'checkpoints_created':False,'optimizer_created':False,'replay':False,'selective_adaptation':False,'numerical_memory_budget':None,'replay_readiness':{'R0':'active','R1':'not entered','R2':'not entered'}}
  save_json(RESULT,report);print(json.dumps({'status':report['status'],'files':len(file_entries),'counts':report['counts'],'token_lengths':post['token_lengths']['unique_selected'],'manifest':report['materialization_manifest'],'validation':rel(RESULT)},indent=2));return 0
 except Exception as exc:
  quarantine=None
  if TMP.exists():
   quarantine=ROOT/'data/processed'/('_quarantine_h2_mini_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));os.replace(TMP,quarantine)
  save_json(RESULT,{'status':'FAIL_QUARANTINED_NOT_APPROVED','created_at':now(),'error':str(exc),'quarantine':rel(quarantine) if quarantine else None,'published':False,'training':False,'replay':False})
  raise
if __name__=='__main__':raise SystemExit(main())