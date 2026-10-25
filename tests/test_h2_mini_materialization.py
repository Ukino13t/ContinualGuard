import hashlib
from pathlib import Path
import pytest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import materialize_h2_mini_from_final_assignments as m

def row(stage,split,label,i):
 eid=f'e-{stage}-{split}-{label}-{i}'
 return {'example_id':eid,'task_id':stage,'split':split,'label':label,'base_document_id':'b-'+eid,'text_sha256':'t-'+eid,'normalized_text_sha256':'n-'+eid,'model_view_sha256':'m-'+eid,'duplicate_group_id':'d-'+eid,'source':{'dataset':'fixture','revision':'r','upstream_split':split,'record_id':eid,'construction':'fixture'},'source_native_id':'s-'+eid,'source_parent_id':'p-'+eid,'exact_group':'x-'+eid,'normalized_group':'xn-'+eid,'model_view_group':'xm-'+eid,'project_semantic_parent_group':'sp-'+eid,'project_template_group':'tp-'+eid,'attack_family':'none' if label==0 else {'AF1':'naive','AF2':'ignore','AF3':'escape_separation','AF4':'fake_completion'}[stage],'injection_position':'none' if label==0 else 'head','injection_id':None if label==0 else 'inj-'+eid,'text':'word'}
def rows(quotas=None):
 q=quotas or {'train':2,'val':1,'test':1}
 return [row(s,sp,y,i) for s in ['AF1','AF2','AF3','AF4'] for sp,n in q.items() for y in [0,1] for i in range(n)]
def test_exact_row_counts():
 r=rows();assert len(m.validate_arm_contract(r,{'train':2,'val':1,'test':1}))==24
def test_source_hash_mismatch_rejected(tmp_path):
 p=tmp_path/'x';p.write_text('x');
 with pytest.raises(ValueError,match='source hash mismatch'):m.verify_file_hash(p,'0'*64)
def test_duplicate_id_rejected():
 r=rows();r[-1]['example_id']=r[0]['example_id']
 with pytest.raises(ValueError,match='duplicate example_id'):m.validate_arm_contract(r,{'train':2,'val':1,'test':1})
def test_invalid_label_rejected():
 r=rows();r[0]['label']=2
 with pytest.raises(ValueError,match='invalid labels'):m.validate_arm_contract(r,{'train':2,'val':1,'test':1})
def test_cross_split_leakage_rejected():
 r=rows();a=next(x for x in r if x['split']=='train');b=next(x for x in r if x['split']=='val');b['base_document_id']=a['base_document_id']
 with pytest.raises(ValueError,match='cross-split/stage leakage'):m.validate_arm_contract(r,{'train':2,'val':1,'test':1})
def test_matched_arm_malicious_mismatch_rejected():
 a=rows();b=[dict(x) for x in a];next(x for x in b if x['label']==1)['injection_position']='tail'
 with pytest.raises(ValueError,match='malicious mismatch'):m.verify_matched_arms(a,b)
def test_stage1_b0_mismatch_rejected():
 a=rows();b=[dict(x) for x in a];x=next(x for x in b if x['label']==0 and x['task_id']=='AF1');x['example_id']='changed'
 with pytest.raises(ValueError,match='Stage-1 B0 mismatch'):m.verify_matched_arms(a,b)
def test_max_length_integrity():
 class Tok:
  def encode(self,text,add_special_tokens=True,truncation=False):return list(range(len(text.split())+2))
 with pytest.raises(ValueError,match='max_length'):m.audit_token_lengths([{'text':'x '*1300}],Tok())
def test_deterministic_materialization(tmp_path):
 r=[{'example_id':'b','v':2},{'example_id':'a','v':1}];p=tmp_path/'x.jsonl';m.write_jsonl(p,r);one=p.read_bytes();m.write_jsonl(p,list(reversed(r)));assert p.read_bytes()==one
def test_checkpoint_independent_materialization(tmp_path,monkeypatch):
 r=[{'example_id':'a','v':1}];p=tmp_path/'x.jsonl';monkeypatch.setenv('CHECKPOINT_PATH','one');m.write_jsonl(p,r);a=hashlib.sha256(p.read_bytes()).hexdigest();monkeypatch.setenv('CHECKPOINT_PATH','two');m.write_jsonl(p,r);assert hashlib.sha256(p.read_bytes()).hexdigest()==a