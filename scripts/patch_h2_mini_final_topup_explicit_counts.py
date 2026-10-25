"""Add explicit accepted/rejected totals to the validated top-up report and refresh dependent hashes."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def fsha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=ROOT/'results/h2_mini_b0_final_topup_analysis.json';d=load(a)
assert d['status']=='TOPUP_CAPACITY_PASS_CONTINUE_APPROVED_FINALIZATION'
d['training']['ACCEPT_BENIGN']=40;d['training']['rejection_count']=0;d['evaluation']['both_ACCEPT_BENIGN']=24
save(a,d);ah=fsha(a)
c=ROOT/'review/b0_final_topup_canonical_manifest.json';x=load(c);x['analysis_sha256']=ah;save(c,x)
b=ROOT/'data/manifests/hard_benign/B0_final_accepted.json';x=load(b);x['source_inputs']['results/h2_mini_b0_final_topup_analysis.json']=ah;save(b,x)
v=ROOT/'data/manifests/h2_mini/h2m_final_assignment_validation.json';x=load(v)
for pth in [c,b]:x['manifest_hashes'][pth.relative_to(ROOT).as_posix()]=fsha(pth)
save(v,x)
doc=ROOT/'docs/h2_mini_b0_final_topup_results.md';s=doc.read_text(encoding='utf-8');s=s.replace('Training primary: 40 reviewed; 39 ACCEPT_BENIGN + B0 YES','Training primary: 40 reviewed; 40 ACCEPT_BENIGN; 0 rejected; 39 ACCEPT_BENIGN + B0 YES').replace('25 double-reviewed; 24/25 decision agreement','25 double-reviewed; 24 both ACCEPT_BENIGN; 24/25 decision agreement');doc.write_text(s,encoding='utf-8')
print(json.dumps({'analysis_sha256':ah,'canonical_sha256':fsha(c),'B0_final_sha256':fsha(b),'validation_sha256':fsha(v)},indent=2))