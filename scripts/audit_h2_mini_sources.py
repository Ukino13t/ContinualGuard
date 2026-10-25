"""Reproducible H2-mini source audit; metadata only, no training or detector inference."""
from __future__ import annotations
import csv,gzip,hashlib,json,math,re,unicodedata
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as sknorm
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]; MAXLEN=1280
ORR="e36d8b80e81837c8a8f264bbb2a49f1b32c7e272"; OAR="fdf72ae0827c1cda404aff25b6603abec9e3399b"; NIR="847ae76cf8fea5ed325429e569ae8cfef022d2e0"; TOKR="8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
ORP=ROOT/"data/raw/hard_benign/or_bench"/ORR; OAP=ROOT/"data/raw/hard_benign/oasst1"/OAR
AUD=ROOT/"data/audits/hard_benign"; MAN=ROOT/"data/manifests/hard_benign"
CUES=[
("C01","override/negative constraint",r"\b(?:ignore|disregard|forget|override|bypass|circumvent|avoid|never|without|do not|don.t|must not|should not|refrain)\b"),
("C02","instruction hierarchy/policy",r"\b(?:system prompt|system message|developer message|previous instructions?|prior instructions?|above instructions?|instructions?|guidelines?|polic(?:y|ies)|rules?|protocols?)\b"),
("C03","role/simulation",r"\b(?:act as|pretend|role[- ]?play|persona|simulate|simulation|hypothetical|fictional|scenario|imagine)\b"),
("C04","output/imperative control",r"\b(?:answer|respond|reply|output|return|write|draft|compose|create|generate|list|outline|describe|explain|discuss|provide|summarize|translate|tell me|show me|help me)\b"),
("C05","secret/protected information",r"\b(?:reveal|expose|leak|disclose|secret|password|credential|access token|api key|private|confidential)\b"),
("C06","tool/code/execution",r"\b(?:execute|run|call|invoke|command|script|shell|terminal|tool|api|code|program|algorithm)\b"),
("C07","encoding/boundary/transformation",r"\b(?:base64|rot13|hexadecimal|decode|encoded|obfuscat\w*|delimiter|separator|code block|xml tag|html tag|quotation|quoted?|translate|translation)\b"),
("C08","context/memory/message",r"\b(?:memory|context|conversation history|chat history|previous|prior|above|below|document|content|prompt|message)\b"),
("C09","authority/privilege",r"\b(?:admin|administrator|developer|system|root|superuser|authority|official)\b"),
("C10","security/prompt meta-language",r"\b(?:security|cybersecurity|safety|safe|malicious|attack|injection|phishing|moderation|jailbreak|prompt engineering|configuration|configure|system design)\b")]
CP=[(x,re.compile(p,re.I)) for x,n,p in CUES]
FRAME=re.compile(r"\b(?:education\w*|training|research|analy[sz]\w*|explain|discuss|describe|translat\w*|quot\w*|fictional|hypothetical|scenario|role[- ]?play|awareness|policy|configuration|security|design|documentation|example)\b",re.I)
OPI=re.compile(r"\b(?:ignore|disregard|forget|override|bypass)\b.{0,50}\b(?:previous|prior|above|system|developer|instructions?|rules?)\b|\b(?:reveal|print|show|leak|expose)\b.{0,50}\b(?:system prompt|secret|credential|password|token)\b|\byou are now\b",re.I|re.S)
SAFE=re.compile(r"\b(?:avoid|prevent|protect|awareness|education\w*|fictional|hypothetical|research|training|analysis|safety|security|not actually|without|do not|don.t)\b",re.I)
HARM=re.compile(r"\b(?:how (?:do|can|could|would) i|give me|provide|write|create|develop|generate|list|explain)\b.{0,90}\b(?:kill|murder|steal|hack|exploit|bomb|suicide|self-harm|rape|child sexual|phishing|malware)\b",re.I|re.S)
PII=re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b|\b(?:\+?\d[\d .()\-]{7,}\d)\b|\b(?:\d[ -]*?){13,19}\b",re.I)
CTX=re.compile(r"^(?:what about|and what|same as|continue|yes[, ]|no[, ]|that one|this one)\b",re.I)
WRE=re.compile(r"\w+",re.UNICODE)

def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def fsha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def norm(s):return " ".join(unicodedata.normalize("NFKC",s).casefold().split())
def words(s):return WRE.findall(norm(s))
def tset(s):return set(words(s))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def pct(v,q):return sorted(v)[max(0,math.ceil(q*len(v))-1)] if v else None
def lens(v):return {"n":len(v),"min":min(v) if v else None,"median":pct(v,.5),"p90":pct(v,.9),"p95":pct(v,.95),"p99":pct(v,.99),"max":max(v) if v else None,"over_1280":sum(x>MAXLEN for x in v)}
def classify(s):
 fam=[];sp=0
 for cid,rx in CP:
  z={(m.start(),m.end()) for m in rx.finditer(s)}
  if z:fam.append(cid);sp+=len(z)
 framed=bool(FRAME.search(s));op=bool(OPI.search(s))
 if op and not framed:lab="ambiguous"
 elif framed and fam and (set(fam)&{"C02","C03","C05","C06","C07","C08","C09","C10"} or len(fam)>=2):lab="B3_candidate"
 elif len(fam)>=2 or sp>=2:lab="B2_candidate"
 elif len(fam)==1:lab="B1_candidate"
 else:lab="unassigned"
 return lab,fam,sp
def corrupt(s):return not isinstance(s,str) or not s.strip() or any(ord(c)<32 and c not in "\r\n\t" for c in s)
def unsafe(s,d=None):
 if HARM.search(s) and not SAFE.search(s):return True
 v=[x for x in (d or {}).values() if isinstance(x,(float,int))]
 return bool(v and max(v)>=.9)
def context(s):return len(words(s))<4 or bool(CTX.search(s.strip()))
def template(s,c=()):
 w=words(s)
 if not w:return ""
 return sha("|".join(w[:4])+"::"+"|".join(w[-4:])+"::"+",".join(c)+"::"+str(len(w)//5)+"::"+"".join(x for x in norm(s) if x in "?!:;()[]{}")[:12])
def mviews(texts,tok):
 hh=[];ll=[]
 for i in range(0,len(texts),1000):
  z=tok(texts[i:i+1000],add_special_tokens=True,truncation=False,padding=False)["input_ids"]
  hh += [sha(",".join(map(str,x))) for x in z];ll += [len(x) for x in z]
  print("tokenized",min(i+1000,len(texts)),len(texts),flush=True)
 return hh,ll
class UF:
 def __init__(self,n):self.p=list(range(n))
 def find(self,x):
  while self.p[x]!=x:self.p[x]=self.p[self.p[x]];x=self.p[x]
  return x
 def union(self,a,b):
  a=self.find(a);b=self.find(b)
  if a!=b:self.p[b]=a
 def groups(self):
  d=defaultdict(list)
  for i in range(len(self.p)):d[self.find(i)].append(i)
  return list(d.values())
def ids_from_groups(gs,n,prefix):
 out=[""]*n
 for g in gs:
  x=prefix+sha("|".join(map(str,g)))[:20]
  for i in g:out[i]=x
 return out
def keygroups(keys,prefix):
 d=defaultdict(list)
 for i,k in enumerate(keys):d[k].append(i)
 gs=list(d.values())
 return ids_from_groups(gs,len(keys),prefix),{"groups":len(gs),"duplicate_groups":sum(len(x)>1 for x in gs),"duplicate_rows_beyond_first":sum(len(x)-1 for x in gs),"max_group_size":max(map(len,gs)) if gs else 0}
def fit_embed(texts):
 hv=HashingVectorizer(n_features=8192,alternate_sign=False,norm="l2",ngram_range=(1,2),token_pattern=r"(?u)\b\w+\b")
 x=hv.transform(texts);sv=TruncatedSVD(48,n_iter=5,random_state=42);e=sknorm(sv.fit_transform(x)).astype(np.float32)
 rng=np.random.RandomState(42);planes=[rng.normal(size=(48,16)).astype(np.float32) for _ in range(8)]
 return hv,sv,e,planes
def xembed(texts,hv,sv):return sknorm(sv.transform(hv.transform(texts))).astype(np.float32)
def lsh(e,planes):
 powers=1<<np.arange(16,dtype=np.uint64);out=[]
 for p in planes:
  sig=(((e@p)>0).astype(np.uint64)*powers).sum(1);d=defaultdict(list)
  for i,x in enumerate(sig.tolist()):d[int(x)].append(i)
  out.append(d)
 return out
def semantic_groups(texts,e,planes):
 ss=[tset(x) for x in texts];u=UF(len(texts));comp=se=le=skip=0
 for d in lsh(e,planes):
  for g in d.values():
   if len(g)>300:skip+=1;continue
   for n,a in enumerate(g):
    for b in g[n+1:]:
     if not ss[a] or not ss[b] or min(len(ss[a]),len(ss[b]))/max(len(ss[a]),len(ss[b]))<.45:continue
     comp+=1;j=len(ss[a]&ss[b])/len(ss[a]|ss[b]);co=float(e[a]@e[b])
     if j>=.82:u.union(a,b);le+=1
     elif co>=.97 and j>=.35:u.union(a,b);se+=1
 rare=defaultdict(list)
 for i,s in enumerate(texts):
  w=words(s)
  for n in range(max(0,len(w)-5)):rare[sha(" ".join(w[n:n+6]))[:20]].append(i)
 re_=0
 for g in rare.values():
  g=sorted(set(g))
  if 2<=len(g)<=8:
   for b in g[1:]:
    a=g[0];j=len(ss[a]&ss[b])/max(1,len(ss[a]|ss[b]))
    if j>=.4:u.union(a,b);re_+=1
 gs=u.groups()
 return ids_from_groups(gs,len(texts),"psp-"),{"method":"8x16-bit LSH over deterministic 48D hashed word/bigram TF-IDF-SVD; cosine>=.97/Jaccard>=.35 or Jaccard>=.82; rare six-word overlap/Jaccard>=.40","groups":len(gs),"multi_record_groups":sum(len(x)>1 for x in gs),"max_group_size":max(map(len,gs)),"candidate_comparisons":comp,"semantic_edges":se,"lexical_edges":le,"rare_phrase_edges":re_,"oversized_lsh_buckets_skipped":skip,"caveat":"Project approximation, not official lineage or a pretrained embedding model."}
def template_groups(texts,cues):
 sig=[template(x,c) for x,c in zip(texts,cues)];d=defaultdict(list)
 for i,x in enumerate(sig):d[x].append(i)
 u=UF(len(texts));ss=[tset(x) for x in texts];edges=0
 for g in d.values():
  for n,a in enumerate(g):
   for b in g[n+1:]:
    if len(ss[a]&ss[b])/max(1,len(ss[a]|ss[b]))>=.55:u.union(a,b);edges+=1
 gs=u.groups()
 return ids_from_groups(gs,len(texts),"ptg-"),sig,{"method":"structural signature plus Jaccard>=.55","groups":len(gs),"multi_record_groups":sum(len(x)>1 for x in gs),"max_group_size":max(map(len,gs)),"edges":edges,"caveat":"Project approximation; official generator templates unavailable."}
def collapse(active,gid,labels):
 d=defaultdict(list)
 for i in active:d[gid[i]].append(i)
 keep=[];removed=conf=0
 for g in d.values():
  if len({labels[i] for i in g})>1:conf+=1;removed+=len(g)
  else:keep.append(min(g));removed+=len(g)-1
 return set(keep),{"groups_seen":len(d),"removed":removed,"cross_stratum_conflict_groups":conf}
def counts(active,labels):
 c=Counter(labels[i] for i in active)
 return {x:c.get(x,0) for x in ["B1_candidate","B2_candidate","B3_candidate"]}
def overlap(qt,qe,qs,tt,te,ts,planes,qid,tid):
 tn=[norm(x) for x in tt];tex=defaultdict(list);tm=defaultdict(list);tsets=[tset(x) for x in tt]
 for j,x in enumerate(tn):tex[x].append(j)
 for j,x in enumerate(ts):tm[x].append(j)
 tb=lsh(te,planes);powers=1<<np.arange(16,dtype=np.uint64);qsg=[((((qe@p)>0).astype(np.uint64))*powers).sum(1) for p in planes]
 shingles=defaultdict(list)
 for j,s in enumerate(tt):
  w=words(s)
  for n in range(max(0,len(w)-4)):shingles[sha(" ".join(w[n:n+5]))[:20]].append(j)
 out=set();mc=Counter();examples=[];qsets=[tset(x) for x in qt]
 for i,s in enumerate(qt):
  n=norm(s);cand=set(tex.get(n,[]));cand.update(tm.get(qs[i],[]))
  for k,d in enumerate(tb):cand.update(d.get(int(qsg[k][i]),[]))
  w=words(s)
  for a in range(max(0,len(w)-4)):cand.update(shingles.get(sha(" ".join(w[a:a+5]))[:20],[]))
  methods=set();best=None
  for j in cand:
   jac=len(qsets[i]&tsets[j])/max(1,len(qsets[i]|tsets[j]));co=float(qe[i]@te[j]);mm=[]
   if n==tn[j]:mm.append("normalized_exact")
   if (len(n)>=25 and n in tn[j]) or (len(tn[j])>=25 and tn[j] in n):mm.append("substring")
   if jac>=.85:mm.append("high_lexical")
   if co>=.97 and jac>=.30:mm.append("semantic_neighbor")
   if qs[i] and qs[i]==ts[j]:mm.append("template")
   if mm:
    methods.update(mm)
    if best is None or (co,jac)>(best[0],best[1]):best=(co,jac,j,mm)
  if methods:
   out.add(i);mc.update(methods)
   if len(examples)<100 and best:examples.append({"query_id":qid[i],"target_id":tid[best[2]],"methods":best[3],"latent_cosine":round(best[0],6),"token_set_jaccard":round(best[1],6)})
 return out,{"affected_queries":len(out),"method_counts":dict(mc),"examples_metadata_only":examples}
def load_ni():
 d={}
 for p in (ROOT/"data/raw/notinject"/NIR/"dataset_server").glob("*.json"):
  split=p.stem.rsplit("_",1)[0]
  for x in json.loads(p.read_text(encoding="utf-8"))["rows"]:
   r=x["row"];d[(split,x["row_idx"])]={"id":split+":"+str(x["row_idx"]),"text":r["prompt"],"category":r["category"]}
 return list(d.values())
def load_af():
 out=[]
 for line in (ROOT/"data/processed/chen_attack_family_shift.jsonl").open(encoding="utf-8"):
  r=json.loads(line)
  if r.get("task_id") in {"AF1","AF2","AF3","AF4"} and r.get("label")==1:out.append({"id":r["example_id"],"text":r["text"]})
 return out
def inventory(base,rel,url,lic,n,schema):
 p=base/rel
 return {"file_name":rel,"local_path":str(p.relative_to(ROOT)).replace("\\","/"),"official_url":url,"bytes":p.stat().st_size,"sha256":fsha(p),"license":lic,"schema":schema,"total_row_count":n,"acquisition_timestamp":datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()}

def main():
 stamp=datetime.now(timezone.utc).isoformat()
 with (ORP/"or-bench-80k.csv").open(encoding="utf-8-sig",newline="") as f:orr=list(csv.DictReader(f))
 if len(orr)!=80359 or list(orr[0])!=["prompt","category"]:raise RuntimeError("OR schema/count mismatch")
 ot=[r["prompt"] for r in orr];classified=[classify(x) for x in ot];labels=[x[0] for x in classified];cues=[x[1] for x in classified];spans=[x[2] for x in classified]
 oa=[]
 with gzip.open(OAP/"2023-04-12_oasst_ready.messages.jsonl.gz","rt",encoding="utf-8") as f:
  for i,line in enumerate(f):
   if line.strip():r=json.loads(line);r["_source_idx"]=i;oa.append(r)
 roots=[r for r in oa if r.get("role")=="prompter" and not r.get("parent_id") and r.get("message_tree_id")==r.get("message_id")]
 at=[r.get("text") or "" for r in roots];ni=load_ni();af=load_af();nt=[x["text"] for x in ni];ft=[x["text"] for x in af]
 print("source rows",len(orr),len(oa),len(roots),len(ni),len(af),flush=True)
 tok=AutoTokenizer.from_pretrained(ROOT/"data/raw/huggingface/microsoft_deberta-v3-base"/TOKR,local_files_only=True,use_fast=False)
 oh,ol=mviews(ot,tok);ah,al=mviews(at,tok);nh,nl=mviews(nt,tok);fh,fl=mviews(ft,tok)
 on=[norm(x) for x in ot];an=[norm(x) for x in at]
 oex,oexs=keygroups(ot,"oex-");onm,onms=keygroups(on,"onm-");omv,omvs=keygroups(oh,"omv-")
 aex,aexs=keygroups(at,"aex-");anm,anms=keygroups(an,"anm-");amv,amvs=keygroups(ah,"amv-")
 print("fit latent semantic approximation",flush=True);hv,sv,oe,planes=fit_embed(on);ae=xembed(an,hv,sv);ne=xembed([norm(x) for x in nt],hv,sv);fe=xembed([norm(x) for x in ft],hv,sv)
 print("group OR",flush=True);osg,osgs=semantic_groups(on,oe,planes)
 print("group OASST",flush=True);asg,asgs=semantic_groups(an,ae,planes)
 otg,ots,otgs=template_groups(on,cues);acues=[classify(x)[1] for x in at];atg,ats,atgs=template_groups(an,acues);nts=[template(x,classify(x)[1]) for x in nt];fts=[template(x,classify(x)[1]) for x in ft]
 oi=["orb-"+sha(ORR+"|"+str(i)+"|"+sha(x))[:24] for i,x in enumerate(ot)];ai=[r["message_id"] for r in roots];nii=[x["id"] for x in ni];fi=[x["id"] for x in af]
 print("cross-corpus audits",flush=True)
 ovn,ron=overlap(on,oe,ots,nt,ne,nts,planes,oi,nii);ovo,roo=overlap(on,oe,ots,at,ae,ats,planes,oi,ai);ovf,rof=overlap(on,oe,ots,ft,fe,fts,planes,oi,fi)
 avn,ran=overlap(an,ae,ats,nt,ne,nts,planes,ai,nii);avo,rao=overlap(an,ae,ats,ot,oe,ots,planes,ai,oi);avf,raf=overlap(an,ae,ats,ft,fe,fts,planes,ai,fi)
 active={i for i,x in enumerate(labels) if x.startswith("B")};seq=[{"step":0,"name":"raw_ontology_labels","counts":counts(active,labels),"removed":0}]
 bad={i for i,x in enumerate(ot) if corrupt(x)};before=len(active);active-=bad;seq.append({"step":1,"name":"empty_corrupt","counts":counts(active,labels),"removed":before-len(active)})
 for s,n,g in [(2,"exact_duplicate",oex),(3,"normalized_duplicate",onm),(4,"model_view_duplicate",omv)]:
  active,z=collapse(active,g,labels);seq.append({"step":s,"name":n,"counts":counts(active,labels),**z})
 seq.append({"step":5,"name":"official_source_parent","counts":counts(active,labels),"removed":0,"note":"Unavailable; row order was not used."})
 active,z=collapse(active,osg,labels);seq.append({"step":6,"name":"project_semantic_parent","counts":counts(active,labels),**z})
 active,z=collapse(active,otg,labels);seq.append({"step":7,"name":"project_template","counts":counts(active,labels),**z})
 uns={i for i,x in enumerate(ot) if unsafe(x)};before=len(active);active-=uns;seq.append({"step":8,"name":"obvious_unsafe_nonbenign","counts":counts(active,labels),"removed":before-len(active),"automated_only":True})
 ctx={i for i,x in enumerate(ot) if context(x)};before=len(active);active-=ctx;seq.append({"step":9,"name":"context_dependent","counts":counts(active,labels),"removed":before-len(active),"automated_only":True})
 for s,n,v,report,note in [(10,"NotInject_contamination",ovn,ron,None),(11,"B_short_source_contamination",ovo,roo,"Whole OASST root pool used because final B-short IDs do not exist."),(12,"AF1_AF4_overlap",ovf,rof,None)]:
  before=len(active);active-=v;row={"step":s,"name":n,"counts":counts(active,labels),"removed":before-len(active),"overlap":report}
  if note:row["note"]=note
  seq.append(row)
 finalo=set(active);fc=counts(finalo,labels)
 flags=[]
 for i,(r,x,l) in enumerate(zip(roots,at,al)):
  q=set()
  if corrupt(x):q.add("empty_corrupt")
  if r.get("deleted") or r.get("review_result") is not True or r.get("tree_state")!="ready_for_export":q.add("source_status")
  if l>MAXLEN:q.add("over_1280")
  if unsafe(x,r.get("detoxify")):q.add("obvious_unsafe")
  if PII.search(x):q.add("possible_PII")
  if context(x):q.add("context_dependent")
  if OPI.search(x) and not FRAME.search(x):q.add("operational_IPI_or_ambiguous")
  if i in avn:q.add("NotInject_overlap")
  if i in avo:q.add("OR_Bench_overlap")
  if i in avf:q.add("AF1_AF4_overlap")
  flags.append(q)
 aa=set(range(len(roots)));aseq=[{"step":0,"name":"root_prompter","count":len(roots),"removed":0}]
 for s,n,ff in [(1,"source_status",{"source_status"}),(2,"length_quality",{"empty_corrupt","over_1280"})]:
  before=len(aa);aa={i for i in aa if not(flags[i]&ff)};aseq.append({"step":s,"name":n,"count":len(aa),"removed":before-len(aa)})
 for s,n,g in [(3,"exact",aex),(4,"normalized",anm),(5,"model_view",amv),(6,"conversation_tree",[r["message_tree_id"] for r in roots]),(7,"project_semantic_parent",asg),(8,"project_template",atg)]:
  aa,z=collapse(aa,g,["ordinary"]*len(roots));aseq.append({"step":s,"name":n,"count":len(aa),**z})
 for s,n,ff in [(9,"unsafe_IPI",{"obvious_unsafe","operational_IPI_or_ambiguous"}),(10,"PII",{"possible_PII"}),(11,"context",{"context_dependent"}),(12,"NotInject",{"NotInject_overlap"}),(13,"OR_Bench",{"OR_Bench_overlap"}),(14,"AF1_AF4",{"AF1_AF4_overlap"})]:
  before=len(aa);aa={i for i in aa if not(flags[i]&ff)};aseq.append({"step":s,"name":n,"count":len(aa),"removed":before-len(aa)})
 finala=set(aa);mult=sum(r["category"]=="Multilingual" for r in ni);short={i for i in finala if min(nl)<=al[i]<=max(nl)};sl=Counter("en" if roots[i].get("lang")=="en" else "multilingual" for i in short);ten=round(120*(len(ni)-mult)/len(ni));target={"english":ten,"multilingual":120-ten};bf=sl["en"]>=target["english"] and sl["multilingual"]>=target["multilingual"];ordinary=len(finala)-(120 if bf else 0)
 ormanifest=[]
 for i,r in enumerate(orr):
  if i not in finalo:continue
  ormanifest.append({"candidate_id":oi[i],"dataset_id":"bench-llm/or-bench","revision":ORR,"source_row_index":i,"native_record_id":None,"category":r["category"],"ontology":"h2m-cue-ontology-v1","candidate_label":labels[i],"cue_family_ids":cues[i],"cue_span_count":spans[i],"raw_text_sha256":sha(ot[i]),"normalized_text_sha256":sha(on[i]),"model_view_sha256":oh[i],"deberta_tokens":ol[i],"exact_group":oex[i],"normalized_group":onm[i],"model_view_group":omv[i],"official_parent":None,"project_semantic_parent_group":osg[i],"project_template_group":otg[i],"unsafe_flag":i in uns,"context_flag":i in ctx,"NotInject_overlap":i in ovn,"OASST_overlap":i in ovo,"AF_overlap":i in ovf,"group_safe_after_automated_audit":i in finalo,"human_review_status":"not_started","final_assignment":None})
 oamanifest=[]
 for i,r in enumerate(roots):
  if i not in finala:continue
  oamanifest.append({"candidate_id":ai[i],"dataset_id":"OpenAssistant/oasst1","revision":OAR,"source_row_index":r["_source_idx"],"message_id":r["message_id"],"message_tree_id":r["message_tree_id"],"parent_id":r.get("parent_id"),"role":r.get("role"),"lang":r.get("lang"),"review_count":r.get("review_count"),"review_result":r.get("review_result"),"deleted":r.get("deleted"),"raw_text_sha256":sha(at[i]),"normalized_text_sha256":sha(an[i]),"model_view_sha256":ah[i],"deberta_tokens":al[i],"exact_group":aex[i],"normalized_group":anm[i],"model_view_group":amv[i],"project_semantic_parent_group":asg[i],"project_template_group":atg[i],"group_safe_ordinary_candidate":True,"b_short_length_language_candidate":i in short,"human_review_status":"not_started","final_assignment":None})
 oschema=[{"name":"prompt","dtype":"string"},{"name":"category","dtype":"string"}];oaschema=[{"name":x,"dtype":"see official dataset card"} for x in ["message_id","parent_id","user_id","created_date","text","role","lang","review_count","review_result","deleted","rank","synthetic","model_name","detoxify","message_tree_id","tree_state","emojis","labels"]]
 ofiles=[inventory(ORP,r,"https://huggingface.co/datasets/bench-llm/or-bench/resolve/"+ORR+"/"+r+"?download=true","CC BY 4.0",len(orr) if r.endswith(".csv") else None,oschema if r.endswith(".csv") else None) for r in ["README.md","or-bench-80k.csv"]]
 oac={"2023-04-12_oasst_ready.messages.jsonl.gz":len(oa),"data/train-00000-of-00001-b42a775f407cee45.parquet":84437,"data/validation-00000-of-00001-134b8fd0c89408b6.parquet":4401}
 oarels=["README.md","LICENSE","2023-04-12_oasst_ready.messages.jsonl.gz","data/train-00000-of-00001-b42a775f407cee45.parquet","data/validation-00000-of-00001-134b8fd0c89408b6.parquet"]
 oafiles=[inventory(OAP,r,"https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/"+OAR+"/"+r+"?download=true","Apache-2.0",oac.get(r),oaschema if r.endswith((".gz",".parquet")) else None) for r in oarels]
 for x in ofiles:x["revision"]=ORR
 for x in oafiles:x["revision"]=OAR
 raw=Counter(labels)
 ora={"status":"audit_complete_no_selection_no_training","audit_timestamp":stamp,"source":{"dataset_id":"bench-llm/or-bench","config":"or-bench-80k","revision":ORR,"license":"CC BY 4.0","files":ofiles},"official_lineage":{"native_record_ids":False,"toxic_seed_ids":False,"rewrite_parent_ids":False,"generator_seed_ids":False,"template_ids":False,"generation_batch_ids":False,"source_parent_relationships":False,"evidence":["Pinned schema has only prompt and category.","Paper states 2,000 toxic seeds/category, deduplicated, five rewrites/seed.","Pinned repository has no lineage sidecar or generation code.","Official Space has evaluation data/code but no parent mapping."],"row_order_not_used":True},"raw_label_counts":dict(raw),"filter_sequence":seq,"grouping":{"exact":oexs,"normalized":onms,"model_view":omvs,"project_semantic":osgs,"project_template":otgs},"overlap":{"NotInject":ron,"OASST_root_pool":roo,"AF1_AF4":rof},"group_safe_counts_pending_human_review":fc,"capacity":{"H2mini_290_each":all(v>=290 for v in fc.values()),"future_650_each":all(v>=650 for v in fc.values())},"length_by_stratum":{x.replace("_candidate",""):lens([ol[i] for i,y in enumerate(labels) if y==x]) for x in ["B1_candidate","B2_candidate","B3_candidate"]},"max_length":MAXLEN,"detector_inference":False,"NotInject_used_for_ontology_or_ranking":False,"final_JSONL":False,"training":False,"risks":["No official parent mapping; residual sibling leakage.","Project semantic/template groups are approximate.","OR labels are moderator-derived and require human review."]}
 oaa={"status":"audit_complete_no_selection_no_training","audit_timestamp":stamp,"source":{"dataset_id":"OpenAssistant/oasst1","revision":OAR,"license":"Apache-2.0","files":oafiles},"all_ready_rows":len(oa),"role_counts":dict(Counter(r.get("role") for r in oa)),"root_prompter_rows":len(roots),"root_parent_null":sum(not r.get("parent_id") for r in roots),"root_message_equals_tree":sum(r.get("message_id")==r.get("message_tree_id") for r in roots),"root_languages":dict(Counter(r.get("lang") for r in roots)),"duplicates":{"exact":aexs,"normalized":anms,"model_view":amvs},"grouping":{"project_semantic":asgs,"project_template":atgs},"flag_counts_nonexclusive":dict(Counter(x for q in flags for x in q)),"filter_sequence":aseq,"overlap":{"NotInject":ran,"OR_Bench":rao,"AF1_AF4":raf},"group_safe_ordinary_capacity_pending_human_review":len(finala),"ordinary_capacity_after_Bshort_reserve":ordinary,"H2m_C0_1160_feasible":ordinary>=1160,"B_short":{"candidate_pool":len(short),"languages":dict(sl),"target_approx":target,"token_range":[min(nl),max(nl)],"feasible_120":bf,"final_selection":False},"length":{"B0":lens([al[i] for i in finala]),"B_short_candidates":lens([al[i] for i in short]),"NotInject":lens(nl)},"fallback_Dolly_used":False,"detector_inference":False,"final_JSONL":False,"training":False}
 prod={"B1_B3_train_single":600,"B1_B3_train_second_10pct":60,"B1_B3_val_double":180,"B1_B3_test_double":360,"Bshort_double":240,"C0_train_single":800,"C0_train_second_10pct":80,"C0_val_double":240,"C0_test_double":480,"total":3040}
 cap={"status":"capacity_estimated_human_review_not_started","benchmark":"H2-mini feasibility only; not full H2-F","counts":{"per_stage":{"train":[200,200],"val":[30,30],"test":[60,60],"class_total":290},"four_stages":{"malicious":1160,"benign":1160,"total":2320},"B1_B2_B3_each":290},"OR_raw":{x.replace("_candidate",""):raw[x] for x in ["B1_candidate","B2_candidate","B3_candidate"]},"OR_group_safe_pending_review":{x.replace("_candidate",""):fc[x] for x in fc},"OASST_group_safe":len(finala),"OASST_after_Bshort":ordinary,"Bshort_pool":len(short),"feasible":{"H2mini_290_each":all(v>=290 for v in fc.values()),"future_650_each":all(v>=650 for v in fc.values()),"C0_1160_plus_Bshort_120":ordinary>=1160 and bf,"scientific_legal":"Conditional on attribution/license compliance and successful human review."},"review":{"production_judgments":prod,"timing_calibration":{"records":50,"two_reviewer_judgments":100,"status":"not_started"},"measured_seconds_per_judgment":None,"projected_hours":None,"formula":"(3040 + calibration + replacements/adjudication) * observed median seconds / 3600","10h_seconds_per_judgment_at_3140":round(36000/3140,2),"15h_seconds_per_judgment_at_3140":round(54000/3140,2)},"split":{"salt":"H2-mini-split-v1-2026-09-13","rule":"Union all native, parent/tree, exact, normalized, model-view, project semantic, and project template identities.","algorithm":"SHA256(source revision || group ID || salt); whole groups; fill test, then val, then train; stop on shortage.","shared_exception":"Only stage-1 B0 anchor IDs across C0/F.","assigned":False},"gate":{"ready_for_human_review":all(v>=290 for v in fc.values()) and ordinary>=1160 and bf,"ready_for_final_materialization_proposal":False,"blocked_by":["human acceptance of ontology/grouping","50-record timing calibration","human record review","post-review re-audit","human approval of proposed hash-only assignment"],"R0":True,"R1":False,"R2":False},"no_training":True,"no_replay":True,"no_final_JSONL":True}
 save(AUD/"or_bench_full_audit.json",ora);save(AUD/"oasst1_full_audit.json",oaa);save(AUD/"h2_mini_capacity.json",cap)
 save(MAN/"h2_mini_candidates.json",{"status":"metadata_only_no_final_assignment","ontology":"h2m-cue-ontology-v1","grouping":"h2m-project-grouping-v1","source_revision":ORR,"records":ormanifest,"raw_text":False,"final_assignments":False})
 save(MAN/"b_short_mini_candidates.json",{"status":"metadata_only_no_final_assignment","source_revision":OAR,"records":oamanifest,"raw_text":False,"final_assignments":False})
 print(json.dumps({"raw":dict(raw),"group_safe":fc,"OASST_roots":len(roots),"OASST_group_safe":len(finala),"Bshort_pool":len(short),"feasible":cap["feasible"]},indent=2),flush=True)
if __name__=="__main__":main()
