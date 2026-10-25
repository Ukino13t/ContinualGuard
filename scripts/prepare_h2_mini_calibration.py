"""Prepare the blinded H2-mini 50-record human-review calibration package."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,importlib.util,json,math,random
from collections import Counter
from datetime import datetime,timezone,timedelta
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REVIEW=ROOT/"review"
OR_REV="e36d8b80e81837c8a8f264bbb2a49f1b32c7e272"
OA_REV="fdf72ae0827c1cda404aff25b6603abec9e3399b"
SEED=20260913
CALIBRATION_ID="h2mini-calibration-v1-20260913"
A_ORDER_SEED=2026091301
B_ORDER_SEED=2026091302
FIELDS=["review_id","text","source","source_category","language","proposed_stratum","reviewer_decision","reviewer_stratum","reason_code","confidence","optional_note","review_seconds"]
DECISIONS=["ACCEPT_BENIGN","REJECT_TRUE_INJECTION","REJECT_AMBIGUOUS_INTENT","REJECT_HARMFUL_OR_UNSAFE","REJECT_CONTEXT_MISSING","REJECT_PRIVACY_OR_RIGHTS","REJECT_QUALITY"]
STRATA=["B0","B1","B2","B3","UNSURE","NOT_APPLICABLE"]
CONFIDENCE=["HIGH","MEDIUM","LOW"]

def sha(s):return hashlib.sha256(s.encode("utf-8")).hexdigest()
def file_sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def save(path,obj):
 path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def hkey(*parts):return sha("|".join(map(str,parts)))
def percentile(v,q):
 x=sorted(v);return x[max(0,math.ceil(q*len(x))-1)]
def length_bin(pool,row):
 vals=[x["deberta_tokens"] for x in pool]
 cuts=[percentile(vals,.25),percentile(vals,.50),percentile(vals,.75)]
 return sum(row["deberta_tokens"]>x for x in cuts)
def select_diverse(pool,n,seed,category_key,cue_key=None):
 remaining=list(pool);chosen=[];cat=Counter();bins=Counter();cues=Counter()
 vals=sorted(x["deberta_tokens"] for x in pool)
 cuts=[percentile(vals,.25),percentile(vals,.50),percentile(vals,.75)]
 while len(chosen)<n:
  scored=[]
  for row in remaining:
   b=sum(row["deberta_tokens"]>x for x in cuts);c=str(category_key(row))
   cf=list(cue_key(row)) if cue_key else []
   cue_score=sum(cues[x] for x in cf)/max(1,len(cf))
   score=(cat[c],bins[b],cue_score,hkey(seed,row["candidate_id"]))
   scored.append((score,row,b,c,cf))
  score,row,b,c,cf=min(scored,key=lambda x:x[0])
  chosen.append(row);remaining.remove(row);cat[c]+=1;bins[b]+=1;cues.update(cf)
 return chosen
def load_audit_module():
 p=ROOT/"scripts/audit_h2_mini_sources.py"
 spec=importlib.util.spec_from_file_location("h2audit",p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return m
def oasst_texts(indices):
 out={}
 with gzip.open(ROOT/"data/raw/hard_benign/oasst1"/OA_REV/"2023-04-12_oasst_ready.messages.jsonl.gz","rt",encoding="utf-8") as f:
  for i,line in enumerate(f):
   if i in indices:out[i]=json.loads(line)["text"]
 return out
def make_review_id(candidate_id):
 return "H2MC-"+hkey(CALIBRATION_ID,SEED,candidate_id)[:12].upper()
def write_csv(path,rows):
 with path.open("w",encoding="utf-8-sig",newline="") as f:
  w=csv.DictWriter(f,fieldnames=FIELDS,extrasaction="ignore",quoting=csv.QUOTE_MINIMAL);w.writeheader();w.writerows(rows)
def main():
 REVIEW.mkdir(parents=True,exist_ok=True)
 or_manifest=json.loads((ROOT/"data/manifests/hard_benign/h2_mini_candidates.json").read_text(encoding="utf-8"))["records"]
 oa_manifest=json.loads((ROOT/"data/manifests/hard_benign/b_short_mini_candidates.json").read_text(encoding="utf-8"))["records"]
 with (ROOT/"data/raw/hard_benign/or_bench"/OR_REV/"or-bench-80k.csv").open(encoding="utf-8-sig",newline="") as f:or_raw=list(csv.DictReader(f))
 audit=load_audit_module()
 tok=audit.AutoTokenizer.from_pretrained(ROOT/"data/raw/huggingface/microsoft_deberta-v3-base"/audit.TOKR,local_files_only=True,use_fast=False)
 selected=[]
 for stratum,count in [("B1",12),("B2",12),("B3",12)]:
  pool=[x for x in or_manifest if x["candidate_label"]==stratum+"_candidate"]
  chosen=select_diverse(pool,count,SEED+int(stratum[1:]),lambda x:x["category"],lambda x:x["cue_family_ids"])
  for x in chosen:selected.append({**x,"source":"OR-Bench","source_category":x["category"],"language":"en","proposed_stratum":stratum,"text":or_raw[x["source_row_index"]]["prompt"],"native_id":None})
 bpool=[x for x in oa_manifest if x["b_short_length_language_candidate"] and x["deberta_tokens"]<=57]
 groups={"en":[x for x in bpool if x["lang"]=="en"],"zh":[x for x in bpool if x["lang"]=="zh"],"other":[x for x in bpool if x["lang"] not in {"en","zh"}]}
 bchosen=[]
 for group,n in [("en",4),("zh",3),("other",3)]:
  if len(groups[group])<n:raise RuntimeError("Insufficient B0 language-diversity pool: "+group)
  bchosen+=select_diverse(groups[group],n,SEED+100+len(bchosen),lambda x:x["lang"])
 texts=oasst_texts({x["source_row_index"] for x in bchosen})
 for x in bchosen:selected.append({**x,"source":"OASST1","source_category":"","language":x["lang"],"proposed_stratum":"B0","text":texts[x["source_row_index"]],"native_id":x["message_id"]})
 boundary=[]
 for i,row in enumerate(or_raw):
  label,cues,spans=audit.classify(row["prompt"])
  if label=="ambiguous":
   cid="orb-"+sha(OR_REV+"|"+str(i)+"|"+sha(row["prompt"]))[:24]
   boundary.append({"candidate_id":cid,"source_row_index":i,"category":row["category"],"cue_family_ids":cues,"cue_span_count":spans,"deberta_tokens":len(tok.encode(row["prompt"],add_special_tokens=True,truncation=False)),"raw_text_sha256":sha(row["prompt"]),"normalized_text_sha256":sha(audit.norm(row["prompt"])),"source":"OR-Bench","source_category":row["category"],"language":"en","proposed_stratum":"BOUNDARY","text":row["prompt"],"native_id":None})
 if len(boundary)!=4:raise RuntimeError(f"Expected four audited ambiguous boundary candidates, found {len(boundary)}")
 selected+=sorted(boundary,key=lambda x:hkey(SEED,"boundary",x["candidate_id"]))
 if Counter(x["proposed_stratum"] for x in selected)!=Counter({"B0":10,"B1":12,"B2":12,"B3":12,"BOUNDARY":4}):raise RuntimeError("Calibration count mismatch")
 rows=[]
 for x in selected:
  rows.append({"review_id":make_review_id(x["candidate_id"]),"text":x["text"],"source":x["source"],"source_category":x["source_category"],"language":x["language"],"proposed_stratum":x["proposed_stratum"],"reviewer_decision":"","reviewer_stratum":"","reason_code":"","confidence":"","optional_note":"","review_seconds":""})
 if len({x["review_id"] for x in rows})!=50:raise RuntimeError("review_id collision")
 ra=list(rows);rb=list(rows);random.Random(A_ORDER_SEED).shuffle(ra);random.Random(B_ORDER_SEED).shuffle(rb)
 if [x["review_id"] for x in ra]==[x["review_id"] for x in rb]:raise RuntimeError("Reviewer orders unexpectedly identical")
 pa=REVIEW/"h2_mini_calibration_reviewer_A.csv";pb=REVIEW/"h2_mini_calibration_reviewer_B.csv";write_csv(pa,ra);write_csv(pb,rb)
 selected_by_id={x["candidate_id"]:x for x in selected}
 meta=[]
 for x in sorted(selected,key=lambda z:(z["proposed_stratum"],z["candidate_id"])):
  meta.append({"review_id":make_review_id(x["candidate_id"]),"candidate_id":x["candidate_id"],"source":x["source"],"source_revision":OA_REV if x["source"]=="OASST1" else OR_REV,"source_native_id":x["native_id"],"source_row_index":x["source_row_index"],"proposed_stratum":x["proposed_stratum"],"source_category":x["source_category"],"language":x["language"],"deberta_tokens":x.get("deberta_tokens"),"cue_family_ids":x.get("cue_family_ids",[]),"text_sha256":sha(x["text"]),"normalized_text_sha256":x.get("normalized_text_sha256",sha(audit.norm(x["text"])))})
 manifest={"calibration_id":CALIBRATION_ID,"status":"worksheets_ready_independent_human_review_not_started","created_at":datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),"sampling_seed":SEED,"reviewer_order_seeds":{"A":A_ORDER_SEED,"B":B_ORDER_SEED},"source_revisions":{"OR-Bench":OR_REV,"OASST1":OA_REV},"selection_counts":dict(Counter(x["proposed_stratum"] for x in selected)),"selection_policy":{"detector_predictions_used":False,"model_confidence_used":False,"NotInject_similarity_used":False,"future_split_stage_assignment_used":False,"B0":"4 English, 3 Chinese, 3 other-language group-safe short OASST candidates; deterministic diversity over language and token-length quartiles","B1_B2_B3":"deterministic greedy diversity over source category, token-length quartile, and cue families","BOUNDARY":"all four audited OR-Bench ambiguous candidates","tie_break":"SHA256(sampling seed || candidate ID)"},"worksheet_columns":FIELDS,"allowed_values":{"reviewer_decision":DECISIONS,"reviewer_stratum":STRATA,"confidence":CONFIDENCE},"records":meta,"worksheets":{"reviewer_A":{"path":str(pa.relative_to(ROOT)).replace("\\","/"),"rows":50,"sha256":file_sha(pa)},"reviewer_B":{"path":str(pb.relative_to(ROOT)).replace("\\","/"),"rows":50,"sha256":file_sha(pb)}},"raw_text_in_manifest":False,"human_judgments_performed":False,"production_review_started":False,"final_assignment":False,"training":False,"replay":False}
 save(REVIEW/"calibration_manifest.json",manifest)
 print(json.dumps({"calibration_id":CALIBRATION_ID,"counts":manifest["selection_counts"],"sampling_seed":SEED,"A_sha256":manifest["worksheets"]["reviewer_A"]["sha256"],"B_sha256":manifest["worksheets"]["reviewer_B"]["sha256"]},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
