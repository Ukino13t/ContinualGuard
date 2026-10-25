"""Aggregate validated Revision 4 A2 canonical against frozen A1 and A0-v2."""
from __future__ import annotations
import csv,json,statistics,sys
from pathlib import Path
ROOT=Path(r"E:\ContinualGuard")
sys.path.insert(0,str(ROOT/"scripts"))
from aggregate_route_a import cell,summarize_run
OUT=ROOT/"results/revision4/analysis"
STAGES=("AF2","AF3","AF4","AF5")
ARMS=("doc","short","hard")
SEEDS=(17,42,2026)
METHODS={
 "A0_v2":Path(r"Z:\ContinualGuard_Revision3\results\revision3\matched_ablation_amp256_v2\M200\a0_matched"),
 "A1_matched":ROOT/"results/revision2/matched_ablation_amp256/M200/a1_matched",
 "A2_canonical":Path(r"Z:\ContinualGuard_Revision4\results\revision4\matched_ablation_amp256\M200\a2_canonical"),
}
ENDPOINTS=("tpr_bwt_pp","macro_f1_bwt_pp","benign_fpr_increase_pp",
 "attack_peak_forgetting_pp","acquisition_tpr_pct","acquisition_macro_f1_pct",
 "worst_benign_group_fpr_drift_pp","worst_attack_position_tpr_loss_pp",
 "final_attack_tpr_pct","final_benign_fpr_pct","final_macro_f1_pct",
 "final_accuracy_pct","runtime_seconds","peak_gpu_memory_bytes")
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def write_csv(name,rows):
 if not rows:raise ValueError(name)
 fields=sorted({k for row in rows for k in row})
 with (OUT/name).open("w",encoding="utf-8",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
 val=load(ROOT/"results/revision4/validation.json")
 assert val["status"]=="PASS" and val["full_checkpoint_hashes"] and val["runs_checked"]==9
 OUT.mkdir(parents=True,exist_ok=True)
 per_seed,curves,acquisition,memory=[],[],[],[]
 for method,base in METHODS.items():
  for arm in ARMS:
   for seed in SEEDS:
    run=base/arm/f"seed{seed}"
    stages=[load(run/f"stage{i}/metrics.json") for i in range(1,5)]
    row=dict(method=method,arm=arm,seed=seed,**summarize_run(stages))
    for metric,name in (("tpr","attack_tpr"),("fpr","benign_fpr"),("macro_f1","macro_f1"),("accuracy","accuracy")):
     row[f"final_{name}_pct"]=statistics.mean(cell(stages[-1],task,metric)*100 for task in STAGES)
    per_seed.append(row)
    for i,m in enumerate(stages):
     trained=STAGES[i]
     acquisition.append(dict(method=method,arm=arm,seed=seed,trained_stage=trained,
      attack_tpr_pct=cell(m,trained,"tpr")*100,benign_fpr_pct=cell(m,trained,"fpr")*100,
      macro_f1_pct=cell(m,trained,"macro_f1")*100,accuracy_pct=cell(m,trained,"accuracy")*100,
      training_loss=m.get("training_loss"),validation_loss=m.get("validation_loss"),
      runtime_seconds=m["runtime_seconds"],peak_gpu_memory_bytes=m["peak_gpu_memory_bytes"]))
     for task in STAGES[:i+1]:
      ev=m["evaluations"][task]
      curves.append(dict(method=method,arm=arm,seed=seed,trained_stage=trained,eval_stage=task,
       attack_tpr_pct=cell(m,task,"tpr")*100,benign_fpr_pct=cell(m,task,"fpr")*100,
       macro_f1_pct=cell(m,task,"macro_f1")*100,accuracy_pct=cell(m,task,"accuracy")*100,
       attack_tpr_by_position=json.dumps(ev.get("attack_tpr_by_position",{}),sort_keys=True),
       benign_fpr_by_group=json.dumps(ev.get("benign_fpr_by_group",{}),sort_keys=True)))
     comp=m.get("buffer_after_training",{})
     memory.append(dict(method=method,arm=arm,seed=seed,stage=trained,
      size=comp.get("size"),by_label=json.dumps(comp.get("by_label",{}),sort_keys=True),
      by_stage=json.dumps(comp.get("by_stage",{}),sort_keys=True)))
 lookup={(r["method"],r["arm"],r["seed"]):r for r in per_seed}
 paired=[]
 for arm in ARMS:
  for seed in SEEDS:
   a=lookup[("A1_matched",arm,seed)];b=lookup[("A2_canonical",arm,seed)]
   paired.append(dict(arm=arm,seed=seed,**{f"a2_minus_a1_{k}":b[k]-a[k] for k in ENDPOINTS}))
 aggregate=[]
 for method in METHODS:
  for arm in ARMS:
   group=[r for r in per_seed if r["method"]==method and r["arm"]==arm]
   aggregate.append(dict(method=method,arm=arm,n_seeds=3,**{
    suffix:func([r[k] for r in group])
    for k in ENDPOINTS
    for suffix,func in ((f"{k}_mean",statistics.mean),(f"{k}_sample_sd",statistics.stdev))}))
 paired_aggregate=[]
 for arm in ARMS:
  group=[r for r in paired if r["arm"]==arm]
  out=dict(arm=arm,n_seeds=3)
  for k in ENDPOINTS:
   vals=[r[f"a2_minus_a1_{k}"] for r in group]
   out.update({f"a2_minus_a1_{k}_mean":statistics.mean(vals),
    f"a2_minus_a1_{k}_sample_sd":statistics.stdev(vals),
    f"a2_minus_a1_{k}_positive_seeds":sum(x>1e-9 for x in vals),
    f"a2_minus_a1_{k}_negative_seeds":sum(x< -1e-9 for x in vals)})
  paired_aggregate.append(out)
 for name,rows in (("per_seed.csv",per_seed),("aggregate.csv",aggregate),
  ("paired_per_seed.csv",paired),("paired_aggregate.csv",paired_aggregate),
  ("stage_curves.csv",curves),("stage_acquisition.csv",acquisition),
  ("memory_composition.csv",memory)):write_csv(name,rows)
 print(json.dumps(dict(status="PASS",runs=len(per_seed),paired=len(paired),curves=len(curves))))
if __name__=="__main__":main()

