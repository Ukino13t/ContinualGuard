"""Generate vector Revision 4 figures from validated aggregate and selector CSVs."""
from __future__ import annotations
import csv,json,statistics
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9})
ROOT=Path(r"E:\ContinualGuard")
DATA=ROOT/"results/revision4/analysis"
OUT=ROOT/"figures/revision4"
STAGES=("AF2","AF3","AF4","AF5")
METHODS=(("A0_v2","A0-v2","#555555"),("A1_matched","A1","#1f77b4"),("A2_canonical","A2-canonical","#d95f02"))
def rows(path):
 with path.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def save(fig,name):
 OUT.mkdir(parents=True,exist_ok=True)
 fig.savefig(OUT/(name+".pdf"),bbox_inches="tight")
 fig.savefig(OUT/(name+".png"),dpi=180,bbox_inches="tight")
 plt.close(fig)
def main():
 val=json.loads((ROOT/"results/revision4/validation.json").read_text(encoding="utf-8"))
 assert val["status"]=="PASS" and val["runs_checked"]==9 and val["full_checkpoint_hashes"]
 curves=rows(DATA/"stage_curves.csv")
 for arm,task,metric,name,title in (
  ("short","AF3","attack_tpr_pct","b_short_attack_forgetting","B-SHORT: AF3 attack recall after later stages"),
  ("hard","AF2","benign_fpr_pct","b_hard_benign_fpr","B-HARD: AF2 benign false-positive rate")):
  fig,ax=plt.subplots(figsize=(6.2,3.35))
  xstages=STAGES[STAGES.index(task):]
  for method,label,color in METHODS:
   means=[];sds=[]
   for stage in xstages:
    vals=[float(r[metric]) for r in curves if r["arm"]==arm and r["method"]==method and r["eval_stage"]==task and r["trained_stage"]==stage]
    assert len(vals)==3,(arm,method,stage)
    means.append(statistics.mean(vals));sds.append(statistics.stdev(vals))
   ax.errorbar(range(len(xstages)),means,yerr=sds,label=label,color=color,marker="o",capsize=3,linewidth=1.6)
  ax.set_xticks(range(len(xstages)),xstages)
  ax.set_ylabel("Attack TPR (%)" if metric=="attack_tpr_pct" else "Benign FPR (%)")
  ax.set_title(title,fontsize=10)
  ax.grid(axis="y",alpha=.22)
  ax.legend(frameon=False,ncol=3,fontsize=8,loc="best")
  fig.tight_layout()
  save(fig,name)
 memory=rows(DATA/"memory_composition.csv")
 fig,ax=plt.subplots(figsize=(8,3.7))
 positions=[];labels=[];values=[]
 for arm in ("doc","short","hard"):
  for method,short in (("A1_matched","A1"),("A2_canonical","A2")):
   records=[r for r in memory if r["arm"]==arm and r["method"]==method and r["stage"]=="AF5"]
   assert len(records)==3
   vals=[statistics.mean(int(json.loads(r["by_stage"]).get(stage,0)) for r in records) for stage in STAGES]
   positions.append(len(positions));labels.append(f"B-{arm.upper()}\n{short}");values.append(vals)
 bottom=[0.0]*len(positions)
 for j,stage in enumerate(STAGES):
  vals=[v[j] for v in values]
  ax.bar(positions,vals,bottom=bottom,label=stage,width=.72)
  bottom=[a+b for a,b in zip(bottom,vals)]
 ax.set_xticks(positions,labels)
 ax.set_ylabel("Final selected records (mean of 3 seeds)")
 ax.set_ylim(0,210);ax.legend(ncol=4,frameon=False,loc="upper center")
 ax.grid(axis="y",alpha=.18);fig.tight_layout();save(fig,"stage_memory_composition")
 audit=rows(ROOT/"results/revision4/selector_audit/candidate_pool_audit.csv")
 fig,ax=plt.subplots(figsize=(6.2,3.35))
 for arm,color in (("doc","#1f77b4"),("short","#d95f02"),("hard","#2ca02c")):
  means=[statistics.mean(float(r["actual_selected_jaccard"]) for r in audit if r["arm"]==arm and r["stage"]==stage) for stage in STAGES]
  ax.plot(STAGES,means,label=f"B-{arm.upper()}",color=color,marker="o")
 ax.set_ylabel("A1/A2 selected-ID Jaccard")
 ax.set_ylim(0,1.03);ax.grid(axis="y",alpha=.2);ax.legend(frameon=False)
 fig.tight_layout();save(fig,"selected_buffer_overlap")
 print(json.dumps({"status":"PASS","vector_figures":4}))
if __name__=="__main__":main()

