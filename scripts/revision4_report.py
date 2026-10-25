"""Render Revision 4 corrected stage-quota report from validated CSVs."""
from __future__ import annotations
import csv,json,statistics
from pathlib import Path
ROOT=Path(r"E:\ContinualGuard")
DATA=ROOT/"results/revision4/analysis"
OUT=ROOT/"reports/revision4/02_corrected_stage_balance.md"
def rows(name):
 with (DATA/name).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def f(row,key):return float(row[key])
def pm(row,key):return f"{f(row,key+'_mean'):+.2f} +/- {f(row,key+'_sample_sd'):.2f}"
def main():
 val=json.loads((ROOT/"results/revision4/validation.json").read_text(encoding="utf-8"))
 assert val["status"]=="PASS" and val["runs_checked"]==9 and val["full_checkpoint_hashes"]
 per=rows("per_seed.csv");paired=rows("paired_per_seed.csv")
 agg=rows("aggregate.csv");pairs=rows("paired_aggregate.csv")
 acq=rows("stage_acquisition.csv");mem=rows("memory_composition.csv")
 audit=json.loads((ROOT/"results/revision4/selector_audit/summary.json").read_text(encoding="utf-8"))
 assert len(per)==27 and len(paired)==9 and len(agg)==9 and len(pairs)==3
 lookup={(r["method"],r["arm"],r["seed"]):r for r in per}
 lines=["# Corrected A1/A2-canonical historical-stage quota ablation","",
  "Status: nine A2-canonical trajectories complete and independently full-hash validated. Frozen A1 is the paired comparator; A0-v2 is the corrected unstratified context. All claims concern Candidate B only.","",
  f"Selector gate: {audit['cells']}/36 canonical actual-pool cells; {audit['same_pool_binding_cells']}/36 same-pool quota-binding cells; 0 duplicate coverage-key cells. Two-stage same-set actual GPU parity matched 400/400 replay IDs and exact model hashes. The intervention changes membership, but its detector effect is evaluated below.","",
  "## Paired per-seed endpoints","",
  "| Arm | Seed | A1 TPR BWT | A2 TPR BWT | A2-A1 TPR BWT | A2-A1 macro-F1 BWT | A2-A1 FPR increase | A2 acquisition TPR | A2 peak forgetting | A2 worst position loss | A2 worst benign-group drift |",
  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
 for r in sorted(paired,key=lambda x:(("doc","short","hard").index(x["arm"]),int(x["seed"]))):
  arm,seed=r["arm"],r["seed"]
  a=lookup[("A1_matched",arm,seed)];b=lookup[("A2_canonical",arm,seed)]
  lines.append("| B-{arm} | {seed} | {a:+.2f} | {b:+.2f} | {dt:+.2f} | {df:+.2f} | {dp:+.2f} | {ac:.2f} | {pk:.2f} | {pos:.2f} | {bg:.2f} |".format(
   arm=arm.upper(),seed=seed,a=f(a,"tpr_bwt_pp"),b=f(b,"tpr_bwt_pp"),
   dt=f(r,"a2_minus_a1_tpr_bwt_pp"),df=f(r,"a2_minus_a1_macro_f1_bwt_pp"),
   dp=f(r,"a2_minus_a1_benign_fpr_increase_pp"),
   ac=f(b,"acquisition_tpr_pct"),pk=f(b,"attack_peak_forgetting_pp"),
   pos=f(b,"worst_attack_position_tpr_loss_pp"),bg=f(b,"worst_benign_group_fpr_drift_pp")))
 lines+=["","## Paired mean +/- sample SD (percentage points)","",
  "| Arm | A2-A1 TPR BWT | A2-A1 macro-F1 BWT | A2-A1 FPR increase | Positive/negative TPR seeds | Positive/negative FPR seeds |",
  "|---|---:|---:|---:|---:|---:|"]
 for r in pairs:
  lines.append(f"| B-{r['arm'].upper()} | {pm(r,'a2_minus_a1_tpr_bwt_pp')} | {pm(r,'a2_minus_a1_macro_f1_bwt_pp')} | {pm(r,'a2_minus_a1_benign_fpr_increase_pp')} | {r['a2_minus_a1_tpr_bwt_pp_positive_seeds']}/{r['a2_minus_a1_tpr_bwt_pp_negative_seeds']} | {r['a2_minus_a1_benign_fpr_increase_pp_positive_seeds']}/{r['a2_minus_a1_benign_fpr_increase_pp_negative_seeds']} |")
 lines+=["","## Three-selector raw aggregate context","",
  "| Arm | Selector | TPR BWT | macro-F1 BWT | FPR increase | Acquisition TPR | Runtime (min) | Peak GPU (GiB) |",
  "|---|---|---:|---:|---:|---:|---:|---:|"]
 for r in agg:
  lines.append(f"| B-{r['arm'].upper()} | {r['method']} | {pm(r,'tpr_bwt_pp')} | {pm(r,'macro_f1_bwt_pp')} | {pm(r,'benign_fpr_increase_pp')} | {f(r,'acquisition_tpr_pct_mean'):.2f} +/- {f(r,'acquisition_tpr_pct_sample_sd'):.2f} | {f(r,'runtime_seconds_mean')/60:.2f} +/- {f(r,'runtime_seconds_sample_sd')/60:.2f} | {f(r,'peak_gpu_memory_bytes_mean')/(1024**3):.3f} +/- {f(r,'peak_gpu_memory_bytes_sample_sd')/(1024**3):.3f} |")
 lines+=["","## A2-canonical stage acquisition (mean +/- sample SD over seeds)","",
  "| Arm | New stage | Attack TPR (%) | Benign FPR (%) | Macro-F1 (%) | Training loss | Validation loss |",
  "|---|---|---:|---:|---:|---:|---:|"]
 for arm in ("doc","short","hard"):
  for stage in ("AF2","AF3","AF4","AF5"):
   group=[r for r in acq if r["method"]=="A2_canonical" and r["arm"]==arm and r["trained_stage"]==stage]
   assert len(group)==3
   def ms(key):
    vals=[f(r,key) for r in group]
    return f"{statistics.mean(vals):.2f} +/- {statistics.stdev(vals):.2f}"
   lines.append(f"| B-{arm.upper()} | {stage} | {ms('attack_tpr_pct')} | {ms('benign_fpr_pct')} | {ms('macro_f1_pct')} | {ms('training_loss')} | {ms('validation_loss')} |")
 lines+=["","## Interpretation and resolution","",
  "Positive TPR/macro-F1 backward transfer is better; negative FPR increase is better. Each 60-malicious/60-benign stage test resolves class-conditional rates in 1.667-pp steps; each 20-item attack-position slice resolves 5 pp. Three fixed seeds support descriptive paired differences and direction counts, not a population significance claim.",
  "Current-stage acquisition, earlier-stage retention and benign false-positive drift are separate quantities. The complete per-stage curves (including position and benign-group JSON fields), acquisition diagnostics, selected-memory stage distributions, and per-seed appendix values are in the sibling CSVs. The four vector plots are in figures/revision4. Historical A1/A2 outcomes remain descriptive because their stored buffer ordering differed.",
  ""]
 OUT.write_text("\n".join(lines),encoding="utf-8")
 print(json.dumps({"status":"PASS","per_seed_pairs":len(paired),"lines":len(lines)}))
if __name__=="__main__":main()

