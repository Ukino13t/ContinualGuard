"""Generate Revision 4 ACL tables only from full-hash-validated CSV results."""
from __future__ import annotations
import csv,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
if HERE.name=="scripts":
 ROOT=HERE.parent
 DATA=ROOT/"results/revision4/analysis"
 TABLES=ROOT/"paper/acl/tables"
 VALIDATION=ROOT/"results/revision4/validation.json"
else:
 ROOT=HERE
 DATA=ROOT/"data/revision4"
 TABLES=ROOT/"tables"
 VALIDATION=DATA/"validation.json"
def rows(name):
 with (DATA/name).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def pm(row,key,signed=True):
 a=float(row[key+"_mean"]);s=float(row[key+"_sample_sd"])
 return f"${a:+.2f}\\pm{s:.2f}$" if signed else f"${a:.2f}\\pm{s:.2f}$"
def write(name,lines):
 (TABLES/name).write_text("\n".join(lines)+"\n",encoding="utf-8")
def main():
 v=json.loads(VALIDATION.read_text(encoding="utf-8"))
 assert v["status"]=="PASS" and v["full_checkpoint_hashes"] and v["runs_checked"]==9
 agg={(r["method"],r["arm"]):r for r in rows("aggregate.csv")}
 pair={r["arm"]:r for r in rows("paired_aggregate.csv")}
 per=rows("paired_per_seed.csv")
 assert len(agg)==9 and len(pair)==3 and len(per)==9
 out=[r"\begin{table*}[t]",r"\centering",r"\small",
  r"\setlength{\tabcolsep}{4pt}",r"\begin{tabular}{@{}llrrrr@{}}",r"\toprule",
  r"Arm & Selector & TPR BWT & Macro-F1 BWT & FPR increase & Acquisition TPR \\",
  r"\midrule"]
 for arm in ("doc","short","hard"):
  for method,label in (("A0_v2","A0-v2"),("A1_matched","A1"),("A2_canonical","A2-canonical")):
   r=agg[(method,arm)]
   out.append(f"B-{arm.upper()} & {label} & {pm(r,'tpr_bwt_pp')} & {pm(r,'macro_f1_bwt_pp')} & {pm(r,'benign_fpr_increase_pp')} & {pm(r,'acquisition_tpr_pct',False)} "+r"\\")
 out.extend([r"\bottomrule",r"\end{tabular}",
  r"\caption{Controlled memory-selector comparison at $M=200$, mean $\pm$ sample SD over seeds 17, 42, and 2026 (percentage points except acquisition TPR, shown in percent). All three use shared SHA-256 ranks and canonical replay-buffer order. A0-v2 has no class quota; A1 adds a 100/100 label quota; A2-canonical adds historical-stage quotas. Positive backward transfer is favorable; negative FPR increase is favorable.}",
  r"\label{tab:r4-rq3}",r"\end{table*}",""])
 write("revision4_rq3.tex",out)
 out=[r"\begin{table}[t]",r"\centering",r"\small",r"\setlength{\tabcolsep}{3.5pt}",
  r"\begin{tabular}{@{}lrrr@{}}",r"\toprule",
  r"Arm & $\Delta$TPR BWT & $\Delta$F1 BWT & $\Delta$FPR incr. \\",r"\midrule"]
 for arm in ("doc","short","hard"):
  r=pair[arm]
  out.append(f"B-{arm.upper()} & {pm(r,'a2_minus_a1_tpr_bwt_pp')} & {pm(r,'a2_minus_a1_macro_f1_bwt_pp')} & {pm(r,'a2_minus_a1_benign_fpr_increase_pp')} "+r"\\")
 out.extend([r"\bottomrule",r"\end{tabular}",
  r"\caption{Paired A2-canonical minus A1 stage-quota effects in percentage points; mean $\pm$ sample SD across three fixed seeds. These methods share ranks, numerical settings, and canonical buffer order. Positive backward transfer and negative FPR-increase differences favor A2.}",
  r"\label{tab:r4-stage-pairs}",r"\end{table}",""])
 write("revision4_stage_balance_pairs.tex",out)
 out=[r"\begin{table*}[t]",r"\centering",r"\small",r"\setlength{\tabcolsep}{4pt}",
  r"\begin{tabular}{@{}llrrr@{}}",r"\toprule",
  r"Arm & Seed & $\Delta$TPR BWT & $\Delta$F1 BWT & $\Delta$FPR increase \\",r"\midrule"]
 for r in sorted(per,key=lambda r:(("doc","short","hard").index(r["arm"]),int(r["seed"]))):
  out.append(f"B-{r['arm'].upper()} & {r['seed']} & {float(r['a2_minus_a1_tpr_bwt_pp']):+.2f} & {float(r['a2_minus_a1_macro_f1_bwt_pp']):+.2f} & {float(r['a2_minus_a1_benign_fpr_increase_pp']):+.2f} "+r"\\")
 out.extend([r"\bottomrule",r"\end{tabular}",
  r"\caption{Every seed-paired A2-canonical minus A1 endpoint (percentage points). One class-conditional test error is 1.667 pp; the three seeds do not support a significance claim.}",
  r"\label{tab:r4-stage-seeds}",r"\end{table*}",""])
 write("revision4_stage_balance_seeds.tex",out)
 print(json.dumps(dict(status="PASS",tables=3)))
if __name__=="__main__":main()

