"""Run the nine predeclared Revision 4 cells with hash and lineage preflight."""
from __future__ import annotations
import hashlib,json,shutil,sys,time
from pathlib import Path
from revision4_train import run

ROOT=Path(r"E:\ContinualGuard")
Z=Path(r"Z:\ContinualGuard_Revision4")
protocol=json.loads((ROOT/"results/revision4/protocol.json").read_text(encoding="utf-8"))
assert protocol["status"]=="FROZEN_BEFORE_REVISION4_GPU"
for relative,expected in protocol["source_sha256"].items():
    got=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
    if got!=expected:raise RuntimeError(f"Protocol hash drift: {relative}")
assert shutil.disk_usage(ROOT).free>3_500_000_000
assert shutil.disk_usage(Z).free>10_000_000_000
for arm,seed in (("doc",2026),("doc",17),("doc",42),
                 ("short",17),("short",42),("short",2026),
                 ("hard",17),("hard",42),("hard",2026)):
    key=f"{arm}/seed{seed}"
    a1=ROOT/"results/revision2/matched_ablation_amp256/M200/a1_matched"/arm/f"seed{seed}"
    m=json.loads((a1/"run_metadata.json").read_text(encoding="utf-8"))
    assert m["status"]=="COMPLETE" and m["completed_stages"]==4
    assert m["config_sha256"]==protocol["paired_a1"][key]["config_sha256"]
    start=time.time()
    print(json.dumps({"event":"run_start","arm":arm,"seed":seed,"time":start}),flush=True)
    path=run(ROOT,"a2_canonical",arm,seed,capacity=200)
    meta=json.loads((path/"run_metadata.json").read_text(encoding="utf-8"))
    assert meta["status"]=="COMPLETE" and meta["completed_stages"]==4
    print(json.dumps({"event":"run_complete","arm":arm,"seed":seed,
                      "seconds":time.time()-start,"path":str(path)}),flush=True)
print(json.dumps({"event":"revision4_all_nine_complete"}),flush=True)

