"""Revision 4 selector-level pretraining gate and actual-pool audit."""
from __future__ import annotations
import csv, copy, json, statistics
from collections import Counter
from pathlib import Path
from continualguard.stronger_pilot_no_replay import CandidateBData, STAGES
from revision2_selector import MatchedBuffer, ranked_entry
from revision4_selector import A2CanonicalBuffer

ROOT=Path(r"E:\ContinualGuard")
A1=ROOT/"results/revision2/matched_ablation_amp256/M200/a1_matched"
A2_OLD=ROOT/"results/revision2/matched_ablation_amp256/M200/a2_matched"
OUT=ROOT/"results/revision4/selector_audit"
OUT.mkdir(parents=True,exist_ok=True)
key=lambda x:(x["label"],x["random_rank_sha256"],x["example_id"])
ids=lambda rows:[x["example_id"] for x in rows]

def jaccard(a,b):
    sa,sb=set(ids(a)),set(ids(b))
    return len(sa&sb)/len(sa|sb)

def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

cells=[]
selected=[]
same_set_replay_positions=0
binding_cells=0
for arm in ("doc","short","hard"):
    data=CandidateBData(ROOT,arm)
    for seed in (17,42,2026):
        a1=MatchedBuffer("a1_matched",seed,200)
        a2=A2CanonicalBuffer("a2_canonical",seed,200)
        for index,stage in enumerate(STAGES,1):
            current=[ranked_entry(r,seed) for r in data.records(stage,"train")]
            previous_a1=copy.deepcopy(a1.entries)
            previous_a2=copy.deepcopy(a2.entries)
            if index==1:
                assert previous_a1==previous_a2==[]
            common=MatchedBuffer("a1_matched",seed,200)
            common.entries=copy.deepcopy(previous_a2)
            common.update(current,tuple(STAGES[:index]))
            pre=Counter(x["task_id"] for x in [*previous_a2,*current])
            diag1=a1.update(current,tuple(STAGES[:index]))
            diag2=a2.update(current,tuple(STAGES[:index]))
            frozen_a1=read_rows(A1/arm/f"seed{seed}"/f"stage{index}"/"buffer.jsonl")
            frozen_a2=read_rows(A2_OLD/arm/f"seed{seed}"/f"stage{index}"/"buffer.jsonl")
            assert ids(a1.entries)==ids(frozen_a1)
            assert set(ids(a2.entries))==set(ids(frozen_a2))
            assert a2.entries==sorted(a2.entries,key=key)
            assert all(x["coverage_key"] for x in [*previous_a2,*current])
            assert diag2["duplicate_coverage_key_count"]==0
            assert len(a2.entries)==200
            if index==1:
                assert ids(a1.entries)==ids(a2.entries)
                s1,s2=a1.sample(4,2),a2.sample(4,2)
                assert ids(s1)==ids(s2)
                same_set_replay_positions+=4
            binding=set(ids(common.entries))!=set(ids(a2.entries))
            binding_cells+=int(binding)
            cells.append(dict(arm=arm,seed=seed,stage=stage,index=index,
                eligible_a1=len(previous_a1)+len(current),
                eligible_a2=len(previous_a2)+len(current),
                previous_a1=len(previous_a1),previous_a2=len(previous_a2),
                current_train=len(current),pre_a1_stage=json.dumps(dict(pre),sort_keys=True),
                a1_stage=json.dumps(dict(Counter(x["task_id"] for x in a1.entries)),sort_keys=True),
                a2_stage=json.dumps(dict(Counter(x["task_id"] for x in a2.entries)),sort_keys=True),
                a1_label=json.dumps(dict(Counter(x["label"] for x in a1.entries)),sort_keys=True),
                a2_label=json.dumps(dict(Counter(x["label"] for x in a2.entries)),sort_keys=True),
                same_pool_quota_binding=binding,
                actual_selected_jaccard=jaccard(a1.entries,a2.entries),
                common_pool_selected_jaccard=jaccard(a1.entries,common.entries),
                canonical_order=True,coverage_duplicates=diag2["duplicate_coverage_key_count"]))
            selected.append(dict(arm=arm,seed=seed,stage=stage,a1_ids=ids(a1.entries),a2_canonical_ids=ids(a2.entries),
                same_pool_a2_ids=ids(common.entries)))
with (OUT/"candidate_pool_audit.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(cells[0]));w.writeheader();w.writerows(cells)
with (OUT/"selected_ids.jsonl").open("w",encoding="utf-8",newline="\n") as f:
    for item in selected:f.write(json.dumps(item,separators=(",",":"))+"\n")
summary=dict(status="PASS",cells=len(cells),same_pool_binding_cells=binding_cells,
    same_set_replay_positions=same_set_replay_positions,
    canonical_cells=sum(r["canonical_order"] for r in cells),
    coverage_duplicate_cells=sum(r["coverage_duplicates"]>0 for r in cells),
    mean_actual_jaccard=statistics.mean(r["actual_selected_jaccard"] for r in cells),
    frozen_a1_order_verified=True,old_a2_membership_verified=True,
    train_only=True,full_training_gate="PENDING two-stage model parity")
(OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
print(json.dumps(summary))


