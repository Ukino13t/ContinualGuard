"""Revision 4 canonical stage-quota selector regression tests."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from continualguard.stronger_pilot_no_replay import CandidateBData,STAGES
from revision2_selector import MatchedBuffer,ranked_entry
from revision4_selector import A2CanonicalBuffer

def test_same_set_canonical_order_and_replay_sequence():
    data=CandidateBData(ROOT,"doc")
    seed=2026
    candidates=[ranked_entry(r,seed) for r in data.records("AF2","train")]
    a1=MatchedBuffer("a1_matched",seed,200)
    a2=A2CanonicalBuffer("a2_canonical",seed,200)
    a1.update(candidates,("AF2",))
    a2.update(candidates,("AF2",))
    assert [r["example_id"] for r in a1.entries]==[r["example_id"] for r in a2.entries]
    s1=[r["example_id"] for _ in range(100) for r in a1.sample(4,2)]
    s2=[r["example_id"] for _ in range(100) for r in a2.sample(4,2)]
    assert s1==s2
    scrambled=A2CanonicalBuffer("a2_canonical",seed,200)
    scrambled.entries=list(a2.entries)
    scrambled.entries[0],scrambled.entries[-1]=scrambled.entries[-1],scrambled.entries[0]
    shifted=[r["example_id"] for _ in range(100) for r in scrambled.sample(4,2)]
    assert shifted!=s1

def test_stage_quota_changes_membership_but_preserves_canonical_storage():
    data=CandidateBData(ROOT,"doc")
    seed=2026
    a1=MatchedBuffer("a1_matched",seed,200)
    a2=A2CanonicalBuffer("a2_canonical",seed,200)
    for stage_index,stage in enumerate(STAGES[:2],1):
        candidates=[ranked_entry(r,seed) for r in data.records(stage,"train")]
        a1.update(candidates,tuple(STAGES[:stage_index]))
        a2.update(candidates,tuple(STAGES[:stage_index]))
    assert {r["example_id"] for r in a1.entries}!={r["example_id"] for r in a2.entries}
    assert a2.entries==sorted(a2.entries,key=lambda r:(r["label"],r["random_rank_sha256"],r["example_id"]))
    assert len([r for r in a2.entries if r["label"]==0])==100
    assert len([r for r in a2.entries if r["label"]==1])==100
    assert len([r for r in a2.entries if r["task_id"]=="AF2"])==100
    assert len([r for r in a2.entries if r["task_id"]=="AF3"])==100

def test_future_or_test_candidate_is_rejected():
    data=CandidateBData(ROOT,"doc")
    seed=17
    row=data.records("AF3","test")[0]
    try:
        ranked_entry(row,seed)
    except ValueError:
        pass
    else:
        raise AssertionError("TEST record entered selector")

