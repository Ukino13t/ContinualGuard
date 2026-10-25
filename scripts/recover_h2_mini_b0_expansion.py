"""Freeze explicitly reconfirmed expansion evidence; no agreement or capacity analysis."""
import csv,gzip,io,json,sys
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter
import prepare_h2_mini_b0_encoding_reconfirmation as p

ROOT=p.ROOT
MANIFEST=ROOT/"review/b0_replenishment_expansion_integrity_recovery_manifest.json"
CANON=p.EXP_DIR/"reviewer_A_batch_01_canonical_utf8.csv"
REQUEST=Path("C:/Users/TLS/.codex/attachments/5b5cf6e6-534d-42d8-9a92-b0efb2d8ade8/\u5df2\u7c98\u8d34\u7684\u6587\u672c.txt")
HISTORY=ROOT/"results/h2_mini_b0_review_history"

def validate_rows(rows,expected,role,require_source):
    errors=[]
    if Counter(r.get("review_id") for r in rows)!=Counter(expected.keys()):
        errors.append({"field":"review_id","problem":"Expected each assigned identity exactly once"})
    for row in rows:
        rid=row.get("review_id");meta=expected.get(rid)
        if meta is None:continue
        for key,value in [("language",meta["language"]),("token_length",str(meta["deberta_tokens"])),("review_role",role)]:
            if row.get(key)!=value:errors.append({"review_id":rid,"field":key,"problem":"Immutable metadata mismatch"})
        if require_source and row.get("source")!="OpenAssistant/oasst1":
            errors.append({"review_id":rid,"field":"source","problem":"Source mismatch"})
        if p.sha(row.get("text",""))!=meta["raw_text_sha256"]:
            errors.append({"review_id":rid,"field":"text","problem":"Immutable source hash mismatch"})
        for key,allowed in [("reviewer_decision",p.DECISIONS),("b0_confirmed",p.B0_VALUES),("confidence",p.CONFIDENCE)]:
            if row.get(key,"").strip() not in allowed:
                errors.append({"review_id":rid,"field":key,"problem":"Missing or invalid required human value"})
        if row.get("reviewer_decision","").strip().startswith("REJECT_") and not row.get("reason_code","").strip():
            errors.append({"review_id":rid,"field":"reason_code","problem":"Rejection reason required"})
    return errors

def file_entry(path):
    return {"path":p.relative(path),**p.fingerprint(path)}

def preserve(path,name):
    dest=HISTORY/name
    assert not dest.exists()
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(path.read_bytes())
    assert p.file_sha(dest)==p.file_sha(path)
    return file_entry(dest)

def main():
    assert Path.cwd().resolve()==ROOT.resolve()
    assert Path(sys.executable).resolve()==(ROOT/".venv/Scripts/python.exe").resolve()
    assert not MANIFEST.exists() and not CANON.exists(),"Never overwrite canonical evidence"
    prep=p.load(p.RESULT);m=p.load(p.EXP_MANIFEST)
    assert prep["status"]=="STOP_AWAITING_A_17_HUMAN_RECONFIRMATIONS"
    for rel,fp in prep["protected_file_snapshots"].items():
        assert p.fingerprint(ROOT/rel)==fp,rel
    original_A=p.EXP_DIR/"reviewer_A_batch_01.csv";B=p.EXP_DIR/"reviewer_B_batch_01.csv"
    for who,path in [("A",original_A),("B",B)]:
        assert p.fingerprint(path)=={k:prep["original_submissions"][who][k] for k in ["sha256","mtime_ns"]}
    expected={r["review_id"]:r for r in m["records"]}
    Aexp={k:v for k,v in expected.items() if v["second_reviewer"]=="A"}
    Bexp={k:v for k,v in expected.items() if v["second_reviewer"]=="B"}
    recon_ids=prep["affected_review_ids"];Rexp={rid:Aexp[rid] for rid in recon_ids}
    before={p.relative(path):p.fingerprint(path) for path in [original_A,B,p.PACKET,p.EXP_MANIFEST]}
    h,rr=p.read_csv(p.PACKET.read_bytes())
    errors=validate_rows(rr,Rexp,"SECOND_REVIEW_RECONFIRMATION",False)
    if h!=p.FIELDS:errors.append({"field":"header","problem":"Unexpected reconfirmation columns"})
    if p.file_sha(p.PACKET)==prep["packet"]["sha256"]:errors.append({"field":"human_fields","problem":"Unchanged blank preparation file"})
    bh,br=p.read_csv(B.read_bytes())
    errors+=validate_rows(br,Bexp,"SECOND_REVIEW",True)
    if bh not in [p.EXP_FIELDS,p.EXP_FIELDS+["review_seconds"]]:errors.append({"file":p.relative(B),"field":"header","problem":"Unexpected columns"})
    if errors:
        print(json.dumps({"integrity_status":"STOP","errors":errors},ensure_ascii=False,indent=2));return 2
    raw=original_A.read_bytes()
    assert raw.decode("cp936",errors="strict")==raw.decode("gb18030",errors="strict")
    ah,ar=p.read_csv(raw,"gb18030")
    assert ah in [p.EXP_FIELDS,p.EXP_FIELDS+["review_seconds"]]
    assert Counter(x["review_id"] for x in ar)==Counter(Aexp.keys())
    old={x["review_id"]:x for x in ar};new={x["review_id"]:x for x in rr}
    intact={rid:Aexp[rid] for rid in Aexp if rid not in new}
    errors=validate_rows([old[rid] for rid in intact],intact,"SECOND_REVIEW",True)
    assert not errors,errors
    assert len(intact)==36 and len(new)==17
    for rid,meta in expected.items():
        assert meta["second_reviewer"]!=meta["primary_reviewer"]
    # Re-read pinned source text for all105 assignments, without source/model judgments.
    source_audit=ROOT/"data/audits/hard_benign/oasst1_full_audit.json"
    source=p.load(source_audit)["source"]
    entry=next(x for x in source["files"] if x["file_name"].endswith(".messages.jsonl.gz"))
    path=ROOT/entry["local_path"]
    assert source["revision"]==p.OA_REV and p.file_sha(path)==entry["sha256"]
    wanted={r["source_row_index"]:rid for rid,r in expected.items()};texts={}
    with gzip.open(path,"rt",encoding="utf-8",errors="strict") as stream:
        for index,line in enumerate(stream):
            if index not in wanted:continue
            rid=wanted[index];meta=expected[rid];row=json.loads(line)
            assert row["message_id"]==meta["source_native_id"]==meta["candidate_id"]
            assert row["message_tree_id"]==meta["source_parent_id"] and row.get("parent_id")==meta["source_immediate_parent_id"]
            assert row["lang"]==meta["language"] and p.sha(row["text"])==meta["raw_text_sha256"]
            texts[rid]=row["text"]
    assert len(texts)==105
    for row in rr+br:assert row["text"]==texts[row["review_id"]]
    canonical=[];provenance=[]
    for rid in m["batches"]["reviewer_A_batch_01.csv"]["review_ids"]:
        if rid in new:
            meta=Aexp[rid];r={"review_id":rid,"text":texts[rid],"source":"OpenAssistant/oasst1",
                "language":meta["language"],"token_length":str(meta["deberta_tokens"]),"review_role":"SECOND_REVIEW",
                **{k:new[rid][k] for k in p.HUMAN}}
            if "review_seconds" in ah:r["review_seconds"]=""
            origin="RECONFIRMED_AFTER_ENCODING_DAMAGE";judgment_source=p.PACKET
        else:
            r=dict(old[rid]);assert r["text"]==texts[rid]
            origin="ORIGINAL_INTACT";judgment_source=original_A
        assert all(r[k]==(new[rid] if rid in new else old[rid])[k] for k in p.HUMAN)
        canonical.append(r)
        provenance.append({"review_id":rid,"provenance":origin,"judgment_source":p.relative(judgment_source),
            "judgment_source_sha256":p.file_sha(judgment_source),"source_revision":p.OA_REV,
            "source_native_id":Aexp[rid]["source_native_id"],"source_row_index":Aexp[rid]["source_row_index"],
            "raw_text_sha256":Aexp[rid]["raw_text_sha256"],"source_text_hash_validated":True,
            "exact_human_fields_sha256":p.sha(json.dumps({k:r[k] for k in p.HUMAN},ensure_ascii=False,sort_keys=True)),
            "old_damaged_judgment_used":False if rid in new else None,
            "timing_provenance":"not_reconfirmed_left_blank" if rid in new else "preserved_original_optional_value"})
    assert not validate_rows(canonical,Aexp,"SECOND_REVIEW",True)
    buf=io.StringIO(newline="");writer=csv.DictWriter(buf,fieldnames=ah,lineterminator="\r\n")
    writer.writeheader();writer.writerows(canonical);payload=buf.getvalue().encode("utf-8-sig")
    assert p.read_csv(payload)[1]==canonical
    stamp=datetime.now(timezone.utc).isoformat()
    old_prep=preserve(p.RESULT,"20261004-v3_before_completed_reconfirmation_recovery.json")
    CANON.write_bytes(payload)
    manifest={"status":"PASS","recovered_at":stamp,"rubric_version":p.POLICY,"source_revision":p.OA_REV,
        "authorization_request_sha256":p.file_sha(REQUEST),"assignment_manifest_sha256":p.file_sha(p.EXP_MANIFEST),
        "original_damaged_A":file_entry(original_A),"Reviewer_B":file_entry(B),
        "reconfirmation_17":file_entry(p.PACKET),"canonical_A":file_entry(CANON),
        "preparation_history":old_prep,"prepared_blank_packet_sha256":prep["packet"]["sha256"],
        "pinned_source":{"path":entry["local_path"],"sha256":entry["sha256"],"revision":p.OA_REV},
        "human_review_provenance":"User explicitly confirms17 human reconfirmations; prepared packet had85 blank human cells. This pipeline copies values only from the named human submissions, never infers judgments.",
        "reconfirmation_fields_complete":17,"original_intact_rows":36,"reconfirmed_rows":17,
        "Reviewer_A_valid":53,"Reviewer_B_valid":52,"total_valid_new":105,
        "all105_source_hashes_verified_against_pinned_source":True,
        "all105_opposite_primary_reviewer":True,"all105_required_fields_complete":True,
        "row_provenance":provenance,"provenance_is_review_metadata_only":True,
        "canonical_review_seconds_policy":"Original36 optional times preserved;17 new rows have no recorded timing, remain blank.",
        "expanded180_gate_run":False,"human_judgments_inferred":False,"adjudication":False,
        "source_model_labels_used":False,"training":False,"replay":False}
    p.save(MANIFEST,manifest)
    result={"status":"PASS","integrity_status":"PASS","created_at":stamp,"A":53,"B":52,"new_second_reviews":105,
        "reconfirmation_validation":"PASS","source_hash_validation":"PASS","assignment_validation":"PASS",
        "canonical_A":file_entry(CANON),"recovery_manifest":file_entry(MANIFEST),
        "original_submissions_and_reconfirmation_preserved":True,"protected_submission_snapshots":before,
        "preparation_history":old_prep,"expanded180_gate_run":False,"training":False,"replay":False}
    p.save(p.RESULT,result)
    assert all(p.fingerprint(ROOT/rel)==fp for rel,fp in before.items())
    assert all(p.fingerprint(ROOT/rel)==fp for rel,fp in prep["protected_file_snapshots"].items())
    print(json.dumps({"integrity_status":"PASS","A":53,"B":52,"reconfirmed":17,
                     "canonical":p.relative(CANON),"manifest":p.relative(MANIFEST),"gate_run":False},indent=2))
    return 0
if __name__=="__main__":sys.exit(main())

