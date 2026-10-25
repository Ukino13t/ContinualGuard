"""Validate recovered expansion provenance before using the canonical A submission."""
from h2_mini_b0_review_common import ROOT,load,file_sha,sha
from recover_h2_mini_b0_expansion import validate_rows
import prepare_h2_mini_b0_encoding_reconfirmation as p

def resolve_recovered(m,root=ROOT,directory=None):
    directory=directory or root/"review/b0_replenishment_expansion"
    path=root/"review/b0_replenishment_expansion_integrity_recovery_manifest.json"
    if not path.exists():return {},set(),None
    r=load(path)
    if r["status"]!="PASS":raise ValueError("Recovery manifest is not PASS")
    if file_sha(root/"review/b0_replenishment_expansion_manifest.json")!=r["assignment_manifest_sha256"]:
        raise ValueError("Assignment manifest differs from recovery evidence")
    allowed={
        "original_damaged_A":"reviewer_A_batch_01.csv","Reviewer_B":"reviewer_B_batch_01.csv",
        "reconfirmation_17":"reviewer_A_encoding_reconfirmation_17.csv",
        "canonical_A":"reviewer_A_batch_01_canonical_utf8.csv"}
    files={}
    for key,name in allowed.items():
        target=directory/name
        if (root/r[key]["path"]).resolve()!=target.resolve() or file_sha(target)!=r[key]["sha256"]:
            raise ValueError("Recovery file path/hash mismatch: "+key)
        files[key]=target
    _,old=p.read_csv(files["original_damaged_A"].read_bytes(),"gb18030")
    _,new=p.read_csv(files["reconfirmation_17"].read_bytes())
    _,canon=p.read_csv(files["canonical_A"].read_bytes())
    old={x["review_id"]:x for x in old};new={x["review_id"]:x for x in new}
    expected={x["review_id"]:x for x in m["records"] if x["second_reviewer"]=="A"}
    problems=validate_rows(canon,expected,"SECOND_REVIEW",True)
    recon_expected={rid:expected[rid] for rid in new}
    problems+=validate_rows(list(new.values()),recon_expected,"SECOND_REVIEW_RECONFIRMATION",False)
    if problems:raise ValueError("Recovery row validation: "+str(problems))
    lineage={x["review_id"]:x for x in r["row_provenance"]}
    if len(canon)!=53 or len(new)!=17 or set(lineage)!=set(expected):
        raise ValueError("Recovery provenance counts/IDs differ")
    for row in canon:
        rid=row["review_id"];reconfirmed=rid in new;source=new[rid] if reconfirmed else old[rid]
        expected_origin="RECONFIRMED_AFTER_ENCODING_DAMAGE" if reconfirmed else "ORIGINAL_INTACT"
        if lineage[rid]["provenance"]!=expected_origin:raise ValueError("Wrong provenance "+rid)
        if any(row[k]!=source[k] for k in p.HUMAN):raise ValueError("Human field differs from named source "+rid)
        if sha(row["text"])!=lineage[rid]["raw_text_sha256"]:raise ValueError("Provenance text mismatch "+rid)
        if not reconfirmed and row!=old[rid]:raise ValueError("Original intact row altered "+rid)
        if reconfirmed and row.get("review_seconds","")!="":raise ValueError("Reconfirmed timing was imputed "+rid)
    return {"reviewer_A_batch_01.csv":files["canonical_A"]},set(allowed.values())-{"reviewer_A_batch_01.csv","reviewer_B_batch_01.csv"},{
        "path":str(path.relative_to(root)).replace("\\","/"),"sha256":file_sha(path),"status":"PASS"}
