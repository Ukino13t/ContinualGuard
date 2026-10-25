"""Package Revision 4 ACL source and hash-only artifacts for Overleaf."""
from __future__ import annotations
import csv,hashlib,json,os,re,shutil,subprocess,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import revision3_package as r3
ROOT=Path(r"E:\ContinualGuard")
PAPER=ROOT/"paper/acl"
DATA=ROOT/"results/revision4/analysis"
DEST=PAPER/"data/revision4"
OUT=ROOT/"paper/ContinualGuard_RouteA_Revision4_Overleaf.zip"
FILES=("per_seed.csv","aggregate.csv","paired_per_seed.csv","paired_aggregate.csv",
       "stage_curves.csv","stage_acquisition.csv","memory_composition.csv")
EXTRAS={"validation.json":"results/revision4/validation.json",
        "protocol.json":"results/revision4/protocol.json",
        "selector_summary.json":"results/revision4/selector_audit/summary.json",
        "candidate_pool_audit.csv":"results/revision4/selector_audit/candidate_pool_audit.csv",
        "selected_ids.jsonl":"results/revision4/selector_audit/selected_ids.jsonl",
        "two_stage_model_parity.json":"results/revision4/two_stage_model_parity.json"}
FIGURES=("b_short_attack_forgetting.pdf","b_hard_benign_fpr.pdf",
         "stage_memory_composition.pdf","selected_buffer_overlap.pdf")
TABLES=("revision4_rq3.tex","revision4_stage_balance_pairs.tex",
        "revision4_stage_balance_seeds.tex")
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk)
 return h.hexdigest()
def main():
 v=json.loads((ROOT/"results/revision4/validation.json").read_text(encoding="utf-8"))
 assert v["status"]=="PASS" and v["full_checkpoint_hashes"] and v["runs_checked"]==9
 qa=json.loads((ROOT/"reports/revision4/visual_qa/page_audit.json").read_text(encoding="utf-8"))
 assert qa["all_word_boxes_inside_page"] and not qa["blank_text_pages"]
 assert qa["latex_error_count"]==0 and not qa["undefined_reference_or_citation_warnings"]
 assert sha(PAPER/"continualguard_acl.pdf")==qa["pdf_sha256"]
 DEST.mkdir(parents=True,exist_ok=True)
 shutil.copy2(ROOT/"scripts/revision4_paper_tables.py",PAPER/"build_revision4_tables.py")
 for name in FILES:shutil.copy2(DATA/name,DEST/name)
 for name,relative in EXTRAS.items():shutil.copy2(ROOT/relative,DEST/name)
 for name in FIGURES:shutil.copy2(ROOT/"figures/revision4"/name,PAPER/"figures"/("revision4_"+name))
 before={name:sha(PAPER/"tables"/name) for name in TABLES}
 subprocess.run([sys.executable,str(PAPER/"build_revision4_tables.py")],cwd=PAPER,check=True)
 after={name:sha(PAPER/"tables"/name) for name in TABLES}
 if before!=after:raise RuntimeError("Packaged CSV cannot regenerate identical Revision 4 tables")
 files=r3.source_files()
 files.extend(DEST/name for name in FILES)
 files.extend(DEST/name for name in EXTRAS)
 files.append(PAPER/"build_revision4_tables.py")
 files=sorted(set(files))
 for file in files:
  if not file.is_file() or not file.resolve().is_relative_to(PAPER.resolve()):
   raise RuntimeError(f"Missing or unsafe package member: {file}")
  relative=file.relative_to(PAPER).as_posix().lower()
  if any(term in relative for term in ("raw","checkpoint","predictions","reviewer","_attempt",".safetensors")):
   raise RuntimeError(f"Disallowed package path: {relative}")
  if file.suffix in (".csv",".jsonl"):
   head=file.read_text(encoding="utf-8").splitlines()[0].lower()
   if "raw_text_stored" in head:
    if file.suffix != ".csv":
     raise RuntimeError(f"Unexpected raw-text flag format: {relative}")
    with file.open(encoding="utf-8",newline="") as handle:
     if any(row["raw_text_stored"].strip().lower() != "false" for row in csv.DictReader(handle)):
      raise RuntimeError(f"Raw text may be stored: {relative}")
    head=head.replace("raw_text_stored","")
   if any(token in head for token in ("prompt_text","reviewer_note","raw_text")):
    raise RuntimeError(f"Disallowed source column: {relative}")
 temp=OUT.with_suffix(".zip.tmp")
 with zipfile.ZipFile(temp,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
  for file in files:archive.write(file,file.relative_to(PAPER).as_posix())
 with zipfile.ZipFile(temp) as archive:
  assert archive.testzip() is None
  assert len(archive.namelist())==len(set(archive.namelist()))
 os.replace(temp,OUT)
 manifest=dict(package=str(OUT),sha256=sha(OUT),bytes=OUT.stat().st_size,
  members=len(files),pdf_sha256=sha(PAPER/"continualguard_acl.pdf"),
  source_allowlist=[f.relative_to(PAPER).as_posix() for f in files],
  contents="ACL sources, vector figures, aggregate CSV/JSON, IDs and hashes only")
 (ROOT/"reports/revision4/overleaf_package_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
 print(json.dumps({k:manifest[k] for k in ("package","sha256","bytes","members")}))
if __name__=="__main__":main()

