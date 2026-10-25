"""Synthetic expansion tests; never alter real review evidence."""
import csv,json,sys,tempfile,unittest
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from h2_mini_b0_expansion_common import *
from h2_mini_b0_review_common import assign_reviews
import analyze_h2_mini_b0_expansion as analyzer
class ExpansionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="synthetic_b0_expansion_",dir=ROOT/"results")
        self.root=Path(self.tmp.name);self.directory=self.root/"worksheets";self.directory.mkdir()
        candidates=[{"candidate_id":f"synthetic-{i}","source":"OASST1","source_revision":OA_REV,
          "language":["en","es","zh"][i%3],"deberta_tokens":10+i%55,"source_row_index":i,
          "source_native_id":f"native-{i}","source_parent_id":f"tree-{i}","raw_text_sha256":sha(f"Synthetic ordinary question {i}.")}
          for i in range(872)]
        self.records,_=assign_reviews(candidates);self.texts={r["review_id"]:f"Synthetic ordinary question {r['source_row_index']}." for r in self.records}
        self.states={}
        for r in self.records:
            for a in r["reviewer_assignments"]:
                a["batch_filename"]=f"old_{a['reviewer']}.csv"
                self.states[(a["reviewer"],r["review_id"])]={"status":"completed","values":self.valid()}
        self.old=old_audit_records(self.records)
        for r in self.old[:8]:self.states[("B",r["review_id"])]["values"]["b0_confirmed"]="NO"
        self.selected,self.frame=expansion_selection(self.records)
        self.baseline=({"manifest":{"records":self.records,"old_B0_pool":{"counts":{"training":234,"training_audited":20,"evaluation":105}}},
          "states":self.states},{},{})
        self.m={"package_id":EXP_PACKAGE,"frozen_dependencies":{},"records":self.selected,
          "old_audited_training_ids":[r["review_id"] for r in self.old],"worksheet_columns":EXP_FIELDS,
          "new_second_review_assignments":105,"batches":{}}
        for who in ["A","B"]:
            rr=[r for r in self.selected if r["second_reviewer"]==who];fn=f"reviewer_{who}_batch_01.csv"
            rows=[{"review_id":r["review_id"],"text":self.texts[r["review_id"]],"source":"OpenAssistant/oasst1","language":r["language"],
              "token_length":str(r["deberta_tokens"]),"review_role":"SECOND_REVIEW",**{f:"" for f in EXP_HUMAN}} for r in rr]
            self.write(fn,rows)
            self.m["batches"][fn]={"reviewer":who,"rows":len(rows),"review_ids":[r["review_id"] for r in rr],"prepared_sha256":file_sha(self.directory/fn)}
        self.path=self.root/"manifest.json";save(self.path,self.m)
    def tearDown(self):
        self.assertTrue(self.root.resolve().is_relative_to((ROOT/"results").resolve()));self.tmp.cleanup()
    def valid(self,**kw):return {"reviewer_decision":"ACCEPT_BENIGN","b0_confirmed":"YES","reason_code":"","confidence":"HIGH","optional_note":"",**kw}
    def write(self,fn,rows,optional=False):
        with (self.directory/fn).open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=EXP_FIELDS+(["review_seconds"] if optional else []));w.writeheader();w.writerows(rows)
    def fill(self,changes=None,all_complete=False):
        for fn in self.m["batches"]:
            with (self.directory/fn).open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
            for row in rows:
                if all_complete:row.update(self.valid())
                if changes and row["review_id"] in changes:row.update(changes[row["review_id"]])
            optional=any("review_seconds" in row for row in rows);self.write(fn,rows,optional)
    def inspect(self):return analyzer.inspect_expansion(self.path,self.directory,self.root,self.baseline)
    def test_selection_uses_metadata_not_answers(self):
        rr=json.loads(json.dumps(self.records))
        for r in rr:
            r.update(reviewer_decision="REJECT_TRUE_INJECTION",b0_confirmed="NO",source_status_verified={"source_review_result":False},
              detector_prediction=1,optional_note="Do not prefer this synthetic candidate")
        selected,_=expansion_selection(list(reversed(rr)))
        self.assertEqual(selected,self.selected)
        self.assertTrue(all("reviewer_decision" not in r and "detector_prediction" not in r for r in selected))
    def test_roles_and_old75_are_preserved(self):
        self.assertEqual(len(self.frame),525);self.assertEqual(Counter(r["second_reviewer"] for r in self.selected),Counter(A=53,B=52))
        self.assertTrue(all(r["second_reviewer"]!=r["primary_reviewer"] for r in self.selected))
        self.assertFalse({r["review_id"] for r in self.old}&{r["review_id"] for r in self.selected})
    def test_blank_package_cannot_pass_gate(self):
        d=self.inspect();self.assertFalse(d["errors"])
        self.assertEqual(len(d["states"]),105);self.assertTrue(all(s["status"]=="pending" for s in d["states"].values()))
        out=analyzer.analyze_expansion(d)
        self.assertEqual(out["status"],"INCOMPLETE_EXPANDED_REVIEW")
        self.assertIsNone(out["expanded_audit"]["B0_confirmation_gate_pass"])
        self.assertIsNone(out["conditional_label_only_capacity"])
    def test_full_denominator_includes_eight_old_disagreements(self):
        self.fill(all_complete=True);out=analyzer.analyze_expansion(self.inspect());audit=out["expanded_audit"]
        self.assertEqual(audit["observed_pairs"],180);self.assertEqual(audit["old_pairs_retained"],75)
        self.assertEqual(audit["full_audit_metrics"]["B0_confirmation"]["disagreements"],8)
        self.assertEqual(audit["full_audit_metrics"]["B0_confirmation"]["disagreement_rate"],8/180)
        self.assertTrue(audit["B0_confirmation_gate_pass"]);self.assertEqual(out["total_complete_judgments"],1324)
        self.assertIsNone(out["optional_timing"]["person_hours"])
        self.assertFalse(out["final_pool_merge_performed"])
    def test_eighteen_pass_nineteen_fail_without_dropping_records(self):
        ids=[r["review_id"] for r in self.selected]
        self.fill({rid:{"b0_confirmed":"NO"} for rid in ids[:10]},all_complete=True)
        out=analyzer.analyze_expansion(self.inspect());self.assertTrue(out["expanded_audit"]["B0_confirmation_gate_pass"])
        self.assertEqual(out["expanded_audit"]["full_audit_metrics"]["B0_confirmation"]["disagreements"],18)
        self.fill({ids[10]:{"b0_confirmed":"NO"}})
        out=analyzer.analyze_expansion(self.inspect());self.assertFalse(out["expanded_audit"]["B0_confirmation_gate_pass"])
        self.assertEqual(out["expanded_audit"]["full_audit_metrics"]["B0_confirmation"]["disagreements"],19)
        self.assertEqual(out["expanded_audit"]["observed_pairs"],180)
    def test_changed_text_is_blocked_before_metrics(self):
        self.fill({self.selected[0]["review_id"]:{"text":"Changed evidence"}},all_complete=True)
        d=self.inspect();self.assertTrue(any(e["message"]=="Immutable source text changed" for e in d["errors"]))
        out=analyzer.analyze_expansion(d);self.assertEqual(out["status"],"INVALID_EXPANSION_INTEGRITY_OR_FIELDS")
        self.assertIsNone(out["conditional_label_only_capacity"])
    def test_old_disagreement_cannot_be_removed_from_manifest(self):
        self.m["old_audited_training_ids"].pop(0);save(self.path,self.m)
        self.assertTrue(any(e["message"]=="Original75-pair denominator changed" for e in self.inspect()["errors"]))
    def test_rejection_requires_reason_but_no_timing(self):
        rid=self.selected[0]["review_id"]
        self.fill({rid:self.valid(reviewer_decision="REJECT_QUALITY",b0_confirmed="NO")},all_complete=True)
        d=self.inspect();self.assertEqual(d["states"][rid]["missing"],["reason_code"])
        self.assertEqual(analyzer.analyze_expansion(d)["completed_new_judgments"],104)
        self.fill({rid:{"reason_code":"QUALITY"}})
        self.assertEqual(analyzer.analyze_expansion(self.inspect())["completed_new_judgments"],105)
    def test_optional_timing_only_reports_actual_values(self):
        rid=self.selected[0]["review_id"];self.fill({rid:{"review_seconds":"31"}},all_complete=True)
        out=analyzer.analyze_expansion(self.inspect())
        self.assertEqual(out["completed_new_judgments"],105)
        self.assertEqual(out["optional_timing"]["completed_judgments"],1);self.assertEqual(out["optional_timing"]["seconds"],31)
    def test_unfinished_expansion_never_decides_gate(self):
        pairs=[(self.valid(),self.valid(b0_confirmed="NO"))]*8+[(self.valid(),self.valid())]*67
        out=summarize_pairs(pairs,[(self.valid(),self.valid())]*104)
        self.assertEqual(out["observed_pairs"],179);self.assertIsNone(out["B0_confirmation_gate_pass"])
if __name__=="__main__":unittest.main()
