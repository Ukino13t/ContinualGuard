"""Synthetic tests: no source data or human review evidence is modified."""
import csv,json,sys,tempfile,unittest
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from h2_mini_b0_review_common import *
import analyze_h2_mini_b0_replenishment as analyzer

class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="synthetic_b0_review_",dir=ROOT/"results")
        self.root=Path(self.tmp.name);self.directory=self.root/"review";self.directory.mkdir()
        base=[{"candidate_id":f"synthetic-{i}","language":["en","es","zh"][i%3],
          "deberta_tokens":10+i%40,"source":"OASST1","source_revision":OA_REV,
          "raw_text_sha256":sha(f"Synthetic ordinary question {i}."),"synthetic_text":f"Synthetic ordinary question {i}."}
          for i in range(872)]
        self.records,self.early=assign_reviews(base)
        self.m={"package_id":PACKAGE,"taxonomy_policy":POLICY,"candidate_source_revision":OA_REV,
          "worksheet_columns":FIELDS,"human_fields":HUMAN,"expected_judgments":1219,"records":self.records,"batches":{}}
        for pk,hk in [("taxonomy_policy_path","taxonomy_policy_sha256"),("candidate_pool_path","candidate_pool_sha256"),("audit_path","audit_sha256")]:
            p=self.root/(pk+".json");save(p,{"synthetic":True});self.m[pk]=p.name;self.m[hk]=file_sha(p)
        old=[{"review_lane":"TRAIN_CANDIDATE","reviewer_evidence":[{},{}] if i<20 else [{}]} for i in range(234)]
        old += [{"review_lane":"VALIDATION_RESERVE" if i<33 else "TEST_RESERVE","reviewer_evidence":[{},{}]} for i in range(105)]
        p=self.root/"old.json";save(p,{"records":old})
        self.m["old_B0_pool"]={"path":p.name,"sha256":file_sha(p),"counts":analyzer.snapshot_counts(old)}
        for w,total in [("A",609),("B",610)]:
            entries=[(r,a) for r in self.records for a in r["reviewer_assignments"] if a["reviewer"]==w];offset=0
            for number,size in enumerate(batch_sizes(total),1):
                fn=f"reviewer_{w}_batch_{number:02d}.csv";rows=[]
                for r,a in entries[offset:offset+size]:
                    a["batch_filename"]=fn
                    rows.append({"review_id":r["review_id"],"text":r["synthetic_text"],"source":"OpenAssistant/oasst1",
                      "language":r["language"],"token_length":str(r["deberta_tokens"]),"review_role":a["review_role"],**{k:"" for k in HUMAN}})
                self.write(fn,rows);self.m["batches"][fn]={"reviewer":w,"rows":size,"review_ids":[r["review_id"] for r in rows],
                  "prepared_sha256":file_sha(self.directory/fn)}
                offset+=size
        self.manifest=self.root/"manifest.json";save(self.manifest,self.m)
    def tearDown(self):
        self.assertTrue(self.root.resolve().is_relative_to((ROOT/"results").resolve()))
        self.tmp.cleanup()
    def write(self,fn,rows):
        with (self.directory/fn).open("w",encoding="utf-8-sig",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
    def change(self,changes):
        for fn in self.m["batches"]:
            with (self.directory/fn).open(encoding="utf-8-sig",newline="") as f:rr=list(csv.DictReader(f))
            who=self.m["batches"][fn]["reviewer"]
            for r in rr:
                if (who,r["review_id"]) in changes:r.update(changes[(who,r["review_id"])])
            self.write(fn,rr)
    def inspect(self):return analyzer.inspect(self.manifest,self.directory,self.root)
    def valid(self,**updates):
        return {"reviewer_decision":"ACCEPT_BENIGN","b0_confirmed":"YES","reason_code":"","confidence":"HIGH","optional_note":"","review_seconds":"12",**updates}
    def audited(self):return next(r for r in self.records if r["review_lane"]=="TRAIN_CANDIDATE" and len(r["reviewer_assignments"])==2)
    def test_assignment_contract_and_determinism(self):
        roles=Counter(a["review_role"] for r in self.records for a in r["reviewer_assignments"])
        self.assertEqual(roles,Counter(PRIMARY=600,SECOND_REVIEW=75,DOUBLE_REVIEW=544));self.assertEqual(len(self.early),32)
        self.assertEqual(Counter(a["reviewer"] for r in self.records for a in r["reviewer_assignments"]),Counter(A=609,B=610))
        rr,early=assign_reviews(list(reversed(self.records)))
        mapping=lambda records:{r["candidate_id"]:[(a["reviewer"],a["review_role"]) for a in r["reviewer_assignments"]] for r in records}
        self.assertEqual(mapping(rr),mapping(self.records));self.assertEqual(early,self.early)
        self.assertEqual(batch_sizes(609),[100,85,85,85,85,85,84])
    def test_blank_is_pending_never_accepted(self):
        d=self.inspect();self.assertEqual(d["errors"],[])
        r=analyzer.analyze(d);self.assertEqual(r["completed_judgments"],0);self.assertEqual(r["capacity"]["new_accepted_unique"],0)
        self.assertEqual(r["status"],"INCOMPLETE_HUMAN_REVIEW");self.assertEqual(r["capacity"]["total_B0"],339)
    def test_missing_second_does_not_become_accepted(self):
        r=self.audited();a=next(a for a in r["reviewer_assignments"] if a["review_role"]=="PRIMARY")
        self.change({(a["reviewer"],r["review_id"]):self.valid()})
        out=analyzer.analyze(self.inspect(),interim=True)
        self.assertEqual(out["training_primary_acceptance"]["ACCEPT_BENIGN"],1)
        self.assertEqual(out["capacity"]["new_accepted_unique"],0);self.assertEqual(out["second_review_agreement"]["decision"]["completed_pairs"],0)
    def test_decision_and_confirmation_are_separate(self):
        r=next(r for r in self.records if r["review_lane"]=="EVALUATION_RESERVE")
        self.change({("A",r["review_id"]):self.valid(),("B",r["review_id"]):self.valid(b0_confirmed="NO")})
        out=analyzer.analyze(self.inspect(),interim=True)
        self.assertEqual(out["double_review_agreement"]["decision"]["agreement_rate"],1)
        self.assertEqual(out["double_review_agreement"]["B0_confirmation"]["disagreement_rate"],1)
        self.assertEqual(out["capacity"]["new_accepted_unique"],0)
    def test_rejection_is_never_rescued_and_gate_stops(self):
        r=self.audited()
        self.change({("A",r["review_id"]):self.valid(),("B",r["review_id"]):self.valid(reviewer_decision="REJECT_TRUE_INJECTION",b0_confirmed="NO",reason_code="INJECTION")})
        out=analyzer.analyze(self.inspect(),interim=True)
        self.assertEqual(out["capacity"]["new_accepted_unique"],0);self.assertTrue(out["training_audit_gate"]["exceeded"])
        self.assertEqual(out["second_review_agreement"]["accept_vs_true_injection"],1)
    def test_full_synthetic_capacity_without_assignment(self):
        self.change({(a["reviewer"],r["review_id"]):self.valid() for r in self.records for a in r["reviewer_assignments"]})
        d=self.inspect();self.assertEqual(d["errors"],[])
        out=analyzer.analyze(d);c=out["capacity"]
        self.assertTrue(out["all_required_judgments_complete"]);self.assertEqual(out["completed_judgments"],1219)
        self.assertEqual((c["total_B0"],c["training_lane_total"],c["independently_audited_training_total"],c["double_evaluation_reserve_total"]),(1211,834,95,377))
        self.assertTrue(c["joint_lane_preserving_feasible"]);self.assertFalse(out["final_IDs_selected"])
        self.assertFalse(out["scientific_finalization_authorized"])
        self.assertTrue(all(r["final_assignment"] is None for r in out["record_outcomes_metadata_only"]))
    def test_shared_pool_cannot_fill_both_eval_quotas(self):
        c=coverage([],{"training":900,"training_audited":90,"evaluation":250})
        self.assertTrue(c["validation_120_capacity_available"]);self.assertTrue(c["test_240_capacity_available"])
        self.assertFalse(c["validation120_and_test240_jointly_available"]);self.assertFalse(c["joint_lane_preserving_feasible"])
        self.assertEqual(c["lane_shortages"]["double_evaluation"],110)
    def test_changed_text_and_role_are_detected(self):
        r=self.records[0];w=r["reviewer_assignments"][0]["reviewer"]
        self.change({(w,r["review_id"]):{"text":"Changed synthetic evidence","review_role":"UNKNOWN"}})
        messages=[e["message"] for e in self.inspect()["errors"]]
        self.assertIn("Source text changed",messages);self.assertIn("Immutable metadata changed",messages)
    def test_missing_row_and_duplicate_are_detected(self):
        fn=next(iter(self.m["batches"]))
        with (self.directory/fn).open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
        rows[-1]=dict(rows[0]);self.write(fn,rows)
        messages=[e["message"] for e in self.inspect()["errors"]]
        self.assertIn("Duplicate reviewer judgment row",messages);self.assertIn("Missing assigned judgment row",messages)
    def test_partial_and_invalid_time_are_not_completed(self):
        a,b=self.records[:2];wa=a["reviewer_assignments"][0]["reviewer"];wb=b["reviewer_assignments"][0]["reviewer"]
        self.change({(wa,a["review_id"]):{"reviewer_decision":"ACCEPT_BENIGN"},
          (wb,b["review_id"]):self.valid(review_seconds="nan")})
        d=self.inspect();self.assertEqual(d["state_counts"].get("partial"),1);self.assertEqual(d["state_counts"].get("invalid"),1)
        out=analyzer.analyze(d,interim=True);self.assertEqual(out["completed_judgments"],0);self.assertEqual(out["status"],"INVALID_INTEGRITY_OR_FIELDS")
    def test_completed_labels_without_timing_stop_without_zero_hours(self):
        self.change({(a["reviewer"],r["review_id"]):self.valid(review_seconds="") for r in self.records for a in r["reviewer_assignments"]})
        out=analyzer.analyze(self.inspect())
        self.assertEqual(out["status"],"INCOMPLETE_HUMAN_REVIEW")
        self.assertEqual(out["completeness"]["populated_human_fields"]["reviewer_decision"],1219)
        self.assertEqual(out["completeness"]["missing_fields"],{"review_seconds":1219})
        self.assertEqual(out["completed_judgments"],0)
        self.assertIsNone(out["timing"]["person_hours"]);self.assertIsNone(out["timing"]["p90_seconds"])
        self.assertFalse(out["final_capacity_assessment"]["performed"])
        self.assertIsNone(out["final_capacity_assessment"]["final_capacity_pass"])
    def test_timing_p90_uses_declared_interpolation(self):
        summary=analyzer.times([10,20,30,40,50])
        self.assertEqual(summary["p90_seconds"],46)
        self.assertEqual(summary["seconds"],150)
    def test_integrity_failure_cannot_create_eligible_records(self):
        r=next(r for r in self.records if len(r["reviewer_assignments"])==1)
        w=r["reviewer_assignments"][0]["reviewer"]
        self.change({(w,r["review_id"]):dict(self.valid(),text="Altered synthetic text")})
        out=analyzer.analyze(self.inspect())
        self.assertEqual(out["status"],"INVALID_INTEGRITY_OR_FIELDS")
        self.assertEqual(out["capacity"]["new_accepted_unique"],0)
    def test_ai_preliminary_status_still_blocks_after_all_timings_filled(self):
        changes={(a["reviewer"],r["review_id"]):self.valid() for r in self.records for a in r["reviewer_assignments"]}
        k=next(k for k in changes if k[0]=="A")
        changes[k]["optional_note"]="AI\u9884\u5ba1\u5efa\u8bae; independent human review pending"
        self.change(changes);out=analyzer.analyze(self.inspect())
        self.assertEqual(out["completed_judgments"],1219)
        self.assertEqual(out["status"],"STOP_UNVERIFIED_HUMAN_REVIEW")
        self.assertEqual(out["review_provenance_gate"]["blocking_flag_count"],1)
        self.assertEqual(out["capacity"]["new_accepted_unique"],0)
        self.assertFalse(out["final_capacity_assessment"]["performed"])
    def test_ai_topic_in_notes_is_not_a_provenance_flag(self):
        r=self.records[0];w=r["reviewer_assignments"][0]["reviewer"]
        self.change({(w,r["review_id"]):self.valid(optional_note="An ordinary question about an AI system.")})
        self.assertEqual(self.inspect()["provenance_flags"],[])
    def make_confirmation(self,data):
        obj={"package_id":data["manifest"]["package_id"],"assignment_manifest_sha256":data["manifest_sha256"],
          "submitted_worksheet_sha256":data["hashes"],"human_statement_source":"CURRENT_USER_MESSAGE",
          "timing_clarification":{"evidence_type":"HUMAN_REPORTED_AGGREGATE_MEAN","per_record_imputation":False,"mean_seconds_per_judgment":25,"judgments_in_scope":1219},
          "reviewer_A_confirmation":{"manual_review_completed":True,"notes_are_stale":True,"stale_note_sha256":list({f["note_sha256"] for f in data["provenance_flags"]})}}
        p=self.root/"synthetic_confirmation.json";save(p,obj);return p
    def test_user_aggregate_confirmation_never_fills_row_seconds(self):
        self.change({(a["reviewer"],r["review_id"]):self.valid(review_seconds="") for r in self.records for a in r["reviewer_assignments"]})
        data=self.inspect();p=self.make_confirmation(data);hashes=dict(data["hashes"])
        data=analyzer.apply_human_confirmation(data,p,self.root);out=analyzer.analyze(data)
        self.assertEqual(out["completed_judgments"],1219)
        self.assertTrue(all(s["values"]["review_seconds"]=="" and s["seconds"] is None for s in data["states"].values()))
        self.assertEqual(self.inspect()["hashes"],hashes)
        self.assertEqual(out["timing"]["human_reported_aggregate"]["estimated_total_person_hours"],1219*25/3600)
        self.assertIsNone(out["timing"]["median_seconds"]);self.assertIsNone(out["timing"]["human_reported_aggregate"]["p90_seconds"])
    def test_user_confirmation_is_bound_to_exact_submissions(self):
        self.change({(a["reviewer"],r["review_id"]):self.valid(review_seconds="") for r in self.records for a in r["reviewer_assignments"]})
        data=self.inspect();p=self.make_confirmation(data)
        r=self.records[0];w=r["reviewer_assignments"][0]["reviewer"]
        self.change({(w,r["review_id"]):{"confidence":"LOW"}})
        with self.assertRaisesRegex(ValueError,"exact submitted files"):analyzer.apply_human_confirmation(self.inspect(),p,self.root)
    def test_user_confirmation_cannot_fill_missing_label_field(self):
        self.change({(a["reviewer"],r["review_id"]):self.valid(review_seconds="",confidence="") for r in self.records for a in r["reviewer_assignments"]})
        data=self.inspect();p=self.make_confirmation(data)
        with self.assertRaisesRegex(ValueError,"another missing judgment field"):analyzer.apply_human_confirmation(data,p,self.root)
    def test_aggregate_timing_keeps_eight_of_seventyfive_disagreement_stop(self):
        changes={(a["reviewer"],r["review_id"]):self.valid(review_seconds="") for r in self.records for a in r["reviewer_assignments"]}
        audit=[r for r in self.records if r["review_lane"]=="TRAIN_CANDIDATE" and len(r["reviewer_assignments"])==2]
        for r in audit[:8]:changes[("B",r["review_id"])]["b0_confirmed"]="NO"
        k=next(k for k in changes if k[0]=="A");changes[k]["optional_note"]="AI\u9884\u5ba1\u5efa\u8bae"
        self.change(changes);data=self.inspect();p=self.make_confirmation(data)
        out=analyzer.analyze(analyzer.apply_human_confirmation(data,p,self.root))
        self.assertEqual(out["completed_judgments"],1219);self.assertEqual(out["status"],"STOP_TRAINING_AUDIT_DISAGREEMENT")
        self.assertEqual(out["second_review_agreement"]["B0_confirmation"]["disagreement_rate"],8/75)
        self.assertEqual(out["review_provenance_gate"]["status"],"RESOLVED_BY_CURRENT_USER_CONFIRMATION")
        self.assertFalse(out["final_capacity_assessment"]["performed"])
    def test_historical_evidence_tampering_detected(self):
        save(self.root/"old.json",{"records":[]})
        self.assertTrue(any("Historical B0 evidence" in e["message"] for e in self.inspect()["errors"]))
    def test_encoding_failure_is_reported(self):
        fn=next(iter(self.m["batches"]));(self.directory/fn).write_bytes(b"\xff\xff")
        self.assertTrue(any(e["message"]=="UTF-8/CSV decoding failed" for e in self.inspect()["errors"]))
if __name__=="__main__":unittest.main()
