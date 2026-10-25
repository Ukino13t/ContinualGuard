"""Synthetic checks for fail-closed review evidence and provenance handling."""
import csv, io, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import analyze_h2_mini_production_review as a
import audit_h2_mini_post_review as p

class ReviewEvidenceTests(unittest.TestCase):
    def test_unresolved_stratum_and_safety_conflicts_are_excluded(self):
        r={"proposed_stratum":"B1"}
        accepted={"reviewer_decision":"ACCEPT_BENIGN","reviewer_stratum":"B1"}
        ordinary={"reviewer_decision":"ACCEPT_BENIGN","reviewer_stratum":"B0"}
        injection={"reviewer_decision":"REJECT_TRUE_INJECTION","reviewer_stratum":"NOT_APPLICABLE"}
        self.assertTrue(a.outcome(r,[accepted])["eligible_in_proposed_stratum"])
        self.assertFalse(a.outcome(r,[accepted,ordinary])["eligible_in_proposed_stratum"])
        self.assertFalse(a.outcome(r,[ordinary,ordinary])["eligible_in_proposed_stratum"])
        self.assertIn("safety_critical_disagreement",a.outcome(r,[accepted,injection])["exclusion_reasons"])

    def test_utf16_transport_preserves_literal_human_fields(self):
        fields=["review_id",*a.HUMAN_FIELDS]
        row=dict(zip(fields,["SYNTHETIC","ACCEPT_BENIGN","B0","  keep spaces  ","HIGH","\u4eba\u5de5\nquoted \"note\", comma","17.00"]))
        buffer=io.StringIO(newline="");w=csv.DictWriter(buffer,fieldnames=fields,lineterminator="\r\n");w.writeheader();w.writerow(row)
        with tempfile.TemporaryDirectory(dir=a.ROOT/"results") as td:
            path=Path(td)/"synthetic.csv";path.write_bytes(buffer.getvalue().encode("utf-16"))
            rr,encoding,_=a.read_csv(path,fields)
        self.assertEqual(encoding,"utf-16");self.assertEqual(rr,[row]);self.assertEqual(a.evidence_hash(rr[0]),a.evidence_hash(row))

    def test_missing_second_and_changed_text_fail_validation(self):
        fields=["review_id","text","source","source_category","language","proposed_stratum","review_role",*a.HUMAN_FIELDS]
        r={"review_id":"SYNTHETIC","source":"fixture","source_category":"","language":"en","proposed_stratum":"B0",
           "raw_text_sha256":a.sha("original synthetic text"),"review_lane":"TRAIN_CANDIDATE",
           "reviewer_assignments":[{"reviewer":"A","review_role":"PRIMARY","batch_filename":"a.csv"},
                                   {"reviewer":"B","review_role":"SECOND_REVIEW","batch_filename":"b.csv"}]}
        row={k:r.get(k,"") for k in fields}
        row.update(text="changed synthetic text",review_role="PRIMARY",reviewer_decision="ACCEPT_BENIGN",
          reviewer_stratum="B0",confidence="HIGH",review_seconds="12")
        with tempfile.TemporaryDirectory(dir=a.ROOT/"results") as td:
            with (Path(td)/"a.csv").open("w",encoding="utf-8-sig",newline="") as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(row)
            _,_,v=a.validate({"records":[r],"worksheet_columns":fields},Path(td))
        self.assertEqual(v["missing_judgments"],1)
        self.assertTrue(any("text hash mismatch" in x["error"] for x in v["issues"]))
        self.assertTrue(any(x.get("filename")=="b.csv" for x in v["issues"]))

    def test_degenerate_kappa_is_not_fake_perfect_score(self):
        self.assertIsNone(a.kappa(["ACCEPT_BENIGN"]*3,["ACCEPT_BENIGN"]*3)["value"])
        self.assertAlmostEqual(a.kappa(["a","a","b","b"],["a","b","a","b"])["value"],0)

    def test_source_local_group_collisions_and_global_content(self):
        rows=[{"review_id":"a","source":"OR","source_revision":"1","review_cohort":"B1","project_semantic_parent_group":"psp-same"},
              {"review_id":"b","source":"OA","source_revision":"2","review_cohort":"B0","project_semantic_parent_group":"psp-same"}]
        uf,hits=p.stored_group_audit(rows)
        self.assertEqual(len(uf.groups()),2);self.assertEqual(hits,[])
        for r in rows:r["raw_text_sha256"]="same-content-hash"
        uf,hits=p.stored_group_audit(rows)
        self.assertEqual(len(uf.groups()),1);self.assertTrue(hits[0]["cross_source"])

    def test_count_target_does_not_imply_review_coverage_or_C0_capacity(self):
        analysis={"by_cohort":{c:{"candidates":360 if c!="B-short" else 150,"benign_acceptance_rate":1} for c in a.COHORTS}}
        rows=[{"review_cohort":"B0","review_lane":"TRAIN_CANDIDATE","reviewer_evidence":[{}]} for _ in range(300)]
        cap=p.capacity_report(rows,analysis)
        self.assertTrue(cap["B0"]["passes_minimum_count"])
        self.assertEqual(cap["B0"]["minimum_additional_double_qualified_for_30val_60test"],90)
        self.assertEqual(cap["C0_all_four_B0_pools"]["shortage"],860)

    def test_short_cross_corpus_exact_overlap_is_detected(self):
        d={"id":"a","raw":"hash","norm":"Hi","normalized_hash":"nh","model_view":"mv","tokens":3,"template":"tp","words":{"hi"}}
        e=dict(d,id="b")
        matrix=p.np.array([[1.,0.]],dtype=p.np.float32)
        report=p.compare([d],matrix,[e],matrix)
        self.assertIn("raw_exact",report["pairs"][0]["methods"])
        self.assertIn("model_view_exact",report["pairs"][0]["methods"])
        self.assertNotIn("substring",report["pairs"][0]["methods"])
        self.assertEqual(p.compare([d],matrix,[d],matrix)["pairs"],[])

if __name__=="__main__":unittest.main()
