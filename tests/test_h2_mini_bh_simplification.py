"""Synthetic tests for proposed BH eligibility and count-only balancing."""
import sys,unittest,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import audit_h2_mini_bh_simplification as bh

class SimplifiedTaxonomyTests(unittest.TestCase):
    def test_bh_ignores_stratum_but_never_overrides_rejection(self):
        a={"reviewer_decision":"ACCEPT_BENIGN","reviewer_stratum":"B0"}
        b={"reviewer_decision":"ACCEPT_BENIGN","reviewer_stratum":"B3"}
        self.assertTrue(bh.benign_eligible([a,b]))
        self.assertTrue(bh.benign_eligible([a]))
        self.assertFalse(bh.benign_eligible([]))
        for code in ["REJECT_TRUE_INJECTION","REJECT_AMBIGUOUS_INTENT","REJECT_HARMFUL_OR_UNSAFE","REJECT_CONTEXT_MISSING","REJECT_PRIVACY_OR_RIGHTS","REJECT_QUALITY"]:
            self.assertFalse(bh.benign_eligible([a,dict(b,reviewer_decision=code)]))

    def rows(self):
        out=[]
        for lane,n,double in [("TRAIN_CANDIDATE",60,True),("TRAIN_CANDIDATE",540,False),("VALIDATION_RESERVE",90,True),("TEST_RESERVE",180,True)]:
            for i in range(n):
                out.append({"review_id":"SYNTHETIC_PRIVATE_ID_"+str(len(out)),"review_lane":lane,
                  "reviewer_evidence":[{},{}] if double else [{}],"source_category":"category-"+str(i%10),
                  "deberta_tokens":20 if i%2 else 40,"language":"en","proposed_stratum":["B1","B2","B3"][i%3],
                  "cue_family_ids":["C01" if i%2 else "C04"]})
        return out

    def test_no_double_counting_training_audit_as_evaluation_reserve(self):
        c=bh.capacities(self.rows())
        self.assertEqual(c["training_lane"],600)
        self.assertEqual(c["all_double_reviewed"],330)
        self.assertEqual(c["validation_test_reserve_double"],270)
        self.assertEqual(c["training_audited_double"],60)

    def test_anonymous_witness_is_balanced_and_contains_no_record_mapping(self):
        rows=self.rows();w=bh.histogram_witness(rows)
        self.assertEqual(w["status"],"ANONYMOUS_METADATA_COUNT_WITNESS_ONLY")
        for count in w["stage_split_counts"].values():self.assertEqual(count,{"train":200,"validation":30,"test":60})
        self.assertTrue(all(v["metadata_bucket_capacity_respected"] for v in w["allocations"].values()))
        self.assertNotIn("SYNTHETIC_PRIVATE_ID",json.dumps(w))
        self.assertEqual(w,bh.histogram_witness(list(reversed(rows))))

    def test_primary_only_cannot_fill_missing_validation_coverage(self):
        rows=self.rows()
        for r in rows:
            if r["review_lane"]=="VALIDATION_RESERVE":
                r["review_lane"]="TRAIN_CANDIDATE";r["reviewer_evidence"]=[{}]
        self.assertNotEqual(bh.histogram_witness(rows)["status"],"ANONYMOUS_METADATA_COUNT_WITNESS_ONLY")

if __name__=="__main__":unittest.main()
