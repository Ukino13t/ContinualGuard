"""Document approved expansion preparation and preserve historical review evidence."""
import json,sys
from datetime import datetime,timezone
from h2_mini_b0_expansion_common import *
import analyze_h2_mini_b0_expansion as analyzer
MARKER="H2_MINI_B0_EXPANSION_PREPARED_20261004"
def prepend(name,body):
    p=ROOT/"docs"/name;s=p.read_text(encoding="utf-8");assert MARKER not in s
    title,rest=s.split("\n",1)
    p.write_text(title+f"\n\n<!-- {MARKER}:BEGIN -->\n"+body+f"\n<!-- {MARKER}:END -->\n"+rest,encoding="utf-8")
def append(path,event):
    rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert not any(r["id"]==event["id"] for r in rows)
    with path.open("a",encoding="utf-8",newline="\n") as f:f.write(json.dumps(event,ensure_ascii=False)+"\n")
def main():
    m=load(EXP_MANIFEST);reserve=load(RESERVE);audit=load(RESERVE_AUDIT);d=analyzer.inspect_expansion()
    assert not d["errors"] and len(d["states"])==105 and all(s["status"]=="pending" for s in d["states"].values())
    assert all(t["unchanged_prepared_bytes"] for t in d["transport"].values())
    assert len(reserve["records"])==65 and not reserve["human_review_worksheets_created"]
    # This one document is explicitly authorized for update by the current request.
    # Preserve its previous exact bytes so earlier audit references remain recoverable.
    amendment=m["authorized_historical_document_update"]
    doc=ROOT/amendment["path"];copy=ROOT/amendment["preserve_prior_copy_at"]
    assert file_sha(doc)==amendment["old_sha256"] and not copy.exists()
    copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(doc.read_bytes())
    assert file_sha(copy)==amendment["old_sha256"]
    guide=ROOT/"review/B0扩审人工审核说明.md"
    guide.write_text("""# B0 扩审人工审核说明

本轮只做105条新增的独立第二次审核，沿用 h2-mini-benign-taxonomy-v2 的普通B0标准，不区分B1/B2/B3。

- A只填写 b0_replenishment_expansion/reviewer_A_batch_01.csv（53条）。
- B只填写 b0_replenishment_expansion/reviewer_B_batch_01.csv（52条）。
- 只看自己的新增表和原文，不查看或讨论首审答案。不要打开旧审核表来参照答案；表中没有提供首审决定、确认、置信度或备注。
- 不重审原有75条，不修改原始已提交文件，不替换或删除分歧记录。
- 不审核65条补量储备；本轮没有为储备生成审核表。
- UTF-8导入CSV，所有列按文本读取，避免自动公式、日期和数字转换。保留原文、ID、来源、语言、长度、SECOND_REVIEW角色和行数。
- 每条只填写 reviewer_decision、b0_confirmed、reason_code、confidence、optional_note。空白标签不能由程序推断。

接受条件保持不变：明确良性、普通短形式用户请求、独立可理解、无提示注入、无范围内有害内容、无隐私/权利问题、无缺失上下文且质量可用。不懂语言或不能可靠判断时先联系协调人，不猜测、不用模型输出代替判断。

允许的 reviewer_decision：
ACCEPT_BENIGN、REJECT_TRUE_INJECTION、REJECT_AMBIGUOUS_INTENT、REJECT_HARMFUL_OR_UNSAFE、REJECT_CONTEXT_MISSING、REJECT_PRIVACY_OR_RIGHTS、REJECT_QUALITY。

b0_confirmed 填 YES / NO / UNSURE。ACCEPT_BENIGN只有同时B0=YES才可能进入普通B0池；明确良性但不是普通B0时如实填NO或UNSURE。不要为了补齐数量而改变判断。

拒绝项必须填写 reason_code，建议分别用 INJECTION、AMBIGUOUS_INTENT、HARMFUL_OR_UNSAFE、CONTEXT_MISSING、PRIVACY_OR_RIGHTS、QUALITY。confidence 填 HIGH / MEDIUM / LOW；optional_note 可留空。

本轮不要求逐条计时。原表没有 review_seconds 列；若自愿记录，可在末尾添加同名列，只填实际计时，空值允许。不要把历史25秒均值批量填入新表。

两位均完成全部105条后通知协调人。预计合计43分45秒（约0.729人时），仅按历史25秒均值规划，不是本轮实测耗时。

协调人在 E:\\ContinualGuard 运行：

~~~powershell
& .\\.venv\\Scripts\\python.exe .\\scripts\\analyze_h2_mini_b0_expansion.py
~~~

分析必须保留原75对，加上新增105对，使用完整180对分母；只看新105条的结果不能替代完整审核。未完成全部105条时不给出门槛通过结论。程序不会改写原始判断、裁决分歧、开展储备审核、生成最终分配或训练。
""",encoding="utf-8")
    head="""## Approved expanded independent review — prepared2026-10-04

The human accepted the1219 completed judgments, A609/B610, aggregate25s timing estimate (approximately8.47 person-hours), and the reported agreement/conflict counts. The frozen10% gate is unchanged. Current75-pair training B0-confirmation disagreement remains8/75=10.67%; no old disagreement is deleted or relabeled.

**Exactly105 additional independent SECOND_REVIEW assignments are prepared, with blank human fields.** A receives53 and B52, one manageable batch each. Each reviewer sees records originally primary-reviewed by the other. After completion, the audit will contain75 original +105 new =180/600 training candidates, exactly30% coverage; cumulative workload will be662 judgments per reviewer. Coverage is currently75/600 until the105 new judgments are actually completed.

- [A batch01 —53 judgments](../review/b0_replenishment_expansion/reviewer_A_batch_01.csv)
- [B batch01 —52 judgments](../review/b0_replenishment_expansion/reviewer_B_batch_01.csv)
- [Expansion manifest](../review/b0_replenishment_expansion_manifest.json)
- [Chinese reviewer guide](../review/B0扩审人工审核说明.md)

Selection uses all525 remaining single-primary training candidates:262 primary-A and263 primary-B. Eligibility here means eligible for an independent audit, **not primary-accepted-only**. Reject/NO/UNSURE primary judgments cannot filter the frame. Fixed seed2026100402 selects language-proportional quotas and within-language token-length quartile coverage with deterministic SHA256 ties. Neither primary answers, source safety labels, model outputs, external-bank results nor subjective apparent difficulty enter selection. No salts were searched.

The11 worksheet columns are review_id, text, source, language, token_length, review_role, reviewer_decision, b0_confirmed, reason_code, confidence and optional_note. No primary answers or ordinal benign strata are exposed. Timing is not required; a voluntary trailing review_seconds column is supported only for actual recorded values. Prior25s timing is never imputed.

Projected marginal human work is105x25=2625seconds, **43minutes45seconds =0.72917 person-hours**. A's projection is22minutes05seconds and B's21minutes40seconds. These are planning estimates, not observations.
"""
    reserve_text=f"""
## Small frozen reserve — metadata only

The [reserve manifest](../data/manifests/hard_benign/b0_small_topup_reserve.json) holds exactly40 training-review reserve candidates and25 evaluation-double-review reserve candidates. All65 are **unreviewed candidates**: no worksheets, human assignments, accepted capacity, final stage IDs or train/validation/test split assignments were created for them.

Source remains pinned OASST1 revision{OA_REV}. All1392 previously reviewed OASST identities (including all872 replenishment candidates, even rejected/disputed records) and their prohibited groups were excluded, leaving7742 metadata-eligible candidates. A fixed100-candidate superset was audited; all100 passed the existing source/group/contamination rules, then65 were selected deterministically.

Checks include native/message/tree IDs, exact/normalized/model-view hashes, source-scoped project semantic/template groups and exhaustive lexical/SVD/template comparisons against2512 prior reviewed records (1590 production +50 calibration +872 replenishment), frozen NotInject339 and all14640 AF1-AF4 records. The prior bank includes all150 B-short reserve candidates, BH999 and existing accepted pools; those are nested subsets, not additive counts. No prohibited overlap was detected in the selected reserve. Existing grouping thresholds were retained; approximate semantic grouping and unavailable official OR parent lineage remain limitations.

The [reserve audit](../data/audits/hard_benign/b0_small_topup_reserve_audit.json) records exclusions, source hashes and fresh tokenization. Reserve maximum length is{audit["reserve_max_tokens"]} tokens; expansion maximum is{audit["expansion_max_tokens"]}; both are within1280. The DeBERTa tokenizer revision remains{src_revision()}. No detector inference or training was used.

The conditional pre-grouping deficit remains at least19 training and10 double-evaluation records (29 total); the new65 reserve candidates do not reduce it. Future expanded review and grouping can change that shortage.
"""
    analysis=r"""
## Analysis after all105 new judgments are completed

~~~powershell
Set-Location E:\ContinualGuard
& .\.venv\Scripts\python.exe .\scripts\analyze_h2_mini_b0_expansion.py
~~~
Output: results/h2_mini_b0_expansion_analysis.json. The script verifies frozen baseline dependencies, immutable worksheet content, opposite-reviewer assignments, exact IDs and every required new label field. It reports separate decision/B0-confirmation agreements, disagreements, Cohen kappa and ACCEPT_BENIGN-versus-REJECT_TRUE_INJECTION conflicts over the **full180 pairs**, retaining every original pair. New-only statistics are descriptive and do not determine the gate. The existing decision-disagreement gate also remains unchanged.

B0-confirmation disagreement must be **<=10% over180**, not over105. Missing/partial new reviews cannot pass. Any full180 failure stops without automatic further expansion, rubric change, adjudication or record removal.

If the full expanded gate passes, continue separately with the approved accepted-evidence merge and complete post-review grouping/contamination audit. Then compute exact800 train/120 validation/240 test/1160 unique capacity, including accepted training-audit coverage. If short, prepare only the minimum-needed review package from the frozen reserve; do not automatically conduct that review. Passing this audit alone does not approve final assignment, materialization or training.

For the initial blank-package check only:

~~~powershell
& .\.venv\Scripts\python.exe .\scripts\analyze_h2_mini_b0_expansion.py --check-package
~~~

The blank-package check passed.10 synthetic tests passed: answer-independent selection, reviewer independence, retention of all75 old pairs and disagreements, exact180-denominator boundary behavior, incomplete-review blocking, content integrity, rejection reasons and optional timing. No new human judgments were generated.

## Scientific status and stop

B0/BH remains the adopted nonordinal taxonomy. B1/B2/B3 remain descriptive metadata only. BH999 stays frozen with sufficient proposed600/90/180 capacity; no new BH review. B-short Option B remains the later120-record unmatched descriptive external short-form benign control targeting42 English/78 multilingual; no new B-short review.

R0 remains active;R1/R2 are not entered. This preparation provides no continual-learning evidence. Future comparisons retain the fixed base model/revision/initialization, datasets/order, seeds, epochs, learning rate, optimizer resets, evaluation protocol and a later approved fixed memory budget. No numerical replay budget is chosen.

Stop now for the two humans to complete only the new53/52 assignments independently. Do not modify original submitted judgments, review the reserve, merge final pools, select final B0/BH/B-short IDs, assign final splits, materialize research JSONL, train, implement replay or selective adaptation.

Recommended next prompt after both new files are complete:

> The two human reviewers have completed all105 independent B0 second-review expansion judgments. Analyze the full75 original +105 new =180-pair audit, preserving all original disagreements and the frozen10% gate. If the full audit passes, merge eligible evidence, run complete post-review grouping/contamination checks and report exact B0 capacity. If capacity is short, prepare only the minimum necessary top-up review package from the frozen40/25 reserve; do not launch review. If the full audit fails, stop without automatic expansion. Do not select final assignments, materialize research JSONL, train, implement replay or choose a numerical memory budget.
"""
    for name in ["h2_mini_b0_replenishment_results.md","h2_mini_b0_replenishment_plan.md","h2_mini_review_protocol.md","open_questions.md","assumptions_and_risks.md"]:
        extra=""
        if name=="h2_mini_review_protocol.md":
            extra="\nThe exact pre-expansion protocol is preserved in [the historical document copy](history/h2_mini_review_protocol_before_expansion_20261004.md), matching its previous audit SHA256. The current user's explicit instruction authorizes this document update; historical calibration/production results and all submitted judgments remain unchanged. Earlier sections below are historical where they conflict with the adopted B0/BH taxonomy or this phase's optional-timing rule.\n"
        prepend(name,head+reserve_text+analysis+extra)
    now=datetime.now(timezone.utc).isoformat()
    decision={"date":now,"id":"D038","decision":"Prepare105 additional independent B0 training second reviews and a65-record metadata-only top-up reserve",
      "authorization":"Current explicit human approval; request SHA256 recorded in the expansion manifest",
      "approved_request_sha256":m["approved_request_sha256"],"new_assignments":105,"A":53,"B":52,
      "original_pairs_retained":75,"future_audit_denominator":180,"training_candidates":600,"future_coverage":.30,
      "B0_confirmation_gate_unchanged":.10,"primary_answer_blind_selection":True,"new_timing_required":False,
      "reserve_training":40,"reserve_evaluation":25,"reserve_worksheets_created":False,"reserve_review_performed":False,
      "human_judgments_performed_by_agent":False,"final_pool_merge":False,"final_assignments":False,"materialization":False,"training":False,"replay":False,
      "replay_readiness":"R0","numerical_memory_budget":None,"next_action":"A and B independently complete their53/52 new SECOND_REVIEW rows; analyze all180 pairs only after all105 new judgments are complete"}
    files=["scripts/h2_mini_b0_expansion_common.py","scripts/prepare_h2_mini_b0_expansion.py","scripts/analyze_h2_mini_b0_expansion.py",
      "scripts/report_h2_mini_b0_expansion_preparation.py","tests/test_h2_mini_b0_expansion.py",".gitignore",
      "review/b0_replenishment_expansion_manifest.json","review/b0_replenishment_expansion/reviewer_A_batch_01.csv (local ignored)",
      "review/b0_replenishment_expansion/reviewer_B_batch_01.csv (local ignored)","review/B0扩审人工审核说明.md",
      "data/manifests/hard_benign/b0_small_topup_reserve.json","data/audits/hard_benign/b0_small_topup_reserve_audit.json",
      "results/h2_mini_b0_expansion_preparation_validation.json",
      *["docs/"+name for name in ["h2_mini_b0_replenishment_results.md","h2_mini_b0_replenishment_plan.md","h2_mini_review_protocol.md",
        "history/h2_mini_review_protocol_before_expansion_20261004.md","open_questions.md","assumptions_and_risks.md","decision_log.md","change_log.md"]],
      "history/decisions.jsonl","history/changes.jsonl"]
    change={"date":now,"id":"C025","files_changed":files,"reason":"Required30% independent training audit expansion approved by human",
      "effect":"Answer-blind105-record second-review package and65 unreviewed reserve metadata records; full180-pair analyzer and human instructions",
      "tests_run":["10 synthetic tests passed","Actual105-row blank package validated","65 reserve candidates pass fixed exclusion/group/contamination audit",
        "127 protected files unchanged; explicitly authorized protocol update preserves its prior exact-byte historical copy"],
      "project_root":str(ROOT),"python":sys.executable,"new_human_judgments":0,"reserve_accepted_capacity_added":0,
      "threshold_relaxed":False,"old_disagreements_removed":False,"training":False,"replay":False}
    prepend("decision_log.md","## D038 — Expanded independent review approved and prepared\n\n~~~json\n"+json.dumps(decision,ensure_ascii=False,indent=2)+"\n~~~\n")
    prepend("change_log.md","## C025 — B0 expansion worksheets and small reserve\n\n~~~json\n"+json.dumps(change,ensure_ascii=False,indent=2)+"\n~~~\n")
    append(ROOT/"history/decisions.jsonl",decision);append(ROOT/"history/changes.jsonl",change)
    unchanged=0
    for path,e in m["historical_evidence_snapshot"].items():
        if path==amendment["path"]:
            assert file_sha(copy)==e["sha256"];continue
        p=ROOT/path;assert file_sha(p)==e["sha256"] and p.stat().st_mtime_ns==e["mtime_ns"],path;unchanged+=1
    verify=analyzer.inspect_expansion()
    assert not verify["errors"] and all(s["status"]=="pending" for s in verify["states"].values())
    validation={"status":"PREPARED_ONLY_STOP_FOR_HUMAN_EXPANSION_REVIEW","at":now,"package":EXP_PACKAGE,
      "expansion_manifest_sha256":file_sha(EXP_MANIFEST),"reserve_manifest_sha256":file_sha(RESERVE),
      "worksheet_sha256":verify["hashes"],"blank_new_judgments":105,"A":53,"B":52,
      "old75_pairs_unchanged":True,"full_future_denominator":180,"future_training_coverage":.30,"gate_unchanged":.10,
      "metadata_only_reserve":{"training":40,"evaluation":25,"reviewed":0,"worksheets":0},
      "synthetic_tests_passed":10,"protected_files_unchanged":unchanged,
      "authorized_protocol_document_update":{"current_path":amendment["path"],"current_sha256":file_sha(doc),
        "previous_exact_copy":rel(copy),"previous_sha256":file_sha(copy)},
      "raw_worksheets_ignored":True,"human_judgments_generated":False,"final_pools_merged":False,
      "final_ids_selected":False,"materialization":False,"training":False,"replay":False,"replay_readiness":"R0",
      "archival_project_accessed":False,"storage_audit_or_cleanup":False}
    save(ROOT/"results/h2_mini_b0_expansion_preparation_validation.json",validation)
    print(json.dumps({"status":validation["status"],"A":53,"B":52,"reserve":validation["metadata_only_reserve"],
      "protected_files_unchanged":unchanged,"authorized_protocol_history_preserved":True,
      "decision":"D038","change":"C025","guide":rel(guide)},indent=2))
def src_revision():return "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
if __name__=="__main__":main()
