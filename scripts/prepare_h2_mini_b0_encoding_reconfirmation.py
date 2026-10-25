"""Prepare only the authorized source-backed blank 17-row reconfirmation packet."""
import argparse, csv, gzip, io, json, os, re, subprocess, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from h2_mini_b0_expansion_common import ROOT, EXP_MANIFEST, EXP_DIR, EXP_FIELDS, EXP_HUMAN
from h2_mini_b0_review_common import load, save, sha, file_sha, DECISIONS, B0_VALUES, CONFIDENCE, OA_REV, POLICY

REQUEST = Path("C:/Users/TLS/.codex/attachments/ac0dff55-c482-4273-b58a-73927b8dc7b2/\u5df2\u7c98\u8d34\u7684\u6587\u672c.txt")
PRIOR = ROOT / "results/h2_mini_b0_expansion_analysis.json"
RESULT = ROOT / "results/h2_mini_b0_expansion_integrity_recovery.json"
PACKET = EXP_DIR / "reviewer_A_encoding_reconfirmation_17.csv"
GUIDE = ROOT / "review/B0\u7f16\u7801\u635f\u574f17\u6761\u91cd\u65b0\u786e\u8ba4\u8bf4\u660e.md"
FIELDS = ["review_id", "text", "language", "token_length", "review_role",
          "reviewer_decision", "b0_confirmed", "reason_code", "confidence", "optional_note"]
HUMAN = FIELDS[5:]
MARKER = "H2_MINI_B0_ENCODING_RECONFIRMATION_PREPARED_20261004"

def relative(path):
    return path.resolve().relative_to(ROOT.resolve()).as_posix()

def fingerprint(path):
    return {"sha256": file_sha(path), "mtime_ns": path.stat().st_mtime_ns}

def valid_human(row):
    return (row.get("reviewer_decision", "").strip() in DECISIONS
            and row.get("b0_confirmed", "").strip() in B0_VALUES
            and row.get("confidence", "").strip() in CONFIDENCE
            and (not row["reviewer_decision"].strip().startswith("REJECT_")
                 or bool(row.get("reason_code", "").strip())))

def read_csv(raw, encoding="utf-8-sig"):
    reader = csv.DictReader(io.StringIO(raw.decode(encoding, errors="strict"), newline=""), strict=True)
    rows = list(reader)
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError("Malformed CSV row")
    return reader.fieldnames, rows

def search_artifacts(expected, original):
    cmd = ["rg", "--files", "--hidden", "--no-ignore", "-g", "!.venv/**", "-g", "!.git/**"]
    run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if run.returncode or run.stderr: raise RuntimeError("Incomplete active-project search: " + run.stderr)
    files = sorted(run.stdout.splitlines())
    assert not any(Path(f).name == "AGENTS.md" for f in files), "Read newly discovered local instructions first"
    workbook_suffixes = {".xlsx", ".xls", ".xlsm", ".xlsb", ".ods", ".et", ".xlk", ".zip"}
    text_suffixes = {".csv", ".tsv", ".txt", ".tmp", ".bak", ".asd", ".wbk"}
    workbooks = []
    candidates = []
    intact = []
    scanned = 0
    for rel in files:
        p = ROOT / rel
        if p.is_symlink(): continue
        assert p.resolve().is_relative_to(ROOT.resolve())
        suffix = p.suffix.lower()
        if suffix in workbook_suffixes:
            workbooks.append(relative(p))
            continue
        if suffix not in text_suffixes: continue
        scanned += 1
        # Header screening avoids reading unrelated large source CSVs or model-result tables.
        with p.open("rb") as stream: head = stream.read(4096)
        if b"review_id" not in head and b"r\x00e\x00v\x00i\x00e\x00w\x00_\x00i\x00d" not in head:
            continue
        raw = p.read_bytes()
        item = {"file": relative(p), "sha256": sha(raw)}
        parsed = None
        for encoding in ["utf-8-sig", "utf-16", "gb18030", "cp936"]:
            try:
                fields, rows = read_csv(raw, encoding)
                if "review_id" not in (fields or []): continue
                parsed = (fields, rows, encoding)
                break
            except (UnicodeError, csv.Error, ValueError): continue
        if parsed is None:
            item["status"] = "not_a_parseable_review_csv"; candidates.append(item); continue
        fields, rows, encoding = parsed
        ids = [r.get("review_id") for r in rows]
        shared = set(ids) & set(expected)
        exact = len(rows) == 53 and Counter(ids) == Counter(expected.keys())
        item.update(rows=len(rows), encoding_for_diagnosis=encoding, shared_A_ids=len(shared),
                    exact_53_A_ids=exact)
        if not exact:
            item["status"] = "not_the_53_assignment_A_submission"
        elif not all(x in fields for x in ["text", *HUMAN]):
            item["status"] = "required_columns_missing"
        else:
            by = {r["review_id"]: r for r in rows}
            source_matches = sum(sha(by[rid]["text"]) == meta["raw_text_sha256"] for rid, meta in expected.items())
            human_matches = all(all(by[rid][k] == original[rid][k] for k in HUMAN) for rid in expected)
            notes_and_optional_timing_match = all(by[rid].get("review_seconds", "") == original[rid].get("review_seconds", "") for rid in expected)
            item.update(source_text_hash_matches=source_matches,
                        exact_human_values_match=human_matches,
                        optional_timing_values_match=notes_and_optional_timing_match,
                        required_human_fields_complete=all(valid_human(row) for row in rows))
            item["status"] = "intact_candidate_requires_provenance_verification" if (
                source_matches == 53 and human_matches and notes_and_optional_timing_match
                and item["required_human_fields_complete"]) else "does_not_pass_intact_recovery_requirements"
            if item["status"].startswith("intact_candidate"): intact.append(item)
        candidates.append(item)
    if workbooks: raise RuntimeError("Workbook/archive candidates require inspection before choosing the packet branch: " + str(workbooks))
    if intact: raise RuntimeError("Intact candidate found; inspect human provenance and perform direct-recovery branch: " + str(intact))
    return {"root": str(ROOT), "archive_searched": False, "follow_symlinks": False,
            "excluded_runtime_directories": [".venv", ".git"], "enumeration_errors": [],
            "text_or_temp_candidate_files_screened": scanned, "workbook_or_archive_candidates": workbooks,
            "review_header_candidates": candidates, "intact_prior_A_artifact_found": False}

def append_event(path, event):
    before = path.read_bytes()
    assert event["id"] not in {json.loads(x)["id"] for x in before.decode("utf-8").splitlines() if x.strip()}
    with path.open("ab") as f:
        if before and not before.endswith(b"\n"): f.write(b"\n")
        f.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
    assert path.read_bytes().startswith(before)

def prepend(path, body):
    original = path.read_bytes()
    assert MARKER.encode() not in original
    line_end = original.index(b"\n") + 1
    insertion = ("\n<!-- " + MARKER + ":BEGIN -->\n" + body + "\n<!-- " + MARKER + ":END -->\n").encode("utf-8")
    path.write_bytes(original[:line_end] + insertion + original[line_end:])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Create authorized blank packet and recovery status; never reconcile judgments")
    args = parser.parse_args()
    assert Path.cwd().resolve() == ROOT.resolve()
    assert Path(sys.executable).resolve() == (ROOT / ".venv/Scripts/python.exe").resolve()
    assert not any(p.exists() for p in [PACKET, GUIDE, RESULT]), "Refuse to overwrite a packet or human evidence"
    m = load(EXP_MANIFEST); prior = load(PRIOR)
    assert prior["status"] == "STOP_EXPANSION_SUBMISSION_INTEGRITY"
    expected = {r["review_id"]: r for r in m["records"] if r["second_reviewer"] == "A"}
    expected_B = {r["review_id"]: r for r in m["records"] if r["second_reviewer"] == "B"}
    assert len(expected) == 53 and len(expected_B) == 52
    ids = re.findall(r"^H2B0-[0-9A-F]{16}$", REQUEST.read_text(encoding="utf-8-sig"), re.M)
    assert len(ids) == len(set(ids)) == 17
    assert set(ids) == {r["review_id"] for r in prior["source_text_mismatches"]}
    assert all(rid in expected for rid in ids)
    protected = {rel: fingerprint(ROOT / rel) for rel in prior["protected_files_before"]}
    assert protected == prior["protected_files_before"], "Previously protected evidence changed"
    protected[relative(PRIOR)] = fingerprint(PRIOR)
    for path in m["frozen_dependencies"]: protected[path] = fingerprint(ROOT / path)
    for path, digest in m["frozen_dependencies"].items(): assert file_sha(ROOT / path) == digest
    submitted = {}
    for who in ["A", "B"]:
        p = EXP_DIR / f"reviewer_{who}_batch_01.csv"
        assert file_sha(p) == prior["worksheet_sha256"][p.name]
        submitted[who] = {"path": relative(p), **fingerprint(p)}
    raw = (EXP_DIR / "reviewer_A_batch_01.csv").read_bytes()
    assert raw.decode("gb18030", errors="strict") == raw.decode("cp936", errors="strict")
    header, old_rows = read_csv(raw, "gb18030")
    assert header in [EXP_FIELDS, EXP_FIELDS + ["review_seconds"]]
    assert Counter(r["review_id"] for r in old_rows) == Counter(expected.keys())
    original = {r["review_id"]: r for r in old_rows}
    corrupted = {rid for rid, row in original.items() if sha(row["text"]) != expected[rid]["raw_text_sha256"]}
    assert corrupted == set(ids)
    intact_ids = sorted(set(expected) - corrupted)
    for rid in intact_ids:
        row = original[rid]; meta = expected[rid]
        assert valid_human(row)
        assert row["language"] == meta["language"] and row["token_length"] == str(meta["deberta_tokens"])
        assert row["source"] == "OpenAssistant/oasst1" and row["review_role"] == "SECOND_REVIEW"
        assert meta["primary_reviewer"] == "B" and meta["second_reviewer"] == "A"
    B_header, B_rows = read_csv((EXP_DIR / "reviewer_B_batch_01.csv").read_bytes())
    assert B_header in [EXP_FIELDS, EXP_FIELDS + ["review_seconds"]]
    assert Counter(r["review_id"] for r in B_rows) == Counter(expected_B.keys())
    for row in B_rows:
        meta = expected_B[row["review_id"]]
        assert sha(row["text"]) == meta["raw_text_sha256"] and valid_human(row)
        assert row["source"] == "OpenAssistant/oasst1" and row["review_role"] == "SECOND_REVIEW"
        assert row["language"] == meta["language"] and row["token_length"] == str(meta["deberta_tokens"])
        assert meta["primary_reviewer"] == "A" and meta["second_reviewer"] == "B"
    search = search_artifacts(expected, original)
    print(json.dumps({"search": "complete", "screened": search["text_or_temp_candidate_files_screened"],
                      "review_header_candidates": len(search["review_header_candidates"]),
                      "intact_prior_A_artifact_found": False, "B_validated": 52, "A_intact_rows": 36,
                      "reconfirmation_required": 17}, ensure_ascii=False), flush=True)
    # Extract original text directly from the pinned local source, without loading any human labels.
    audit_path = ROOT / "data/audits/hard_benign/oasst1_full_audit.json"
    source = load(audit_path)["source"]
    entry = next(x for x in source["files"] if x["file_name"] == "2023-04-12_oasst_ready.messages.jsonl.gz")
    source_path = ROOT / entry["local_path"]
    assert source["revision"] == OA_REV and file_sha(source_path) == entry["sha256"]
    wanted = {expected[rid]["source_row_index"]: rid for rid in ids}
    texts = {}
    with gzip.open(source_path, "rt", encoding="utf-8", errors="strict") as stream:
        for index, line in enumerate(stream):
            if index not in wanted: continue
            row = json.loads(line); rid = wanted[index]; meta = expected[rid]
            assert meta["source_revision"] == OA_REV
            assert row["message_id"] == meta["source_native_id"] == meta["candidate_id"]
            assert row["message_tree_id"] == meta["source_parent_id"]
            assert row.get("parent_id") == meta["source_immediate_parent_id"]
            assert row["lang"] == meta["language"] and sha(row["text"]) == meta["raw_text_sha256"]
            texts[rid] = row["text"]
    assert len(texts) == 17
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    import audit_h2_mini_sources as source_tools
    tok = AutoTokenizer.from_pretrained(ROOT / "data/raw/huggingface/microsoft_deberta-v3-base" / source_tools.TOKR,
                                       local_files_only=True, use_fast=False)
    model_hashes, token_counts = source_tools.mviews([texts[rid] for rid in ids], tok)
    records = []
    packet_rows = []
    for rid, model_hash, length in zip(ids, model_hashes, token_counts):
        meta = expected[rid]; text = texts[rid]
        assert length == meta["deberta_tokens"] <= 1280
        assert model_hash == meta["model_view_sha256"]
        assert sha(source_tools.norm(text)) == meta["normalized_text_sha256"]
        packet_rows.append({"review_id": rid, "text": text, "language": meta["language"],
                            "token_length": str(length), "review_role": "SECOND_REVIEW_RECONFIRMATION",
                            **{key: "" for key in HUMAN}})
        records.append({key: meta[key] for key in ["review_id", "source_revision", "source_native_id", "source_parent_id",
                        "source_row_index", "language", "deberta_tokens", "raw_text_sha256",
                        "normalized_text_sha256", "model_view_sha256"]})
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\r\n")
    writer.writeheader(); writer.writerows(packet_rows)
    payload = buf.getvalue().encode("utf-8-sig")
    h, rr = read_csv(payload)
    assert payload.startswith(b"\xef\xbb\xbf") and h == FIELDS and len(rr) == 17
    assert [r["review_id"] for r in rr] == ids
    assert all(sha(r["text"]) == expected[r["review_id"]]["raw_text_sha256"] for r in rr)
    assert all(all(r[k] == "" for k in HUMAN) for r in rr)
    assert all(fingerprint(ROOT / rel) == value for rel, value in protected.items())
    if not args.prepare:
        print(json.dumps({"check": "PASS", "packet_rows": 17, "blank_human_cells": 85,
                          "max_tokens": max(token_counts), "writes": False, "gate_run": False}))
        return
    now = datetime.now(timezone.utc).isoformat()
    for name, event_id in [("decisions", "D040"), ("changes", "C027")]:
        assert not any(json.loads(x)["id"] == event_id for x in
                       (ROOT / "history" / (name + ".jsonl")).read_text(encoding="utf-8").splitlines() if x.strip())
    PACKET.write_bytes(payload)
    guide = """# B0编码损坏17条重新确认说明

本轮由审核员A独立重新审核17条记录，仅限本次CSV所列ID。其余36条A记录和52条B记录不需重审。

上一份A提交表在传输或保存过程中发生了原文字符损坏，部分非ASCII字符被写成问号。现已从固定版本原始数据恢复这17条的完整原文，逐条验证了来源ID、原文哈希和词元长度。本次是证据恢复，不是重新讨论整个审核标准，也不因补齐数量而调整标签。

## 请打开这个文件

[b0_replenishment_expansion/reviewer_A_encoding_reconfirmation_17.csv](b0_replenishment_expansion/reviewer_A_encoding_reconfirmation_17.csv)

文件为UTF-8带BOM。使用Excel/WPS的“从文本/CSV导入”，明确选择UTF-8（65001）和逗号分隔，将各列按文本导入。保留全部10列、17条记录及原文换行。可启用自动换行以查看完整原文。若出现乱码、异常问号或文字不可理解，先联系协调人，不猜测原文。

## 独立填写

- 请只看本轮恢复后的完整原文，重新作出判断，不参照审核员B的答案，也不要复制旧损坏表的答案。
- 五个人工填写字段初始全部为空：reviewer_decision、b0_confirmed、reason_code、confidence、optional_note。
- 不修改review_id、text、language、token_length、review_role；角色固定为SECOND_REVIEW_RECONFIRMATION。
- 每条必须填写新的reviewer_decision、b0_confirmed、confidence。拒绝时必须填写reason_code；optional_note可留空。
- 本轮不要求逐条计时，不需添加review_seconds，也不要把历史25秒均值填到每条记录。
- 不新增、删除或替换记录；不使用来源标签或模型输出代替自己的判断。

## 冻结标准保持不变

沿用h2-mini-benign-taxonomy-v2的普通B0标准：明确良性、普通短形式用户请求、独立可理解，无提示注入、无范围内有害内容、无隐私或权利问题、无缺失上下文且质量可用。不再区分B1/B2/B3。不能可靠理解语言或语境时联系协调人，不猜测。

reviewer_decision允许值：

- ACCEPT_BENIGN
- REJECT_TRUE_INJECTION
- REJECT_AMBIGUOUS_INTENT
- REJECT_HARMFUL_OR_UNSAFE
- REJECT_CONTEXT_MISSING
- REJECT_PRIVACY_OR_RIGHTS
- REJECT_QUALITY

b0_confirmed填写YES、NO或UNSURE。明确良性但不符合普通B0时如实填NO或UNSURE。
confidence填写HIGH、MEDIUM或LOW。
拒绝理由reason_code建议分别使用INJECTION、AMBIGUOUS_INTENT、HARMFUL_OR_UNSAFE、CONTEXT_MISSING、PRIVACY_OR_RIGHTS、QUALITY。不要为满足审核一致率而修改自己的判断。

## 保存和提交

保存完成的本轮表为“CSV UTF-8（带BOM）”，保留同一文件名，不覆盖旧的reviewer_A_batch_01.csv或reviewer_B_batch_01.csv。不要选择ANSI或普通本地编码CSV。保存后重新以UTF-8打开，确认所有语言文字完整，再通知协调人“17条重新确认已完成”。

损坏文件中的原判断继续保留为历史证据；这17条最终只能使用本轮新的明确人工判断，除非后来找到经验证的完整已审核原件。不要自行拼接53条表，不运行旧分析脚本。

## 协调人后续步骤（尚未执行）

完成后，先核验本轮17条的ID、原文哈希、语言、长度、角色和人工必填字段。验证成功才可将原A表中36条原文完整记录的原判断，与本轮17条新人工判断合并为单独的53条canonical UTF-8文件。逐条记录判断来源以及原损坏文件、本轮确认文件和最终文件的哈希，保留原提交不变。

本次仅准备重新确认材料，不进行人工判断、不生成53条canonical文件、不计算180对分歧率、不做容量认证或最终分配、不物化研究JSONL、不训练、不实施回放。R0 active；R1/R2尚未进入。
"""
    GUIDE.write_text(guide, encoding="utf-8")
    record_provenance = [
        {"review_id": rid, "future_judgment_source": "original_A_intact_row",
         "source_file": submitted["A"]["path"], "source_file_sha256": submitted["A"]["sha256"],
         "human_fields_sha256": sha(json.dumps({k: original[rid].get(k, "") for k in [*HUMAN, "review_seconds"]},
                                            ensure_ascii=False, sort_keys=True))}
        for rid in intact_ids
    ] + [
        {"review_id": rid, "future_judgment_source": "new_explicit_A_reconfirmation_required",
         "old_damaged_judgment_usable": False, "completed_reconfirmation_file_sha256": None}
        for rid in ids
    ]
    report = {
        "status": "STOP_AWAITING_A_17_HUMAN_RECONFIRMATIONS", "created_at": now,
        "authorization_request_sha256": file_sha(REQUEST), "prior_integrity_report_sha256": file_sha(PRIOR),
        "assignment_manifest_sha256": file_sha(EXP_MANIFEST), "taxonomy_policy": POLICY,
        "project_root": str(ROOT), "python": sys.executable, "search": search,
        "original_submissions": submitted, "original_submissions_modified": False,
        "intact_prior_A_artifact_found": False, "source_text_for_all53_resolvable": True,
        "all53_judgments_provenance_safely_recovered": False, "reconfirmation_required": True,
        "A_assigned": 53, "A_intact_original_rows": 36, "B_assigned_and_validated": 52,
        "new_human_reconfirmations_required": 17, "new_human_reconfirmations_completed": 0,
        "affected_review_ids": ids, "pinned_source": {
            "revision": OA_REV, "path": entry["local_path"], "sha256": entry["sha256"],
            "source_audit_sha256": file_sha(audit_path), "selected_text_hashes_verified": 17},
        "tokenizer_revision": source_tools.TOKR, "max_packet_tokens": max(token_counts),
        "over1280": sum(n > 1280 for n in token_counts), "truncation": False,
        "packet": {"path": relative(PACKET), "sha256": file_sha(PACKET), "encoding": "utf-8-sig",
                   "columns": FIELDS, "rows": 17, "blank_human_cells": 85, "records_metadata_only": records,
                   "all_raw_normalized_model_view_hashes_verified": True,
                   "prepared_status": "ALL_HUMAN_FIELDS_BLANK", "old_judgments_prefilled": False,
                   "gitignore_rule": "review/b0_replenishment_expansion/"},
        "chinese_instructions": relative(GUIDE), "guide_sha256": file_sha(GUIDE),
        "row_provenance_plan": record_provenance, "canonical_file": None,
        "recovered_file": None, "recovery_integrity_status": "STOP_PENDING_HUMAN_RECONFIRMATION",
        "packet_integrity_validation": "PASS", "expanded180_gate_run": False,
        "pool_merge": False, "final_ID_selection": False, "research_JSONL_materialization": False,
        "training": False, "replay": False, "numerical_memory_budget": None,
        "readiness": {"R0": "active", "R1": "not_entered", "R2": "not_entered"},
        "archive_accessed": False, "storage_audit_or_cleanup": False,
        "source_or_model_labels_used_to_adjudicate": False, "human_judgments_inferred": False,
        "protected_file_snapshots": protected, "protected_files_unchanged": True,
        "next_human_action": "Reviewer A independently completes exactly17 rows using intact source text and frozen ordinary-B0 rubric; no reference to B answers; save UTF-8 BOM and notify coordinator.",
        "subsequent_reconciliation_policy": "Only after17 complete validated reconfirmations:36 original intact A judgments plus17 new human judgments, separate canonical53-row CSV and per-row provenance; revalidate A53/B52, stop before180-pair gate."
    }
    assert "review/b0_replenishment_expansion/" in (ROOT / ".gitignore").read_text(encoding="utf-8")
    save(RESULT, report)
    body = f"""## Submission evidence recovery only — {now}

**STOP_AWAITING_A_17_HUMAN_RECONFIRMATIONS.** The human accepted the transport/source-text
failure and authorized a search of the active project, followed by a focused blank packet if no intact
prior human-reviewed copy exists.

The active-project search found no qualifying intact53-assignment A artifact. It screened
{search['text_or_temp_candidate_files_screened']} CSV/text/temporary candidates, including
{len(search['review_header_candidates'])} with review headers; no XLSX/XLS/WPS/archive recovery
candidate was present. Hidden project directories were included; the runtime .venv and .git were
excluded. No archive outside the active root was searched. Earlier review batches cannot substitute
for these independent second-review judgments.

A's original53-row and B's52-row submissions remain byte-for-byte and modification-time unchanged.
B52 still validate; A36 source-intact rows and their exact human values are preserved.
The remaining17 original source texts are now restored **only into a blank reconfirmation packet**,
directly from pinned OASST1 revision {OA_REV}, with source/native IDs, raw/normalized/model-view
hashes and token counts verified. There are17 rows,10 columns and85 blank human cells; UTF-8 BOM.
No damaged-file judgments, notes or B answers are prefilled.

- [Reviewer A17-row reconfirmation CSV](../review/b0_replenishment_expansion/reviewer_A_encoding_reconfirmation_17.csv)
- [Chinese instructions](../review/{GUIDE.name})
- [Recovery provenance and validation](../results/h2_mini_b0_expansion_integrity_recovery.json)

All53 source texts are traceable to immutable candidates, but the full53-row reviewed submission is
**not yet provenance-safely recovered**. No recovered/canonical53-row file exists. Packet integrity
is PASS; submission recovery is STOP pending17 fresh explicit human judgments.
The damaged judgments remain historical evidence and cannot be reused for these17 records.

Reviewer A must independently read the restored text, keep the frozen ordinary-B0 rubric, avoid B's
answers, fill fresh decision/B0/confidence and rejection reasons where required, save UTF-8 BOM,
and notify the coordinator. Subsequent reconciliation will preserve the36 intact original judgments
and use only the17 new explicit confirmations, with row-level provenance and all file hashes.
The original analyzer must not be run directly against this mixed submission/packet directory.

The180-pair gate remains uncomputed. No pool merge, capacity certification, final IDs, research JSONL,
training, replay or memory budget. R0 active; R1/R2 not entered. Earlier integrity results below remain
historical; the current next action is the authorized17-row reconfirmation, not a new general recovery request.
"""
    prepend(ROOT / "docs/h2_mini_b0_expansion_results.md", body)
    append_event(ROOT / "history/decisions.jsonl", {
        "date": now, "id": "D040", "decision": "Prepare source-backed blank17-row A reconfirmation after no intact active-project artifact found",
        "authorization_request_sha256": file_sha(REQUEST), "result": relative(RESULT),
        "A_intact_original_rows": 36, "B_validated": 52, "pending_new_A_reconfirmations": 17,
        "original_submissions_preserved": True, "old_damaged_judgments_reused": False,
        "full180_gate_run": False, "canonical53_created": False, "training": False, "replay": False,
        "replay_readiness": "R0", "numerical_memory_budget": None})
    append_event(ROOT / "history/changes.jsonl", {
        "date": now, "id": "C027", "decision_id": "D040",
        "files_changed": ["scripts/prepare_h2_mini_b0_encoding_reconfirmation.py", relative(PACKET),
                          relative(GUIDE), relative(RESULT), "docs/h2_mini_b0_expansion_results.md",
                          "history/decisions.jsonl", "history/changes.jsonl"],
        "validation": ["Pinned source/hash/native-ID verification", "17 UTF-8 BOM rows and85 blank human fields",
                       "Fresh offline tokenizer verifies all17 model-view hashes and token lengths",
                       "Original submissions and protected evidence hashes/mtimes unchanged"],
        "scope": "Submission evidence recovery preparation only; no inferred human judgments or expanded audit"})
    assert read_csv(PACKET.read_bytes())[1] == rr
    assert all(fingerprint(ROOT / rel) == value for rel, value in protected.items())
    assert all(fingerprint(ROOT / item["path"]) == {k: item[k] for k in ["sha256", "mtime_ns"]}
               for item in submitted.values())
    assert not any((EXP_DIR / name).exists() for name in [
        "reviewer_A_batch_01_recovered_utf8.csv", "reviewer_A_batch_01_canonical_utf8.csv"])
    print(json.dumps({"status": report["status"], "packet": relative(PACKET), "rows": 17,
                      "blank_human_cells": 85, "max_tokens": max(token_counts),
                      "original_submissions_unchanged": True, "expanded180_gate_run": False,
                      "history": ["D040", "C027"]}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

