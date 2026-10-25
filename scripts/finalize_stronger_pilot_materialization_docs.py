from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(r"E:\ContinualGuard")
TARGET = ROOT / "data/processed/stronger_pilot_candidate_b"
MANIFEST_PATH = TARGET / "materialization_manifest.json"
VALIDATION_PATH = ROOT / "results/stronger_pilot_materialization_validation.json"
DOCS = ROOT / "docs"
MARKER = "STRONGER_PILOT_CANDIDATE_B_MATERIALIZED_20261005"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def upsert(path: Path, body: str) -> None:
    begin = f"<!-- {MARKER}:BEGIN -->"
    end = f"<!-- {MARKER}:END -->"
    block = f"{begin}\n{body.rstrip()}\n{end}"
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if begin in old and end in old:
        prefix, rest = old.split(begin, 1)
        _, suffix = rest.split(end, 1)
        new = prefix.rstrip() + "\n\n" + block + suffix
    else:
        new = old.rstrip() + ("\n\n" if old.strip() else "") + block + "\n"
    write(path, new)


manifest = load(MANIFEST_PATH)
validation = load(VALIDATION_PATH)
assert manifest["status"] == "MATERIALIZATION_PASS"
assert validation["status"] == "PASS_READY_FOR_HUMAN_NO_REPLAY_TRAINING_APPROVAL"
assert validation["all_checks_pass"] is True
assert len(manifest["files"]) == 36
for item in manifest["files"]:
    path = ROOT / item["relative_path"]
    assert sha(path) == item["sha256"]
    assert len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]) == item["row_count"]

validation["tests"] = {
    "relevant_materialization_tests": {
        "path": "tests/test_stronger_pilot_candidate_b_materialization.py",
        "passed": 12,
        "failed": 0,
        "skipped": 0,
        "warnings": 0,
        "command": "python -m pytest tests/test_stronger_pilot_candidate_b_materialization.py -q -p no:cacheprovider",
    },
    "full_project_tests": {
        "passed": 114,
        "failed": 0,
        "skipped": 0,
        "warnings": 1,
        "command": "python -m pytest -q -p no:cacheprovider",
        "note": "The warning is the existing torch.jit FutureWarning under Python 3.14.",
    },
}
save_json(VALIDATION_PATH, validation)
validation_hash = sha(VALIDATION_PATH)
manifest["validation_artifact_sha256"] = validation_hash
manifest["post_materialization_tests"] = validation["tests"]
save_json(MANIFEST_PATH, manifest)
manifest_hash = sha(MANIFEST_PATH)

next_prompt = f"""Approve Candidate B no-replay execution only from materialization manifest SHA-256 `{manifest_hash}` and validation SHA-256 `{validation_hash}`. Use `microsoft/deberta-v3-base` revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`, `max_length=1280`, seeds 17/42/2026, arms B-DOC/B-SHORT/B-HARD, and order AF2→AF3→AF4→AF5. Use the frozen H2-mini training configuration: one epoch per stage, batch size 4, learning rate `1e-5`, constant scheduler, optimizer reset at every stage, model weights carried forward, and current-stage training data only. Run exactly 9 sequential runs and save 36 stage-end checkpoints, raw predictions, confusion counts, losses, runtime, peak GPU memory, accuracy/macro-F1/attack-TPR/benign-FPR matrices, acquisition, retention, BWT, peak forgetting, and head/middle/tail TPR. Keep NotInject and B-short evaluation-only. Apply the unchanged acquisition and R0/R1/R2 gates after all runs. Do not implement replay or selective adaptation, choose a replay memory budget, change assignments, or retune hyperparameters. Stop for human review after the complete no-replay result."""

report_doc = f"""# Candidate B materialization report

## Outcome

Local Candidate B JSONL materialization completed and passed every validation. Status: **PASS — READY FOR HUMAN NO-REPLAY TRAINING APPROVAL**. No detector training or checkpoint evaluation occurred.

- Materializer: `scripts/materialize_stronger_pilot_candidate_b.py`
- Output root: `data/processed/stronger_pilot_candidate_b/`
- JSONL files: 36, with 12 in each of `b_doc`, `b_short`, and `b_hard`
- Per arm: 2,320 records
- Total arm memberships: 6,960
- Unique selected records: 3,770
- Manifest SHA-256: `{manifest_hash}`
- Validation SHA-256: `{validation_hash}`

## Count contract

Every arm has four stages of 580 records. Every stage contains train 400 (200/200), validation 60 (30/30), and test 120 (60/60) for benign/malicious labels. Labels remain binary: 0 benign and 1 malicious indirect prompt injection.

## Integrity and matching

All authoritative proposal, source, revision, and source-file hashes match their frozen values. Raw, normalized, and model-view hashes match for every written row. The 1,160 malicious records are identical across all three arms; all arms share the same 290-record D0-1 Stage-1 anchor; B-SHORT and B-HARD share the same 290-record B0-2 Stage-2 anchor. Other benign sharing is absent.

Cross-stage/split native identity, document/tree, exact/normalized/model-view, semantic-parent, template-group, and payload isolation checks pass. The 1,160 D0 clean/malicious base-document pairs in B-DOC are intentional same-stage/same-split matched pairs. AF5 has 290 selected records, all with the three declared components; complete templates are split-exclusive and exact identity overlap with AF2–AF4 is zero.

## External banks and token lengths

`external/references.json` contains paths, hashes, roles, and prohibited uses only. It does not copy raw NotInject or B-short text. NotInject remains 339 records and B-short 120 records, both evaluation-only. Their overlap with selected benchmark records is zero under the audited hashes.

Across 3,770 unique selected records, pinned-tokenizer lengths are min 6, median 119, p95 261, p99 354, and max 491. Zero records exceed 1,280 or require truncation.

## Tests and stop condition

Focused materialization tests: 12 passed, 0 failed, 0 skipped, 0 warnings. The full project suite passed 114 tests with 0 failures and one existing `torch.jit` FutureWarning under Python 3.14. The focused tests cover deterministic serialization, counts, hash rejection, labels, matched anchors, cross-stage duplicates, length limits, AF5 composition policy, prediction-independent selection, and published artifact hashes.

No optimizer, epoch, prediction, Candidate B evaluation, checkpoint, replay, selective adaptation, or numerical memory budget was created. R0 remains active; R1 and R2 are not entered.

## Exact recommended next prompt

> {next_prompt}
"""
write(DOCS / "stronger_pilot_materialization_report.md", report_doc)

upsert(DOCS / "stronger_pilot_assignment_plan.md", f"""## Materialization completed

The frozen hash-only proposals have been resolved without changing membership. Thirty-six JSONL files were atomically published under `data/processed/stronger_pilot_candidate_b/`. The materialization manifest SHA-256 is `{manifest_hash}` and the validation artifact SHA-256 is `{validation_hash}`. All 21 declared materialization checks pass.

The next gate is human approval of no-replay execution. Materialization does not authorize training.
""")

upsert(DOCS / "stronger_pilot_candidate_b_protocol.md", f"""## Materialized benchmark status

Candidate B is locally materialized and fully validated. The exact assignments, three-arm structure, binary labels, AF5 compositional interpretation, matched anchors, and external-bank roles remain unchanged. Manifest: `{manifest_hash}`. Validation: `{validation_hash}`.

Status: **READY FOR HUMAN NO-REPLAY TRAINING APPROVAL**. No model execution occurred.
""")

upsert(DOCS / "stronger_pilot_no_replay_protocol.md", f"""## Final pre-execution state

The local dataset is available under `data/processed/stronger_pilot_candidate_b/` and passed all materialization checks. Freeze manifest `{manifest_hash}` and validation `{validation_hash}` for any approved run.

Future execution remains exactly nine runs: three arms × seeds 17, 42, and 2026. Use one epoch per stage, batch size 4, learning rate `1e-5`, constant scheduler, optimizer reset at every stage, carried model weights, current-stage-only data, and the fixed DeBERTa revision. Primary contrasts and acquisition/R0/R1/R2 gates remain unchanged. Training still requires human approval.
""")

upsert(DOCS / "replay_readiness_review.md", """## Candidate B materialization readiness update

Candidate B has passed local raw-text materialization and post-materialization validation. This establishes dataset readiness only. **R0 remains active; R1 and R2 are not entered.** Replay, selective adaptation, and numerical memory-budget selection remain blocked. The next possible action is human approval of the fixed nine-run no-replay baseline.
""")

upsert(DOCS / "replay_comparison_protocol.md", """## Candidate B materialization prerequisite satisfied

The stronger-pilot dataset is now materially fixed and validated, but no no-replay evidence exists. Replay eligibility still requires a separately approved nine-run no-replay execution that passes acquisition and reaches R1 or R2 under the unchanged criteria. No replay code or budget is authorized.
""")

upsert(DOCS / "decision_log.md", f"""## D048 — Certify Candidate B local materialization (2026-10-05)

Certify 36 atomically published Candidate B JSONL files, 6,960 arm memberships, 3,770 unique records, and all post-materialization integrity checks. Freeze manifest `{manifest_hash}` and validation `{validation_hash}` for subsequent review.

Candidate B is ready for human no-replay training approval. This decision does not authorize training, evaluation, replay, selective adaptation, or a numerical memory budget. R0 remains active.
""")

upsert(DOCS / "change_log.md", """## C036 — Materialize and validate Candidate B locally (2026-10-05)

Added the guarded materializer and focused tests, explicitly git-ignored the raw output root, resolved all frozen assignments from pinned local sources, atomically published 36 JSONL files plus external reference metadata, created the materialization manifest and validation artifact, and synchronized protocol, readiness, risk, question, decision, and change records.

No model training, model evaluation, checkpoint, optimizer state, replay, selective adaptation, or memory-budget choice was produced.
""")

upsert(DOCS / "assumptions_and_risks.md", """## Candidate B materialization risks

| ID | Assumption or risk | Current treatment |
|---|---|---|
| R090 | Raw Candidate B text is locally materialized. | Entire output root is git-ignored; manifests and reports contain hashes and metadata only. |
| R091 | External banks could accidentally enter training. | Only path-reference metadata is present under `external/`; roles and prohibited uses are explicit. |
| R092 | Materialization could drift from frozen assignments. | Every row is verified against source/native identity and raw/normalized/model-view hashes; proposal hashes are frozen in the manifest. |
| R093 | A valid dataset may still produce ceiling saturation. | Preserve R0 and run only the separately approved no-replay protocol before any replay decision. |
""")

upsert(DOCS / "open_questions.md", """## Candidate B post-materialization questions

| ID | Question | Current status |
|---|---|---|
| Q056 | May Candidate B be locally materialized? | Resolved: approved, completed, and validated on 2026-10-05. |
| Q057 | May the nine Candidate B no-replay runs begin? | Open human approval; dataset is ready. |
| Q058 | May replay or selective adaptation be implemented? | No; blocked until no-replay results reach R1/R2 and are reviewed. |
| Q059 | What numerical replay memory budget should be used? | Unselected and blocked. |
""")

decision = {
    "date": validation["created_at"],
    "id": "D048",
    "decision": "Certify Candidate B local JSONL materialization for human no-replay training approval",
    "materializer": "scripts/materialize_stronger_pilot_candidate_b.py",
    "counts": validation["counts"],
    "manifest_sha256": manifest_hash,
    "validation_sha256": validation_hash,
    "checks_passed": len(validation["checks"]),
    "tests": validation["tests"],
    "status": validation["status"],
    "readiness": validation["replay_readiness"],
    "training": False,
    "evaluation": False,
    "replay": False,
    "selective_adaptation": False,
    "numerical_memory_budget": None,
    "next_prompt": next_prompt,
}

change = {
    "date": validation["created_at"],
    "id": "C036",
    "decision_id": "D048",
    "reason": "Resolve the approved frozen Candidate B assignments into local git-ignored JSONL and validate actual bytes before training",
    "files_changed": [
        ".gitignore",
        "scripts/materialize_stronger_pilot_candidate_b.py",
        "tests/test_stronger_pilot_candidate_b_materialization.py",
        "data/processed/stronger_pilot_candidate_b/**",
        "results/stronger_pilot_materialization_validation.json",
        "docs/stronger_pilot_materialization_report.md",
        "docs/stronger_pilot_assignment_plan.md",
        "docs/stronger_pilot_no_replay_protocol.md",
        "docs/stronger_pilot_candidate_b_protocol.md",
        "docs/replay_readiness_review.md",
        "docs/replay_comparison_protocol.md",
        "docs/decision_log.md",
        "docs/change_log.md",
        "docs/assumptions_and_risks.md",
        "docs/open_questions.md",
        "history/decisions.jsonl",
        "history/changes.jsonl",
    ],
    "validation": [
        "36/36 JSONL files and hashes verified",
        "6960 arm memberships and 3770 unique records verified",
        "21/21 materialization checks PASS",
        "12 focused tests passed; full suite 114 passed, 0 failed, 0 skipped, 1 existing warning",
        "max token length 491; zero truncation-required records",
    ],
    "manifest_sha256": manifest_hash,
    "validation_sha256": validation_hash,
    "scope": "Local JSONL materialization and validation only",
    "training": False,
    "evaluation": False,
    "replay": False,
    "selective_adaptation": False,
    "numerical_memory_budget": None,
}


def append_jsonl_unique(path: Path, record: dict) -> None:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [item.get("id") for item in records]
    if record["id"] in ids:
        records[ids.index(record["id"])] = record
    else:
        records.append(record)
    write(path, "\n".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in records))


append_jsonl_unique(ROOT / "history/decisions.jsonl", decision)
append_jsonl_unique(ROOT / "history/changes.jsonl", change)

print(json.dumps({
    "status": "PASS",
    "manifest_sha256": manifest_hash,
    "validation_sha256": validation_hash,
    "tests": validation["tests"],
    "decision": "D048",
    "change": "C036",
}, indent=2))
