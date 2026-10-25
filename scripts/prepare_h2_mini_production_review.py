"""Prepare the H2-mini production human-review package after calibration PASS.

This script selects deterministic, group-safe review candidates and creates local
raw-text CSV worksheets plus a no-text manifest. It does not assign final dataset
splits, perform judgments, materialize research JSONL, run models, or implement replay.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review"
OUT = REVIEW / "production"
OR_REV = "e36d8b80e81837c8a8f264bbb2a49f1b32c7e272"
OA_REV = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
SEED = 2026092501
PACKAGE_ID = "h2mini-production-review-v1-20260925"
RUBRIC_VERSION = "h2-mini-review-rubric-v1"
ORDER_SEEDS = {"A": 2026092511, "B": 2026092512}
TARGETS = {"B0": 360, "B1": 360, "B2": 360, "B3": 360, "B-short": 150}
REVIEW_FIELDS = [
    "review_id", "text", "source", "source_category", "language",
    "proposed_stratum", "review_role", "reviewer_decision",
    "reviewer_stratum", "reason_code", "confidence", "optional_note",
    "review_seconds",
]
GROUP_FIELDS_OR = [
    "exact_group", "normalized_group", "model_view_group",
    "project_semantic_parent_group", "project_template_group",
]
GROUP_FIELDS_OA = [
    "message_id", "message_tree_id", "exact_group", "normalized_group",
    "model_view_group", "project_semantic_parent_group",
    "project_template_group",
]


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_key(*parts):
    return sha("|".join(map(str, parts)))


def review_id(candidate_id, cohort):
    return "H2PR-" + deterministic_key(PACKAGE_ID, SEED, cohort, candidate_id)[:16].upper()


def group_keys(record, source):
    fields = GROUP_FIELDS_OA if source == "OASST1" else GROUP_FIELDS_OR
    return {
        f"{source}:{field}:{record[field]}"
        for field in fields
        if record.get(field)
    }


def can_claim(record, source, claimed):
    return not (group_keys(record, source) & claimed)


def claim(record, source, claimed):
    keys = group_keys(record, source)
    if keys & claimed:
        raise RuntimeError(f"group collision for {record['candidate_id']}")
    claimed.update(keys)


def percentile(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def length_bins(pool):
    values = [row["deberta_tokens"] for row in pool]
    cuts = [percentile(values, 0.25), percentile(values, 0.50), percentile(values, 0.75)]
    return cuts


def choose_diverse(pool, count, seed, source, claimed, cue_field=None):
    candidates = [row for row in pool if can_claim(row, source, claimed)]
    if len(candidates) < count:
        raise RuntimeError(f"insufficient group-safe pool: need {count}, have {len(candidates)}")
    cuts = length_bins(candidates)
    chosen = []
    bins = Counter()
    cues = Counter()
    while len(chosen) < count:
        scored = []
        for row in candidates:
            if not can_claim(row, source, claimed):
                continue
            length_bin = sum(row["deberta_tokens"] > cut for cut in cuts)
            cue_ids = list(row.get(cue_field, [])) if cue_field else []
            cue_score = sum(cues[cue] for cue in cue_ids) / max(1, len(cue_ids))
            score = (
                bins[length_bin],
                cue_score,
                deterministic_key(seed, row["candidate_id"]),
            )
            scored.append((score, row, length_bin, cue_ids))
        if not scored:
            raise RuntimeError("group-safe selection exhausted before target")
        _, row, length_bin, cue_ids = min(scored, key=lambda item: item[0])
        chosen.append(row)
        candidates.remove(row)
        claim(row, source, claimed)
        bins[length_bin] += 1
        cues.update(cue_ids)
    return chosen


def language_quotas(pool, total):
    counts = Counter(row["lang"] for row in pool)
    languages = sorted(counts)
    base = {language: 1 if counts[language] >= 5 else 0 for language in languages}
    remaining = total - sum(base.values())
    if remaining < 0:
        base = {language: 0 for language in languages}
        remaining = total
    weight_total = sum(counts.values())
    raw = {language: remaining * counts[language] / weight_total for language in languages}
    quotas = {language: base[language] + math.floor(raw[language]) for language in languages}
    left = total - sum(quotas.values())
    order = sorted(
        languages,
        key=lambda language: (
            -(raw[language] - math.floor(raw[language])),
            deterministic_key(SEED, "language", language),
        ),
    )
    for language in order[:left]:
        quotas[language] += 1
    return {language: quota for language, quota in quotas.items() if quota}


def select_oasst(pool, total, cohort, claimed):
    quotas = language_quotas(pool, total)
    selected = []
    for language, quota in sorted(quotas.items()):
        language_pool = [row for row in pool if row["lang"] == language]
        selected.extend(
            choose_diverse(
                language_pool,
                quota,
                f"{SEED}:{cohort}:{language}",
                "OASST1",
                claimed,
            )
        )
    if len(selected) != total:
        raise RuntimeError(f"{cohort}: expected {total}, selected {len(selected)}")
    return selected, quotas


def load_oasst_texts(indices):
    found = {}
    path = (
        ROOT / "data/raw/hard_benign/oasst1" / OA_REV
        / "2023-04-12_oasst_ready.messages.jsonl.gz"
    )
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index in indices:
                found[index] = json.loads(line)
    if set(found) != set(indices):
        raise RuntimeError("not all selected OASST rows were found")
    return found


def write_review_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def assign_review(rows_by_cohort):
    assignments = {row["review_id"]: [] for rows in rows_by_cohort.values() for row in rows}
    lanes = {}
    for index, stratum in enumerate(("B0", "B1", "B2", "B3")):
        rows = list(rows_by_cohort[stratum])
        random.Random(SEED + 100 + index).shuffle(rows)
        validation = rows[:37]
        test = rows[37:112]
        training = rows[112:]
        if not (len(training) == 248 and len(validation) == 37 and len(test) == 75):
            raise RuntimeError("unexpected review-lane allocation")
        for row in validation:
            lanes[row["review_id"]] = "VALIDATION_RESERVE"
            assignments[row["review_id"]] += [
                {"reviewer": "A", "review_role": "DOUBLE_REVIEW"},
                {"reviewer": "B", "review_role": "DOUBLE_REVIEW"},
            ]
        for row in test:
            lanes[row["review_id"]] = "TEST_RESERVE"
            assignments[row["review_id"]] += [
                {"reviewer": "A", "review_role": "DOUBLE_REVIEW"},
                {"reviewer": "B", "review_role": "DOUBLE_REVIEW"},
            ]
        primary_a = training[:124]
        primary_b = training[124:]
        for row in primary_a:
            lanes[row["review_id"]] = "TRAIN_CANDIDATE"
            assignments[row["review_id"]].append(
                {"reviewer": "A", "review_role": "PRIMARY"}
            )
        for row in primary_b:
            lanes[row["review_id"]] = "TRAIN_CANDIDATE"
            assignments[row["review_id"]].append(
                {"reviewer": "B", "review_role": "PRIMARY"}
            )
        audit_a_count = 12 if index % 2 == 0 else 13
        audit_b_count = 13 if index % 2 == 0 else 12
        audit_for_a = sorted(
            primary_b,
            key=lambda row: deterministic_key(SEED, stratum, "audit-A", row["review_id"]),
        )[:audit_a_count]
        audit_for_b = sorted(
            primary_a,
            key=lambda row: deterministic_key(SEED, stratum, "audit-B", row["review_id"]),
        )[:audit_b_count]
        for row in audit_for_a:
            assignments[row["review_id"]].append(
                {"reviewer": "A", "review_role": "SECOND_REVIEW"}
            )
        for row in audit_for_b:
            assignments[row["review_id"]].append(
                {"reviewer": "B", "review_role": "SECOND_REVIEW"}
            )

    for row in rows_by_cohort["B-short"]:
        lanes[row["review_id"]] = "B_SHORT"
        assignments[row["review_id"]] = [
            {"reviewer": "A", "review_role": "DOUBLE_REVIEW"},
            {"reviewer": "B", "review_role": "DOUBLE_REVIEW"},
        ]
    return assignments, lanes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    calibration = json.loads((REVIEW / "calibration_manifest.json").read_text(encoding="utf-8"))
    calibration_analysis = json.loads(
        (REVIEW / "h2_mini_calibration_analysis.json").read_text(encoding="utf-8")
    )
    if calibration_analysis["calibration_gate"]["recommendation"] != "PASS":
        raise RuntimeError("calibration gate is not PASS")
    calibration_pairs = {
        (row["source"], row["source_row_index"]) for row in calibration["records"]
    }

    or_manifest = json.loads(
        (ROOT / "data/manifests/hard_benign/h2_mini_candidates.json").read_text(
            encoding="utf-8"
        )
    )["records"]
    oa_manifest = json.loads(
        (ROOT / "data/manifests/hard_benign/b_short_mini_candidates.json").read_text(
            encoding="utf-8"
        )
    )["records"]
    claimed = set()
    for row in or_manifest:
        if ("OR-Bench", row["source_row_index"]) in calibration_pairs:
            claimed.update(group_keys(row, "OR-Bench"))
    for row in oa_manifest:
        if ("OASST1", row["source_row_index"]) in calibration_pairs:
            claimed.update(group_keys(row, "OASST1"))
    or_manifest = [
        row for row in or_manifest
        if ("OR-Bench", row["source_row_index"]) not in calibration_pairs
        and row["group_safe_after_automated_audit"]
    ]
    oa_manifest = [
        row for row in oa_manifest
        if ("OASST1", row["source_row_index"]) not in calibration_pairs
        and row["group_safe_ordinary_candidate"]
    ]

    rows_by_cohort = {}

    bshort_pool = [row for row in oa_manifest if row["b_short_length_language_candidate"]]
    bshort, bshort_quotas = select_oasst(
        bshort_pool, TARGETS["B-short"], "B-short", claimed
    )
    rows_by_cohort["B-short"] = bshort

    b0, b0_quotas = select_oasst(oa_manifest, TARGETS["B0"], "B0", claimed)
    rows_by_cohort["B0"] = b0

    category_quotas = {}
    for stratum in ("B1", "B2", "B3"):
        pool = [
            row for row in or_manifest
            if row["candidate_label"] == f"{stratum}_candidate"
        ]
        categories = sorted({row["category"] for row in pool})
        if TARGETS[stratum] % len(categories):
            raise RuntimeError(f"{stratum}: target is not divisible by category count")
        per_category = TARGETS[stratum] // len(categories)
        selected = []
        category_quotas[stratum] = {}
        for category in categories:
            category_pool = [row for row in pool if row["category"] == category]
            chosen = choose_diverse(
                category_pool,
                per_category,
                f"{SEED}:{stratum}:{category}",
                "OR-Bench",
                claimed,
                cue_field="cue_family_ids",
            )
            selected.extend(chosen)
            category_quotas[stratum][category] = len(chosen)
        rows_by_cohort[stratum] = selected

    or_raw_path = ROOT / "data/raw/hard_benign/or_bench" / OR_REV / "or-bench-80k.csv"
    with or_raw_path.open("r", encoding="utf-8-sig", newline="") as handle:
        or_raw = list(csv.DictReader(handle))
    oa_indices = {
        row["source_row_index"]
        for cohort in ("B0", "B-short")
        for row in rows_by_cohort[cohort]
    }
    oa_raw = load_oasst_texts(oa_indices)

    prepared = {}
    for cohort, selected in rows_by_cohort.items():
        prepared[cohort] = []
        for row in selected:
            if cohort in ("B0", "B-short"):
                raw = oa_raw[row["source_row_index"]]
                text = raw["text"]
                source = "OASST1"
                source_category = ""
                language = row["lang"]
                proposed = "B0"
                native_id = row["message_id"]
                parent_id = row["parent_id"]
                source_parent = row["message_tree_id"]
                cue_ids = []
            else:
                raw = or_raw[row["source_row_index"]]
                text = raw["prompt"]
                source = "OR-Bench"
                source_category = row["category"]
                language = "en"
                proposed = cohort
                native_id = row.get("native_record_id")
                parent_id = None
                source_parent = row.get("official_parent")
                cue_ids = row["cue_family_ids"]
            if text_sha := sha(text):
                if text_sha != row["raw_text_sha256"]:
                    raise RuntimeError(f"raw text hash mismatch: {row['candidate_id']}")
            rid = review_id(row["candidate_id"], cohort)
            prepared[cohort].append({
                "review_id": rid,
                "candidate_id": row["candidate_id"],
                "text": text,
                "source": source,
                "source_category": source_category,
                "language": language,
                "proposed_stratum": proposed,
                "review_cohort": cohort,
                "source_revision": row["revision"],
                "source_native_id": native_id,
                "source_parent_id": source_parent,
                "source_immediate_parent_id": parent_id,
                "source_row_index": row["source_row_index"],
                "raw_text_sha256": row["raw_text_sha256"],
                "normalized_text_sha256": row["normalized_text_sha256"],
                "model_view_sha256": row["model_view_sha256"],
                "exact_group": row["exact_group"],
                "normalized_group": row["normalized_group"],
                "model_view_group": row["model_view_group"],
                "project_semantic_parent_group": row["project_semantic_parent_group"],
                "project_template_group": row["project_template_group"],
                "cue_family_ids": cue_ids,
                "deberta_tokens": row["deberta_tokens"],
            })

    all_rows = [row for rows in prepared.values() for row in rows]
    if len(all_rows) != sum(TARGETS.values()):
        raise RuntimeError("production candidate total mismatch")
    if len({row["review_id"] for row in all_rows}) != len(all_rows):
        raise RuntimeError("review_id collision")

    assignments, lanes = assign_review(prepared)
    worksheet_rows = {"A": [], "B": []}
    row_lookup = {row["review_id"]: row for row in all_rows}
    for rid, assigned in assignments.items():
        base = row_lookup[rid]
        for assignment in assigned:
            worksheet_rows[assignment["reviewer"]].append({
                "review_id": rid,
                "text": base["text"],
                "source": base["source"],
                "source_category": base["source_category"],
                "language": base["language"],
                "proposed_stratum": base["proposed_stratum"],
                "review_role": assignment["review_role"],
                "reviewer_decision": "",
                "reviewer_stratum": "",
                "reason_code": "",
                "confidence": "",
                "optional_note": "",
                "review_seconds": "",
            })

    worksheet_manifest = {}
    assignment_locations = defaultdict(list)
    for reviewer in ("A", "B"):
        rows = worksheet_rows[reviewer]
        random.Random(ORDER_SEEDS[reviewer]).shuffle(rows)
        files = []
        for batch_number, start in enumerate(range(0, len(rows), 100), start=1):
            batch = rows[start:start + 100]
            name = f"reviewer_{reviewer}_batch_{batch_number:02d}.csv"
            path = OUT / name
            write_review_csv(path, batch)
            files.append({"path": f"review/production/{name}", "rows": len(batch), "sha256": file_sha(path)})
            for row in batch:
                assignment_locations[row["review_id"]].append({
                    "reviewer": reviewer,
                    "review_role": row["review_role"],
                    "batch_filename": name,
                })
        worksheet_manifest[reviewer] = {
            "judgments": len(rows),
            "batches": len(files),
            "files": files,
        }

    manifest_records = []
    for row in sorted(all_rows, key=lambda item: item["review_id"]):
        manifest_records.append({
            key: value for key, value in row.items() if key != "text"
        } | {
            "review_lane": lanes[row["review_id"]],
            "reviewer_assignments": sorted(
                assignment_locations[row["review_id"]],
                key=lambda item: item["reviewer"],
            ),
            "final_train_validation_test_assignment": None,
        })

    manifest = {
        "package_id": PACKAGE_ID,
        "status": "production_review_worksheets_ready_human_review_not_started",
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "rubric_version": RUBRIC_VERSION,
        "calibration": {
            "calibration_id": calibration_analysis["calibration_id"],
            "agreement_rate": calibration_analysis["reviewer_decision"]["agreement_rate"],
            "decision_kappa": calibration_analysis["reviewer_decision"]["cohens_kappa"]["value"],
            "stratum_agreement_rate": calibration_analysis["reviewer_stratum"]["agreement_rate"],
            "sampling_seed": calibration_analysis["calibration_sampling_seed"],
            "manifest_sha256": calibration_analysis["calibration_manifest_sha256"],
            "reviewer_pseudonyms": calibration_analysis["reviewer_pseudonyms"],
        },
        "candidate_sampling_seed": SEED,
        "reviewer_order_seeds": ORDER_SEEDS,
        "source_revisions": {"OR-Bench": OR_REV, "OASST1": OA_REV},
        "candidate_counts": TARGETS,
        "candidate_total": len(all_rows),
        "selection_policy": {
            "calibration_records_excluded": True,
            "detector_outputs_used": False,
            "model_confidence_used": False,
            "NotInject_similarity_used": False,
            "replay_performance_used": False,
            "final_split_used": False,
            "OR_B1_B3": "36 candidates from each of 10 source categories per stratum, with deterministic token-length and cue-family diversity",
            "OASST1": "deterministic source-language proportional allocation with rare-language coverage and token-length diversity",
            "group_exclusion": GROUP_FIELDS_OR + GROUP_FIELDS_OA,
        },
        "review_lane_policy": {
            "per_B0_B3_stratum": {
                "TRAIN_CANDIDATE": 248,
                "VALIDATION_RESERVE": 37,
                "TEST_RESERVE": 75,
                "total": 360,
                "note": "Review lanes preserve proportional oversampling; they are not final train/validation/test assignments.",
            },
            "training_primary_per_reviewer_per_stratum": 124,
            "training_second_review_audit_per_stratum": 25,
            "double_review_per_stratum": 112,
            "B_short_double_review": 150,
        },
        "planned_final_counts": {
            "per_stratum": {"train": 200, "validation": 30, "test": 60, "total": 290},
            "B_short": 120,
            "assigned": False,
        },
        "language_quotas": {"B0": b0_quotas, "B-short": bshort_quotas},
        "OR_category_quotas": category_quotas,
        "worksheet_columns": REVIEW_FIELDS,
        "worksheets": worksheet_manifest,
        "records": manifest_records,
        "raw_text_in_manifest": False,
        "human_judgments_performed": False,
        "final_assignment": False,
        "research_jsonl_materialized": False,
        "training": False,
        "replay": False,
        "scope_note": "This package follows the current explicit target of one 360-candidate B0 pool plus B1-B3 and B-short. Additional ordinary-benign candidates for a separate four-stage H2m-C0 arm are not selected here.",
    }
    path = REVIEW / "production_review_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "package_id": PACKAGE_ID,
        "candidate_counts": TARGETS,
        "reviewer_judgments": {
            reviewer: worksheet_manifest[reviewer]["judgments"]
            for reviewer in ("A", "B")
        },
        "batches": {
            reviewer: worksheet_manifest[reviewer]["batches"]
            for reviewer in ("A", "B")
        },
        "manifest": str(path.relative_to(ROOT)).replace("\\", "/"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
