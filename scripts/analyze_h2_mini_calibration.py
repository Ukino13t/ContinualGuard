"""Analyze two completed independent H2-mini calibration worksheets.

The input worksheets are never modified. Immutable candidate text is verified against
the calibration manifest; if exactly one worksheet has an encoding-damaged text cell,
the hash-matching copy is used in memory and the recovery is recorded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

FIELDS = [
    "review_id", "text", "source", "source_category", "language",
    "proposed_stratum", "reviewer_decision", "reviewer_stratum",
    "reason_code", "confidence", "optional_note", "review_seconds",
]
DECISIONS = [
    "ACCEPT_BENIGN", "REJECT_TRUE_INJECTION", "REJECT_AMBIGUOUS_INTENT",
    "REJECT_HARMFUL_OR_UNSAFE", "REJECT_CONTEXT_MISSING",
    "REJECT_PRIVACY_OR_RIGHTS", "REJECT_QUALITY",
]
STRATA = ["B0", "B1", "B2", "B3", "UNSURE", "NOT_APPLICABLE"]
BENIGN_STRATA = ["B0", "B1", "B2", "B3"]
CONFIDENCE = ["HIGH", "MEDIUM", "LOW"]
METADATA_FIELDS = ["source", "source_category", "language", "proposed_stratum"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_sheet(path: Path):
    data = path.read_bytes()
    errors = []
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = data.decode(encoding)
            reader = csv.DictReader(io.StringIO(text, newline=""))
            if reader.fieldnames != FIELDS:
                raise ValueError(
                    f"{path}: columns must exactly equal {FIELDS}; got {reader.fieldnames}"
                )
            rows = list(reader)
            break
        except UnicodeDecodeError as error:
            errors.append(f"{encoding}: {error}")
    else:
        raise ValueError(f"{path}: unable to decode worksheet ({'; '.join(errors)})")
    if len(rows) != 50:
        raise ValueError(f"{path}: expected 50 rows, found {len(rows)}")
    ids = [row["review_id"] for row in rows]
    duplicates = [rid for rid, count in Counter(ids).items() if count > 1]
    if any(not rid for rid in ids) or duplicates:
        raise ValueError(
            f"{path}: review_id values must be 50 unique nonblank identifiers; "
            f"duplicates={duplicates}"
        )
    return rows, encoding, sha256_bytes(data)


def load_manifest(path: Path):
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    records = {row["review_id"]: row for row in manifest["records"]}
    if len(records) != 50:
        raise ValueError(f"{path}: expected 50 unique manifest records, found {len(records)}")
    return manifest, records, sha256_bytes(manifest_bytes)


def check_pair(a_rows, b_rows, manifest_records):
    a = {row["review_id"]: dict(row) for row in a_rows}
    b = {row["review_id"]: dict(row) for row in b_rows}
    expected_ids = set(manifest_records)
    if set(a) != set(b):
        raise ValueError("Reviewer worksheets do not contain the same review_id set")
    if set(a) != expected_ids:
        missing = sorted(expected_ids - set(a))
        unknown = sorted(set(a) - expected_ids)
        raise ValueError(f"Worksheet IDs differ from manifest; missing={missing}, unknown={unknown}")

    warnings = []
    for rid in sorted(a):
        manifest_row = manifest_records[rid]
        for field in METADATA_FIELDS:
            expected = manifest_row[field]
            if a[rid][field] != expected or b[rid][field] != expected:
                raise ValueError(
                    f"{rid}: immutable metadata field {field!r} differs from manifest"
                )

        expected_hash = manifest_row["text_sha256"]
        a_matches = text_sha256(a[rid]["text"]) == expected_hash
        b_matches = text_sha256(b[rid]["text"]) == expected_hash
        if not a_matches and not b_matches:
            raise ValueError(
                f"{rid}: candidate text in both worksheets differs from manifest hash"
            )
        if a_matches and b_matches:
            continue
        canonical_text = a[rid]["text"] if a_matches else b[rid]["text"]
        damaged_reviewer = "B" if a_matches else "A"
        source_reviewer = "A" if a_matches else "B"
        a[rid]["text"] = canonical_text
        b[rid]["text"] = canonical_text
        warnings.append({
            "type": "immutable_text_encoding_recovery",
            "review_id": rid,
            "damaged_reviewer": damaged_reviewer,
            "canonical_text_source": source_reviewer,
            "human_judgments_changed": False,
        })
    return a, b, warnings


def validate_completion(label, rows):
    warnings = []
    incomplete = []
    for line_number, row in enumerate(rows, start=2):
        rid = row["review_id"]
        for field in ("reviewer_decision", "reviewer_stratum", "confidence"):
            if not row[field].strip():
                incomplete.append(f"{label} row {line_number} {rid}: missing {field}")
        if row["reviewer_decision"] and row["reviewer_decision"] not in DECISIONS:
            incomplete.append(
                f"{label} row {line_number} {rid}: invalid reviewer_decision "
                f"{row['reviewer_decision']!r}"
            )
        if row["reviewer_stratum"] and row["reviewer_stratum"] not in STRATA:
            incomplete.append(
                f"{label} row {line_number} {rid}: invalid reviewer_stratum "
                f"{row['reviewer_stratum']!r}"
            )
        if row["confidence"] and row["confidence"] not in CONFIDENCE:
            incomplete.append(
                f"{label} row {line_number} {rid}: invalid confidence "
                f"{row['confidence']!r}"
            )
        if row["review_seconds"].strip():
            try:
                seconds = float(row["review_seconds"])
                if not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError
            except ValueError:
                incomplete.append(
                    f"{label} row {line_number} {rid}: review_seconds must be "
                    "blank or a positive number"
                )

        decision = row["reviewer_decision"]
        stratum = row["reviewer_stratum"]
        if decision == "ACCEPT_BENIGN" and stratum not in BENIGN_STRATA + ["UNSURE"]:
            warnings.append(f"{label} {rid}: accepted record has stratum {stratum!r}")
        if decision and decision != "ACCEPT_BENIGN" and stratum != "NOT_APPLICABLE":
            warnings.append(f"{label} {rid}: rejected record normally uses NOT_APPLICABLE")
        if decision and decision != "ACCEPT_BENIGN" and not row["reason_code"].strip():
            warnings.append(f"{label} {rid}: rejected record has blank reason_code")
        if (
            decision == "ACCEPT_BENIGN"
            and stratum == "UNSURE"
            and not row["optional_note"].strip()
        ):
            warnings.append(f"{label} {rid}: accepted UNSURE record has no explanatory note")
    return incomplete, warnings


def confusion(rows_a, rows_b, field, labels, eligibility=None):
    matrix = {left: {right: 0 for right in labels} for left in labels}
    used = 0
    for rid, left in rows_a.items():
        right = rows_b[rid]
        if eligibility and not eligibility(left, right):
            continue
        matrix[left[field]][right[field]] += 1
        used += 1
    return {
        "reviewer_A_rows": labels,
        "reviewer_B_columns": labels,
        "counts": matrix,
        "n": used,
    }


def kappa(left, right):
    n = len(left)
    if n < 2:
        return {"value": None, "reason": "fewer than two paired observations"}
    labels = sorted(set(left) | set(right))
    if len(labels) < 2:
        return {
            "value": None,
            "reason": "only one observed category; chance agreement is degenerate",
        }
    observed = sum(a == b for a, b in zip(left, right)) / n
    ca, cb = Counter(left), Counter(right)
    expected = sum((ca[label] / n) * (cb[label] / n) for label in labels)
    if math.isclose(expected, 1.0):
        return {"value": None, "reason": "expected chance agreement equals one"}
    return {"value": (observed - expected) / (1.0 - expected), "reason": None}


def p90(values):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.90 * len(ordered)) - 1)]


def agreement_group(rows_a, rows_b, key):
    groups = defaultdict(list)
    for rid, left in rows_a.items():
        groups[key(left)].append((left, rows_b[rid]))
    output = {}
    for group, pairs in sorted(groups.items(), key=lambda item: str(item[0])):
        decision_matches = sum(
            a["reviewer_decision"] == b["reviewer_decision"] for a, b in pairs
        )
        stratum_matches = sum(
            a["reviewer_stratum"] == b["reviewer_stratum"] for a, b in pairs
        )
        output[str(group) if str(group) else "(blank)"] = {
            "n": len(pairs),
            "decision_agreement": decision_matches / len(pairs),
            "stratum_agreement": stratum_matches / len(pairs),
        }
    return output


def timing(rows):
    values = [
        float(row["review_seconds"])
        for row in rows.values()
        if row["review_seconds"].strip()
    ]
    if not values:
        return {
            "usable_records": 0,
            "missing_records": len(rows),
            "mean_seconds": None,
            "median_seconds": None,
            "p90_seconds": None,
            "total_seconds": None,
            "total_hours": None,
        }
    return {
        "usable_records": len(values),
        "missing_records": len(rows) - len(values),
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "p90_seconds": p90(values),
        "total_seconds": sum(values),
        "total_hours": sum(values) / 3600.0,
    }


def classify_disagreement(a, b):
    decision_diff = a["reviewer_decision"] != b["reviewer_decision"]
    stratum_diff = a["reviewer_stratum"] != b["reviewer_stratum"]
    confidence_diff = a["confidence"] != b["confidence"]
    benign_vs_reject = (
        (a["reviewer_decision"] == "ACCEPT_BENIGN")
        ^ (b["reviewer_decision"] == "ACCEPT_BENIGN")
    )
    safety_critical = {
        a["reviewer_decision"], b["reviewer_decision"]
    } == {"ACCEPT_BENIGN", "REJECT_TRUE_INJECTION"}
    return {
        "decision_disagreement": decision_diff,
        "benign_vs_reject": benign_vs_reject,
        "safety_critical_true_injection_vs_accept": safety_critical,
        "stratum_only": stratum_diff and not decision_diff,
        "confidence_only": confidence_diff and not decision_diff and not stratum_diff,
        "any_reportable": decision_diff or stratum_diff or confidence_diff,
    }


def write_disagreements(path, rows_a, rows_b):
    fields = [
        "review_id", "text", "source", "proposed_stratum",
        "reviewer_A_decision", "reviewer_B_decision",
        "reviewer_A_stratum", "reviewer_B_stratum",
        "reviewer_A_confidence", "reviewer_B_confidence",
        "reviewer_A_note", "reviewer_B_note",
    ]
    rows = []
    categories = Counter()
    for rid in sorted(rows_a):
        a, b = rows_a[rid], rows_b[rid]
        flags = classify_disagreement(a, b)
        for name, value in flags.items():
            if name != "any_reportable" and value:
                categories[name] += 1
        if not flags["any_reportable"]:
            continue
        rows.append({
            "review_id": rid,
            "text": a["text"],
            "source": a["source"],
            "proposed_stratum": a["proposed_stratum"],
            "reviewer_A_decision": a["reviewer_decision"],
            "reviewer_B_decision": b["reviewer_decision"],
            "reviewer_A_stratum": a["reviewer_stratum"],
            "reviewer_B_stratum": b["reviewer_stratum"],
            "reviewer_A_confidence": a["confidence"],
            "reviewer_B_confidence": b["confidence"],
            "reviewer_A_note": a["optional_note"],
            "reviewer_B_note": b["optional_note"],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows), dict(categories)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("review/calibration_manifest.json"),
    )
    parser.add_argument(
        "--analysis-json",
        type=Path,
        default=Path("review/h2_mini_calibration_analysis.json"),
    )
    parser.add_argument(
        "--disagreements",
        type=Path,
        default=Path("review/h2_mini_calibration_disagreements.csv"),
    )
    args = parser.parse_args()

    try:
        manifest, manifest_records, manifest_hash = load_manifest(args.manifest)
        a_list, encoding_a, input_hash_a = read_sheet(args.reviewer_a)
        b_list, encoding_b, input_hash_b = read_sheet(args.reviewer_b)
        rows_a, rows_b, identity_warnings = check_pair(
            a_list, b_list, manifest_records
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))

    incomplete_a, warnings_a = validate_completion("reviewer A", a_list)
    incomplete_b, warnings_b = validate_completion("reviewer B", b_list)
    if incomplete_a or incomplete_b:
        status = {
            "status": "INCOMPLETE",
            "completed_record_counts": {
                "reviewer_A": sum(
                    all(row[field].strip() for field in (
                        "reviewer_decision", "reviewer_stratum", "confidence"
                    ))
                    for row in a_list
                ),
                "reviewer_B": sum(
                    all(row[field].strip() for field in (
                        "reviewer_decision", "reviewer_stratum", "confidence"
                    ))
                    for row in b_list
                ),
            },
            "missing_or_invalid_fields": incomplete_a + incomplete_b,
            "warnings": identity_warnings + warnings_a + warnings_b,
            "outputs_written": False,
        }
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 2

    rids = sorted(rows_a)
    decisions_a = [rows_a[rid]["reviewer_decision"] for rid in rids]
    decisions_b = [rows_b[rid]["reviewer_decision"] for rid in rids]
    strata_a = [rows_a[rid]["reviewer_stratum"] for rid in rids]
    strata_b = [rows_b[rid]["reviewer_stratum"] for rid in rids]
    n = len(rids)
    decision_matches = sum(a == b for a, b in zip(decisions_a, decisions_b))
    stratum_matches = sum(a == b for a, b in zip(strata_a, strata_b))
    one_accept = sum(
        (a == "ACCEPT_BENIGN") ^ (b == "ACCEPT_BENIGN")
        for a, b in zip(decisions_a, decisions_b)
    )
    both_accept = sum(
        a == b == "ACCEPT_BENIGN" for a, b in zip(decisions_a, decisions_b)
    )
    neither_accept = n - one_accept - both_accept
    true_injection_disagreements = sum(
        (a == "REJECT_TRUE_INJECTION") ^ (b == "REJECT_TRUE_INJECTION")
        for a, b in zip(decisions_a, decisions_b)
    )
    safety_critical = sum(
        {a, b} == {"REJECT_TRUE_INJECTION", "ACCEPT_BENIGN"}
        for a, b in zip(decisions_a, decisions_b)
    )

    by_proposed = agreement_group(
        rows_a, rows_b, lambda row: row["proposed_stratum"]
    )
    by_source = agreement_group(rows_a, rows_b, lambda row: row["source"])
    by_category = agreement_group(
        {
            rid: row
            for rid, row in rows_a.items()
            if row["source_category"]
        },
        rows_b,
        lambda row: row["source_category"],
    )

    operational_details = {}
    for label in ("B1", "B2", "B3"):
        pairs = [
            (rows_a[rid], rows_b[rid])
            for rid in rids
            if rows_a[rid]["proposed_stratum"] == label
        ]
        accepted_pairs = [
            (a, b)
            for a, b in pairs
            if a["reviewer_decision"] == b["reviewer_decision"] == "ACCEPT_BENIGN"
            and a["reviewer_stratum"] in BENIGN_STRATA
            and b["reviewer_stratum"] in BENIGN_STRATA
        ]
        accepted_stratum_agreement = (
            sum(
                a["reviewer_stratum"] == b["reviewer_stratum"]
                for a, b in accepted_pairs
            )
            / len(accepted_pairs)
            if accepted_pairs
            else None
        )
        operational_details[label] = {
            "decision_agreement": by_proposed[label]["decision_agreement"],
            "both_accepted_with_B0_B3_labels": len(accepted_pairs),
            "accepted_stratum_agreement": accepted_stratum_agreement,
            "passes_proxy": (
                by_proposed[label]["decision_agreement"] >= 0.80
                and accepted_stratum_agreement is not None
                and accepted_stratum_agreement >= 0.70
            ),
        }
    operationally_understandable = all(
        item["passes_proxy"] for item in operational_details.values()
    )

    source_rates = [
        item["decision_agreement"] for item in by_source.values()
    ]
    source_specific_misunderstanding = (
        any(item["decision_agreement"] < 0.80 for item in by_source.values())
        or (
            max(source_rates) - min(source_rates) > 0.20
            if len(source_rates) > 1
            else False
        )
    )

    raw_agreement = decision_matches / n
    if safety_critical > 0 or raw_agreement < 0.80:
        gate = "FAIL"
        recommendation = (
            "Do not begin production review; resolve safety-critical disagreements "
            "and revise the rubric before recalibration."
        )
    elif (
        raw_agreement >= 0.90
        and operationally_understandable
        and not source_specific_misunderstanding
    ):
        gate = "PASS"
        recommendation = (
            "Proceed to production-review preparation under "
            "h2-mini-review-rubric-v1."
        )
    else:
        gate = "REVISE"
        recommendation = (
            "Revise the ambiguous rubric clauses, then run another independent "
            "30-50 record calibration before production review."
        )

    timing_a = timing(rows_a)
    timing_b = timing(rows_b)
    combined_seconds = [
        float(rows[rid]["review_seconds"])
        for rows in (rows_a, rows_b)
        for rid in rids
        if rows[rid]["review_seconds"].strip()
    ]
    if combined_seconds:
        mean_seconds = statistics.fmean(combined_seconds)
        median_seconds = statistics.median(combined_seconds)
        combined_timing = {
            "usable_judgments": len(combined_seconds),
            "missing_judgments": 2 * n - len(combined_seconds),
            "mean_seconds": mean_seconds,
            "median_seconds": median_seconds,
            "p90_seconds": p90(combined_seconds),
            "total_person_hours": sum(combined_seconds) / 3600.0,
        }
        projected = {
            "planned_production_judgments": 3040,
            "person_hours_at_median_speed": 3040 * median_seconds / 3600.0,
            "person_hours_at_mean_speed": 3040 * mean_seconds / 3600.0,
            "estimated_elapsed_hours_with_two_reviewers_at_median_speed":
                3040 * median_seconds / 7200.0,
            "estimated_elapsed_hours_with_two_reviewers_at_mean_speed":
                3040 * mean_seconds / 7200.0,
        }
        projected["size_reduction_review_flag"] = (
            projected["person_hours_at_median_speed"] > 15
            or projected["person_hours_at_mean_speed"] > 15
        )
    else:
        combined_timing = {
            "usable_judgments": 0,
            "missing_judgments": 2 * n,
            "mean_seconds": None,
            "median_seconds": None,
            "p90_seconds": None,
            "total_person_hours": None,
        }
        projected = {
            "planned_production_judgments": 3040,
            "person_hours_at_median_speed": None,
            "person_hours_at_mean_speed": None,
            "estimated_elapsed_hours_with_two_reviewers_at_median_speed": None,
            "estimated_elapsed_hours_with_two_reviewers_at_mean_speed": None,
            "size_reduction_review_flag": None,
            "reason": "No usable review_seconds values; workload was not invented.",
        }

    disagreement_count, disagreement_types = write_disagreements(
        args.disagreements, rows_a, rows_b
    )
    result = {
        "status": "COMPLETE",
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "calibration_id": manifest["calibration_id"],
        "calibration_sampling_seed": manifest["sampling_seed"],
        "calibration_manifest_sha256": manifest_hash,
        "rubric_version": (
            "h2-mini-review-rubric-v1" if gate == "PASS" else None
        ),
        "reviewer_pseudonyms": {"reviewer_A": "A", "reviewer_B": "B"},
        "input_worksheets": {
            "reviewer_A": {
                "path": str(args.reviewer_a),
                "encoding": encoding_a,
                "sha256": input_hash_a,
            },
            "reviewer_B": {
                "path": str(args.reviewer_b),
                "encoding": encoding_b,
                "sha256": input_hash_b,
            },
        },
        "completed_record_counts": {"reviewer_A": 50, "reviewer_B": 50},
        "missing_field_warnings": [],
        "identity_warnings": identity_warnings,
        "review_warnings": warnings_a + warnings_b,
        "reviewer_decision": {
            "agreement_count": decision_matches,
            "agreement_rate": raw_agreement,
            "disagreement_count": n - decision_matches,
            "disagreement_rate": 1.0 - raw_agreement,
            "confusion_matrix": confusion(
                rows_a, rows_b, "reviewer_decision", DECISIONS
            ),
            "cohens_kappa": kappa(decisions_a, decisions_b),
        },
        "accept_benign": {
            "both_accept": both_accept,
            "binary_agreement_count": both_accept + neither_accept,
            "binary_agreement_rate": (both_accept + neither_accept) / n,
            "one_accept_only": one_accept,
            "neither_accept": neither_accept,
        },
        "true_injection": {
            "disagreement_count": true_injection_disagreements,
            "safety_critical_true_injection_vs_accept_benign": safety_critical,
        },
        "reviewer_stratum": {
            "agreement_count": stratum_matches,
            "agreement_rate": stratum_matches / n,
            "confusion_matrix_all_labels": confusion(
                rows_a, rows_b, "reviewer_stratum", STRATA
            ),
            "B0_B1_B2_B3_confusion_matrix": confusion(
                rows_a,
                rows_b,
                "reviewer_stratum",
                BENIGN_STRATA,
                eligibility=lambda a, b: (
                    a["reviewer_stratum"] in BENIGN_STRATA
                    and b["reviewer_stratum"] in BENIGN_STRATA
                ),
            ),
            "cohens_kappa": kappa(strata_a, strata_b),
        },
        "disagreement_types": disagreement_types,
        "agreement_by_proposed_stratum": by_proposed,
        "agreement_by_source": by_source,
        "agreement_by_source_category": by_category,
        "timing": {
            "reviewer_A": timing_a,
            "reviewer_B": timing_b,
            "combined_judgments": combined_timing,
        },
        "calibration_gate": {
            "recommendation": gate,
            "raw_agreement_threshold": 0.90,
            "operational_understanding_proxy": {
                "definition": (
                    "For each proposed B1/B2/B3: decision agreement >= 0.80 "
                    "and reviewer-stratum agreement >= 0.70 among jointly "
                    "accepted records with B0-B3 labels."
                ),
                "details": operational_details,
                "passes": operationally_understandable,
            },
            "source_specific_misunderstanding_proxy": {
                "definition": (
                    "Flag if any source decision agreement is below 0.80 or "
                    "the gap between source agreement rates exceeds 0.20."
                ),
                "flagged": source_specific_misunderstanding,
            },
            "recommendation_text": recommendation,
        },
        "projected_production_review_time": projected,
        "disagreement_output": {
            "path": str(args.disagreements),
            "rows": disagreement_count,
            "contains_raw_text": True,
        },
    }

    args.analysis_json.parent.mkdir(parents=True, exist_ok=True)
    args.analysis_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "COMPLETE",
        "gate": gate,
        "analysis_json": str(args.analysis_json),
        "disagreements_csv": str(args.disagreements),
        "disagreements": disagreement_count,
        "identity_warnings": identity_warnings,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
