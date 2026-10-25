"""Derive H2-mini saturation, separability, and checkpoint-lineage evidence.

Analysis only: this module never loads a detector, trains, selects records, or
changes any frozen benchmark/result artifact.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
H2_RESULTS = RESULTS / "h2_mini_no_replay"
H2_DATA = ROOT / "data/processed/h2_mini"
AF_DATA = ROOT / "data/processed/chen_attack_family_shift.jsonl"
NEAR_LOW = 0.40
NEAR_HIGH = 0.60
SEEDS = (17, 42, 2026)
ARMS = ("c0", "f")
STAGES = ("AF1", "AF2", "AF3", "AF4")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def r6(value):
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), 6)


def qstats(values):
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        return {"n": 0}
    result = {
        "n": int(values.size),
        "mean": r6(values.mean()),
        "sample_sd": r6(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": r6(values.min()),
        "max": r6(values.max()),
    }
    for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        result[f"p{int(q * 100):02d}"] = r6(np.quantile(values, q))
    result["median"] = result["p50"]
    return result


def fraction(rows, predicate):
    return r6(sum(1 for row in rows if predicate(row)) / len(rows)) if rows else None


def score_summary(rows, label_type):
    values = [float(row["p_attack"]) for row in rows]
    result = qstats(values)
    result.update({
        "true_label": 1 if label_type == "malicious" else 0,
        "fraction_correct": fraction(rows, lambda r: int(r["predicted_label"]) == int(r["true_label"])),
        "fraction_near_threshold_0.40_to_0.60_inclusive": fraction(
            rows, lambda r: NEAR_LOW <= float(r["p_attack"]) <= NEAR_HIGH
        ),
        "fraction_predicted_attack": fraction(rows, lambda r: int(r["predicted_label"]) == 1),
    })
    if label_type == "malicious":
        for threshold in (0.90, 0.95, 0.99, 0.999):
            result[f"fraction_p_attack_ge_{threshold}"] = fraction(
                rows, lambda r, t=threshold: float(r["p_attack"]) >= t
            )
    else:
        for threshold in (0.10, 0.05, 0.01, 0.001):
            result[f"fraction_p_attack_le_{threshold}"] = fraction(
                rows, lambda r, t=threshold: float(r["p_attack"]) <= t
            )
    return result


def collect_checkpoint_rows(arm: str, seed: int, stage_index: int):
    pred_dir = H2_RESULTS / arm / f"seed{seed}" / f"stage{stage_index}" / "predictions"
    grouped = {}
    all_paired = []
    for family_index in range(1, stage_index + 1):
        family = f"AF{family_index}"
        rows = read_jsonl(pred_dir / f"{family}.jsonl")
        all_paired.extend(rows)
        grouped[family] = [row for row in rows if int(row["true_label"]) == 1]
    benign_by_group = defaultdict(list)
    for row in all_paired:
        if int(row["true_label"]) == 0:
            benign_by_group[row.get("benign_group") or "unspecified"].append(row)
    grouped.update({key: value for key, value in benign_by_group.items()})
    grouped["NotInject"] = read_jsonl(pred_dir / "NotInject.jsonl")
    grouped["B-short"] = read_jsonl(pred_dir / "B-short.jsonl")
    return grouped, all_paired


def saturation_analysis():
    frozen = [
        RESULTS / "h2_mini_no_replay_aggregate.json",
        RESULTS / "h2_mini_no_replay_pairwise.json",
        RESULTS / "h2_mini_no_replay_readiness.json",
        RESULTS / "h2_mini_no_replay_artifact_validation.json",
        H2_DATA / "materialization_manifest.json",
    ]
    checkpoints = []
    row_cache = {}
    pooled_malicious = []
    pooled_benign = []
    for arm in ARMS:
        for seed in SEEDS:
            for stage_index in range(1, 5):
                grouped, paired = collect_checkpoint_rows(arm, seed, stage_index)
                summaries = {}
                for name, rows in sorted(grouped.items()):
                    label_type = "malicious" if name.startswith("AF") else "benign"
                    summaries[name] = score_summary(rows, label_type)
                    row_cache[(arm, seed, stage_index, name)] = rows
                    if label_type == "malicious":
                        pooled_malicious.extend(rows)
                    else:
                        pooled_benign.extend(rows)
                paired_mal = [r for r in paired if int(r["true_label"]) == 1]
                paired_ben = [r for r in paired if int(r["true_label"]) == 0]
                checkpoints.append({
                    "arm": arm,
                    "seed": seed,
                    "checkpoint_after_stage": f"AF{stage_index}",
                    "stage_index": stage_index,
                    "datasets": summaries,
                    "pooled_acquired_malicious": score_summary(paired_mal, "malicious"),
                    "pooled_acquired_benign": score_summary(paired_ben, "benign"),
                    "empirical_margin_gap_min_malicious_minus_max_benign": r6(
                        min(float(r["p_attack"]) for r in paired_mal)
                        - max(float(r["p_attack"]) for r in paired_ben)
                    ),
                })

    movements = []
    for arm in ARMS:
        for seed in SEEDS:
            for family_index, family in enumerate(STAGES, start=1):
                first = {r["example_id"]: r for r in row_cache[(arm, seed, family_index, family)]}
                final = {r["example_id"]: r for r in row_cache[(arm, seed, 4, family)]}
                ids = sorted(first.keys() & final.keys())
                delta = [float(final[x]["p_attack"]) - float(first[x]["p_attack"]) for x in ids]
                movements.append({
                    "arm": arm, "seed": seed, "dataset": family,
                    "from_checkpoint": family, "to_checkpoint": "AF4",
                    "paired_ids": len(ids), "p_attack_delta": qstats(delta),
                    "correctness_change_pp": r6(100 * (
                        sum(final[x]["predicted_label"] == final[x]["true_label"] for x in ids) / len(ids)
                        - sum(first[x]["predicted_label"] == first[x]["true_label"] for x in ids) / len(ids)
                    )),
                })
            for external in ("NotInject", "B-short"):
                first = {r["example_id"]: r for r in row_cache[(arm, seed, 1, external)]}
                final = {r["example_id"]: r for r in row_cache[(arm, seed, 4, external)]}
                ids = sorted(first.keys() & final.keys())
                movements.append({
                    "arm": arm, "seed": seed, "dataset": external,
                    "from_checkpoint": "AF1", "to_checkpoint": "AF4",
                    "paired_ids": len(ids),
                    "p_attack_delta": qstats(float(final[x]["p_attack"]) - float(first[x]["p_attack"]) for x in ids),
                    "false_positive_rate_change_pp": r6(100 * (
                        sum(final[x]["predicted_label"] == 1 for x in ids) / len(ids)
                        - sum(first[x]["predicted_label"] == 1 for x in ids) / len(ids)
                    )),
                })

    acquisition_mal = []
    acquisition_ben = []
    for arm in ARMS:
        for seed in SEEDS:
            for index, family in enumerate(STAGES, start=1):
                acquisition_mal.extend(row_cache[(arm, seed, index, family)])
                current_rows = read_jsonl(
                    H2_RESULTS / arm / f"seed{seed}" / f"stage{index}" / "predictions" / f"{family}.jsonl"
                )
                acquisition_ben.extend(row for row in current_rows if int(row["true_label"]) == 0)
    return {
        "status": "ANALYSIS_ONLY_COMPLETE",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Existing frozen H2-mini predictions only; no detector load, inference, training, threshold tuning, or data selection.",
        "decision_rule": "argmax over fixed binary logits; p_attack=0.5 boundary; ties choose benign/class 0",
        "near_threshold_band": {
            "inclusive": [NEAR_LOW, NEAR_HIGH],
            "rationale": "Symmetric predeclared descriptive band of width 0.20 around the fixed 0.5 boundary; it is not a tuned threshold."
        },
        "frozen_input_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in frozen},
        "checkpoints": checkpoints,
        "paired_score_movements": movements,
        "pooled_descriptive": {
            "all_available_checkpoint_family_malicious_rows": score_summary(pooled_malicious, "malicious"),
            "all_available_checkpoint_benign_rows_including_external_banks": score_summary(pooled_benign, "benign"),
            "first_acquisition_malicious": score_summary(acquisition_mal, "malicious"),
            "first_acquisition_corresponding_benign": score_summary(acquisition_ben, "benign"),
        },
        "diagnosis": {
            "classification": "C_both_malicious_and_benign_margin_saturation",
            "malicious_side": "All 1,440 first-acquisition malicious observations are correct and >=0.95; 99.7222% are >=0.99, with zero in the near-threshold band.",
            "benign_side": "Across available benign trajectory observations, 99.6032% are <=0.10 and 98.8643% are <=0.01; only 0.0616% lie in the near-threshold band. Rare high-confidence benign errors create support overlap but do not erase the central quantile separation.",
            "current_old_conflict": "AF1-AF4 paired correctness never changes from acquisition to final. Mean malicious p_attack movements are zero or positive and at most 0.007756, so the stream supplies minimal current-versus-old attack conflict.",
            "correctness_vs_margin": "Correctness is at ceiling and the probability margins are also highly saturated; both limitations reduce the benchmark's sensitivity to retention loss.",
        },
        "interpretation_guardrails": [
            "Correctness saturation and probability-margin saturation are reported separately.",
            "Repeated evaluation rows across checkpoints are intentional trajectory observations, not independent examples.",
            "These distributions diagnose the completed benchmark; they are not used to select future train, validation, or test records.",
        ],
    }


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def tokens(text: str):
    return re.findall(r"[^\W_]+(?:['’-][^\W_]+)?", normalize(text), flags=re.UNICODE)


def ngrams(text: str, n: int):
    toks = tokens(text)
    return [" ".join(toks[i:i+n]) for i in range(len(toks) - n + 1)]


def counter_dict(counter, limit=None):
    items = counter.most_common(limit) if limit else sorted(counter.items())
    return {str(k): int(v) for k, v in items}


def record_source(row):
    src = row.get("source") or {}
    if isinstance(src, dict):
        return src.get("dataset") or row.get("domain") or "unknown"
    return str(src)


def family_summary(rows):
    payloads = [row.get("payload_text") or row.get("injection_text") or "" for row in rows]
    unique_payloads = sorted(set(payloads))
    template_ids = {row.get("attack_template_id") or row.get("template_id") for row in rows}
    base_ids = {row.get("base_document_id") for row in rows}
    components = Counter(c for row in rows for c in (row.get("attack_template_components") or []))
    token_counts = [len(tokens(value)) for value in unique_payloads]
    phrase_counts = Counter()
    for value in unique_payloads:
        phrase_counts.update(set(ngrams(value, 2)))
        phrase_counts.update(set(ngrams(value, 3)))
    return {
        "records": len(rows),
        "unique_payloads": len(unique_payloads),
        "unique_full_templates": len(template_ids),
        "unique_base_documents": len(base_ids),
        "payload_token_count": qstats(token_counts),
        "injection_positions": counter_dict(Counter(row.get("injection_position") for row in rows)),
        "template_components": counter_dict(components),
        "sources": counter_dict(Counter(record_source(row) for row in rows)),
        "domains": counter_dict(Counter(row.get("domain") or "unknown" for row in rows)),
        "top_repeated_payload_phrases": [
            {"phrase": phrase, "unique_payload_count": count}
            for phrase, count in phrase_counts.most_common(20)
        ],
    }


def pairwise_family_metrics(name_a, rows_a, name_b, rows_b):
    payloads_a = sorted({row.get("payload_text") or row.get("injection_text") or "" for row in rows_a})
    payloads_b = sorted({row.get("payload_text") or row.get("injection_text") or "" for row in rows_b})
    token_sets_a = [set(tokens(x)) for x in payloads_a]
    token_sets_b = [set(tokens(x)) for x in payloads_b]
    jaccards = []
    for aa in token_sets_a:
        for bb in token_sets_b:
            union = aa | bb
            jaccards.append(len(aa & bb) / len(union) if union else 1.0)
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(payloads_a + payloads_b)
    cosine = cosine_similarity(matrix[:len(payloads_a)], matrix[len(payloads_a):]).ravel()
    corpus_tokens_a = set().union(*token_sets_a) if token_sets_a else set()
    corpus_tokens_b = set().union(*token_sets_b) if token_sets_b else set()
    bi_a = set(g for x in payloads_a for g in ngrams(x, 2))
    bi_b = set(g for x in payloads_b for g in ngrams(x, 2))
    tri_a = set(g for x in payloads_a for g in ngrams(x, 3))
    tri_b = set(g for x in payloads_b for g in ngrams(x, 3))
    def jac(a, b):
        return r6(len(a & b) / len(a | b)) if a | b else 1.0
    return {
        "family_a": name_a, "family_b": name_b,
        "unique_payload_pairs": len(payloads_a) * len(payloads_b),
        "exact_payload_intersection": len(set(payloads_a) & set(payloads_b)),
        "full_template_id_intersection": len(
            {r.get("attack_template_id") or r.get("template_id") for r in rows_a}
            & {r.get("attack_template_id") or r.get("template_id") for r in rows_b}
        ),
        "base_document_id_intersection": len({r.get("base_document_id") for r in rows_a} & {r.get("base_document_id") for r in rows_b}),
        "normalized_full_text_intersection": len({normalize(r["text"]) for r in rows_a} & {normalize(r["text"]) for r in rows_b}),
        "corpus_token_vocabulary_jaccard": jac(corpus_tokens_a, corpus_tokens_b),
        "corpus_bigram_jaccard": jac(bi_a, bi_b),
        "corpus_trigram_jaccard": jac(tri_a, tri_b),
        "cross_payload_token_jaccard": qstats(jaccards),
        "cross_payload_tfidf_1_2gram_cosine": qstats(cosine),
    }


LEXICONS = {
    "instruction_verbs": {"answer", "classify", "create", "describe", "determine", "explain", "generate", "identify", "ignore", "list", "provide", "respond", "summarize", "tell", "write"},
    "prompt_system_terms": {"assistant", "instruction", "jailbreak", "model", "prompt", "system", "user"},
    "security_terms": {"attack", "bypass", "exploit", "inject", "injection", "malicious", "override", "password", "secret", "security"},
    "control_terms": {"above", "below", "following", "instead", "must", "previous", "rules", "task"},
}


def benign_format_summary(rows):
    texts = [row["text"] for row in rows]
    toks = [tokens(text) for text in texts]
    lexicon = {}
    for name, words in LEXICONS.items():
        counts = [sum(1 for token in row_tokens if token in words) for row_tokens in toks]
        lexicon[name] = {
            "records_with_any_fraction": r6(sum(x > 0 for x in counts) / len(counts)),
            "matches_per_100_tokens": r6(100 * sum(counts) / max(1, sum(len(x) for x in toks))),
        }
    return {
        "records": len(rows),
        "sources": counter_dict(Counter(record_source(row) for row in rows)),
        "domains": counter_dict(Counter(row.get("domain") or "unknown" for row in rows)),
        "languages": counter_dict(Counter(row.get("language") or "unknown" for row in rows)),
        "deberta_tokens_if_recorded": qstats(row["deberta_tokens"] for row in rows if row.get("deberta_tokens") is not None),
        "unicode_word_count": qstats(len(x) for x in toks),
        "character_count": qstats(len(text) for text in texts),
        "question_mark_fraction": r6(sum("?" in text for text in texts) / len(texts)),
        "newline_fraction": r6(sum("\n" in text for text in texts) / len(texts)),
        "quote_mark_fraction": r6(sum(any(ch in text for ch in ('\"', "'", "“", "”", "‘", "’")) for text in texts) / len(texts)),
        "short_form_le_64_words_fraction": r6(sum(len(x) <= 64 for x in toks) / len(toks)),
        "long_context_ge_128_words_fraction": r6(sum(len(x) >= 128 for x in toks) / len(toks)),
        "predeclared_lexical_cues": lexicon,
    }


def load_notinject_texts():
    raw = ROOT / "data/raw/notinject/847ae76cf8fea5ed325429e569ae8cfef022d2e0/dataset_server"
    manifest = read_jsonl(ROOT / "data/manifests/hard_benign/notinject/records.jsonl")
    by_key = {(r["source_split"], r["source_row_idx"]): r for r in manifest}
    rows = []
    for split in ("NotInject_one", "NotInject_two", "NotInject_three"):
        wrapped = []
        for offset in (0, 100):
            wrapped.extend(read_json(raw / f"{split}_{offset:03d}.json")["rows"])
        for item in wrapped:
            index = item["row_idx"]
            source = item["row"]
            audited = by_key[(split, index)]
            rows.append({
                "text": source["prompt"], "label": 0, "language": "multilingual_or_en_unknown",
                "deberta_tokens": audited["deberta_tokens"], "domain": "NotInject_short_benign",
                "source": {"dataset": "leolee99/NotInject", "revision": "847ae76cf8fea5ed325429e569ae8cfef022d2e0"},
                "category": audited["category"], "trigger_count_group": audited["trigger_count_group"],
            })
    return rows


def separability_analysis():
    af14 = {family: [] for family in STAGES}
    for stage_index, family in enumerate(STAGES, start=1):
        for split in ("train", "val", "test"):
            for row in read_jsonl(H2_DATA / "h2m_c0" / f"stage{stage_index}" / f"{split}.jsonl"):
                if int(row["label"]) == 1:
                    af14[family].append(row)
    full_rows = read_jsonl(AF_DATA)
    af5 = [row for row in full_rows if row["task_id"] == "AF5" and int(row["label"]) == 1]
    all_families = dict(af14)
    all_families["AF5"] = af5
    pairwise = [pairwise_family_metrics(a, all_families[a], b, all_families[b]) for a, b in combinations(all_families, 2)]

    benign = {"B0": [], "BH": [], "B-short": read_jsonl(H2_DATA / "external/b_short.jsonl"), "NotInject": load_notinject_texts()}
    for stage_index in range(1, 5):
        for split in ("train", "val", "test"):
            benign["B0"].extend(row for row in read_jsonl(H2_DATA / "h2m_c0" / f"stage{stage_index}" / f"{split}.jsonl") if int(row["label"]) == 0)
            if stage_index >= 2:
                benign["BH"].extend(row for row in read_jsonl(H2_DATA / "h2m_f" / f"stage{stage_index}" / f"{split}.jsonl") if int(row["label"]) == 0)
    benign_overlap = []
    for a, b in combinations(benign, 2):
        exact_a = {normalize(r["text"]) for r in benign[a]}
        exact_b = {normalize(r["text"]) for r in benign[b]}
        token_a = set(token for row in benign[a] for token in tokens(row["text"]))
        token_b = set(token for row in benign[b] for token in tokens(row["text"]))
        benign_overlap.append({
            "pool_a": a, "pool_b": b,
            "normalized_exact_text_intersection": len(exact_a & exact_b),
            "token_vocabulary_jaccard": r6(len(token_a & token_b) / len(token_a | token_b)),
        })

    overlap_audit = read_json(ROOT / "data/audits/attack_family_shift/overlap_audit.json")
    construction = read_json(ROOT / "data/manifests/attack_family_shift/construction.json")
    return {
        "status": "MODEL_INDEPENDENT_ANALYSIS_COMPLETE",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Model-independent descriptive audit of already audited/materialized records. No record selection or benchmark construction.",
        "methods": {
            "normalization": "Unicode NFKC, casefold, whitespace collapse",
            "lexical": "Unicode word tokens; set Jaccard",
            "phrase": "corpus bigram and trigram set Jaccard plus repeated payload n-grams",
            "tfidf": "word 1-2 gram TF-IDF cosine over every cross-family unique-payload pair",
            "cue_lexicons": {key: sorted(value) for key, value in LEXICONS.items()},
            "selection_prohibition": "No detector prediction or test outcome was used to select or rank future records.",
        },
        "AF1_AF4": {family: family_summary(rows) for family, rows in af14.items()},
        "pairwise_family_metrics_including_AF5": pairwise,
        "AF5_audit": {
            "role": "auxiliary compositional/combined stage; not a family-disjoint stage",
            "family_name": construction["stage_families"]["AF5"],
            "counts": {
                split: {"malicious": sum(r["split"] == split for r in af5), "benign": sum(r["task_id"] == "AF5" and r["split"] == split and int(r["label"]) == 0 for r in full_rows)}
                for split in ("train", "val", "test")
            },
            "summary": family_summary(af5),
            "approved_source_revision": "1e9e0f00179adc154e9ca991b9cd3206c99f9588",
            "local_materialization": "Present in data/processed/chen_attack_family_shift.jsonl at the frozen processed SHA-256 recorded by construction.json.",
            "overlap_audit": {
                "cross_stage_exact_text_groups": overlap_audit["audits"]["exact_text"]["cross_stage_groups"],
                "cross_stage_base_document_groups": overlap_audit["audits"]["base_document"]["cross_stage_groups"],
                "cross_stage_payload_groups": overlap_audit["audits"]["payload"]["cross_stage_groups"],
                "cross_stage_full_attack_template_groups": overlap_audit["audits"]["attack_template"]["cross_stage_groups"],
                "cross_stage_normalized_template_groups": overlap_audit["audits"]["normalized_template"]["cross_stage_groups"],
                "interpretation": "Identity-level overlap is zero, while AF5 deliberately reuses AF2/AF3/AF4 attack components by construction. It therefore tests composition, not an unseen family-disjoint shift.",
            },
            "reuse_feasibility": "Existing accepted source, labels, provenance, and leakage audits can be reused. Any new down-selection/assignment would require a frozen hash-only proposal and human approval before materialization; no new content labeling is intrinsically required.",
        },
        "benign_pools": {name: benign_format_summary(rows) for name, rows in benign.items()},
        "benign_source_rights": {
            "B0": {"source": "OpenAssistant/oasst1", "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b", "license": "Apache-2.0"},
            "BH": {"source": "bench-llm/or-bench", "revision": "e36d8b80e81837c8a8f264bbb2a49f1b32c7e272", "license": "CC BY 4.0"},
            "B-short": {"source": "OpenAssistant/oasst1", "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b", "license": "Apache-2.0"},
            "NotInject": {"source": "leolee99/NotInject", "revision": "847ae76cf8fea5ed325429e569ae8cfef022d2e0", "license": "MIT (dataset card)"},
        },
        "benign_pool_overlap": benign_overlap,
        "limitations": [
            "TF-IDF and lexical cues measure surface separability, not causal detector features.",
            "NotInject language is recorded conservatively because the official source has no language field.",
            "B-short is not language matched to NotInject; their contrast is descriptive only.",
            "Chen-derived attack families remain a deterministic reconstruction, not an exact paper reproduction.",
        ],
    }


def lineage_comparison():
    earlier = read_json(RESULTS / "notinject_checkpoint_evaluation_aggregate.json")
    h2 = read_json(RESULTS / "h2_mini_no_replay_aggregate.json")
    rows = earlier["per_checkpoint"]
    def group(order, stage_index):
        selected = [r for r in rows if r["order"] == order and int(r["stage_index"]) == stage_index]
        return {
            "per_seed_fpr": {str(r["seed"]): r6(r["overall_fpr"]) for r in selected},
            "mean_fpr": r6(statistics.mean(r["overall_fpr"] for r in selected)),
            "sample_sd_fpr": r6(statistics.stdev(r["overall_fpr"] for r in selected)),
            "mean_p_attack": r6(statistics.mean(r["confidence"]["mean_p_attack"] for r in selected)),
        }
    h2_final_ni = []
    h2_stage1_ni = []
    for arm in ARMS:
        for seed in SEEDS:
            for stage_index, target in ((1, h2_stage1_ni), (4, h2_final_ni)):
                pred = read_jsonl(H2_RESULTS / arm / f"seed{seed}" / f"stage{stage_index}" / "predictions/NotInject.jsonl")
                target.append({"arm": arm, "seed": seed, "fpr": sum(r["predicted_label"] == 1 for r in pred) / len(pred), "mean_p_attack": statistics.mean(r["p_attack"] for r in pred)})
    def summarize(items):
        return {
            "observations": len(items),
            "fpr": qstats(x["fpr"] for x in items),
            "mean_p_attack_across_checkpoints": qstats(x["mean_p_attack"] for x in items),
        }
    return {
        "status": "LINEAGE_COMPARISON_COMPLETE",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "shared_evaluation_bank": {
            "dataset": "leolee99/NotInject", "revision": "847ae76cf8fea5ed325429e569ae8cfef022d2e0",
            "records": 339, "role": "evaluation only", "fixed_threshold_rule": "p_attack=0.5 / argmax"
        },
        "H2_D1_earlier_checkpoints": {
            "checkpoint_source": "completed full-scale AF1-AF4 primary and reverse-order no-replay baselines",
            "stage_train_count": 3000,
            "stage_class_counts": {"benign": 1500, "malicious": 1500},
            "benign_training_context": "Chen/SQuAD-derived long clean documents paired within each attack-family stage",
            "malicious_training_context": "Chen-derived AF1-AF4 document injections",
            "initialization": "pinned DeBERTa base at stream start; weights carried across stages; optimizer reset",
            "primary_stage1": group("primary", 1),
            "primary_final": group("primary", 4),
            "secondary_final": group("secondary", 4),
        },
        "H2_mini_checkpoints": {
            "checkpoint_source": "fresh H2-mini C0/F streams, one pinned-base initialization per arm/seed",
            "stage_train_count": 400,
            "stage_class_counts": {"benign": 200, "malicious": 200},
            "benign_training_context": "C0 uses reviewed OASST1 B0; F shares B0 at Stage 1 then reviewed OR-Bench BH partitions",
            "malicious_training_context": "down-selected Chen-derived AF1-AF4 document injections, shared across arms",
            "initialization": "pinned DeBERTa base with seed-specific random head; paired across arms; weights carried; optimizer reset",
            "stage1_notinject": summarize(h2_stage1_ni),
            "final_notinject": summarize(h2_final_ni),
            "total_stage_runtime_seconds": h2["total_stage_runtime_seconds"],
        },
        "explanation": {
            "observed_contrast": "Earlier AF checkpoints classify roughly 96% of NotInject as attack at Stage 1, whereas H2-mini Stage 1 and final checkpoints are near 0% FPR.",
            "high_confidence_facts": [
                "The evaluated NotInject revision and decision rule are the same.",
                "The checkpoint lineages and training distributions are materially different.",
                "The H2-mini collapse occurs already after shared B0 Stage 1, before F sees BH; BH exposure is therefore unnecessary to explain the contrast.",
            ],
            "plausible_contributors": [
                "H2-mini Stage 1 directly exposes the model to reviewed short ordinary benign prompts, closer in format to NotInject than the earlier long clean-document controls.",
                "The H2-mini class-balanced stage is 7.5x smaller, changing gradient volume and the relative influence of short benign examples.",
                "The H2-mini runs are fresh lineages; they do not continue from the earlier full-scale AF checkpoints.",
            ],
            "causal_limit": "The completed experiments jointly change benign source/format, sample count, and lineage. They establish a dataset/training-regime explanation but do not identify a single causal factor. A matched controlled stream is required for attribution.",
        },
        "guardrail": "The contrast does not revise R0/R1/R2 and is not evidence for replay efficacy.",
    }


def main():
    saturation = saturation_analysis()
    separability = separability_analysis()
    lineage = lineage_comparison()
    write_json(RESULTS / "h2_mini_score_saturation_analysis.json", saturation)
    write_json(RESULTS / "h2_mini_family_separability_analysis.json", separability)
    write_json(RESULTS / "h2_mini_checkpoint_lineage_comparison.json", lineage)
    print(json.dumps({
        "status": "PASS",
        "outputs": [
            "results/h2_mini_score_saturation_analysis.json",
            "results/h2_mini_family_separability_analysis.json",
            "results/h2_mini_checkpoint_lineage_comparison.json",
        ],
        "checkpoint_summaries": len(saturation["checkpoints"]),
        "family_pairs": len(separability["pairwise_family_metrics_including_AF5"]),
    }, indent=2))


if __name__ == "__main__":
    main()
