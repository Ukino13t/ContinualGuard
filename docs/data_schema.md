# JSONL schema and provenance

One UTF-8 JSON object per complete candidate document.

| Required field | Type | Meaning |
|---|---|---|
| text | nonempty string | Exact untrusted document, no appended metadata |
| label | integer 0/1 | Benign/IPI; booleans rejected |
| task_id | string | Stage identifier, excluded from input |
| attack_family | string | Technique; none for benign; unknown allowed if explicit |
| domain | string | Verified domain or declared proxy |
| split | train/val/test | Immutable membership |
| source | object | dataset, revision, upstream_split, record_id, construction: all nonempty strings |
| example_id | string | Globally unique record ID |
| base_document_id | string | Original document lineage shared by clean/injected variants |
| injection_id | string/null | Stable payload identity; null only for benign |
| injection_position | enum | head/middle/tail/unknown for injection; none for benign |
| text_sha256 | hex string | SHA-256 of exact UTF-8 text |

Optional: attack_goal (distinct from technique), template_id, attack_template_id, normalized_template_hash, attack_template_components, language, benign_group, annotation_status, duplicate_group_id, injection_span (half-open original-text character offsets), injection_text, payload_text, base_document_text, upstream_context_sha256, payload_source, and chen_fields. `attack_template_id` is the canonical family-stream field; `template_id` is retained for backward compatibility. `injection_id` identifies the underlying payload before wrapping, while `normalized_template_hash` identifies the complete wrapper with a payload placeholder. `payload_source` preserves the payload artifact/record/hash separately from the base-document source. `chen_fields` retains useful released fields without exposing them to the model. Unknown metadata must be explicit, never guessed from words. Chen's attack output is not our binary label.

Preserve upstream IDs and split; namespace IDs across sources. Assign split at base-document/duplicate-group level before variant construction. Our loader validates already materialized data; a real-source adapter must certify original document/payload identity and cannot assume IDs alone prove independence. No real-data generation adapter is fabricated in this milestone.

Reject schema/hash errors, repeated IDs, cross-split base/duplicate/text/payload groups, unknown stages, missing primary split classes, and historical train-document reuse across stages. Shared test payloads across stages are allowed but reported as correlated evaluations. Audit normalized exact text and token-set Jaccard >=0.9 near-duplicate candidates. Cross-split candidates block; review may resolve false positives in a versioned preparation manifest, never silently lower checks. Token-set similarity is conservative and incomplete; it is not a semantic duplicate guarantee.

Audit original base documents and payload text when available; missing lineage content remains a provenance risk even if provided IDs pass. No global learned vocabulary or normalization statistics. The evaluator may hold historical tests; trainer gets current-stage train only. No test/future-derived class weights, thresholds or selections. Optional NotInject is a separate benign diagnostic, not a primary two-class stage.
