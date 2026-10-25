# Candidate B stronger security-bidirectional pilot protocol

## Status and scope

Candidate B is selected. This document fixes the benchmark design before raw-text materialization or model execution. The current state is **READY FOR HUMAN MATERIALIZATION APPROVAL**. No stronger-pilot JSONL has been materialized, no model has been trained, no replay or selective adaptation has been implemented, and the replay memory budget remains unselected.

The detector remains `microsoft/deberta-v3-base` at revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`, with a fixed binary head, label 0 = benign, label 1 = indirect prompt injection, and `max_length=1280`. The benchmark must not be tuned to produce forgetting.

## Exact three-arm benchmark

| Stage | Shared malicious half | B-DOC benign half | B-SHORT benign half | B-HARD benign half |
|---|---|---|---|---|
| 1 | AF2 Ignore | D0-1 | exact same D0-1 | exact same D0-1 |
| 2 | AF3 Escape-Separation | D0-2 | B0-2 | exact same B0-2 |
| 3 | AF4 Fake Completion | D0-3 | B0-3 | BH-1 |
| 4 | AF5 Combined | D0-4 | B0-4 | BH-2 |

Each stage has 200 malicious and 200 benign training records, 30 per class for validation, and 60 per class for test: 580 records per stage, 2,320 memberships per arm, and 6,960 memberships across the three arms. Intentional cross-arm anchors are counted once when reporting 3,770 unique selected records.

For each malicious stage, the head/middle/tail targets are 67/66/67 for training, 10/10/10 for validation, and 20/20/20 for test. The 1,160 malicious assignments are selected once and copied exactly across arms, including IDs, labels, family, stage, split, position, and hashes.

## Scientific interpretation

AF5 contains Ignore, Escape-Separation, and Fake Completion components. Stage 4 is therefore a **compositional/combined attack shift**, not a fourth independent or family-disjoint attack family. The sequence tests stability after simple components are later combined.

The predeclared contrasts are:

1. **B-SHORT minus B-DOC:** effect associated with changing from Chen-derived document context to OASST1 ordinary short prompts.
2. **B-HARD minus B-SHORT:** incremental effect associated with introducing reviewed OR-Bench hard-benign pressure after a common Stage 2.
3. **Within-arm retention:** malicious attack-TPR retention from first acquisition and benign FPR increase from first acquisition, with macro-F1, accuracy, TNR, peak forgetting, and BWT as secondary measures.

These comparisons are matched more closely than Candidate A because all three arms start identically, B-SHORT and B-HARD remain identical through Stage 2, and B-DOC provides a stable document-context control. Source, language, domain, and format still differ; B-HARD minus B-SHORT does not identify a pure semantic-hardness effect. BH is a source-defined hard-benign group, not an ordinal difficulty label.

## Acquisition and replay-readiness gates

Each stage must reach attack TPR >=85%, benign TNR >=85%, and paired macro-F1 >=85% in at least two of three seeds. The existing R0/R1/R2 criteria are frozen. R1 continues to require a harmful direction in at least two seeds, mean degradation of at least 2 percentage points on a predeclared primary metric, and persistent or localized matched evidence. R2 retains harmful direction in all three seeds, at least 5 percentage points mean degradation, replication or matched evidence, and no acquisition, threshold, or label confound.

NotInject (339 records) and B-short (120 records) remain frozen evaluation-only banks. They may not affect training, validation, thresholds, selection, balancing, or method choice.

<!-- STRONGER_PILOT_CANDIDATE_B_MATERIALIZED_20261005:BEGIN -->
## Materialized benchmark status

Candidate B is locally materialized and fully validated. The exact assignments, three-arm structure, binary labels, AF5 compositional interpretation, matched anchors, and external-bank roles remain unchanged. Manifest: `81962e092e1f8502a537d675582549df94f128f5ebf493b1ed0b59540ec3e8f4`. Validation: `15f2fae9b34595bf586124092eb77b01ffbda44b93f6dd8fc0e78c25581a376a`.

Status: **READY FOR HUMAN NO-REPLAY TRAINING APPROVAL**. No model execution occurred.
<!-- STRONGER_PILOT_CANDIDATE_B_MATERIALIZED_20261005:END -->
