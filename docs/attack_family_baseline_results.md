# Attack-family sequential baseline results

Run date: 2026-09-12. Status: **three-seed primary no-replay baseline complete; replay remains blocked for human review.**

## Protocol and evidence boundary

The run uses the already audited Chen-derived deterministic reconstruction at SHA-256 `4b4efaf7640f31f5f12af2f2b3a40b91bbb71789c91c059d21ce4af1e6a9e43a`. The primary order is AF1 Naive → AF2 Ignore → AF3 Escape-Separation → AF4 Fake Completion. AF5 Combined is excluded, Variant B is not constructed, and every stage trains only its 3,000 current-stage records. The fixed detector is `microsoft/deberta-v3-base`, max length 1280, labels 0 benign/1 indirect prompt injection, one epoch per stage, batch size 4, learning rate 1e-5, optimizer reset at every boundary, and carried model weights. Seeds are 17, 42, and 2026.

Each test family has 180 benign and 180 injected examples; each injected position has support 60. Matrix cells below are percentages in `accuracy / macro-F1 / attack TPR / benign FPR` order.

## Seed 17 complete matrix

| After training | AF1 | AF2 | AF3 | AF4 |
|---|---|---|---|---|
| AF1 | 100.000/100.000/100.000/0.000 | — | — | — |
| AF2 | 100.000/100.000/100.000/0.000 | 100.000/100.000/100.000/0.000 | — | — |
| AF3 | 99.722/99.722/100.000/0.556 | 100.000/100.000/100.000/0.000 | 100.000/100.000/100.000/0.000 | — |
| AF4 | 100.000/100.000/100.000/0.000 | 99.722/99.722/100.000/0.556 | 100.000/100.000/100.000/0.000 | 100.000/100.000/100.000/0.000 |

## Seed 42 complete matrix

| After training | AF1 | AF2 | AF3 | AF4 |
|---|---|---|---|---|
| AF1 | 100.000/100.000/100.000/0.000 | — | — | — |
| AF2 | 96.944/96.942/93.889/0.000 | 100.000/100.000/100.000/0.000 | — | — |
| AF3 | 100.000/100.000/100.000/0.000 | 100.000/100.000/100.000/0.000 | 100.000/100.000/100.000/0.000 | — |
| AF4 | 99.722/99.722/99.444/0.000 | 100.000/100.000/100.000/0.000 | 99.722/99.722/99.444/0.000 | 100.000/100.000/100.000/0.000 |

## Seed 2026 complete matrix

| After training | AF1 | AF2 | AF3 | AF4 |
|---|---|---|---|---|
| AF1 | 100.000/100.000/100.000/0.000 | — | — | — |
| AF2 | 100.000/100.000/100.000/0.000 | 100.000/100.000/100.000/0.000 | — | — |
| AF3 | 100.000/100.000/100.000/0.000 | 99.722/99.722/100.000/0.556 | 100.000/100.000/100.000/0.000 | — |
| AF4 | 99.722/99.722/99.444/0.000 | 100.000/100.000/100.000/0.000 | 100.000/100.000/100.000/0.000 | 100.000/100.000/100.000/0.000 |

## Mean ± sample SD matrix

Each metric component is reported in percentage points as `mean ± sample SD`.

| After training | AF1 | AF2 | AF3 | AF4 |
|---|---|---|---|---|
| AF1 | 100.000 ± 0.000 / 100.000 ± 0.000 / 100.000 ± 0.000 / 0.000 ± 0.000 | — | — | — |
| AF2 | 98.981 ± 1.764 / 98.981 ± 1.766 / 97.963 ± 3.528 / 0.000 ± 0.000 | 100.000 ± 0.000 / 100.000 ± 0.000 / 100.000 ± 0.000 / 0.000 ± 0.000 | — | — |
| AF3 | 99.907 ± 0.160 / 99.907 ± 0.160 / 100.000 ± 0.000 / 0.185 ± 0.321 | 99.907 ± 0.160 / 99.907 ± 0.160 / 100.000 ± 0.000 / 0.185 ± 0.321 | 100.000 ± 0.000 / 100.000 ± 0.000 / 100.000 ± 0.000 / 0.000 ± 0.000 | — |
| AF4 | 99.815 ± 0.160 / 99.815 ± 0.160 / 99.630 ± 0.321 / 0.000 ± 0.000 | 99.907 ± 0.160 / 99.907 ± 0.160 / 100.000 ± 0.000 / 0.185 ± 0.321 | 99.907 ± 0.160 / 99.907 ± 0.160 / 99.815 ± 0.321 / 0.000 ± 0.000 | 100.000 ± 0.000 / 100.000 ± 0.000 / 100.000 ± 0.000 / 0.000 ± 0.000 |

## Acquisition and optimization diagnostics

| Stage | Attack TPR (%) | Benign FPR (%) | Macro-F1 (%) | Training loss | Validation loss | Runtime (s) | Peak allocated GPU (GiB) |
|---|---|---|---|---|---|---|---|
| AF1 | 100.000 ± 0.000 | 0.000 ± 0.000 | 100.000 ± 0.000 | 0.107 ± 0.010 | 0.007 ± 0.012 | 387.119 ± 9.158 | 3.529 ± 0.005 |
| AF2 | 100.000 ± 0.000 | 0.000 ± 0.000 | 100.000 ± 0.000 | 0.003 ± 0.004 | 0.000 ± 0.000 | 397.762 ± 14.903 | 3.533 ± 0.001 |
| AF3 | 100.000 ± 0.000 | 0.000 ± 0.000 | 100.000 ± 0.000 | 0.026 ± 0.017 | 0.000 ± 0.000 | 371.690 ± 18.754 | 3.527 ± 0.003 |
| AF4 | 100.000 ± 0.000 | 0.000 ± 0.000 | 100.000 ± 0.000 | 0.011 ± 0.004 | 0.000 ± 0.000 | 387.172 ± 26.884 | 3.529 ± 0.005 |

All four families reach 100.000 ± 0.000% attack TPR and macro-F1 with 0.000 ± 0.000% benign FPR at first acquisition. The primary result therefore does not show poor new-family acquisition. AF1 validation loss is seed-sensitive (mean 0.007151 ± 0.012234), driven by seed 2026 at 0.021277 despite unchanged hard classification metrics; this is confidence/optimization variability rather than acquisition failure.

## Final retention, BWT, FPR, and peak forgetting

| Earlier family | Final attack-TPR change (pp) | Final macro-F1 change (pp) | Final benign-FPR change (pp) | Peak attack-TPR forgetting (pp) | Peak macro-F1 forgetting (pp) |
|---|---|---|---|---|---|
| AF1 | -0.370 ± 0.321 | -0.185 ± 0.160 | 0.000 ± 0.000 | 2.222 ± 3.379 | 1.205 ± 1.605 |
| AF2 | 0.000 ± 0.000 | -0.093 ± 0.160 | 0.185 ± 0.321 | 0.000 ± 0.000 | 0.185 ± 0.160 |
| AF3 | -0.185 ± 0.321 | -0.093 ± 0.160 | 0.000 ± 0.000 | 0.185 ± 0.321 | 0.093 ± 0.160 |

At the AF4 endpoint, mean attack-TPR BWT is -0.185 ± 0.185 pp, macro-F1 BWT is -0.123 ± 0.053 pp, and accuracy BWT is -0.123 ± 0.053 pp. Mean benign-FPR increase is 0.062 ± 0.107 pp.

The largest mean degradation is AF1 across the AF1→AF2 transition: attack TPR changes by −2.037 ± 3.528 pp across seeds (0.000, −6.111, 0.000). AF1 then changes by +2.037 ± 3.528 pp across AF2→AF3 and fully recovers in every seed at the AF3 endpoint. AF3→AF4 produces the largest consistent-direction final AF1 change, −0.370 ± 0.321 pp (0.000, −0.556, −0.556), but the effect is small and occurs in two of three seeds. No attack-TPR decline is consistent across all seeds.

Malicious and benign behavior move differently. Seed 42 has the large temporary AF1 TPR loss with zero benign FPR throughout. Seed 17 has no attack-TPR loss but transient/final 0.556-point benign-FPR changes on different earlier families. Seed 2026 has a small final AF1 TPR loss and a transient AF2 benign-FPR increase that returns to zero.

## Position-specific attack TPR

### head

| After training | AF1 | AF2 | AF3 | AF4 |
|---|---|---|---|---|
| AF1 | 100.000 ± 0.000 | — | — | — |
| AF2 | 98.889 ± 1.925 | 100.000 ± 0.000 | — | — |
| AF3 | 100.000 ± 0.000 | 100.000 ± 0.000 | 100.000 ± 0.000 | — |
| AF4 | 99.444 ± 0.962 | 100.000 ± 0.000 | 100.000 ± 0.000 | 100.000 ± 0.000 |

### middle

| After training | AF1 | AF2 | AF3 | AF4 |
|---|---|---|---|---|
| AF1 | 100.000 ± 0.000 | — | — | — |
| AF2 | 95.556 ± 7.698 | 100.000 ± 0.000 | — | — |
| AF3 | 100.000 ± 0.000 | 100.000 ± 0.000 | 100.000 ± 0.000 | — |
| AF4 | 99.444 ± 0.962 | 100.000 ± 0.000 | 100.000 ± 0.000 | 100.000 ± 0.000 |

### tail

| After training | AF1 | AF2 | AF3 | AF4 |
|---|---|---|---|---|
| AF1 | 100.000 ± 0.000 | — | — | — |
| AF2 | 99.444 ± 0.962 | 100.000 ± 0.000 | — | — |
| AF3 | 100.000 ± 0.000 | 100.000 ± 0.000 | 100.000 ± 0.000 | — |
| AF4 | 100.000 ± 0.000 | 100.000 ± 0.000 | 99.444 ± 0.962 | 100.000 ± 0.000 |

The worst position-specific cell is seed 42 AF1-middle after AF2 at 86.667% TPR, a 13.333-point temporary loss; head is 96.667% and tail 98.333% in that same cell. All three return to 100% after AF3. At the final AF4 endpoint, the only mean position reductions are AF1-head 99.444 ± 0.962%, AF1-middle 99.444 ± 0.962%, and AF3-tail 99.444 ± 0.962%; all other family-position cells are 100%.

## Comparison and decision

The family stream is more interpretable than the earlier SQuAD→TriviaQA document pilot because family, document, payload, and position are explicitly separated, and it exposes a seed-42 AF1-middle vulnerability that an aggregate score would hide. It does not create stronger consistent final retention pressure: the earlier pilot's mean S1 attack-TPR change was −1.210 ± 1.459 pp, whereas this stream's final attack-TPR BWT is only −0.185 ± 0.185 pp. Its strongest pressure is transient and seed-dependent: AF1 peak attack-TPR forgetting is 2.222 ± 3.379 pp, with a 6.111-point seed-42 maximum that recovers after AF3.

These results are not evidence of catastrophic forgetting. Acquisition is saturated, final retention loss is small, and the largest drop is not consistent across seeds and reverses at the next stage. The current evidence is therefore insufficient to justify a broad replay experiment as the next scientific step. A predeclared secondary-order control can test order-dependent transfer; a separately approved harder benchmark or frozen hard-benign diagnostic may be needed before replay would address a demonstrated dominant limitation.

## Runtime, storage, and validation

| Seed | Run ID | Runtime (s) | Runtime (min) | Peak allocated GPU (GiB) |
|---|---|---|---|---|
| 17 | attack-family-sequential-seed17-20260912T014336-2910b46f3f | 1487.17 | 24.79 | 3.534 |
| 42 | attack-family-sequential-seed42-20260912T053138-74409c4987 | 1583.90 | 26.40 | 3.533 |
| 2026 | attack-family-sequential-seed2026-20260912T055836-450461f09b | 1609.58 | 26.83 | 3.533 |

Mean runtime is 1560.21 ± 64.55 seconds. Final lightweight logical project size is 103,894,368,394 bytes (103.894368 GB / 96.759171 GiB). This exceeds the 100,000,000,000-byte threshold, so human storage review is required. No full audit or cleanup was performed.

Validation passed for the complete test suite (33 tests), compileall, pip check, three-run artifact/lineage verification, and loading all 12 stage-end model plus resume-state checkpoints. The retained Python 3.14/PyTorch JIT warning is unchanged.

## Remaining human decisions

- Decide whether to run the predeclared secondary order AF1→AF4→AF3→AF2 to test order sensitivity.
- Decide whether the benchmark needs a harder acquisition/retention condition before replay, while keeping Variant B and hard-benign evolution separate until approved.
- Decide whether AF5 Combined should later be evaluated only as an auxiliary controlled-overlap diagnostic.
- Review storage now that the logical project size exceeds 100,000,000,000 bytes; a full audit still requires separate approval.
- Keep all replay and selective-adaptation methods blocked pending review.
