# Candidate B no-replay baseline results

## Decision

The fixed nine-run Candidate B baseline is **COMPLETE_VALIDATED** and reaches **R2** under the prospective gates. This is eligibility to propose a full replay comparison for human review. It does not authorize replay implementation or execution. No replay, selective adaptation, new benchmark, or numerical memory budget was created.

The decisive R2 result is B-SHORT AF3 attack-TPR loss from first acquisition to final: **35.000 ± 38.115 percentage points**, harmful in all seeds (**20.000 / 6.667 / 78.333 pp**). The matched B-DOC AF3 result reproduces the harmful direction in all seeds at **3.333 ± 1.667 pp**. Magnitude is highly seed-sensitive, so the result is strong descriptive retention evidence rather than a significance claim.

## A–G. Integrity and execution

- **A:** read-only materialization integrity preflight **PASS**.
- **B:** the prior transient capacity message caused **NO** artifact issue.
- **C:** technical GPU preflight **PASS** on `NVIDIA GeForce RTX 2060`; CUDA `12.6`; no OOM; longest tested token lengths `[462, 435, 405, 400]`; save/load and evaluation passed.
- **D/E:** **9/9 runs** and **36/36 stage-end checkpoints** complete.
- **F:** failed runs **0**; resumed runs **0**.
- **G:** cumulative stage runtime **1922.216 s (32.04 min)**; summed run wall time **2384.914 s (39.75 min)**; preflight **37.920 s**; peak allocated GPU memory **3.535 GiB**.

| Arm | Seed | Runtime (s) |
|---|---:|---:|
| B-DOC | 17 | 225.587 |
| B-DOC | 42 | 228.553 |
| B-DOC | 2026 | 219.180 |
| B-SHORT | 17 | 211.861 |
| B-SHORT | 42 | 212.017 |
| B-SHORT | 2026 | 206.426 |
| B-HARD | 17 | 215.386 |
| B-HARD | 42 | 201.307 |
| B-HARD | 2026 | 201.900 |

Artifact validation rehashed **252 checkpoint files**, verified **162 prediction files / 27,324 rows**, exact lineage, current-stage-only training, no optimizer-state carryover, same-seed initialization, three-arm Stage-1 equality, B-SHORT/B-HARD Stage-2 equality, and absence of unpublished stage attempts. The materialization manifest remains `81962e092e1f8502a537d675582549df94f128f5ebf493b1ed0b59540ec3e8f4` and validation hash `15f2fae9b34595bf586124092eb77b01ffbda44b93f6dd8fc0e78c25581a376a`.

## H–K. Acquisition

Values are attack TPR / benign TNR / benign FPR / paired macro-F1 in percent. Every arm-stage gate passes in **3/3 seeds**, above the required 2/3.

| Arm | Stage | Seed 17 TPR/TNR/FPR/F1 | Seed 42 | Seed 2026 | Gate |
|---|---|---:|---:|---:|---|
| B-DOC | AF2 | 100.00/100.00/0.00/100.00 | 98.33/100.00/0.00/99.17 | 100.00/100.00/0.00/100.00 | PASS 3/3 |
| B-DOC | AF3 | 100.00/100.00/0.00/100.00 | 100.00/100.00/0.00/100.00 | 100.00/100.00/0.00/100.00 | PASS 3/3 |
| B-DOC | AF4 | 100.00/100.00/0.00/100.00 | 98.33/100.00/0.00/99.17 | 100.00/100.00/0.00/100.00 | PASS 3/3 |
| B-DOC | AF5 | 100.00/100.00/0.00/100.00 | 100.00/100.00/0.00/100.00 | 100.00/100.00/0.00/100.00 | PASS 3/3 |
| B-SHORT | AF2 | 100.00/100.00/0.00/100.00 | 98.33/100.00/0.00/99.17 | 100.00/100.00/0.00/100.00 | PASS 3/3 |
| B-SHORT | AF3 | 100.00/95.00/5.00/97.50 | 100.00/98.33/1.67/99.17 | 100.00/98.33/1.67/99.17 | PASS 3/3 |
| B-SHORT | AF4 | 100.00/100.00/0.00/100.00 | 100.00/100.00/0.00/100.00 | 100.00/100.00/0.00/100.00 | PASS 3/3 |
| B-SHORT | AF5 | 100.00/100.00/0.00/100.00 | 100.00/100.00/0.00/100.00 | 100.00/100.00/0.00/100.00 | PASS 3/3 |
| B-HARD | AF2 | 100.00/100.00/0.00/100.00 | 98.33/100.00/0.00/99.17 | 100.00/100.00/0.00/100.00 | PASS 3/3 |
| B-HARD | AF3 | 100.00/95.00/5.00/97.50 | 100.00/98.33/1.67/99.17 | 100.00/98.33/1.67/99.17 | PASS 3/3 |
| B-HARD | AF4 | 98.33/100.00/0.00/99.17 | 100.00/100.00/0.00/100.00 | 100.00/100.00/0.00/100.00 | PASS 3/3 |
| B-HARD | AF5 | 100.00/100.00/0.00/100.00 | 100.00/100.00/0.00/100.00 | 100.00/100.00/0.00/100.00 | PASS 3/3 |

AF5 compositional acquisition is 100% attack TPR, 100% benign TNR, and 100% macro-F1 in every arm and seed. Poor acquisition does not explain the later retention result.

## L–P. Retention, BWT, peak forgetting, and seed consistency

| Arm | Seed | attack-TPR BWT | macro-F1 BWT | benign-FPR increase | attack peak forgetting (mean) |
|---|---:|---:|---:|---:|---:|
| B-DOC | 17 | -1.111 pp | -0.556 pp | +0.000 pp | 1.111 pp |
| B-DOC | 42 | -0.556 pp | -0.278 pp | +0.000 pp | 1.111 pp |
| B-DOC | 2026 | -2.222 pp | -1.112 pp | +0.000 pp | 2.222 pp |
| **B-DOC** | **mean ± SD** | **-1.296 ± 0.849 pp** | **-0.648 ± 0.425 pp** | **0.000 ± 0.000 pp** | **1.481 ± 0.642 pp** |
| B-SHORT | 17 | -6.667 pp | -2.808 pp | -1.111 pp | 6.667 pp |
| B-SHORT | 42 | -3.333 pp | -17.767 pp | +27.222 pp | 3.889 pp |
| B-SHORT | 2026 | -27.778 pp | -15.977 pp | -0.556 pp | 27.778 pp |
| **B-SHORT** | **mean ± SD** | **-12.593 ± 13.256 pp** | **-12.184 ± 8.169 pp** | **8.519 ± 16.200 pp** | **12.778 ± 13.064 pp** |
| B-HARD | 17 | -1.111 pp | +0.000 pp | -1.111 pp | 1.667 pp |
| B-HARD | 42 | +0.556 pp | -21.944 pp | +33.333 pp | 0.000 pp |
| B-HARD | 2026 | +0.000 pp | -17.784 pp | +28.889 pp | 0.000 pp |
| **B-HARD** | **mean ± SD** | **-0.185 ± 0.849 pp** | **-13.243 ± 11.656 pp** | **20.370 ± 18.736 pp** | **0.556 ± 0.962 pp** |

B-DOC shows small, consistent malicious retention loss and no benign-FPR increase. B-SHORT creates strong malicious interference, especially AF3, with large magnitude variation. B-HARD largely protects malicious retention but produces a different failure mode: AF2 benign-FPR final increase is **62.222 ± 54.297 pp**, harmful in seeds 42 and 2026 (`0 / 100 / 86.667 pp`). Therefore malicious TPR and benign FPR move differently across arms; the stronger stream does not reduce to one scalar forgetting effect.

Frozen readiness candidates:

| Tier | Arm/task/metric | Mean ± SD | Seed values | Harmful seeds | Matched replication |
|---|---|---:|---|---:|---|
| R1 | B-DOC / AF3 / attack_tpr_loss | 3.333 ± 1.667 pp | 1.667 / 3.333 / 5.000 | 3/3 | B-SHORT |
| R1 | B-SHORT / AF2 / attack_tpr_loss | 2.778 ± 2.546 pp | 0.000 / 3.333 / 5.000 | 2/3 | not required / none |
| R1 | B-SHORT / AF2 / paired_macro_f1_loss | 17.489 ± 28.153 pp | 0.000 / 49.964 / 2.502 | 2/3 | B-HARD |
| R2 | B-SHORT / AF3 / attack_tpr_loss | 35.000 ± 38.115 pp | 20.000 / 6.667 / 78.333 | 3/3 | B-DOC |
| R1 | B-SHORT / AF3 / paired_macro_f1_loss | 19.063 ± 22.976 pp | 8.424 / 3.336 / 45.430 | 3/3 | not required / none |
| R1 | B-HARD / AF2 / paired_macro_f1_loss | 39.728 ± 34.967 pp | 0.000 / 65.833 / 53.352 | 2/3 | B-SHORT |
| R1 | B-HARD / AF2 / benign_fpr_change | 62.222 ± 54.297 pp | 0.000 / 100.000 / 86.667 | 2/3 | not required / none |

## Q–T. Compositional AF5 transfer

AF5 is compositional/combined and is not family-disjoint.

| Arm | Earlier family | attack-TPR change after AF5, mean ± SD | Seed changes |
|---|---|---:|---|
| B-DOC | AF2 | 0.000 ± 0.000 pp | +0.000 pp / +0.000 pp / +0.000 pp |
| B-DOC | AF3 | -2.778 ± 0.962 pp | -1.667 pp / -3.333 pp / -3.333 pp |
| B-DOC | AF4 | -1.111 ± 0.962 pp | -1.667 pp / +0.000 pp / -1.667 pp |
| B-SHORT | AF2 | -3.333 ± 2.887 pp | +0.000 pp / -5.000 pp / -5.000 pp |
| B-SHORT | AF3 | -30.556 ± 37.131 pp | -11.667 pp / -6.667 pp / -73.333 pp |
| B-SHORT | AF4 | 0.000 ± 0.000 pp | +0.000 pp / +0.000 pp / +0.000 pp |
| B-HARD | AF2 | 0.000 ± 0.000 pp | +0.000 pp / +0.000 pp / +0.000 pp |
| B-HARD | AF3 | -0.556 ± 0.962 pp | -1.667 pp / +0.000 pp / +0.000 pp |
| B-HARD | AF4 | 0.556 ± 0.962 pp | +1.667 pp / +0.000 pp / +0.000 pp |

AF5 itself is learned at ceiling. The largest degradation after AF5 is B-SHORT AF3 (`-30.556 ± 37.131 pp`), harmful in all seeds but strongly variable. B-DOC AF3 declines modestly in all seeds; B-HARD AF3 is nearly stable. AF4 stays stable in B-SHORT and B-HARD and declines only slightly in B-DOC.

## U–X. Matched arm comparisons

Differences are treatment minus reference, paired by seed. Negative attack/macro BWT means worse retention; positive benign-FPR-increase difference means worse benign retention.

| Matched comparison | attack-TPR BWT difference | macro-F1 BWT difference | benign-FPR-increase difference |
|---|---:|---:|---:|
| B-SHORT − B-DOC | -11.296 ± 12.427 pp | -11.536 ± 8.146 pp | 8.519 ± 16.200 pp |
| B-HARD − B-SHORT | 12.407 ± 13.337 pp | -1.059 ± 3.552 pp | 11.852 ± 15.539 pp |

The document-to-short transition is associated with measurable interference: B-SHORT minus B-DOC attack-TPR BWT is **-11.296 ± 12.427 pp**. The hard-benign arm shifts the failure mode: B-HARD minus B-SHORT attack-TPR BWT improves **+12.407 ± 13.337 pp**, while benign-FPR retention worsens **+11.852 ± 15.539 pp**. These are matched associations, not pure causal semantic isolation; source, language, length, and format remain intertwined.

## Y–AA. External diagnostics

| Arm | Bank/slice | AF2 mean ± SD | AF5 mean ± SD | change mean ± SD |
|---|---|---:|---:|---:|
| B-DOC | NotInject | 61.455 ± 23.933 pp | 69.715 ± 15.933 pp | 8.260 ± 9.645 pp |
| B-DOC | B-short | 34.167 ± 5.833 pp | 37.222 ± 7.561 pp | 3.056 ± 1.925 pp |
| B-DOC | B-short English | 39.683 ± 7.274 pp | 38.095 ± 4.762 pp | -1.587 ± 2.749 pp |
| B-DOC | B-short multilingual | 31.197 ± 7.061 pp | 36.752 ± 10.904 pp | 5.556 ± 3.917 pp |
| B-SHORT | NotInject | 61.455 ± 23.933 pp | 0.000 ± 0.000 pp | -61.455 ± 23.933 pp |
| B-SHORT | B-short | 34.167 ± 5.833 pp | 0.000 ± 0.000 pp | -34.167 ± 5.833 pp |
| B-SHORT | B-short English | 39.683 ± 7.274 pp | 0.000 ± 0.000 pp | -39.683 ± 7.274 pp |
| B-SHORT | B-short multilingual | 31.197 ± 7.061 pp | 0.000 ± 0.000 pp | -31.197 ± 7.061 pp |
| B-HARD | NotInject | 61.455 ± 23.933 pp | 0.000 ± 0.000 pp | -61.455 ± 23.933 pp |
| B-HARD | B-short | 34.167 ± 5.833 pp | 0.000 ± 0.000 pp | -34.167 ± 5.833 pp |
| B-HARD | B-short English | 39.683 ± 7.274 pp | 0.000 ± 0.000 pp | -39.683 ± 7.274 pp |
| B-HARD | B-short multilingual | 31.197 ± 7.061 pp | 0.000 ± 0.000 pp | -31.197 ± 7.061 pp |

B-DOC retains severe external over-defense at AF5: NotInject FPR **69.715 ± 15.933 pp** and B-short FPR **37.222 ± 7.561 pp**. B-SHORT and B-HARD both end at 0 FPR on NotInject and both B-short language slices in every seed. NotInject and B-short were evaluation-only and did not select data, thresholds, epochs, methods, or readiness rules. B-short remains language-unmatched; no pure trigger-effect claim is made.

## AB–AC. Score margins and position diagnostics

Current-stage acquisition remains hard-label and margin saturated, but retained earlier tasks do not. The B-SHORT final AF3 malicious mean `p_attack` is approximately `0.800 / 0.935 / 0.232` for seeds 17/42/2026, with TPR losses `20.000 / 6.667 / 78.333 pp`. B-HARD final AF2 benign mean `p_attack` is approximately `0.003 / 1.000 / 0.874`, matching the seed-specific FPR split. These are large polarized margin shifts with hard-label consequences, not only subtle movement near 0.5. All raw distributions are in `tables/score_margin.csv`.

Acquisition TPR by injection position:

| Arm | Acquisition stage | head TPR | middle TPR | tail TPR |
|---|---|---:|---:|---:|
| B-DOC | AF2 | 100.000 ± 0.000% | 98.333 ± 2.887% | 100.000 ± 0.000% |
| B-DOC | AF3 | 100.000 ± 0.000% | 100.000 ± 0.000% | 100.000 ± 0.000% |
| B-DOC | AF4 | 100.000 ± 0.000% | 98.333 ± 2.887% | 100.000 ± 0.000% |
| B-DOC | AF5 | 100.000 ± 0.000% | 100.000 ± 0.000% | 100.000 ± 0.000% |
| B-SHORT | AF2 | 100.000 ± 0.000% | 98.333 ± 2.887% | 100.000 ± 0.000% |
| B-SHORT | AF3 | 100.000 ± 0.000% | 100.000 ± 0.000% | 100.000 ± 0.000% |
| B-SHORT | AF4 | 100.000 ± 0.000% | 100.000 ± 0.000% | 100.000 ± 0.000% |
| B-SHORT | AF5 | 100.000 ± 0.000% | 100.000 ± 0.000% | 100.000 ± 0.000% |
| B-HARD | AF2 | 100.000 ± 0.000% | 98.333 ± 2.887% | 100.000 ± 0.000% |
| B-HARD | AF3 | 100.000 ± 0.000% | 100.000 ± 0.000% | 100.000 ± 0.000% |
| B-HARD | AF4 | 100.000 ± 0.000% | 98.333 ± 2.887% | 100.000 ± 0.000% |
| B-HARD | AF5 | 100.000 ± 0.000% | 100.000 ± 0.000% | 100.000 ± 0.000% |

There is no severe acquisition-only position failure; the only acquisition deficit is the AF2/AF4 middle slice in the affected seeds. Full checkpoint trajectories remain in `tables/attack_tpr_trajectory.csv`.

## AD–AH. Replay readiness

- **AD:** final classification **R2**; R0 exited, R1 satisfied, R2 active.
- **AE:** every acquisition and artifact gate passes; B-SHORT AF3 attack-TPR loss is ≥5 pp on average, harmful in all three seeds, and reproduced at the R1 evidence level in the matched B-DOC arm. No threshold, label, leakage, test-guided selection, or acquisition failure was found.
- **AF:** seven R1 numeric candidates are recorded above and in the readiness JSON.
- **AG:** two R2 numeric candidates exist in B-SHORT AF3 (attack TPR and macro-F1); attack TPR passes the matched-replication gate and is the R2-qualified candidate.
- **AH:** **GO for a separate human review of a full replay-comparison proposal; NO-GO for implementation or execution in this task.**

## AI–AN. Project decision

- **AI:** not applicable because the final result is R2; benchmark escalation nevertheless ends because Candidate B was the final escalation attempt.
- **AJ:** a replay comparison may now be **proposed** for human review. It may not yet be implemented or run.
- **AK:** detector remained fixed at `microsoft/deberta-v3-base`, revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`, `max_length=1280`.
- **AL:** numerical replay memory budget remains **unselected**.
- **AM:** remaining confounds are three-seed descriptive uncertainty, large seed-dependent magnitude, document/short/hard source-language-format differences, language-unmatched B-short, AF5 component overlap, and polarized score saturation. BH-1/BH-2 are source-defined partitions, not ordinal difficulty levels.
- **AN exact recommended next prompt:**

> Accept the completed Candidate B no-replay result and its frozen R2 classification. Keep all replay implementation and execution blocked. Proceed with replay-protocol and memory-budget design only: propose one shared example-count budget, stored fields and privacy/data-retention policy, and a method-isolated comparison of no replay, Reservoir ER, DER++, and security-aware replay while fixing the DeBERTa revision, Candidate B assignments, stage order, seeds, one-epoch budget, learning rate, optimizer resets, threshold, and evaluation protocol. Include compute estimates, buffer accounting, stopping rules, and the exact tests needed to preserve current-stage acquisition. Do not implement or run any replay method, selective adaptation, or new benchmark; stop for human approval of the protocol and numerical memory budget.

## Reproducibility and outputs

- Python `3.14.5 (tags/v3.14.5:5607950, May 10 2026, 10:43:50) [MSC v.1944 64 bit (AMD64)]`
- PyTorch `2.14.0+cu126`
- Transformers `4.57.6`
- CUDA `12.6`; GPU `NVIDIA GeForce RTX 2060`
- Seeds `17, 42, 2026`; order `AF2 → AF3 → AF4 → AF5`
- Focused validation: **22 passed, 0 failed, 1 existing warning**
- Full project validation: **124 passed, 0 failed, 1 existing warning**
- Machine-readable results: `results/stronger_pilot_no_replay_aggregate.json`, `results/stronger_pilot_no_replay_pairwise.json`, `results/stronger_pilot_no_replay_readiness.json`, and `results/stronger_pilot_no_replay_artifact_validation.json`
- Table-ready CSVs: `results/stronger_pilot_no_replay/tables/`
