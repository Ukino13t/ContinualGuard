# Candidate B Generic Loss-aware Replay control results

## Validated scope

The final optional control completed **9/9 runs and 36/36 stage checkpoints**, with zero failed and zero resumed runs. Artifact validation and fairness validation are **PASS**. Validation hashed 252 checkpoint files and verified 162 prediction files containing 27,324 rows. Initializations match the corresponding frozen M0 runs; Candidate B hashes, checkpoint lineage, memory reconstruction, TRAIN-only identity/hash resolution, stage quotas, replay exposure, and absence of future/evaluation/external memory entries all pass.

The detector, revision, Candidate B arms, AF2->AF3->AF4->AF5 order, seeds 17/42/2026, one epoch, batch 4, AdamW `1e-5`, constant schedule, clipping 1.0, FP16, optimizer resets, carried weights, fixed threshold, evaluation sets, `M=200`, four-example replay batch, and `(current_CE+replay_CE)/2` objective are frozen. The only intended method change is the pre-update TRAIN buffer-ranking score. Every stage buffer is exactly 100 benign plus 100 malicious; stage quotas are 100, then 50/50, then 34/33/33, then 25/25/25/25 per label.

## Acquisition and run summaries

All 12 arm-stage acquisition gates pass in 3/3 seeds. Values below are mean +/- sample SD in percentage points; FPR increase is signed harm when positive.

| Arm | attack-TPR BWT | macro-F1 BWT | benign-FPR increase | attack peak forgetting | runtime | score cost | peak GPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| B-DOC | 0.000 +/- 0.556 | 0.093 +/- 0.424 | -0.185 +/- 0.321 | 0.185 +/- 0.321 | 6.857 +/- 0.528 min | 0.310 +/- 0.024 min | 3.540 +/- 0.004 GiB |
| B-SHORT | 0.370 +/- 0.321 | -0.465 +/- 1.056 | 1.296 +/- 2.245 | 0.370 +/- 0.642 | 6.395 +/- 0.316 min | 0.326 +/- 0.086 min | 3.536 +/- 0.002 GiB |
| B-HARD | 0.741 +/- 0.849 | 0.278 +/- 0.556 | 0.185 +/- 0.321 | 0.000 +/- 0.000 | 6.102 +/- 0.500 min | 0.276 +/- 0.020 min | 3.538 +/- 0.001 GiB |

Each run performs 300 replay forward passes after Stage 1. Loss-aware is modestly slower than Security-aware (6.28/5.88/5.82 min on B-DOC/B-SHORT/B-HARD) because it adds float32 per-example CE scoring; memory allocation is effectively unchanged.

## Malicious retention

### Decisive B-SHORT AF3

| Seed | acquisition TPR | final TPR | final loss | peak loss |
|---:|---:|---:|---:|---:|
| 17 | 100.000 | 100.000 | 0.000 | 0.000 |
| 42 | 100.000 | 100.000 | 0.000 | 0.000 |
| 2026 | 95.000 | 96.667 | -1.667 | 3.333 |
| mean +/- SD | 98.333 +/- 2.887 | 98.889 +/- 1.925 | -0.556 +/- 0.962 | 1.111 +/- 1.925 |

Five-method final attack-TPR loss: No Replay `35.000 +/- 38.115`, Reservoir `-0.556 +/- 2.546`, DER++ `0.000 +/- 1.667`, Security-aware `1.111 +/- 0.962`, and Loss-aware `-0.556 +/- 0.962 pp`.

### AF2/AF3/AF4 after AF5

| Arm/task | attack-TPR loss | macro-F1 loss | benign-FPR change |
|---|---:|---:|---:|
| B-DOC AF2 | -0.556 +/- 0.962 | -0.278 +/- 0.481 | 0.000 +/- 0.000 |
| B-DOC AF3 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| B-DOC AF4 | 0.556 +/- 0.962 | 0.000 +/- 0.833 | -0.556 +/- 0.962 |
| B-SHORT AF2 | -0.556 +/- 0.962 | 1.951 +/- 2.690 | 4.444 +/- 6.310 |
| B-SHORT AF3 | -0.556 +/- 0.962 | -0.556 +/- 0.481 | -0.556 +/- 0.962 |
| B-SHORT AF4 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| B-HARD AF2 | -0.556 +/- 0.962 | 0.278 +/- 1.273 | 1.111 +/- 1.925 |
| B-HARD AF3 | -1.111 +/- 1.925 | -0.834 +/- 0.834 | -0.556 +/- 0.962 |
| B-HARD AF4 | -0.556 +/- 0.962 | -0.278 +/- 0.481 | 0.000 +/- 0.000 |

## Benign retention

| Seed | D0-1 acquisition/final/change | B0-2 acquisition/final/change | BH-1 final/change | BH-2 final |
|---:|---:|---:|---:|---:|
| 17 | 0/3.333/+3.333 | 3.333/1.667/-1.667 | 0/0 | 0 |
| 42 | 0/0/0 | 1.667/1.667/0 | 0/0 | 0 |
| 2026 | 0/0/0 | 1.667/1.667/0 | 0/0 | 0 |

Loss-aware does not reproduce Security-aware's perfect D0-1 result: Security-aware is `0/0/0 pp`, while Loss-aware is `3.333/0/0 pp`. Mean B-HARD earlier-task benign-FPR increase is No Replay `20.370 +/- 18.736`, Reservoir `1.481 +/- 3.060`, DER++ `0.370 +/- 1.156`, Security-aware `-0.556 +/- 0.962`, and Loss-aware `0.185 +/- 0.321 pp`.

Security-aware minus Loss-aware B-HARD benign-FPR increase is `-2.222/0/0 pp` (mean `-0.741 +/- 1.283`), favoring Security-aware only in seed 17. On B-SHORT AF3 attack loss, Security-aware minus Loss-aware is `+1.667/0/+3.333 pp` (mean `+1.667 +/- 1.667`), favoring Loss-aware in two seeds and tying in one. Directions do not reverse, but nonzero effects are sparse.

## Memory and score diagnostics

All final buffers contain 200 metadata-only identities, 100/100 labels, and 50 examples from each stage. Full family, benign-group, source, position, coverage-key, and loss-quantile distributions for all nine final buffers are in `results/loss_aware_memory_composition.json`. Across final buffers, generic-loss means range 0.793-1.408, medians 0.0056-0.6575, p90 4.464-5.209, p95 4.882-5.242, and p99 4.966-5.899.

Loss-aware versus Security-aware selected-memory overlap falls from 193.0/200 (Jaccard 0.932) after AF2 to 193.3 (0.935) after AF3, 163.8 (0.695) after AF4, and 140.9 (0.545) after AF5. Final benign overlap is 80.1/100 (Jaccard 0.672); malicious overlap is 60.8/100 (0.438). Per-arm/seed/stage/label and retained-task overlap is machine-readable.

The core score interpretation is algebraic: within the fixed label/stage strata and a common model state, `true-label CE = -log(1-security_risk)`. The two scores therefore have the same exact-arithmetic ordering. The frozen implementations differ operationally because Loss-aware casts FP16 logits to float32 before CE/softmax, while the existing Security-aware run applied softmax to FP16 logits. At AF2, actual-method overall Spearman rho is 0.999 and score top-200 overlap is 200/200, yet FP16 ties reduce structured-buffer overlap to 193/200. Later comparisons additionally include diverged incoming model states caused by prior selections: overall rho falls to 0.754 at AF4 and 0.239 at AF5; AF5 malicious rho is undefined in 8/9 runs because Security-aware risk is constant after FP16 saturation. These later correlations are descriptive trajectory diagnostics, not a pure score-function contrast.

## External diagnostics

Final Loss-aware values:

| Arm | NotInject FPR | B-short FPR | English | multilingual |
|---|---:|---:|---:|---:|
| B-DOC | 88.496 +/- 7.208 | 57.778 +/- 10.012 | 56.349 +/- 11.745 | 58.547 +/- 9.792 |
| B-SHORT | 0.098 +/- 0.170 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| B-HARD | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |

B-DOC remains severely over-defensive. Security-aware minus Loss-aware at AF5 is `-13.471 +/- 34.790 pp` on NotInject and `-11.111 +/- 28.823 pp` on B-short; the large SD indicates mixed seed behavior. External banks remain diagnostic-only and did not affect selection or tuning.

## Interpretation and stop

The predeclared descriptive classification is **Case C**: Loss-aware improves the decisive malicious-retention endpoint relative to Security-aware but is worse on the decisive D0-1 benign endpoint. This is a sparse, seed-dependent trade-off, not a significance claim.

The result does **not** establish a security-specific semantic ranking advantage. Under the fixed class balance, Security-aware risk and true-label CE are monotone transforms; observed method differences originate with numeric precision/tie handling and then compound through different replay trajectories. The strongest supported claim remains that structured, class/stage-balanced hard-example replay repairs the dominant Candidate B no-replay failures, while small changes in which hard examples are retained can move malicious and benign stability in different directions. Security-aware retains the best observed B-HARD benign control; Loss-aware matches Reservoir on the decisive B-SHORT AF3 mean loss.

Do not claim literature reproduction, statistical significance, universal method superiority, causal security-score specificity, cross-detector generality, or deployment readiness. Limits include n=3, one detector, one memory/exposure budget, constructed Candidate B, compositional AF5, source/language/format arm differences, saturated scores, fixed 60-per-class tests, and diagnostic-only mismatched external banks.

No further experiment is recommended or authorized. The next step is final figures, descriptive statistical summary, and report writing from the frozen five-method artifacts.
