# Candidate B primary replay comparison results

## Validated scope

The fixed comparison completed 27/27 new runs and 108/108 new stage-end checkpoints with 0 failed runs and 0 resumed runs. Artifact and fairness validation are **PASS** and **PASS**. M0 reuses the previously validated nine-run no-replay baseline.

All methods use the pinned DeBERTa-v3-base revision, Candidate B bytes and splits, AF2→AF3→AF4→AF5 order, seeds 17/42/2026, one epoch, current batch 4, AdamW at `1e-5`, optimizer resets, carried model weights, fixed threshold, and the same evaluation sets. Each replay method has `M=200` historical TRAIN examples and a four-example replay batch for every nonempty-memory current batch.

DER++ uses `alpha=beta=0.5`. The paper reports this symmetric setting as stable; the project divides the summed objective by two to align overall loss scale. See [Buzzega et al.](https://papers.neurips.cc/paper_files/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf) and the authors' [Mammoth repository](https://github.com/aimagelab/mammoth). No coefficient was tuned on Candidate B or external diagnostics.

## Retention summary

Values are mean ± sample SD across three seeds in percentage points. Attack-TPR and macro-F1 BWT are signed higher-is-better changes; benign-FPR increase is signed harm when positive.

| Method | Arm | Attack-TPR BWT | Macro-F1 BWT | Benign-FPR increase | Attack peak forgetting |
|---|---|---:|---:|---:|---:|
| No Replay | B-DOC | -1.296 ± 0.849 pp | -0.648 ± 0.425 pp | 0.000 ± 0.000 pp | 1.481 ± 0.642 pp |
| No Replay | B-SHORT | -12.593 ± 13.256 pp | -12.184 ± 8.169 pp | 8.519 ± 16.200 pp | 12.778 ± 13.064 pp |
| No Replay | B-HARD | -0.185 ± 0.849 pp | -13.243 ± 11.656 pp | 20.370 ± 18.736 pp | 0.556 ± 0.962 pp |
| Reservoir ER | B-DOC | 0.370 ± 0.642 pp | 0.185 ± 0.321 pp | 0.000 ± 0.000 pp | 0.000 ± 0.000 pp |
| Reservoir ER | B-SHORT | 0.370 ± 1.156 pp | 0.278 ± 0.556 pp | -0.185 ± 0.321 pp | 0.185 ± 0.321 pp |
| Reservoir ER | B-HARD | 0.185 ± 1.398 pp | -0.653 ± 0.983 pp | 1.481 ± 3.060 pp | 0.370 ± 0.642 pp |
| DER++ | B-DOC | -0.741 ± 0.642 pp | -0.370 ± 0.321 pp | 0.000 ± 0.000 pp | 0.926 ± 0.849 pp |
| DER++ | B-SHORT | 0.370 ± 0.642 pp | 0.185 ± 0.161 pp | 0.000 ± 0.556 pp | 0.370 ± 0.321 pp |
| DER++ | B-HARD | -0.185 ± 1.697 pp | -0.278 ± 0.481 pp | 0.370 ± 1.156 pp | 1.296 ± 0.849 pp |
| Security-aware | B-DOC | 0.185 ± 0.321 pp | 0.093 ± 0.160 pp | 0.000 ± 0.000 pp | 0.000 ± 0.000 pp |
| Security-aware | B-SHORT | -0.556 ± 1.111 pp | -0.557 ± 1.004 pp | 0.556 ± 2.546 pp | 0.926 ± 1.156 pp |
| Security-aware | B-HARD | -0.556 ± 1.111 pp | 0.000 ± 0.278 pp | -0.556 ± 0.962 pp | 0.926 ± 0.849 pp |

## Decisive B-SHORT AF3 trajectory

| Method | Seed | Acquisition TPR | Final TPR | Final loss | Peak loss |
|---|---:|---:|---:|---:|---:|
| No Replay | 17 | 100.000 | 80.000 | 20.000 pp | 20.000 pp |
| No Replay | 42 | 100.000 | 93.333 | 6.667 pp | 6.667 pp |
| No Replay | 2026 | 100.000 | 21.667 | 78.333 pp | 78.333 pp |
| Reservoir ER | 17 | 100.000 | 100.000 | 0.000 pp | 0.000 pp |
| Reservoir ER | 42 | 96.667 | 100.000 | -3.333 pp | 0.000 pp |
| Reservoir ER | 2026 | 98.333 | 96.667 | 1.667 pp | 1.667 pp |
| DER++ | 17 | 100.000 | 100.000 | 0.000 pp | 0.000 pp |
| DER++ | 42 | 100.000 | 98.333 | 1.667 pp | 1.667 pp |
| DER++ | 2026 | 93.333 | 95.000 | -1.667 pp | 1.667 pp |
| Security-aware | 17 | 100.000 | 98.333 | 1.667 pp | 1.667 pp |
| Security-aware | 42 | 100.000 | 100.000 | 0.000 pp | 0.000 pp |
| Security-aware | 2026 | 98.333 | 96.667 | 1.667 pp | 3.333 pp |

Aggregate final B-SHORT AF3 losses:

- No Replay: 35.000 ± 38.115 pp.
- Reservoir ER: -0.556 ± 2.546 pp.
- DER++: 0.000 ± 1.667 pp.
- Security-aware: 1.111 ± 0.962 pp.

## Acquisition gates

- **No Replay:** 12/12 arm-stage gates pass; failures: none.
- **Reservoir ER:** 12/12 arm-stage gates pass; failures: none.
- **DER++:** 12/12 arm-stage gates pass; failures: none.
- **Security-aware:** 12/12 arm-stage gates pass; failures: none.

## B-HARD benign retention

Mean signed benign-FPR increase across the three earlier tasks:

- No Replay: 20.370 ± 18.736 pp.
- Reservoir ER: 1.481 ± 3.060 pp.
- DER++: 0.370 ± 1.156 pp.
- Security-aware: -0.556 ± 0.962 pp.

Partition-level B0-2, BH-1, and BH-2 trajectories for every seed and checkpoint are in `aggregate.table_ready.b_hard_benign_fpr_trajectory`.

## External diagnostics at AF5

| Method | Arm | NotInject FPR | B-short FPR | B-short English FPR | B-short multilingual FPR |
|---|---|---:|---:|---:|---:|
| No Replay | B-DOC | 69.715 ± 15.933 pp | 37.222 ± 7.561 pp | 38.095 ± 4.762 pp | 36.752 ± 10.904 pp |
| No Replay | B-SHORT | 0.000 ± 0.000 pp | 0.000 ± 0.000 pp | 0.000 ± 0.000 pp | 0.000 ± 0.000 pp |
| No Replay | B-HARD | 0.000 ± 0.000 pp | 0.000 ± 0.000 pp | 0.000 ± 0.000 pp | 0.000 ± 0.000 pp |
| Reservoir ER | B-DOC | 92.527 ± 4.686 pp | 63.333 ± 4.640 pp | 61.111 ± 7.654 pp | 64.530 ± 4.121 pp |
| Reservoir ER | B-SHORT | 0.393 ± 0.681 pp | 0.000 ± 0.000 pp | 0.000 ± 0.000 pp | 0.000 ± 0.000 pp |
| Reservoir ER | B-HARD | 0.688 ± 0.451 pp | 0.000 ± 0.000 pp | 0.000 ± 0.000 pp | 0.000 ± 0.000 pp |
| DER++ | B-DOC | 71.878 ± 24.816 pp | 43.333 ± 21.651 pp | 42.857 ± 16.667 pp | 43.590 ± 24.460 pp |
| DER++ | B-SHORT | 1.672 ± 2.384 pp | 0.000 ± 0.000 pp | 0.000 ± 0.000 pp | 0.000 ± 0.000 pp |
| DER++ | B-HARD | 1.278 ± 1.703 pp | 0.000 ± 0.000 pp | 0.000 ± 0.000 pp | 0.000 ± 0.000 pp |
| Security-aware | B-DOC | 75.025 ± 31.255 pp | 46.667 ± 19.221 pp | 45.238 ± 12.599 pp | 47.436 ± 22.790 pp |
| Security-aware | B-SHORT | 0.197 ± 0.341 pp | 0.000 ± 0.000 pp | 0.000 ± 0.000 pp | 0.000 ± 0.000 pp |
| Security-aware | B-HARD | 0.000 ± 0.000 pp | 0.000 ± 0.000 pp | 0.000 ± 0.000 pp | 0.000 ± 0.000 pp |

These banks remained diagnostic-only and did not affect memory, coefficients, thresholds, or stopping.

## Memory behavior

- **Reservoir ER:** final buffers 200/200 in all nine runs; benign range 91–107, malicious range 93–109, auxiliary logit bytes range 0–0.
- **DER++:** final buffers 200/200 in all nine runs; benign range 97–104, malicious range 96–103, auxiliary logit bytes range 1600–1600.
- **Security-aware:** final buffers 200/200 in all nine runs; benign range 100–100, malicious range 100–100, auxiliary logit bytes range 0–0.

Security-aware buffers target and achieve 100 benign plus 100 malicious examples, use deterministic per-stage quotas, and rank within quota by frozen pre-update risk with coverage and hash tie-breaking. Full risk quantiles, high-risk counts, stage quotas, families, groups, positions, and sources are in `results/replay_memory_composition.json`.

## Runtime and memory

| Method | Arm | Runtime | Peak allocated GPU memory | Replay forward passes |
|---|---|---:|---:|---:|
| No Replay | B-DOC | 3.74 ± 0.08 min | 3.530 ± 0.005 GiB | 0 |
| No Replay | B-SHORT | 3.50 ± 0.05 min | 3.531 ± 0.004 GiB | 0 |
| No Replay | B-HARD | 3.44 ± 0.13 min | 3.532 ± 0.003 GiB | 0 |
| Reservoir ER | B-DOC | 6.34 ± 0.16 min | 3.532 ± 0.005 GiB | 300 |
| Reservoir ER | B-SHORT | 5.98 ± 0.02 min | 3.533 ± 0.004 GiB | 300 |
| Reservoir ER | B-HARD | 6.03 ± 0.06 min | 3.534 ± 0.003 GiB | 300 |
| DER++ | B-DOC | 6.13 ± 0.49 min | 3.533 ± 0.004 GiB | 300 |
| DER++ | B-SHORT | 5.72 ± 0.49 min | 3.532 ± 0.004 GiB | 300 |
| DER++ | B-HARD | 5.56 ± 0.33 min | 3.531 ± 0.004 GiB | 300 |
| Security-aware | B-DOC | 6.28 ± 0.06 min | 3.539 ± 0.005 GiB | 300 |
| Security-aware | B-SHORT | 5.88 ± 0.09 min | 3.536 ± 0.002 GiB | 300 |
| Security-aware | B-HARD | 5.82 ± 0.02 min | 3.540 ± 0.003 GiB | 300 |

## Validation and limits

Validation hashed 756 checkpoint files and checked 486 raw-prediction files containing 81972 rows. Initialization matched M0, Candidate B hashes matched, every memory identity resolved to an arm-local TRAIN record with matching text hash, all stage lineages passed, and no future-stage, validation, test, NotInject, or B-short record entered memory. No unpublished attempt directory remained.

There are only three seeds. Candidate B is a constructed, reviewed pilot; AF5 is compositional rather than family-disjoint; source/language/format factors differ among arms after declared anchors. Differences support a method comparison within this protocol and should not be overstated as broad deployment evidence.

## Machine-readable outputs

- `results/replay_comparison_aggregate.json`
- `results/replay_comparison_pairwise.json`
- `results/replay_memory_composition.json`
- `results/replay_fairness_validation.json`
- `results/replay_artifact_validation.json`

## Method comparison and conclusion

All three replay methods materially repair the two decisive M0 failures while preserving acquisition. For B-SHORT AF3, final attack-TPR loss falls from `35.000 +/- 38.115 pp` under No Replay to `-0.556 +/- 2.546 pp` for Reservoir, `0.000 +/- 1.667 pp` for DER++, and `1.111 +/- 0.962 pp` for Security-aware. The M0 loss is harmful in 3/3 seeds; all replay endpoints are within 3.333 pp of their acquisition values.

Security-aware does not beat Reservoir on B-SHORT AF3: its paired final-loss difference is `+1.667 / +3.333 / 0.000 pp` for seeds 17/42/2026, where positive means more loss. Relative to DER++, the differences are `+1.667 / -1.667 / +3.333 pp`, so directions are mixed. Relative to M0, Security-aware reduces loss by `18.333 / 6.667 / 76.667 pp`, consistently in all seeds.

The benign side favors Security-aware. In B-HARD, average earlier-task benign-FPR change is `-0.556 +/- 0.962 pp`, versus `+1.481 +/- 3.060 pp` for Reservoir, `+0.370 +/- 1.156 pp` for DER++, and `+20.370 +/- 18.736 pp` for M0. The decisive D0-1 final FPR increase is `0 / 0 / 0 pp` for Security-aware, `0 / 15 / 0 pp` for Reservoir, `0 / 0 / 5 pp` for DER++, and `0 / 100 / 86.667 pp` for M0. B0-2 improves or stays level; BH-1 and newly acquired BH-2 remain at 0 FPR in every method and seed.

The smallest position-specific attack TPR observed anywhere is 10% for M0 at B-SHORT seed 2026, AF3-middle after AF5. The replay minima are 90% for Reservoir, 85% for DER++, and 90% for Security-aware. Replay therefore removes the severe hidden position failure rather than only improving an averaged score.

AF5 remains adequately acquired for every method and arm. After AF5, Reservoir has the smallest B-SHORT AF3 component change (`-0.556 pp` mean); Security-aware changes it by `-1.111 pp`, while M0 changes it by `-30.556 pp`. This again supports replay generally without establishing Security-aware as the best malicious-retention method.

External diagnostics are mixed. On B-SHORT and B-HARD, final NotInject FPR remains between 0 and 1.672% and B-short FPR is 0 for all replay methods. On B-DOC, all methods retain high external over-defense: final NotInject FPR is 69.715% for M0, 92.527% for Reservoir, 71.878% for DER++, and 75.025% for Security-aware; final B-short FPR is 37.222%, 63.333%, 43.333%, and 46.667%, respectively. Reservoir therefore has the largest B-DOC external regression. These banks were never used for selection or tuning.

Security-aware selection behaves as designed: every final buffer is exactly 100 benign/100 malicious and 50 examples from each contributed stage. Across final buffers, mean security risk is 0.356 and mean p90 is 0.993. Reservoir final benign counts range from 91 to 107 and DER++ from 97 to 104; their stage composition varies naturally. DER++ stores 1,600 auxiliary float32 logit bytes at full capacity.

The primary question has a mixed answer. Security-aware Replay is supported relative to No Replay and provides the best B-HARD benign-retention control, but it is not supported as a uniformly better trade-off than both generic methods. Reservoir and DER++ are sufficient to solve the dominant B-SHORT malicious-retention failure, and Reservoir is slightly better there. The proposed method is promising for bidirectional benign protection, not a dominant replacement for generic replay.

One optional loss-aware buffer-selection control is scientifically worth designing because it could test whether Security-aware's benign advantage comes from the security-specific two-direction score or from generic difficulty retention. It is lower priority than accepting the primary result, requires a separately frozen protocol and human approval, and must not trigger a budget sweep or selective adaptation.

<!-- LOSS_AWARE_FINAL_D053:BEGIN -->
## Final Generic Loss-aware control (2026-10-06)

The approved final optional method completed 9/9 runs and 36/36 checkpoints with artifact and fairness PASS; all 12 acquisition gates pass in 3/3 seeds. B-SHORT AF3 final attack loss is `-0.556 +/- 0.962 pp`, while B-HARD benign-FPR increase is `0.185 +/- 0.321 pp`. Security-aware remains better on D0-1 (`0/0/0` versus `3.333/0/0 pp`), while Loss-aware is better on B-SHORT AF3 in two seeds and ties in one. The descriptive outcome is Case C.

True-label CE and security risk are monotone transforms within fixed label quotas, so the control cannot establish an independent semantic benefit for the security score. Observed differences start with frozen FP16/FP32 precision and tie behavior and then compound through model trajectories. The experimental program is closed; proceed to final figures and report writing only.
<!-- LOSS_AWARE_FINAL_D053:END -->
