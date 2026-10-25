# Final descriptive statistics

Status: **EXPERIMENTS COMPLETE / FINAL ANALYSIS COMPLETE / REPORT WRITING ONLY**  
Frozen methods: M0 No Replay, M1 Reservoir ER, M2 DER++, M3 Security-aware Replay, M4 Generic Loss-aware Replay. All values below are descriptive means ± sample SD over seeds 17, 42, and 2026. Percentage-point changes are signed unless explicitly labeled as loss.

## Primary outcomes

| Method | B-SHORT AF3 final TPR loss (pp) | B-HARD FPR increase (pp) | D0-1 final FPR change (pp) | B-HARD final NotInject FPR (pp) |
|---|---|---|---|---|
| No Replay | 35.000 ± 38.115 | 20.370 ± 18.736 | 62.222 ± 54.297 | 0.000 ± 0.000 |
| Reservoir ER | -0.556 ± 2.546 | 1.481 ± 3.060 | 5.000 ± 8.660 | 0.688 ± 0.451 |
| DER++ | 0.000 ± 1.667 | 0.370 ± 1.156 | 1.667 ± 2.887 | 1.278 ± 1.703 |
| Security-aware | 1.111 ± 0.962 | -0.556 ± 0.962 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Loss-aware | -0.556 ± 0.962 | 0.185 ± 0.321 | 1.111 ± 1.925 | 0.000 ± 0.000 |

The decisive retention stressor is B-SHORT AF3 after AF5. No Replay loses **35.000 ± 38.115 pp** of attack TPR. Reservoir ER and Generic Loss-aware each have a mean final change of +0.556 pp (reported as a loss of -0.556 pp); DER++ is 0.000 ± 1.667 pp; Security-aware loses 1.111 ± 0.962 pp. The three replay methods from the primary comparison and the generic selector control all sharply reduce the large, seed-unstable No Replay loss.

On B-HARD, No Replay increases benign FPR by **20.370 ± 18.736 pp**. Reservoir ER reduces that to 1.481 ± 3.060 pp, DER++ to 0.370 ± 1.156 pp, Security-aware to -0.556 ± 0.962 pp, and Generic Loss-aware to 0.185 ± 0.321 pp. Negative change means the final benign FPR is lower than at acquisition.

## Per-seed primary values

| Method | Seed | AF3 final TPR loss | B-HARD FPR increase | D0-1 final FPR change |
|---|---|---|---|---|
| No Replay | 17 | 20.000 | -1.111 | 0.000 |
| No Replay | 42 | 6.667 | 33.333 | 100.000 |
| No Replay | 2026 | 78.333 | 28.889 | 86.667 |
| Reservoir ER | 17 | 0.000 | -0.556 | 0.000 |
| Reservoir ER | 42 | -3.333 | 5.000 | 15.000 |
| Reservoir ER | 2026 | 1.667 | 0.000 | 0.000 |
| DER++ | 17 | 0.000 | -0.556 | 0.000 |
| DER++ | 42 | 1.667 | 0.000 | 0.000 |
| DER++ | 2026 | -1.667 | 1.667 | 5.000 |
| Security-aware | 17 | 1.667 | -1.667 | 0.000 |
| Security-aware | 42 | 0.000 | 0.000 | 0.000 |
| Security-aware | 2026 | 1.667 | 0.000 | 0.000 |
| Loss-aware | 17 | 0.000 | 0.556 | 3.333 |
| Loss-aware | 42 | 0.000 | 0.000 | 0.000 |
| Loss-aware | 2026 | -1.667 | 0.000 | 0.000 |

## Acquisition and stability

All five methods pass all 12 predeclared acquisition gates (three arms × four stages). This rules out failed current-stage learning as the explanation for the primary retention differences. The No Replay B-SHORT AF3 effect is directionally harmful in all three seeds, but its magnitude ranges from 6.667 to 78.333 pp. Replay methods remove most of that variability.

## Security-aware versus generic loss-aware

Generic Loss-aware has lower mean B-SHORT AF3 final TPR loss than Security-aware by 1.667 pp, with paired seed differences in Security-aware minus Loss-aware loss of +1.667, 0.000, and +3.333 pp. On B-HARD overall benign-FPR change, Security-aware is lower by 0.741 pp. On D0-1 final FPR change, the loss-aware values are 3.333, 0, and 0 pp while Security-aware is 0 in every seed.

These are small n=3 descriptive differences. They do not establish a unique security-aware selection benefit. For binary true-label cross-entropy, the loss-aware score is mathematically `-log(1 - security_risk)` under a common model state, so exact-arithmetic rankings are identical. Recorded differences arise from numeric precision/ties and then-diverged training states.

## Source tables

The complete machine-readable results are in `results/final_tables/five_method_master_table.csv` and Appendix Tables A–I. No inferential tests are reported because the experiment used three fixed seeds and was designed for controlled descriptive comparison.
