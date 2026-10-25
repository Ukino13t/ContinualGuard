# Final claims audit

Status: **EXPERIMENTS COMPLETE / FINAL ANALYSIS COMPLETE / REPORT WRITING ONLY**

| ID | Intended report claim | Classification | Exact evidence | Reporting limit |
|---|---|---|---|---|
| A | Continual prompt-injection detection can suffer retention failure under distribution shift. | **SUPPORTED within Candidate B** | No Replay loses B-SHORT AF3 attack TPR by 20.000, 6.667, and 78.333 pp after AF5; mean 35.000 ± 38.115 pp. | Candidate B is constructed; do not generalize prevalence or magnitude to deployment. |
| B | Replay materially reduces malicious-task forgetting. | **SUPPORTED** | The four replay means for B-SHORT AF3 final TPR loss are -0.556, 0.000, 1.111, and -0.556 pp, versus 35.000 pp for No Replay; every method passes 12/12 acquisition gates. | Descriptive n=3 result at M=200, not a universal method ranking. |
| C | Replay materially reduces benign false-positive drift. | **SUPPORTED** | B-HARD overall FPR increase falls from 20.370 ± 18.736 pp for No Replay to between -0.556 and 1.481 pp for replay methods. | Report overall and D0-1 partition changes separately. |
| D | Security-aware Replay is better than No Replay. | **SUPPORTED** | AF3 loss is 1.111 ± 0.962 versus 35.000 ± 38.115 pp; B-HARD drift is -0.556 ± 0.962 versus 20.370 ± 18.736 pp; acquisition is preserved. | Supported on frozen Candidate B endpoints, without a significance claim. |
| E | Security-aware Replay is better than Reservoir ER. | **PARTIALLY SUPPORTED / MIXED** | Security-aware is better on B-HARD drift (-0.556 versus 1.481 pp) and D0-1 (0 versus 5.000 pp), while Reservoir is better on B-SHORT AF3 loss (-0.556 versus 1.111 pp). | No uniform superiority. |
| F | Security-aware Replay is better than DER++. | **PARTIALLY SUPPORTED / MIXED** | Security-aware is better on B-HARD drift (-0.556 versus 0.370 pp) and D0-1 (0 versus 1.667 pp); DER++ is better on AF3 loss (0 versus 1.111 pp). | Differences are small and seed-sensitive. |
| G | Security-aware Replay is better than Generic Loss-aware Replay. | **PARTIALLY SUPPORTED / MIXED** | Security-aware is better on B-HARD drift (-0.556 versus 0.185 pp) and D0-1 (0 versus 1.111 pp); Loss-aware is better on AF3 loss (-0.556 versus 1.111 pp). | Neither dominates both security directions. |
| H | Security-specific scoring has an independent semantic advantage. | **NOT SUPPORTED** | Under a common binary model state, true-label CE = `-log(1-security_risk)`, a strictly increasing transform. Actual overlap declines with FP16/FP32 ties and diverged trajectories. | Do not interpret operational selection differences as semantic uniqueness. |
| I | Security-aware Replay improves bidirectional security stability. | **PARTIALLY SUPPORTED** | It strongly improves both primary directions versus No Replay and has the best observed B-HARD benign retention, but generic methods also repair both failures. | Use “promising” and benchmark-specific wording. |
| J | Candidate B demonstrates universal real-world deployment robustness. | **NOT SUPPORTED** | One detector, three seeds, one memory budget, constructed data, and diagnostic external sets cannot establish deployment robustness. | Avoid universal, production, or deployment-validity claims. |

Historical documents retain the status that was true when they were written. Current summary documents and the final package use the final frozen interpretation.
