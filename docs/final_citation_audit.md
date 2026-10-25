# Final citation audit

Status: **PASS — 15 literature-dependent claims checked against the project's reviewed local sources.**

| # | Literature-dependent claim | Citation | Support status | Notes / scope control |
|---:|---|---|---|---|
| 1 | IPI can be delivered through retrieved or externally supplied data without direct adversarial access to the application prompt. | [1] Greshake et al. | SUPPORTED | Threat-model claim only; no claim that [1] studies detector continual learning. |
| 2 | Chen et al. study detection and removal of IPI in external documents. | [2] Chen et al. | SUPPORTED | Static IPI work; represented neither as continual learning nor as exact reproduction. |
| 3 | Ignore, Escape-Separation, Fake Completion, and Combined are operational constructions documented by Chen et al. | [2] Chen et al. | SUPPORTED | Project deterministic sampling and AF2–AF5 stream are project choices. |
| 4 | PIGuard identifies over-defense in prompt guard models. | [3] Li et al. | SUPPORTED | Static guardrail finding; not a CL or replay result. |
| 5 | NotInject contains 339 benign trigger-rich records for over-defense evaluation. | [3] Li et al.; pinned official dataset audit | SUPPORTED | Project keeps it diagnostic-only; the paper does not prescribe that project choice. |
| 6 | CAPTURE evaluates context-aware direct prompt injection and benign over-defense. | [4] Kholkar and Ahuja | SUPPORTED | Explicitly identified as direct/single-turn PI, not IPI. No CAPTURE record is used. |
| 7 | Replay can improve stability under non-stationary experience. | [5] Rolnick et al. | SUPPORTED WITH SCOPE LIMIT | CLEAR is reinforcement learning; only the general replay/stability principle transfers. |
| 8 | Simple Experience Replay is a strong continual-learning baseline. | [6], [7] Buzzega et al. | SUPPORTED | The project's reservoir implementation and M=200 are project choices. |
| 9 | DER matches historical logits sampled along the training trajectory. | [6] Buzzega et al. | SUPPORTED | Correctly attributed to DER/DER++; no IPI claim attached. |
| 10 | DER++ includes replay-label supervision in addition to logit replay. | [6] Buzzega et al. | SUPPORTED | Project loss normalization by `1+alpha+beta` is separately labeled as a project choice. |
| 11 | Catastrophic forgetting and loss of plasticity/current acquisition are distinguishable problems. | [8] Elsayed and Mahmood | SUPPORTED | Project acquisition gates are motivated by, but not copied from, that paper. |
| 12 | The reviewed prompt-injection papers do not solve the project's staged continual IPI setting. | [2], [3], [4] | SUPPORTED BY SCOPE AUDIT | Chen and PIGuard are static; CAPTURE is direct PI. This is a bounded statement about reviewed literature. |
| 13 | The reviewed continual-learning papers do not study IPI or benign over-defense. | [5], [6], [7], [8] | SUPPORTED BY SCOPE AUDIT | Avoids claiming exhaustive absence across all literature. |
| 14 | Security-aware risk is theoretically superior to true-label CE. | None | NOT SUPPORTED / EXCLUDED | Project algebra shows monotone equivalence under a common binary model state. |
| 15 | Candidate B or the five-method study establishes deployment robustness. | None | NOT SUPPORTED / EXCLUDED | Constructed benchmark, one detector, three seeds, and diagnostic external banks preclude this claim. |

## Category separation

- **Literature:** references [1]–[8] support threat models, static guardrail findings, and generic continual-learning mechanisms.
- **Project design:** Candidate B, R0/R1/R2 gates, M=200, class/stage quotas, coverage keys, loss normalization, stage order, and diagnostic-only policies are project choices.
- **Project result:** all numerical outcomes, including `35.000 ± 38.115 pp` and `-0.556 ± 0.962 pp`, come from frozen project artifacts and require no literature citation.

No citation is used to imply that prior work already solves continual IPI detection. No direct-PI source is relabeled as IPI, and no project-specific choice is presented as a literature fact.
