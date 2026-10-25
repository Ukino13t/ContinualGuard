# Final limitations

- The comparison has three fixed seeds. Means and sample SDs describe these runs; no statistical significance claim is made.
- Candidate B is a controlled constructed benchmark. It does not estimate real-world attack or benign prevalence.
- The Chen-derived attack pipeline is a deterministic reconstruction rather than an exact reproduction of the original study.
- AF5 is compositional and reuses earlier attack components, so later-stage effects cannot be interpreted as fully unseen attack-family evolution.
- B-DOC, B-SHORT, and B-HARD differ in source, length, language, and semantic properties. Arm effects are package effects rather than isolated causal effects of one factor.
- NotInject and B-short are external over-defense diagnostics; zero FPR on these finite sets does not establish deployment safety or calibration.
- The single 200-example memory budget does not establish a capacity curve or optimal budget.
- The study uses one replay exposure ratio: one replay minibatch per current minibatch when memory is nonempty.
- Each stage test has 60 examples per class, so one example changes a class-conditional rate by 1.667 percentage points.
- DER++ uses 1,600 bytes of auxiliary FP32 logits at full capacity in addition to the common example memory. Runtime and storage comparisons should retain this distinction.
- Security-risk and true-label CE scores are monotonic transforms for binary labels under the same model state. This design cannot establish an independent semantic advantage for the security-aware score.
- Numeric precision, tie behavior, and model-state divergence affect selector comparisons. Recorded score correlations and buffer overlaps expose these implementation effects.
- Score saturation is substantial at later stages, especially for malicious AF5 examples, reducing rank resolution and making tie handling consequential.
- The external B-short set is language-mismatched across slices (42 English and 78 multilingual records), so aggregate FPR is composition-dependent.
- The detector architecture, tokenizer, threshold, stage order, and one-epoch schedule are fixed. Conclusions may change under other architectures, thresholds, orders, or optimization budgets.
- There is no deployment-scale validation.
- Raw source datasets remain local under their source terms and are not redistributed through Git.
