# Candidate B Generic Loss-aware Replay control protocol

## Status and question

This final optional method was approved after review of the four-method Candidate B comparison. It tests whether the benign-retention advantage of Security-aware Replay comes from its bidirectional security score or from retaining generically difficult examples. This is a project control inspired by generic difficulty-based replay principles; it is not a reproduction of BRS, LARS, MIR, or another published method.

## Frozen common factors

The detector is `microsoft/deberta-v3-base` revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`, with the same binary head, tokenizer, `max_length=1280`, initializations, and fixed threshold. Candidate B remains byte-identical: B-DOC, B-SHORT, and B-HARD; AF2 -> AF3 -> AF4 -> AF5; seeds 17, 42, and 2026.

Training remains one epoch per stage, current batch 4, AdamW at `1e-5`, constant scheduler, gradient clipping 1.0, CUDA FP16, optimizer reset per stage, and carried model weights. Evaluation remains the acquired paired stage tests plus diagnostic-only NotInject and B-short at every endpoint.

Memory is exactly `M=200` historical TRAIN examples. Replay exposure is one four-example historical batch for every four-example current batch when memory is nonempty. Stage 1 has no replay. The classifier objective is exactly `(current_CE + replay_CE)/2`, matching Reservoir and Security-aware Replay.

## Only intended change

Before each stage, the incoming model scores every current-stage TRAIN record. For example `x` with true binary label `y`, the frozen generic difficulty is:

`generic_difficulty(x) = cross_entropy(float32(model_preupdate(x).logits), y)`.

The same logits also produce a security-risk field for post-hoc score relationship analysis, but security risk cannot affect Loss-aware selection. No class-specific transformation is applied to generic difficulty.

After stage training, memory is selected from the previous memory plus incoming scored candidates. It targets 100 benign and 100 malicious records. Within each label half, quota is split evenly over contributed stages with remainders assigned to earlier stages. Within each stage/label quota, candidates are ordered by descending generic difficulty. The implementation uses the identical Security-aware coverage key, one-per-key-first procedure, and `SHA-256(seed|security-aware|example_id)` tie-break. The sole structural difference is the ranking field.

Old scores are never recomputed. Buffer selection cannot access validation, test, NotInject, B-short, future stages, future model states, or completed comparison outcomes. The buffer is updated only after training, so no current-stage record is replayed within its own stage.

## Frozen execution and interpretation

Run exactly nine Loss-aware experiments and 36 stage checkpoints. Existing M0, Reservoir, DER++, and Security-aware results are reused. Acquisition must retain attack TPR, benign TNR, and paired macro-F1 of at least 85% in at least two of three seeds.

The primary comparison is Security-aware versus Loss-aware for B-HARD benign-FPR retention and B-SHORT AF3 attack retention. Report each seed, mean, sample SD, paired differences, buffer overlap, score rank correlations, external diagnostics, runtime, and memory. Classify the result as Case A, B, C, or D without adding a post-hoc threshold. No later method, budget sweep, detector ablation, selective adaptation, or benchmark redesign is authorized.

<!-- LOSS_AWARE_FINAL_D053:BEGIN -->
## Final Generic Loss-aware control (2026-10-06)

The approved final optional method completed 9/9 runs and 36/36 checkpoints with artifact and fairness PASS; all 12 acquisition gates pass in 3/3 seeds. B-SHORT AF3 final attack loss is `-0.556 +/- 0.962 pp`, while B-HARD benign-FPR increase is `0.185 +/- 0.321 pp`. Security-aware remains better on D0-1 (`0/0/0` versus `3.333/0/0 pp`), while Loss-aware is better on B-SHORT AF3 in two seeds and ties in one. The descriptive outcome is Case C.

True-label CE and security risk are monotone transforms within fixed label quotas, so the control cannot establish an independent semantic benefit for the security score. Observed differences start with frozen FP16/FP32 precision and tie behavior and then compound through model trajectories. The experimental program is closed; proceed to final figures and report writing only.
<!-- LOSS_AWARE_FINAL_D053:END -->
