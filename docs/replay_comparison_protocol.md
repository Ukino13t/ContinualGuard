# Candidate B primary replay comparison protocol

## Status and scientific question

This protocol was frozen before replay execution on 2026-10-05. Candidate B satisfied the existing R2 entry rule and the human approved the full primary comparison. The question is whether replay selection improves malicious retention and benign false-positive retention without impairing new-stage acquisition under a fixed detector, benchmark, training policy, memory capacity, and replay exposure.

## Frozen common factors

- Detector: `microsoft/deberta-v3-base`, revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`, fixed binary head, `max_length=1280`.
- Data: materialization manifest `81962e092e1f8502a537d675582549df94f128f5ebf493b1ed0b59540ec3e8f4`; arms B-DOC, B-SHORT, B-HARD; order AF2, AF3, AF4, AF5.
- Seeds: 17, 42, 2026. Each replay run must match the corresponding frozen no-replay initialization hash.
- Training: one epoch per stage, current batch 4, AdamW at `1e-5`, constant scheduler, gradient clipping 1.0, CUDA FP16, optimizer reset at each stage, model weights carried.
- Memory: at most 200 historical TRAIN examples. Auxiliary logits and metadata are separately reported and never purchase extra examples.
- Exposure: when memory is nonempty, uniformly sample four stored historical examples without replacement for every four-example current minibatch. Stage 1 has no replay.
- Evaluation: the frozen paired stage tests plus diagnostic-only NotInject and B-short evaluations after each stage. Selection never accesses validation, test, NotInject, or B-short records.

M0 reuses the validated no-replay result. The 27 new runs comprise M1 Reservoir ER, M2 DER++, and M3 project Security-aware Replay across all three arms and seeds.

## Frozen losses

Reservoir ER and Security-aware Replay use exactly `(current_CE + replay_CE) / 2`. Their classifier objective is identical; buffer retention is the intended changed variable.

DER++ uses

`(current_CE + 0.5 * replay_logit_MSE + 0.5 * replay_CE) / 2`.

The DER++ paper defines the current-label, replay-logit, and replay-label terms and reports that `alpha=beta=0.5` is stable. Its published objective is a sum. This project adds the predeclared division by `1+alpha+beta=2` solely to control overall loss scale across methods. Historical logits are the two online logits captured for a current sample before its optimizer update. No coefficient was tuned on Candidate B or any evaluation set. Sources: [Buzzega et al., *Dark Experience for General Continual Learning*](https://papers.neurips.cc/paper_files/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf) and the authors' [Mammoth implementation repository](https://github.com/aimagelab/mammoth).

## Buffer rules

Reservoir uses standard unstratified online reservoir replacement over the encountered TRAIN stream and uniform replay sampling.

DER++ uses the same reservoir identities and capacity, with each retained entry also storing its two historical logits and label.

Security-aware Replay is a project method. Before training each stage, it scores that stage's TRAIN examples using the incoming model. A malicious example has risk `1-p_attack`; a benign example has risk `p_attack`. After stage training, it recomputes memory from the previous memory plus the scored incoming candidates. It targets exactly 100 benign and 100 malicious entries when capacity permits. Within each label half it splits quota evenly over all contributed stages, assigning any remainder to earlier stages. Within each stage/label quota it orders candidates by descending frozen risk and SHA-256 tie-break, taking one per coverage key before additional members. Coverage keys use existing family, benign-group, position, source, and parent/template metadata. No raw text is serialized in buffer audit files; training text is reconstructed from the frozen local TRAIN files by identity and hash.

The buffer is updated only after a stage. A current-stage example is never replayed during that same stage. Checkpoints serialize the buffer and both deterministic buffer RNG states for exact continuation.

## Safety and interpretation

Acquisition remains adequate when attack TPR, benign TNR, and paired macro-F1 each reach 85% in at least two of three seeds. Results report every seed, means, sample SDs, and paired differences. With three seeds, magnitude, direction consistency, and matched comparisons carry more weight than p-values.

Security-aware Replay is promising only if it materially reduces decisive retention losses, improves the security trade-off over Reservoir, preserves acquisition, avoids moving harm between malicious TPR and benign FPR, and shows reasonable consistency. The comparison does not authorize a memory sweep, weighted replay loss, loss-aware replay, selective adaptation, or a new benchmark.

<!-- LOSS_AWARE_CONTROL_D052_D053:BEGIN -->
## Final optional Generic Loss-aware control

D052 approved exactly one final project control. It reuses Candidate B and the fixed detector, seeds, training, `M=200`, 4:4 exposure, 100/100 labels, stage quotas, coverage, tie-break, replay loss, and evaluation. It ranks by frozen pre-update true-label CE and is not BRS, LARS, MIR, or another literature reproduction.

D053 records completion and the algebraic limit discovered during final analysis: within a fixed label quota, `CE=-log(1-risk)`. Fairness is PASS for the frozen implementation, but actual differences cannot isolate a non-monotone security-specific ranking. No additional method, sweep, adaptation, detector ablation, or benchmark is permitted.
<!-- LOSS_AWARE_CONTROL_D052_D053:END -->
