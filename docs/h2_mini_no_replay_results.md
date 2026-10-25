# H2-mini no-replay baseline results

Status: **COMPLETE, VALIDATED, R0 ACTIVE**  
Date: 2026-10-05  
Detector: `microsoft/deberta-v3-base` at `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`  
Arms: H2m-C0 and H2m-F; seeds 17/42/2026; order AF1 -> AF2 -> AF3 -> AF4

## Scope and execution

The approved current-stage-only no-replay experiment completed all 6 runs and all 24 stage-end checkpoints. There were no failed or resumed runs. Each stage used one epoch, batch size 4, AdamW at 1e-5, constant scheduling, max gradient norm 1, float16 CUDA, optimizer reset, and sequentially carried model weights. No historical sample, replay, selective adaptation, test-guided selection, or numerical replay memory budget was used.

Technical preflight: **PASS**. It exercised the actual tokenizer, largest selected batch, forward/backward pass, optimizer step, max_length=1280, checkpoint save/load, and evaluation path without CUDA OOM. Preflight peak allocated GPU memory was 3.967 GB.

Independent artifact validation: **PASS**. It recomputed 168 checkpoint-file hashes across 24 checkpoints, verified 108 prediction files containing 18,216 rows, checked checkpoint lineage and the absence of optimizer state, and found no unpublished stage attempt. Same-seed C0/F initialization hashes match, and all three Stage-1 paired model hashes, evaluations, external diagnostics, training losses, and validation losses are identical.

| Arm | Seed | Recorded stage runtime (s) | Recorded run wall time (s) | Peak allocated GPU GB |
|---|---:|---:|---:|---:|
| H2m-C0 | 17 | 206.021 | 248.912 | 3.794 |
| H2m-C0 | 42 | 211.506 | 263.485 | 3.788 |
| H2m-C0 | 2026 | 212.024 | 263.198 | 3.787 |
| H2m-F | 17 | 207.587 | 257.968 | 3.794 |
| H2m-F | 42 | 210.571 | 261.052 | 3.789 |
| H2m-F | 2026 | 210.336 | 261.428 | 3.788 |

Total recorded stage runtime: **1258.045 s**. Summed recorded run wall time: **1556.044 s**. Including the 43.251 s preflight gives **1599.295 s (26.655 min)**.

## Acquisition

Values are seed 17 / 42 / 2026 in percent. Every arm-stage passes the frozen criterion in all 3 seeds.

| Arm | Stage | Attack TPR | Benign TNR | Benign FPR | Paired macro-F1 | Passing seeds |
|---|---|---:|---:|---:|---:|---:|
| H2m-C0 | AF1 | 100.000 / 100.000 / 100.000 | 98.333 / 98.333 / 100.000 | 1.667 / 1.667 / 0.000 | 99.167 / 99.167 / 100.000 | 3/3 |
| H2m-C0 | AF2 | 100.000 / 100.000 / 100.000 | 98.333 / 98.333 / 98.333 | 1.667 / 1.667 / 1.667 | 99.167 / 99.167 / 99.167 | 3/3 |
| H2m-C0 | AF3 | 100.000 / 100.000 / 100.000 | 100.000 / 100.000 / 100.000 | 0.000 / 0.000 / 0.000 | 100.000 / 100.000 / 100.000 | 3/3 |
| H2m-C0 | AF4 | 100.000 / 100.000 / 100.000 | 100.000 / 100.000 / 100.000 | 0.000 / 0.000 / 0.000 | 100.000 / 100.000 / 100.000 | 3/3 |
| H2m-F | AF1 | 100.000 / 100.000 / 100.000 | 98.333 / 98.333 / 100.000 | 1.667 / 1.667 / 0.000 | 99.167 / 99.167 / 100.000 | 3/3 |
| H2m-F | AF2 | 100.000 / 100.000 / 100.000 | 100.000 / 100.000 / 100.000 | 0.000 / 0.000 / 0.000 | 100.000 / 100.000 / 100.000 | 3/3 |
| H2m-F | AF3 | 100.000 / 100.000 / 100.000 | 100.000 / 100.000 / 100.000 | 0.000 / 0.000 / 0.000 | 100.000 / 100.000 / 100.000 | 3/3 |
| H2m-F | AF4 | 100.000 / 100.000 / 100.000 | 100.000 / 100.000 / 100.000 | 0.000 / 0.000 / 0.000 | 100.000 / 100.000 / 100.000 | 3/3 |

H2m-F acquired every attack and benign stage adequately. All head, middle, and tail attack-position TPR values at every acquired checkpoint were at least **100.000%**.

## Retention and BWT

Signed changes are later minus first acquisition. Positive FPR change is harmful; positive TPR/macro-F1/accuracy BWT is beneficial. Values are seed 17 / 42 / 2026 percentage points.

| Arm | Attack-TPR BWT | Paired macro-F1 BWT | Paired accuracy BWT | Benign-FPR increase BWT | Benign-FPR mean +/- sample SD |
|---|---:|---:|---:|---:|---:|
| H2m-C0 | 0.000 / 0.000 / 0.000 | 0.278 / 0.278 / 0.000 | 0.278 / 0.278 / 0.000 | -0.556 / -0.556 / 0.000 | -0.370 +/- 0.321 |
| H2m-F | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / -0.278 | 0.000 / 0.000 / -0.278 | 0.000 / 0.000 / 0.556 | 0.185 +/- 0.321 |

| Arm | Earlier task | Final attack-TPR change | Final macro-F1 change | Final benign-FPR change | Peak benign-FPR increase |
|---|---|---:|---:|---:|---:|
| H2m-C0 | AF1 | 0.000 / 0.000 / 0.000 | 0.833 / 0.833 / 0.000 | -1.667 / -1.667 / 0.000 | 0.000 / 0.000 / 1.667 |
| H2m-C0 | AF2 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 1.667 / 0.000 |
| H2m-C0 | AF3 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| H2m-F | AF1 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / -0.833 | 0.000 / 0.000 / 1.667 | 0.000 / 0.000 / 1.667 |
| H2m-F | AF2 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| H2m-F | AF3 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |

Attack-TPR peak forgetting is **0.000 points for every family, arm, and seed**. The maximum paired macro-F1 peak forgetting is **0.834 points**. The maximum benign peak-FPR increase is **1.667 points**. These are small/transient effects, not catastrophic forgetting.

H2m-C0 mean attack-TPR BWT is 0.000 +/- 0.000 points and macro-F1 BWT is +0.185 +/- 0.160 points. H2m-F mean attack-TPR BWT is 0.000 +/- 0.000 points and macro-F1 BWT is -0.093 +/- 0.160 points. H2m-F benign-FPR increase BWT is +0.185 +/- 0.321 points. Its only harmful final acquisition-relative benign change is AF1 for seed 2026 (+1.667 points); seeds 17 and 42 are unchanged.

## External diagnostics

Each cell is mean +/- sample SD across seeds in percentage points.

| Arm | Checkpoint | NotInject FPR | B-short FPR | B-short English FPR | B-short multilingual FPR |
|---|---|---:|---:|---:|---:|
| H2m-C0 | AF1 | 0.098 +/- 0.170 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| H2m-C0 | AF2 | 0.197 +/- 0.341 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| H2m-C0 | AF3 | 0.492 +/- 0.451 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| H2m-C0 | AF4 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| H2m-F | AF1 | 0.098 +/- 0.170 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| H2m-F | AF2 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| H2m-F | AF3 | 0.393 +/- 0.451 | 1.389 +/- 1.273 | 3.968 +/- 3.637 | 0.000 +/- 0.000 |
| H2m-F | AF4 | 0.000 +/- 0.000 | 0.833 +/- 0.833 | 2.381 +/- 2.381 | 0.000 +/- 0.000 |

Final NotInject FPR is 0 in every arm/seed. Intermediate NotInject FPR is at most 0.885 points and does not worsen persistently. H2m-C0 B-short FPR is 0 at every checkpoint. Final H2m-F B-short FPR is 0.833 +/- 0.833 points: seed values 1.667 / 0.833 / 0.000. The final errors are confined to English (2.381 +/- 2.381 points); multilingual FPR is 0. B-short is language-unmatched, so this difference is descriptive and not a pure trigger-word causal effect. NotInject does not remain worse than B-short after H2-mini training.

## Matched arm comparison

Final-checkpoint differences are H2m-F minus H2m-C0, mean +/- sample SD in percentage points.

| Evaluated stage | Attack TPR | Paired macro-F1 | Benign FPR | Benign TNR |
|---|---:|---:|---:|---:|
| AF1 | 0.000 | -0.833 +/- 0.000 | 1.667 +/- 0.000 | -1.667 +/- 0.000 |
| AF2 | 0.000 | 0.833 +/- 0.000 | -1.667 +/- 0.000 | 1.667 +/- 0.000 |
| AF3 | 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| AF4 | 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |

Persistent hard-benign pressure does not change malicious retention: every paired attack-TPR difference is zero. At the final checkpoint it moves AF1 benign FPR by +1.667 points in all seeds and AF2 benign FPR by -1.667 points in all seeds; the corresponding paired macro-F1 differences are -0.833 and +0.833 points. These are cross-arm endpoint differences, while acquisition-relative AF1 degradation occurs in only one seed. They remain below the frozen R1 threshold.

## Scientific questions

1. **Acquisition:** yes. H2m-F passes all attack-TPR, benign-TNR, and paired macro-F1 gates in 3/3 seeds at all four stages.
2. **Malicious retention:** hard-benign pressure does not increase attack-family forgetting; attack-TPR BWT and peak forgetting are exactly zero.
3. **Benign retention:** no consistent acquisition-relative degradation is established. The largest mean harmful candidate is H2m-F AF1 benign FPR at +0.556 +/- 0.962 points, harmful in 1/3 seeds.
4. **Over-defense:** final NotInject FPR is 0 in both arms. There is no persistent F-arm improvement to attribute to hard-benign pressure.
5. **B-short:** C0 remains at 0; F ends at 0.833 +/- 0.833 points, English-only and seed-sensitive. NotInject is not substantially worse than B-short.
6. **Stability/order:** malicious behavior is fully stable across seeds. Small benign/external changes are seed- or partition-sensitive. This experiment does not include an H2 reverse-order control, so general order symmetry is unresolved.

## Replay readiness

**R0 remains active; R1 and R2 are not entered.** All evidence and acquisition gates pass, but no predeclared primary retention metric has harmful direction in at least 2/3 seeds together with mean degradation of at least 2.0 points. There are zero R1 numeric candidates and zero R2 numeric candidates. Replay is not scientifically justified; neither a narrow replay sensitivity study nor a full replay comparison is eligible.

The result is better described as ceiling saturation with small benign transfer effects. Severe external over-defense observed in earlier checkpoints is not persistent here and is not forgetting evidence.

## Artifacts and tests

- `results/h2_mini_no_replay/`: 6 run directories, 24 checkpoints, raw predictions, confusion counts, metrics, configs, hashes, and matrices.
- `results/h2_mini_no_replay_aggregate.json`: full trajectories and aggregate retention.
- `results/h2_mini_no_replay_pairwise.json`: paired F-minus-C0 comparisons.
- `results/h2_mini_no_replay_readiness.json`: frozen gate application.
- `results/h2_mini_no_replay_artifact_validation.json`: independent artifact validation.
- `results/h2_mini_no_replay/tables/`: eight machine-readable figure/table CSVs.
- Complete project test suite: **102 passed, 0 failed, 1 warning**. The warning is the known PyTorch/Python 3.14 future-compatibility warning.

## Remaining limitations

The 60-per-class stage tests give coarse 1.667-point increments. Malicious performance is ceiling-saturated, limiting sensitivity to retention pressure. H2m-F and C0 intentionally differ in benign source/composition after Stage 1; this supports the operational comparison but leaves residual source/domain confounding. B-short has unmatched language composition and only 42 English examples. The malicious stream is Chen-derived project construction rather than an exact paper reproduction. With three seeds, paired differences are descriptive and do not support strong significance claims.

## Exact recommended next prompt

> Accept the completed H2-mini no-replay results and retain R0. Keep replay, selective adaptation, and numerical memory-budget selection blocked. Record that all acquisition gates passed, attack-TPR BWT was 0 in both arms, H2m-F benign-FPR increase BWT was +0.185 +/- 0.321 percentage points, no R1 threshold was reached, and final NotInject FPR was 0 in both arms while H2m-F final B-short FPR was 0.833 +/- 0.833 percentage points and confined to the English slice. Proceed with benchmark-design analysis only: diagnose ceiling saturation and propose an independently audited stronger security-bidirectional stream with predeclared acquisition/retention gates, provenance, licenses, leakage controls, and matched controls. Do not materialize or train a new benchmark, implement replay or selective adaptation, or choose a replay memory budget until the design is reviewed by a human.
