from __future__ import annotations

import hashlib, json
from pathlib import Path

ROOT=Path(r"E:\ContinualGuard")


def append_once(path,marker,text):
    old=path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in old: raise RuntimeError(f"marker already present: {marker} in {path}")
    path.write_text(old.rstrip()+"\n\n"+text.strip()+"\n",encoding="utf-8")


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


NEXT=("Accept the completed and validated five-method Candidate B analysis and freeze the experimental corpus and artifacts. "
      "Proceed with final figures, descriptive statistical summary, and report writing only: render the predeclared five-method B-SHORT AF3 and B-HARD benign-retention comparisons; Security-aware versus Loss-aware paired plots; memory-overlap and score-relationship diagnostics; external-diagnostic and runtime tables; and report every seed, mean plus sample SD, descriptive Case C, rank-equivalence/numeric-precision limits, and Candidate B limitations. "
      "Do not run further training, another replay method, a memory or coefficient sweep, selective adaptation, a detector ablation, or a benchmark change. Stop after the final report package for human review.")


RESULT=r'''# Candidate B Generic Loss-aware Replay control results

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
'''


SHORT=r'''<!-- LOSS_AWARE_FINAL_D053:BEGIN -->
## Final Generic Loss-aware control (2026-10-06)

The approved final optional method completed 9/9 runs and 36/36 checkpoints with artifact and fairness PASS; all 12 acquisition gates pass in 3/3 seeds. B-SHORT AF3 final attack loss is `-0.556 +/- 0.962 pp`, while B-HARD benign-FPR increase is `0.185 +/- 0.321 pp`. Security-aware remains better on D0-1 (`0/0/0` versus `3.333/0/0 pp`), while Loss-aware is better on B-SHORT AF3 in two seeds and ties in one. The descriptive outcome is Case C.

True-label CE and security risk are monotone transforms within fixed label quotas, so the control cannot establish an independent semantic benefit for the security score. Observed differences start with frozen FP16/FP32 precision and tie behavior and then compound through model trajectories. The experimental program is closed; proceed to final figures and report writing only.
<!-- LOSS_AWARE_FINAL_D053:END -->'''


PROTOCOL=r'''<!-- LOSS_AWARE_CONTROL_D052_D053:BEGIN -->
## Final optional Generic Loss-aware control

D052 approved exactly one final project control. It reuses Candidate B and the fixed detector, seeds, training, `M=200`, 4:4 exposure, 100/100 labels, stage quotas, coverage, tie-break, replay loss, and evaluation. It ranks by frozen pre-update true-label CE and is not BRS, LARS, MIR, or another literature reproduction.

D053 records completion and the algebraic limit discovered during final analysis: within a fixed label quota, `CE=-log(1-risk)`. Fairness is PASS for the frozen implementation, but actual differences cannot isolate a non-monotone security-specific ranking. No additional method, sweep, adaptation, detector ablation, or benchmark is permitted.
<!-- LOSS_AWARE_CONTROL_D052_D053:END -->'''


RISKS=r'''<!-- LOSS_AWARE_FINAL_D053:BEGIN -->
## Final-control assumptions and risks (2026-10-06)

- Three seeds and 60 examples per class make one error equal 1.667 pp; observed paired effects are descriptive.
- True-label CE and the bidirectional security risk are exact monotone transforms inside the frozen label strata. Numeric precision/ties, followed by model-state divergence, are the operative distinction.
- AF5 is compositional; Candidate B is constructed; arm differences include source, language, length, and format.
- Score saturation makes some correlations undefined, especially AF5 malicious risk (8/9 runs). Null values are preserved rather than imputed.
- B-DOC external over-defense remains severe and seed-sensitive. NotInject and B-short never selected memory or tuning choices.
- This is one detector, one memory budget, one replay ratio, and one epoch policy. No broad superiority or deployment claim follows.
<!-- LOSS_AWARE_FINAL_D053:END -->'''


QUESTIONS=r'''<!-- LOSS_AWARE_FINAL_D053:BEGIN -->
## Status after the final optional control (2026-10-06)

The optional-method question is resolved: Generic Loss-aware Replay is complete and yields descriptive Case C, with the algebraic rank-equivalence limit. No experimental approval remains open and no further method is recommended.

Remaining work is editorial: choose final figures from the predeclared table-ready JSON, present every seed and mean +/- sample SD, state the precision/tie and model-trajectory limitation, preserve external diagnostics as non-tuning evidence, and prepare the final report for human review.
<!-- LOSS_AWARE_FINAL_D053:END -->'''


DECISIONS=r'''<!-- LOSS_AWARE_D052_D053:BEGIN -->
## D052 - Approve and freeze the final Generic Loss-aware Replay control

Approve exactly nine runs/36 checkpoints with fixed Candidate B, detector, seeds, optimization, `M=200`, replay exposure/loss, class/stage/coverage/tie structure, and evaluation. Rank by pre-update true-label CE only. Do not add another experiment.

## D053 - Accept the validated final control as descriptive Case C

Accept 9/9 runs, 36/36 checkpoints, artifact PASS, fairness PASS, and all acquisition gates. Loss-aware improves B-SHORT AF3 retention relative to Security-aware, while Security-aware retains the better D0-1 benign result. Record that CE and security risk are monotone transforms under label stratification; the result does not establish independent security-score semantics. Close experimental expansion and move to final writing.
<!-- LOSS_AWARE_D052_D053:END -->'''


CHANGE=r'''<!-- LOSS_AWARE_FINAL_C039:BEGIN -->
## C039 - Implement, execute, validate, and report the final Loss-aware control

Added the frozen config/protocol, implementation, runner, validator/summarizer, focused/result tests, 9 runs, 36 checkpoints, 162 prediction files/27,324 rows, per-run matrices/confusion counts, six required aggregate artifacts, five-method table-ready outputs, score/overlap diagnostics, and final documentation. Validation hashed 252 checkpoint files; 14 focused tests and 146 full-project tests pass. No existing method was rerun. No later method, sweep, adaptation, detector ablation, or benchmark was started.
<!-- LOSS_AWARE_FINAL_C039:END -->'''


def main():
    (ROOT/"docs/loss_aware_control_results.md").write_text(RESULT,encoding="utf-8")
    append_once(ROOT/"docs/replay_comparison_results.md","LOSS_AWARE_FINAL_D053:BEGIN",SHORT)
    append_once(ROOT/"docs/replay_comparison_protocol.md","LOSS_AWARE_CONTROL_D052_D053:BEGIN",PROTOCOL)
    append_once(ROOT/"docs/replay_readiness_review.md","LOSS_AWARE_FINAL_D053:BEGIN",SHORT)
    append_once(ROOT/"docs/two_week_experiment_plan.md","LOSS_AWARE_FINAL_D053:BEGIN",SHORT)
    append_once(ROOT/"docs/assumptions_and_risks.md","LOSS_AWARE_FINAL_D053:BEGIN",RISKS)
    append_once(ROOT/"docs/open_questions.md","LOSS_AWARE_FINAL_D053:BEGIN",QUESTIONS)
    append_once(ROOT/"docs/decision_log.md","LOSS_AWARE_D052_D053:BEGIN",DECISIONS)
    append_once(ROOT/"docs/change_log.md","LOSS_AWARE_FINAL_C039:BEGIN",CHANGE)
    append_once(ROOT/"docs/loss_aware_control_protocol.md","LOSS_AWARE_FINAL_D053:BEGIN",SHORT)
    decisions=[
      {"date":"2026-10-06T00:00:00+00:00","id":"D052","decision":"Approve and freeze exactly one final Candidate B Generic Loss-aware Replay control","runs":9,"checkpoints":36,"score":"pre-update true-label cross-entropy","memory_examples":200,"label_target":[100,100],"only_intended_difference":"buffer ranking score","no_further_experiment":True},
      {"date":"2026-10-06T03:00:00+00:00","id":"D053","decision":"Accept the validated final Generic Loss-aware Replay control as descriptive Case C and close experimental expansion","execution":{"runs":9,"checkpoints":36,"failed":0,"resumed":0,"checkpoint_files_hashed":252,"prediction_files":162,"prediction_rows":27324},"validation":{"artifact":"PASS","fairness":"PASS","focused_tests":"14 passed","full_suite":"146 passed"},"b_short_AF3_loss_pp":{"loss_aware":[-0.5555555555555574,0.9622504486493791],"security_aware":[1.1111111111111107,0.962250448649376]},"b_hard_benign_FPR_increase_pp":{"loss_aware":[0.1851851851851852,0.3207501495497921],"security_aware":[-0.5555555555555556,0.9622504486493763]},"D0_1_loss_aware_change_pp":[3.3333333333333335,0.0,0.0],"case":"C","rank_equivalence":"Within fixed label/stage quotas, true-label CE=-log(1-security-risk); observed differences begin with numeric precision/ties and compound through trajectories","security_specific_advantage_established":False,"readiness":{"R0":"exited","R1":"satisfied","R2":"active and acted upon","criteria_changed":False},"next_prompt":NEXT}
    ]
    with (ROOT/"history/decisions.jsonl").open("a",encoding="utf-8",newline="\n") as h:
        for row in decisions: h.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
    changed=["configs/loss_aware_control.yaml","src/continualguard/loss_aware_control.py","src/continualguard/loss_aware_summary.py","src/continualguard/replay_candidate_b.py","scripts/run_loss_aware_control.py","scripts/summarize_loss_aware_control.py","scripts/finalize_loss_aware_docs.py","tests/test_loss_aware_control.py","tests/test_loss_aware_results.py","results/replay_comparison_candidate_b/loss_aware/**","results/loss_aware_control_aggregate.json","results/loss_aware_security_pairwise.json","results/loss_aware_memory_composition.json","results/loss_aware_score_relationship.json","results/loss_aware_fairness_validation.json","results/loss_aware_artifact_validation.json","docs/loss_aware_control_protocol.md","docs/loss_aware_control_results.md","docs/replay_comparison_results.md","docs/replay_comparison_protocol.md","docs/replay_readiness_review.md","docs/two_week_experiment_plan.md","docs/decision_log.md","docs/change_log.md","docs/assumptions_and_risks.md","docs/open_questions.md","history/decisions.jsonl","history/changes.jsonl"]
    code=changed[:9]
    change={"date":"2026-10-06T03:00:00+00:00","id":"C039","decision_id":"D053","reason":"Implement, execute, independently validate, aggregate, and document the approved final Generic Loss-aware Replay control","files_changed":changed,"validation":["all-method and Loss-aware CUDA preflight PASS","14 focused tests passed","9/9 runs and 36/36 checkpoints complete","252 checkpoint files hash-verified","162 prediction files and 27,324 rows verified","capacity/exposure/quota/lineage/no-leakage fairness PASS","146 full-project tests passed with workspace-root basetemp","0 failed, 0 resumed, 0 unpublished attempts"],"code_config_hashes":{name:sha(ROOT/name) for name in code},"scope":"Final optional method only; no prior method rerun, budget sweep, selective adaptation, detector ablation, or benchmark change"}
    with (ROOT/"history/changes.jsonl").open("a",encoding="utf-8",newline="\n") as h: h.write(json.dumps(change,ensure_ascii=False,separators=(",",":"))+"\n")
    (ROOT/"results/loss_aware_next_prompt.txt").write_text(NEXT+"\n",encoding="utf-8")


if __name__=="__main__": main()
