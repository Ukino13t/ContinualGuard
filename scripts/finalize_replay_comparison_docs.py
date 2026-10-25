from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")


def append_once(path: Path, marker: str, text: str):
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in old:
        raise RuntimeError(f"Marker already exists in {path}: {marker}")
    path.write_text(old.rstrip() + "\n\n" + text.strip() + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


next_prompt = (
    "Accept the completed Candidate B primary replay comparison and its mixed method conclusion. "
    "Record that all three replay methods preserve all 12 acquisition gates and materially repair the M0 failures, "
    "while Security-aware Replay is strongly better than No Replay but is not uniformly superior to Reservoir ER or DER++: "
    "Reservoir is best on B-SHORT AF3 malicious retention, whereas Security-aware is best on B-HARD D0-1 benign-FPR retention. "
    "Keep the detector, Candidate B, seeds, M=200 budget, replay exposure, loss normalization, evaluation, and R0/R1/R2 criteria frozen. "
    "Proceed with design and source verification only for one optional loss-aware buffer-selection baseline using the existing infrastructure "
    "and the same classification replay loss as Reservoir; predeclare its training-only score timing, deterministic tie-breaking, memory rule, "
    "and tests. Do not implement or run it, run a memory sweep, change the benchmark, or add selective adaptation until that exact optional protocol is reviewed by a human."
)


result_conclusion = r"""
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
"""


readiness = r"""
<!-- REPLAY_COMPARISON_FINAL_D051:BEGIN -->
## Candidate B primary replay comparison (2026-10-06)

The frozen R2 evidence state remains active and has now been acted upon through the approved primary comparison. Execution completed 27/27 new replay runs and 108/108 new checkpoints with no failures or resumes. Artifact and fairness validation pass.

All methods pass all 12 acquisition gates. M0 B-SHORT AF3 attack-TPR loss (`35.000 +/- 38.115 pp`) falls to `-0.556 +/- 2.546 pp` for Reservoir, `0.000 +/- 1.667 pp` for DER++, and `1.111 +/- 0.962 pp` for Security-aware. M0 B-HARD benign-FPR increase (`20.370 +/- 18.736 pp`) falls to `1.481 +/- 3.060`, `0.370 +/- 1.156`, and `-0.556 +/- 0.962 pp`, respectively.

Security-aware is strongly supported relative to No Replay but is not uniformly superior to generic replay. Reservoir is best on decisive malicious retention; Security-aware is best on D0-1 benign-FPR retention. R0/R1/R2 criteria were not changed. No new benchmark, memory sweep, optional method, or selective adaptation is authorized by this result.
<!-- REPLAY_COMPARISON_FINAL_D051:END -->
"""


plan = r"""
<!-- REPLAY_COMPARISON_FINAL_D051:BEGIN -->
## Primary replay comparison complete (2026-10-06)

The Candidate B primary comparison is complete: M0 reused plus 27 new runs, 108 new stage checkpoints, all artifacts and fairness checks PASS. The fixed two-week primary work package has no remaining run.

The next optional branch is one loss-aware buffer-selection control, motivated only as a specificity check for the mixed Security-aware result. Its exact literature source, training-only score timing, deterministic memory rule, same-CE loss, and tests must be designed and reviewed before implementation or execution. A memory-budget sweep, benchmark change, and selective adaptation remain out of scope.
<!-- REPLAY_COMPARISON_FINAL_D051:END -->
"""


risks = r"""
<!-- REPLAY_COMPARISON_FINAL_D051:BEGIN -->
## Candidate B replay-comparison limits (2026-10-06)

- Only three seeds are available; sample SD and direction consistency are descriptive, not broad significance claims.
- The result is specific to one fixed detector, one M=200 budget, one 4:4 exposure ratio, and one one-epoch policy. No budget or coefficient sweep was run.
- Candidate B is a deterministic project construction; AF5 is compositional rather than family-disjoint, and the arm differences after declared anchors include source/language/format factors.
- Security-aware group coverage uses available project metadata. A fallback identity key can make coverage effectively record-level when no stronger parent/template key exists.
- Security-aware improves B-HARD benign retention but is slightly worse than Reservoir on B-SHORT AF3 malicious retention; the method claim must remain bidirectional and mixed.
- Final B-DOC external over-defense remains high for every method and is worst for Reservoir. NotInject and B-short are diagnostic-only and cannot justify tuning.
- DER++ `alpha=beta=0.5` follows the paper's reported stable setting; division by two is the project's predeclared scale normalization, not a literature-standard requirement.
<!-- REPLAY_COMPARISON_FINAL_D051:END -->
"""


questions = r"""
<!-- REPLAY_COMPARISON_FINAL_D051:BEGIN -->
## Questions after the primary replay comparison (2026-10-06)

1. Does the human accept the mixed conclusion: Security-aware is better than M0 and best for B-HARD benign retention, but not uniformly better than Reservoir or DER++?
2. Is one optional loss-aware buffer-selection baseline worth the additional nine runs to test security-score specificity?
3. If yes, which single authoritative loss-aware rule and coefficient-free deterministic tie policy should be frozen before implementation?
4. Does the persistent B-DOC external over-defense warrant a later separately designed study, without using external diagnostics for replay selection or tuning?
5. Is cross-detector or larger-stream replication needed before making claims beyond this Candidate B pilot?
<!-- REPLAY_COMPARISON_FINAL_D051:END -->
"""


decision_doc = r"""
<!-- REPLAY_COMPARISON_FINAL_D051:BEGIN -->
## D051 鈥?Accept the validated Candidate B primary replay comparison (2026-10-06)

Accept 27/27 new runs, 108/108 checkpoints, artifact PASS, and fairness PASS. All replay methods preserve all 12 acquisition gates and materially repair M0. Security-aware is supported relative to M0 and is best on B-HARD benign-FPR retention, but Reservoir is best on B-SHORT AF3 attack retention; Security-aware is therefore promising but not uniformly superior to generic replay. Generic replay is sufficient for the dominant malicious-retention failure.

Keep the detector, Candidate B, seeds, M=200, exposure, losses, evaluation, and R0/R1/R2 rules frozen. Do not infer significance from n=3. No optional loss-aware method, memory sweep, new benchmark, or selective adaptation is authorized without a new human decision.
<!-- REPLAY_COMPARISON_FINAL_D051:END -->
"""


change_doc = r"""
<!-- REPLAY_COMPARISON_FINAL_C038:BEGIN -->
## C038 鈥?Implement, execute, validate, and report the Candidate B primary replay comparison (2026-10-06)

Added the frozen replay config, Reservoir/DER++/Security-aware implementation, runner, independent validator/aggregator, report renderer, focused and result tests, and protocol/results documentation. Completed 27 runs and 108 checkpoints. Generated 486 raw-prediction files with 81,972 rows, per-run matrices and confusion outputs, five aggregate/fairness/artifact JSON files, memory composition, paired comparisons, and table-ready outputs. Full checkpoint hashing, initialization, data, provenance, capacity, exposure, and leakage checks pass.

M0 was reused. No loss-aware method, budget sweep, benchmark change, or selective adaptation was implemented or run.
<!-- REPLAY_COMPARISON_FINAL_C038:END -->
"""


def main():
    append_once(ROOT / "docs/replay_comparison_results.md", "## Method comparison and conclusion", result_conclusion)
    append_once(ROOT / "docs/replay_readiness_review.md", "REPLAY_COMPARISON_FINAL_D051:BEGIN", readiness)
    append_once(ROOT / "docs/two_week_experiment_plan.md", "REPLAY_COMPARISON_FINAL_D051:BEGIN", plan)
    append_once(ROOT / "docs/assumptions_and_risks.md", "REPLAY_COMPARISON_FINAL_D051:BEGIN", risks)
    append_once(ROOT / "docs/open_questions.md", "REPLAY_COMPARISON_FINAL_D051:BEGIN", questions)
    append_once(ROOT / "docs/decision_log.md", "REPLAY_COMPARISON_FINAL_D051:BEGIN", decision_doc)
    append_once(ROOT / "docs/change_log.md", "REPLAY_COMPARISON_FINAL_C038:BEGIN", change_doc)

    decision = {
        "date": "2026-10-05T21:45:00+00:00", "id": "D051",
        "decision": "Accept the validated Candidate B primary replay comparison with a mixed Security-aware conclusion",
        "execution": {"new_runs": 27, "new_checkpoints": 108, "failed": 0, "resumed": 0, "prediction_files": 486, "prediction_rows": 81972},
        "fairness": {"memory_examples": 200, "replay_batch": 4, "capacity": "PASS", "exposure": "PASS", "data_seed_config": "PASS", "no_evaluation_leakage": "PASS"},
        "acquisition": "all 12 arm-stage gates pass for all four methods",
        "b_short_AF3_loss_pp": {"no_replay": [35.0, 38.115322086764245], "reservoir": [-0.5555555555555574, 2.545875386086575], "derpp": [0.0, 1.6666666666666663], "security_aware": [1.1111111111111107, 0.962250448649376]},
        "b_hard_benign_FPR_increase_pp": {"no_replay": [20.370370370370374, 18.735762770284055], "reservoir": [1.4814814814814814, 3.0597614151589454], "derpp": [0.37037037037037035, 1.156481110814518], "security_aware": [-0.5555555555555556, 0.9622504486493763]},
        "conclusion": "Security-aware is strongly supported versus No Replay and best for B-HARD benign retention, but Reservoir is best for decisive B-SHORT malicious retention; Security-aware is not uniformly superior to generic replay",
        "generic_replay_sufficient": True,
        "optional_loss_aware": "worth protocol design as a lower-priority specificity control; separately human-gated",
        "readiness": {"R0": "exited", "R1": "satisfied", "R2": "active and acted upon", "criteria_changed": False},
        "next_prompt": next_prompt,
    }
    with (ROOT / "history/decisions.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(decision, ensure_ascii=False, separators=(",", ":")) + "\n")

    changed = [
        "configs/replay_comparison_candidate_b.yaml", "src/continualguard/replay_candidate_b.py", "src/continualguard/replay_summary.py",
        "scripts/run_replay_comparison_candidate_b.py", "scripts/summarize_replay_comparison_candidate_b.py", "scripts/render_replay_comparison_report.py",
        "tests/test_replay_candidate_b.py", "tests/test_replay_candidate_b_results.py", "results/replay_comparison_candidate_b/**",
        "results/replay_comparison_aggregate.json", "results/replay_comparison_pairwise.json", "results/replay_memory_composition.json",
        "results/replay_fairness_validation.json", "results/replay_artifact_validation.json", "docs/replay_comparison_protocol.md",
        "docs/replay_comparison_results.md", "docs/replay_readiness_review.md", "docs/two_week_experiment_plan.md", "docs/decision_log.md",
        "docs/change_log.md", "docs/assumptions_and_risks.md", "docs/open_questions.md", "history/decisions.jsonl", "history/changes.jsonl",
    ]
    change = {
        "date": "2026-10-05T21:45:00+00:00", "id": "C038", "decision_id": "D051",
        "reason": "Implement, execute, independently validate, aggregate, and document the approved Candidate B primary replay comparison",
        "files_changed": changed,
        "validation": ["all-method CUDA preflight PASS", "focused pre-experiment tests 25 passed before execution", "27/27 runs and 108/108 checkpoints complete", "756 checkpoint files hash-verified", "486 prediction files and 81,972 rows verified", "capacity/exposure/data-seed-config/no-leakage fairness PASS", "0 failed, 0 resumed, 0 unpublished attempts"],
        "code_config_hashes": {name: sha(ROOT / name) for name in changed[:8]},
        "scope": "Primary M0/M1/M2/M3 comparison only; no loss-aware method, budget sweep, new benchmark, or selective adaptation",
    }
    with (ROOT / "history/changes.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(change, ensure_ascii=False, separators=(",", ":")) + "\n")
    (ROOT / "results/replay_comparison_candidate_b/next_prompt.txt").write_text(next_prompt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

