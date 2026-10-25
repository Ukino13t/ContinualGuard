# Reproducibility checklist

For every run verify. The following items passed for all six approved real-baseline runs on 2026-09-10:

- [x] Python and dependency versions saved
- [x] Dependency lock verified for target platform
- [x] Model checkpoint/revision and architecture hash recorded
- [x] Python/NumPy/Torch random seeds and states saved
- [x] Deterministic settings and precision recorded
- [x] Data manifest and dataset hashes saved
- [x] Explicit stage order saved
- [x] Fully resolved configuration saved
- [x] Raw predictions without source text and confusion counts saved
- [x] Stage-end checkpoints and resume metadata saved
- [x] Four metric matrices saved
- [x] Exclusion manifest saved, even when empty
- [x] Git unavailable reason saved (workspace has no Git repository)
- [x] Hardware metadata saved
- [x] Markdown and JSONL experiment histories updated

This checklist applies to the six completed real-baseline runs. See each run metadata and summary plus `results/real_baseline_artifact_validation.json`.

Milestone 1G completed source, tokenizer, processed-data, exclusion, split-overlap, span-integrity and deterministic-rebuild checks. Model-weight download/hash verification, forward/backward preflights, human approval of `max_length=1280`, and the first real sequential/joint baselines are complete. Replay remains deliberately incomplete pending human review.
