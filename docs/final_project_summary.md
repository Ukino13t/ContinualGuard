# Final project summary

Status: **EXPERIMENTS COMPLETE / FINAL ANALYSIS COMPLETE / REPORT WRITING ONLY**

The project developed and audited a continual-learning evaluation for indirect prompt-injection detection, then used it to compare No Replay, Reservoir ER, DER++, Security-aware Replay, and a matched Generic Loss-aware control. Candidate B supplied two complementary stressors: malicious-family retention on B-SHORT AF3 and benign over-defense on B-HARD.

At the fixed 200-example budget, every replay method sharply improved both stressors relative to No Replay while preserving current-stage acquisition. Reservoir ER was already strong; DER++ and Security-aware did not resolve a clear decisive advantage over it. Generic Loss-aware matched or slightly improved malicious retention relative to Security-aware but was slightly worse on the benign boundary. Because binary security risk and true-label cross-entropy induce the same exact ranking under a shared model state, the study does not establish an independent semantic benefit for Security-aware selection.

The completed contribution is a provenance-aware, human-reviewed benchmark and a fully paired five-method descriptive comparison that separates acquisition, malicious retention, benign false positives, external over-defense, memory composition, runtime, and numeric selector behavior. The result supports replay as a class of interventions for this benchmark and limits method-specific claims to the observed fixed setting.

Primary artifacts:

- `results/final_tables/five_method_master_table.csv`
- `results/final_tables/reproducibility_table.csv`
- `results/final_tables/appendix_A_acquisition_per_seed.csv` through `appendix_I_artifact_validation.csv`
- `results/final_figures/` PNG/PDF figure pairs
- `docs/final_descriptive_statistics.md`
- `docs/final_claims_audit.md`
- `docs/final_results_narrative.md`
- `docs/final_discussion_narrative.md`
- `docs/final_method_description.md`
- `docs/final_limitations.md`
- `docs/final_experiment_timeline.md`
- `docs/final_contribution_statement.md`
- `docs/final_consistency_audit.md`

No further experiment is part of the approved project state.

## Final report-writing package

- `docs/final_report_draft_en.md`
- `docs/final_report_draft_zh.md`
- `docs/final_citation_audit.md`
- `docs/final_appendix_plan.md`
- `docs/final_report_reduction_plan.md`

The English draft is approximately 4,007 words. Fifteen literature-dependent claims were audited. No experiment, checkpoint evaluation, benchmark generation, or numerical result changed during report assembly.
