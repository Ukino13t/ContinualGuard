from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(r"E:\ContinualGuard")
DOCS = ROOT / "docs"
TABLES = ROOT / "results" / "final_tables"


def load_master() -> dict[str, dict[str, str]]:
    with (TABLES / "five_method_master_table.csv").open(encoding="utf-8", newline="") as handle:
        return {row["method_id"]: row for row in csv.DictReader(handle)}


MASTER = load_master()


REFERENCES_EN = """## References

[1] K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and M. Fritz. “Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection.” 2023.

[2] Y. Chen, H. Li, Y. Sui, Y. He, Y. Liu, Y. Song, and B. Hooi. “Can Indirect Prompt Injection Attacks Be Detected and Removed?” *Proceedings of ACL 2025*, pp. 18189–18206, 2025.

[3] H. Li, X. Liu, N. Zhang, and C. Xiao. “PIGuard: Prompt Injection Guardrail via Mitigating Overdefense for Free.” *Proceedings of ACL 2025*, pp. 30420–30437, 2025.

[4] G. Kholkar and R. Ahuja. “CAPTURE: Context-Aware Prompt Injection Testing and Robustness Enhancement.” *Proceedings of the Workshop on Large Language Model Security (LLMSEC)*, 2025.

[5] D. Rolnick, A. Ahuja, J. Schwarz, T. P. Lillicrap, and G. Wayne. “Experience Replay for Continual Learning.” *Advances in Neural Information Processing Systems 32*, 2019.

[6] P. Buzzega, M. Boschini, A. Porrello, D. Abati, and S. Calderara. “Dark Experience for General Continual Learning: A Strong, Simple Baseline.” *Advances in Neural Information Processing Systems 33*, 2020.

[7] P. Buzzega, M. Boschini, A. Porrello, and S. Calderara. “Rethinking Experience Replay: A Bag of Tricks for Continual Learning.” *25th International Conference on Pattern Recognition (ICPR)*, 2020, doi:10.1109/ICPR48806.2021.9412614.

[8] M. Elsayed and A. R. Mahmood. “Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning.” *International Conference on Learning Representations (ICLR)*, 2024.
"""


EN = rf"""# ContinualGuard: Continual Indirect Prompt-Injection Detection under Malicious and Benign Distribution Shift

**Project status:** EXPERIMENTS COMPLETE / FINAL ANALYSIS COMPLETE / REPORT WRITING ONLY

## Abstract

Indirect prompt injection (IPI) places malicious instructions inside external content later processed by a language-model application. Existing detector evaluations are mainly static, although deployed guards may face sequential changes in attack form, source, format, and benign instruction-like language. We study whether a fixed IPI detector retains earlier malicious behavior and stable benign acceptance while acquiring later stages. We first qualify two simpler streams: an attack-family sequence and a human-reviewed hard-benign stream. Both largely saturate and do not justify replay under prospectively frozen entry criteria. We then introduce Candidate B, a matched three-arm benchmark combining document, short-form, and hard-benign contexts with the sequence Ignore, Escape-Separation, Fake Completion, and compositional Combined attacks. Candidate B exposes both malicious forgetting and benign false-positive drift. With a fixed DeBERTa-v3-base detector, three seeds, one epoch per stage, and a 200-example historical memory, we compare No Replay, Reservoir ER, DER++, Security-aware Replay, and a Generic Loss-aware control. No Replay loses 35.000 ± 38.115 percentage points of B-SHORT AF3 attack recall and increases B-HARD benign false-positive rate by 20.370 ± 18.736 points. All replay methods substantially reduce both failures while passing every acquisition gate. Security-aware Replay gives the strongest observed B-HARD benign-retention control, but generic methods equal or exceed it on some malicious-retention endpoints. Because the security-risk score and true-label cross-entropy are monotone transforms under a common binary model state, the evidence supports structured hard-example replay rather than a uniquely security-specific ranking advantage. The study provides a controlled security-bidirectional continual evaluation, not deployment validation.

## 1. Introduction

Retrieval-augmented generation (RAG) and tool-integrated language-model applications process text that may originate outside the user's control. Indirect prompt injection exploits this boundary by placing instructions in retrieved content rather than sending them directly as the user prompt [1]. Chen et al. showed that IPI can be formulated as binary detection over external documents and studied both detection and removal under static benchmark conditions [2]. The deployment problem is broader: a detector may be updated as attack patterns and input distributions change, and each update can alter behavior on earlier inputs.

This creates two security-relevant error directions. A detector can forget earlier malicious patterns, increasing false negatives, or become over-defensive, rejecting benign text because it contains instruction-like cues. PIGuard documents the latter problem in static prompt-guard evaluation and introduces the 339-record NotInject bank for benign trigger-rich diagnostics [3]. CAPTURE likewise motivates context-sensitive evaluation and reports both missed attacks and benign false positives, although its malicious task is direct, single-turn prompt injection rather than IPI [4]. Neither line of work evaluates staged continual adaptation of an IPI detector.

Continual learning offers a useful framework because it separates plasticity—the acquisition of new behavior—from stability—the preservation of earlier behavior. Replay is a natural intervention: bounded historical examples are interleaved with current data to reduce interference [5–7]. The central question of this project is therefore: **How robust is indirect prompt-injection detection under sequential distribution shift, and can replay mitigate both malicious-task forgetting and benign over-defense?**

This work makes three contributions:

1. It defines a controlled continual IPI evaluation framework that measures current-stage acquisition, earlier malicious retention, and benign false-positive drift separately.
2. It develops Candidate B, a matched document/short-form/hard-benign benchmark that exposes failures hidden by simpler ceiling-saturated streams.
3. It provides a controlled five-method replay comparison at fixed detector and memory budget. Replay substantially repairs the dominant failures; structured hard-example selection changes the malicious-versus-benign stability trade-off, with Security-aware Replay giving the strongest observed B-HARD benign retention but no universal method superiority.

## 2. Background and Related Work

### 2.1 Indirect prompt injection and static detection

IPI differs from direct prompt injection in both delivery and threat model: the adversarial instruction resides in data retrieved or otherwise supplied to an LLM-integrated application [1]. Chen et al. construct injected external-document datasets, evaluate discriminative detection, and define operational attack constructions including Ignore, Escape-Separation, Fake Completion, and Combined attacks [2]. Their work is the malicious-data and detector reference for this project, but it is not a continual-learning study. The project uses a deterministic reconstruction from pinned released artifacts and does not claim exact reproduction.

### 2.2 Guardrails and over-defense

Prompt guards may rely on superficial instruction-associated cues and flag benign text. PIGuard studies this static over-defense problem and releases NotInject, whose benign examples intentionally contain common injection-related trigger words [3]. CAPTURE examines context-aware attack and benign behavior for direct, single-turn prompt injection [4]. We use PIGuard to motivate hard-benign evaluation and use NotInject only as a frozen diagnostic bank. CAPTURE informs the need for context-sensitive safe examples; no CAPTURE record is used, and its malicious benchmark is not relabeled as IPI.

### 2.3 Continual learning and replay

Continual learning studies non-stationary data where new learning can overwrite previous behavior. CLEAR demonstrates replay principles in reinforcement learning, emphasizing stability and plasticity under bounded memory [5]; only these general principles transfer here. Experience Replay (ER) is a strong generic rehearsal baseline in supervised continual learning [6,7]. Dark Experience Replay preserves historical logits, and DER++ additionally uses replay labels [6]. These papers motivate Reservoir ER and DER++ as baselines, but none studies continual IPI detection or benign over-defense. Work on loss of plasticity further cautions that poor current-stage learning and catastrophic forgetting are distinct phenomena [8], motivating explicit acquisition gates.

### 2.4 Research gap

The reviewed security work evaluates static guardrails, while the reviewed continual-learning work uses non-security image or reinforcement-learning settings. This project addresses the intersection: sequential IPI adaptation evaluated in both malicious and benign directions, with replay authorized only after a prospectively qualified benchmark exhibits sufficient retention pressure.

## 3. Problem Formulation

Let stages arrive in order `AF2 → AF3 → AF4 → AF5`. At stage $t$, a binary detector $f_\theta(x)$ is trained on current-stage examples with label 0 for benign content and label 1 for IPI. Model weights carry forward, while the AdamW optimizer is reset. After every stage, the model is evaluated on every seen test stage.

Acquisition is the performance on a stage immediately after learning it. Retention is the change from this acquisition checkpoint to later checkpoints. For higher-is-better metrics, backward transfer (BWT) is the signed mean final-minus-acquisition change over earlier stages. Peak forgetting is the largest nonnegative decline from the best prior checkpoint. Benign stability is measured by false-positive-rate (FPR) increase. Attack true-positive rate (TPR), benign FPR, macro-F1, and accuracy are retained separately; a high attack TPR cannot compensate for unacceptable benign rejection.

The project uses three evidence states. R0 means no replay justification; R1 records limited retention evidence; R2 requires a predeclared harmful magnitude and seed consistency plus matched evidence. These gates govern whether replay may be studied, not whether a method “wins.”

## 4. Methodology

### 4.1 Detector and continual update

The detector is the fixed `microsoft/deberta-v3-base` binary sequence classifier at revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`. Text is tokenized to `max_length=1280`. Each stage uses one epoch of AdamW at learning rate $10^{{-5}}$, batch size 4, constant scheduling, FP16 CUDA training, and a fresh optimizer. The decision rule is argmax over two logits, equivalent to a $p_\text{{attack}}=0.5$ boundary with ties assigned to class 0.

### 4.2 Prospective benchmark evolution

The first family-separated stream used AF1 Naive, AF2 Ignore, AF3 Escape-Separation, and AF4 Fake Completion. Acquisition was near perfect and retention effects were small or transient, including under a secondary order. Replay remained unjustified. H2-mini then introduced independently reviewed benign pressure, but both its control and hard-benign arms had attack-TPR BWT of 0 and did not cross the frozen entry thresholds. H2-mini therefore remained R0, and no replay experiment was run on it.

Candidate B was the next and final benchmark escalation. It preserved the fixed detector and matched malicious assignment while strengthening benign/source contrasts and adding the compositional AF5 stage. Its no-replay result reached R2. Benchmark escalation stopped at that point, before replay methods were selected or compared.

### 4.3 Human-reviewed benign data

The hard-benign pipeline used two independent reviewer identities. A 50-record calibration produced 96% exact decision agreement and no benign-versus-true-injection conflict. Production review then screened ordinary short-form and OR-Bench-derived over-refusal candidates for benign eligibility, unsafe or ambiguous intent, missing context, privacy/rights concerns, and true injection. Large reviewer disagreement over proposed B1/B2/B3 *strata* led to a prospective simplification: B1/B2/B3 were retained only as descriptive provenance, while the final operational taxonomy used ordinary benign (B0) and source-defined hard benign (BH). BH-1/BH-2/BH-3 are balanced disjoint partitions, not ordinal difficulty levels.

Validation and test records required independent double acceptance; training used primary screening plus independent audits. Acceptance conflicts remained excluded. Later targeted B0 review supplied sufficient capacity for four disjoint B0 partitions. A transport-encoding incident affecting reviewer text was recovered through source-hash and human-confirmation procedures without inferring or changing judgments; this is retained as reproducibility detail rather than substantive evidence.

### 4.4 Candidate B design

Candidate B contains three four-stage arms. Each arm-stage has 400 training, 60 validation, and 120 test examples, balanced by label; each test therefore contains 60 examples per class.

- **B-DOC:** document-context benign examples at every stage.
- **B-SHORT:** the shared Stage-1 document anchor, followed by ordinary short-form benign examples.
- **B-HARD:** identical to B-SHORT through Stage 2, followed by reviewed hard-benign partitions.

All arms are identical at Stage 1, and B-SHORT and B-HARD are identical through Stage 2. They share malicious IDs, splits, families, and injection positions. The malicious sequence is AF2 Ignore, AF3 Escape-Separation, AF4 Fake Completion, and AF5 Combined. AF5 composes earlier attack components and is explicitly **not** family-disjoint. Matched anchors improve interpretation, but source, language, length, and format remain bundled; arm differences are not pure causal effects of one property.

### 4.5 Replay methods

**No Replay (M0)** trains on current-stage data only. **Reservoir ER (M1)** maintains a standard unstratified online reservoir and samples it uniformly. **DER++ (M2)** uses the same reservoir identities while retaining two historical logits per example. Its loss combines current CE, replay-label CE, and replay-logit MSE with `alpha=beta=0.5`; the project divides by $1+\alpha+\beta$ to control loss scale. This normalization is a project choice, not a claim from [6].

**Security-aware Replay (M3)** uses a 200-example buffer with 100 benign and 100 malicious records. Historical-stage quotas are deterministic, one example per available coverage key is preferred before filling, and hashes resolve ties. Scores are frozen before the stage update on TRAIN data only:

\[
r(x,y)=
\begin{{cases}}
1-p_\text{{attack}}(x), & y=1 \\ 
p_\text{{attack}}(x), & y=0.
\end{{cases}}
\]

The training loss is the same CE replay loss as Reservoir, $(L_\text{{current}}+L_\text{{replay}})/2$. Evaluation data, external banks, and future stages never enter selection.

**Generic Loss-aware Replay (M4)** is a matched control. It keeps the detector, capacity, 100/100 balance, stage quotas, coverage rule, replay loss, exposure, and deterministic tie structure fixed, changing only the pre-update selection score to true-label cross-entropy:

\[
d(x,y)=\operatorname{{CE}}(f_{{\theta,\mathrm{{pre}}}}(x),y).
\]

For a common binary model state and fixed label,

\[
d(x,y)=-\log(1-r(x,y)).
\]

The relationship is strictly monotone in exact arithmetic. Observed buffer differences arise first from FP16/FP32 precision and ties and then compound as retained examples and model trajectories diverge. M4 therefore tests operational selector behavior; it prevents attributing the result to a uniquely security-specific semantic score.

### 4.6 Metrics and prospective gates

Primary metrics are acquisition attack TPR, benign FPR, macro-F1, accuracy, signed BWT, peak forgetting, and FPR increase. Attack TPR is also reported by head/middle/tail injection position. NotInject and a 120-record B-short bank are external diagnostic-only evaluations. They are never used for thresholds, epochs, selection, replay design, or R0/R1/R2 decisions.

## 5. Experimental Setup

| Item | Frozen value |
|---|---|
| Detector | `microsoft/deberta-v3-base` |
| Revision | `8ccc9b6f36199bec6961081d44eb72fb3f7353f3` |
| Seeds | 17, 42, 2026 |
| Stage order | AF2 → AF3 → AF4 → AF5 |
| Epochs | 1 per stage |
| Batch size | 4 |
| Learning rate | $10^{{-5}}$ |
| Optimizer | AdamW; reset at each stage |
| Maximum length | 1280 |
| Decision threshold | Binary argmax; $p_\text{{attack}}=0.5$ boundary |
| Replay memory | M=200 historical examples |
| Replay batch | 4; one replay batch per current minibatch when nonempty |
| Precision/device | FP16 CUDA 12.6; NVIDIA GeForce RTX 2060 |
| Software | Python 3.14.5; PyTorch 2.14.0+cu126; Transformers 4.57.6 |

The final corpus contains 45 method-arm-seed runs and 180 stage-end checkpoints: 9/36 for No Replay, 27/108 for Reservoir, DER++, and Security-aware, and 9/36 for Loss-aware. All artifact and fairness validations pass, with zero failed or resumed runs. All five methods pass all 12 arm-stage acquisition gates. Results are descriptive means ± sample SD over the three fixed seeds; no p-values or significance claims are used.

## 6. Results

### 6.1 Benchmark qualification

The initial AF1–AF4 stream did not create sustained replay pressure. H2-mini acquired every stage and produced attack-TPR BWT of 0 in both arms, leaving it at R0. Candidate B changed the evidence: No Replay produced a consistent B-SHORT AF3 attack-TPR loss and a separate B-HARD benign-FPR shift. The B-SHORT AF3 loss was harmful in all seeds and replicated in direction in B-DOC, satisfying R2 and prospectively justifying replay.

### 6.2 Five-method quantitative summary

| Method | B-SHORT AF3 final TPR loss (pp) | B-HARD FPR increase (pp) | B-HARD D0-1 final FPR change (pp) | Acquisition gates |
|---|---:|---:|---:|---:|
| No Replay | 35.000 ± 38.115 | 20.370 ± 18.736 | 62.222 ± 54.297 | 12/12 |
| Reservoir ER | -0.556 ± 2.546 | 1.481 ± 3.060 | 5.000 ± 8.660 | 12/12 |
| DER++ | 0.000 ± 1.667 | 0.370 ± 1.156 | 1.667 ± 2.887 | 12/12 |
| Security-aware | 1.111 ± 0.962 | -0.556 ± 0.962 | 0.000 ± 0.000 | 12/12 |
| Generic Loss-aware | -0.556 ± 0.962 | 0.185 ± 0.321 | 1.111 ± 1.925 | 12/12 |

Negative “loss” denotes positive backward transfer; negative FPR increase denotes a lower final FPR than at acquisition. The full five-method table includes BWT, peak forgetting, external diagnostics, runtime, GPU memory, and auxiliary storage.

### 6.3 Malicious retention

The decisive malicious result is B-SHORT AF3. No Replay loses **35.000 ± 38.115 pp** from acquisition to the final AF5 checkpoint, with seed losses of **20.000, 6.667, and 78.333 pp**. The large SD indicates unstable magnitude, while all three directions are harmful. Reservoir ER, DER++, Security-aware, and Loss-aware reduce the mean loss to **-0.556 ± 2.546**, **0.000 ± 1.667**, **1.111 ± 0.962**, and **-0.556 ± 0.962 pp**, respectively. Security-aware is not the best method on this endpoint; the generic methods are equally strong or slightly better descriptively.

![B-SHORT AF3 retention](../results/final_figures/b_short_af3_retention.png)

*Figure 1. B-SHORT AF3 attack TPR after acquisition and later stages. Thin lines show seeds and bold lines show means. No Replay exhibits a large, seed-sensitive endpoint decline; all replay methods largely preserve AF3. The plot does not imply a unique best replay selector.*

### 6.4 Benign retention and over-defense

The principal benign result is B-HARD FPR increase. No Replay reaches **20.370 ± 18.736 pp** overall. Reservoir, DER++, Security-aware, and Loss-aware reduce this to **1.481 ± 3.060**, **0.370 ± 1.156**, **-0.556 ± 0.962**, and **0.185 ± 0.321 pp**. The shared D0-1 partition shows the sharper failure: No Replay changes by 0, 100, and 86.667 pp across seeds. Security-aware remains 0/0/0; Loss-aware is 3.333/0/0; Reservoir is 0/15/0; and DER++ is 0/0/5 pp. Security-aware therefore provides the best observed hard-benign boundary retention, but the evidence is three-seed and descriptive.

![B-HARD benign retention](../results/final_figures/b_hard_benign_retention.png)

*Figure 2. B-HARD benign FPR trajectories for D0-1, B0-2, and BH-1 after their acquisition. The large No Replay D0-1 shift is mostly prevented by replay; later partitions remain near zero.*

![Malicious versus benign trade-off](../results/final_figures/malicious_benign_tradeoff.png)

*Figure 3. Mean B-SHORT AF3 attack-TPR loss versus B-HARD benign-FPR increase, with sample-SD bars. Lower is better on both axes. Replay methods cluster near the origin but occupy different malicious/benign trade-offs.*

### 6.5 Security-aware versus generic replay

Security-aware is strongly supported relative to No Replay in both primary directions. Its comparison with Reservoir and DER++ is mixed. Reservoir has lower mean AF3 attack loss, whereas Security-aware has lower B-HARD FPR drift. DER++ likewise has slightly lower AF3 loss, while Security-aware has lower overall and D0-1 benign drift. These differences do not establish equivalence or superiority; they show that generic replay is already a strong baseline.

Buffer analysis helps explain the methods. Reservoir and DER++ are not forced to retain exact class/stage quotas, whereas Security-aware and Loss-aware are. The Security-aware and Loss-aware selected-buffer Jaccard overlap declines from approximately 0.93 at AF2/AF3 to 0.69 at AF4 and 0.55 at AF5, exposing operational divergence even though the ideal scores are monotone-related.

![Memory selection comparison](../results/final_figures/memory_selection_comparison.png)

*Figure 4. Final label and historical-stage composition of replay buffers, plus Security-aware/Loss-aware selected-memory overlap. No raw prompt text is displayed.*

### 6.6 Generic Loss-aware control: Case C

The final control yields **Case C**, a sparse trade-off rather than dominance. Security-aware minus Loss-aware B-SHORT AF3 attack loss is **+1.667 ± 1.667 pp**, favoring Loss-aware; paired differences are +1.667, 0, and +3.333 pp. Security-aware minus Loss-aware B-HARD benign-FPR increase is approximately **-0.741 ± 1.283 pp**, favoring Security-aware; the nonzero difference occurs in one seed. The D0-1 contrast has the same sparse structure: Security-aware is 0/0/0, while Loss-aware is 3.333/0/0 pp.

The exact score identity constrains causal interpretation. Security risk and true-label CE rank examples identically in exact arithmetic under the same model state. Recorded all-example Spearman correlation is near one at AF2/AF3 and falls at AF4/AF5 as numeric ties and then-diverged trajectories accumulate. Thus, current evidence supports structured hard-example replay, not a uniquely security-specific ranking principle.

![Security and loss score relationship](../results/final_figures/security_loss_score_relationship.png)

*Figure 5. Left: the exact binary relation (\mathrm{{CE}}=-\log(1-r)). Right: recorded correlation and selected-buffer overlap under the actual FP16/FP32 implementation. Later operational divergence does not invalidate the analytic monotonic relationship.*

### 6.7 External diagnostics

B-DOC remains severely over-defensive after AF5. Across the five methods, final NotInject FPR ranges from **69.715 ± 15.933 pp** for No Replay to **92.527 ± 4.686 pp** for Reservoir; final B-short FPR ranges from **37.222 ± 7.561** to **63.333 ± 4.640 pp**. B-SHORT and B-HARD are much lower: final B-short FPR is zero for every method, while replay-method NotInject means range from 0.098 to 1.672 pp in B-SHORT and from 0 to 1.278 pp in B-HARD. Cross-bank differences are not causal because the banks differ in source and language. They were never used for selection or tuning.

![External over-defense diagnostics](../results/final_figures/external_overdefense.png)

*Figure 6. Final NotInject and B-short FPR by arm and method. B-DOC remains severely over-defensive, while B-SHORT/B-HARD are near zero. These external banks are diagnostic-only and are not method-selection evidence.*

## 7. Discussion

The results show why continual security evaluation must preserve both error directions. A detector that remembers attacks but increasingly blocks benign inputs is not stable, and a detector with low benign FPR can still forget an earlier attack family. Candidate B exposes both cases under one matched framework. Current-stage acquisition remains strong throughout, so the dominant differences are retention effects rather than inability to learn later stages.

The benchmark-development path also matters. The family-separated and H2-mini streams largely ceiling-saturate; running replay there would have encouraged conclusions from small or transient differences. The prospective R0/R1/R2 policy delayed replay until Candidate B produced an interpretable failure. Candidate B's matched anchors then help locate two pressures: the document-to-short transition aligns with substantial AF3 interference, while hard-benign exposure aligns with benign-boundary movement. These remain associations because format, source, language, length, and semantics are not isolated individually.

Replay is highly effective in this setting. All four replay methods preserve acquisition and remove most of the large No Replay deviations. Reservoir's strong performance cautions against assuming that additional machinery is necessary. DER++ adds historical-logit information but shows no decisive advantage at the fixed budget. Security-aware's strongest evidence is B-HARD benign protection. Loss-aware's strong malicious retention and the analytic score relationship show that class/stage balance, coverage, and retaining difficult examples may matter as much as the score formula itself.

For future RAG guardrails, the practical implication is methodological rather than deployment-ready: evaluate guards after sequential updates, maintain separate malicious and benign retention matrices, audit replay data provenance, and compare structured selectors against strong generic controls. External diagnostics should remain isolated from training and selection. B-DOC's severe NotInject and B-short FPR also shows that replay cannot repair a benign context never represented in the evolving training distribution.

## 8. Limitations

This study uses one detector architecture and revision, three seeds, one stage order, one epoch per stage, one replay exposure ratio, and one memory budget (M=200). It therefore does not establish cross-detector generalization, a memory-performance frontier, or population-level uncertainty. The 60 malicious and 60 benign test examples per stage produce a coarse 1.667-pp class-conditional resolution; all comparisons are descriptive and make no statistical-significance claim.

Candidate B is constructed. AF5 is compositional rather than family-disjoint. The three arms retain source, format, length, language, and semantic confounds. B-short contains 42 English and 78 multilingual examples and is not language-matched to every arm. OR-Bench lacks official rewrite-parent IDs, and project semantic grouping is an approximation. The Chen-derived data are a deterministic reconstruction rather than an exact reproduction.

Security and loss scores saturate at later stages, particularly on malicious AF5 records, so numeric precision and tie handling affect operational selection. Security-aware and Loss-aware are monotone-equivalent only under a common binary model state; later models differ because earlier selections differ. NotInject and B-short are finite, mismatched diagnostic banks, not causal trigger tests or deployment evaluations. Raw-source license and provenance caveats remain. No deployment-scale, live-RAG, cross-detector, memory-budget, or replay-exposure validation is claimed.

## 9. Conclusion

Continual IPI evaluation reveals retention failures that static or ceiling-saturated benchmarks can miss. Candidate B exposes both malicious forgetting and benign over-defense under matched sequential conditions. At a fixed 200-example memory, Reservoir ER, DER++, Security-aware Replay, and Generic Loss-aware Replay all substantially reduce the dominant failures without preventing new-stage acquisition. Security-aware achieves the strongest observed B-HARD benign retention, but it is not uniformly superior, and the Loss-aware control rules out a claim of uniquely security-specific score semantics. The defensible result is that structured replay improves continual security stability and that buffer design affects the balance between malicious retention and benign acceptance. Future work should test additional detectors, budgets, exposure ratios, stage orders, and more realistic evolving streams.

{REFERENCES_EN}

## Appendix pointers

The complete five-method table, reproducibility table, and per-seed Appendices A–I are stored under `results/final_tables/`. Human-review procedures, calibration, encoding recovery, leakage audits, provenance, memory composition, external slices, and artifact validation are assigned in `docs/final_appendix_plan.md`.
"""


REFERENCES_ZH = """## 参考文献

[1] K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and M. Fritz. “Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection.” 2023.

[2] Y. Chen, H. Li, Y. Sui, Y. He, Y. Liu, Y. Song, and B. Hooi. “Can Indirect Prompt Injection Attacks Be Detected and Removed?” *Proceedings of ACL 2025*, pp. 18189–18206, 2025.

[3] H. Li, X. Liu, N. Zhang, and C. Xiao. “PIGuard: Prompt Injection Guardrail via Mitigating Overdefense for Free.” *Proceedings of ACL 2025*, pp. 30420–30437, 2025.

[4] G. Kholkar and R. Ahuja. “CAPTURE: Context-Aware Prompt Injection Testing and Robustness Enhancement.” *Proceedings of the Workshop on Large Language Model Security (LLMSEC)*, 2025.

[5] D. Rolnick, A. Ahuja, J. Schwarz, T. P. Lillicrap, and G. Wayne. “Experience Replay for Continual Learning.” *Advances in Neural Information Processing Systems 32*, 2019.

[6] P. Buzzega, M. Boschini, A. Porrello, D. Abati, and S. Calderara. “Dark Experience for General Continual Learning: A Strong, Simple Baseline.” *Advances in Neural Information Processing Systems 33*, 2020.

[7] P. Buzzega, M. Boschini, A. Porrello, and S. Calderara. “Rethinking Experience Replay: A Bag of Tricks for Continual Learning.” *25th International Conference on Pattern Recognition (ICPR)*, 2020, doi:10.1109/ICPR48806.2021.9412614.

[8] M. Elsayed and A. R. Mahmood. “Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning.” *International Conference on Learning Representations (ICLR)*, 2024.
"""


ZH = rf"""# ContinualGuard：恶意与良性分布漂移下的持续间接提示注入检测

**项目状态：** 实验完成 / 最终分析完成 / 仅限报告撰写

## 摘要

间接提示注入（Indirect Prompt Injection, IPI）把恶意指令置于语言模型应用随后处理的外部内容中。现有检测器通常在固定分布上评估，但真实更新过程可能连续遭遇攻击形式、来源、格式以及含可疑指令式措辞的良性输入变化。本项目研究固定 IPI 检测器在学习新阶段时能否保留既有恶意行为并维持良性接受边界。我们先按冻结的前瞻准入标准考察攻击家族序列和经人工复核的 H2-mini；两者基本达到性能天花板，均不足以支持 replay。随后构建三臂匹配的 Candidate B，将文档、短文本与困难良性上下文同 Ignore、Escape-Separation、Fake Completion 和组合式 Combined 攻击相结合。Candidate B 同时暴露恶意遗忘与良性误报漂移。在固定 DeBERTa-v3-base、三个种子、每阶段一轮训练和 200 条历史样本内存下，我们比较 No Replay、Reservoir ER、DER++、Security-aware Replay 与 Generic Loss-aware Replay。No Replay 的 B-SHORT AF3 攻击召回损失为 35.000 ± 38.115 个百分点，B-HARD 良性误报率增加 20.370 ± 18.736 个百分点；全部 replay 方法都大幅压低两类失效并通过全部 acquisition gate。Security-aware 在 B-HARD 上取得观察到的最佳良性保留，但通用方法在部分恶意保留端点上相当或更好。由于共同二分类模型状态下安全风险与真实标签交叉熵互为单调变换，现有证据支持的是结构化困难样本 replay，而非安全特定排序分数的独特语义优势。本研究提供受控的双向安全持续评估，但不构成部署验证。

## 1. 引言

RAG 与工具集成式语言模型应用会处理用户无法控制来源的文本。IPI 的攻击指令位于被检索内容中，而不是由用户直接提交 [1]。Chen 等把外部文档上的 IPI 建模为二分类检测，并在静态基准上研究检测与移除 [2]。部署更新场景更复杂：攻击样式和输入分布随时间变化，每次更新都可能改变检测器对旧输入的判断。

因此存在两个同等重要的安全错误方向。检测器可能遗忘早期恶意模式，从而增加漏报；也可能过度防御，因为良性文本包含指令式线索而增加误报。PIGuard 在静态 guard 评估中研究后一问题，并发布 339 条含触发词的良性 NotInject 诊断集 [3]。CAPTURE 同样强调上下文中的漏报与误报，但其恶意任务是直接、单轮提示注入，而非 IPI [4]。这些工作均未研究 IPI 检测器的阶段式持续适应。

持续学习把可塑性与稳定性区分开来：前者指新知识 acquisition，后者指保留旧知识。Replay 通过在当前数据中混入有界历史样本来缓解干扰 [5–7]。本项目的核心问题是：**IPI 检测在连续分布漂移下有多稳健，replay 能否同时缓解恶意任务遗忘与良性过度防御？**

本项目有三项贡献：

1. 建立同时测量新阶段 acquisition、早期恶意保留和良性误报漂移的持续 IPI 评估框架。
2. 构建文档、短文本、困难良性三臂匹配的 Candidate B，暴露简单天花板基准所掩盖的保留失效。
3. 在固定检测器和 M=200 内存下完成五方法受控比较。Replay 修复主要失效；结构化困难样本选择改变恶意/良性稳定性权衡，其中 Security-aware 的 B-HARD 良性保留最佳，但不具有普遍方法优势。

## 2. 背景与相关工作

### 2.1 间接提示注入与静态检测

IPI 的攻击指令来自被检索或外部提供的数据，交付方式和威胁模型均不同于直接注入 [1]。Chen 等构造外部文档注入数据，评估判别式检测，并定义 Ignore、Escape-Separation、Fake Completion 和 Combined 等操作性攻击形式 [2]。本项目以该工作作为恶意数据和检测器参考，但 Chen 不是持续学习研究；项目数据是固定发布物上的确定性重构，而非精确复现。

### 2.2 Guardrail 与过度防御

Prompt guard 可能依赖表面指令线索，把良性文本误判为恶意。PIGuard 研究该静态过度防御现象并发布 NotInject [3]。CAPTURE 研究直接单轮提示注入中的上下文攻击与良性行为 [4]。本项目用 PIGuard 支持困难良性动机，NotInject 仅作为冻结诊断集；CAPTURE 仅支持上下文设计动机，没有使用其记录，也没有把其恶意任务改称 IPI。

### 2.3 持续学习与 replay

CLEAR 在强化学习中展示有界 replay 对稳定性与可塑性的作用，本项目仅借用其一般原理 [5]。ER 是监督式持续学习中的强通用基线 [6,7]；Dark Experience Replay 保存历史 logits，DER++ 进一步使用 replay 标签 [6]。这些文献支持基线选择，但未研究持续 IPI 或良性过度防御。关于可塑性损失的研究还表明，新阶段学习不足与灾难性遗忘应分开诊断 [8]，因此本项目显式设置 acquisition gate。

### 2.4 本项目填补的空白

经审核的安全文献主要评估静态 guardrail，经审核的持续学习文献则使用非安全图像或强化学习设置。本项目研究两者交叉：在 replay 获准前先用前瞻标准确认基准具有足够保留压力，并分别评估恶意和良性方向。

## 3. 问题形式化

阶段顺序为 `AF2 → AF3 → AF4 → AF5`。阶段 $t$ 到来时，二分类检测器 $f_\theta(x)$ 仅用当前阶段训练数据更新；标签 0 表示良性，标签 1 表示 IPI。模型权重跨阶段保留，AdamW 在每阶段重置。每次更新后，检测器在所有已见测试阶段上评估。

Acquisition 是首次学习某阶段后的即时性能；retention 是该性能到后续检查点的变化。高者为优指标的 BWT 为旧阶段“最终减 acquisition”的有符号平均值；peak forgetting 是历史最好值到后续值的最大非负下降。良性稳定性以 FPR increase 衡量。Attack TPR、benign FPR、macro-F1 和 accuracy 分开报告，避免用高攻击召回掩盖高良性误报。

冻结的证据状态包括 R0、R1、R2。R0 表示 replay 尚无科学依据；R1 表示有限保留证据；R2 要求预先规定的有害幅度、种子一致性及匹配证据。它们决定是否可以研究 replay，不用于宣告某方法获胜。

## 4. 方法

### 4.1 检测器与持续更新

检测器固定为 `microsoft/deberta-v3-base` 二分类器，revision 为 `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`，`max_length=1280`。每阶段使用一轮 AdamW、学习率 $10^{{-5}}$、batch size 4、常数调度、FP16 CUDA，并重置优化器。决策为两类 logits 的 argmax，等价于 $p_\text{{attack}}=0.5$ 边界；平局归入良性类。

### 4.2 前瞻式基准演化

最初的 AF1–AF4 家族分离流具有近乎完美 acquisition，保留下降较小或短暂，反向顺序控制也未形成 replay 依据。H2-mini 随后加入独立人工复核的困难良性压力，但控制臂和困难良性臂的 attack-TPR BWT 都为 0，未跨越冻结准入阈值，因而保持 R0，未运行 replay。

Candidate B 是下一次也是最后一次基准升级。它保持检测器与恶意分配不变，加强良性来源/格式对比，并加入组合式 AF5。其 No Replay 结果达到 R2；此后停止基准升级，并按预先冻结配置比较 replay 方法。

### 4.3 人工复核的良性数据

困难良性流程由两个独立 reviewer 身份执行。50 条校准的精确决策一致率为 96%，且没有“接受为良性”与“真实注入”冲突。生产复核检查良性资格、危险或歧义意图、上下文缺失、隐私/权利以及真实注入。reviewer 对候选 B1/B2/B3 *分层* 存在大幅分歧，因此在训练前简化分类：B1/B2/B3 仅保留为描述性来源信息，最终操作分类采用普通良性 B0 与来源定义的困难良性 BH。BH-1/2/3 是平衡且互斥的分区，不表示难度递增。

验证和测试记录必须获得两个 reviewer 的独立接受；训练记录采用全面初审加独立抽审，任何接受/拒绝冲突仍被排除。后续定向 B0 复核补足四个互斥 B0 分区。一项 reviewer 文本传输编码问题通过来源哈希与人工确认恢复，没有推断或改变判断；该事件仅作为复现细节保留。

### 4.4 Candidate B

Candidate B 包含三个四阶段臂。每个臂/阶段有 400 条训练、60 条验证和 120 条测试样本，标签均衡；每个测试集因此包含 60 条良性与 60 条恶意样本。

- **B-DOC：** 每阶段均使用文档上下文良性样本。
- **B-SHORT：** Stage 1 使用共享文档锚点，之后改用普通短文本良性样本。
- **B-HARD：** 与 B-SHORT 在 Stage 2 结束前完全一致，随后引入经复核的困难良性分区。

所有臂在 Stage 1 相同，B-SHORT 与 B-HARD 在 Stage 2 结束前相同；恶意 ID、split、family 和注入位置匹配。恶意顺序为 AF2 Ignore、AF3 Escape-Separation、AF4 Fake Completion 和 AF5 Combined。AF5 复用早期攻击组件，**不是** family-disjoint。匹配锚点提高了解释性，但来源、语言、长度、格式和语义仍相互纠缠，臂差异不是单因素纯因果效应。

### 4.5 Replay 方法

**M0 No Replay** 仅训练当前阶段。**M1 Reservoir ER** 使用无分层在线 reservoir 并均匀采样。**M2 DER++** 使用同一 reservoir identity，并为每条记录保存两个历史 logits；其 loss 用 `alpha=beta=0.5` 组合 current CE、replay-label CE 与 replay-logit MSE。项目为控制总 loss 尺度而除以 $1+\alpha+\beta$，这是项目选择，不是文献 [6] 的原始事实。

**M3 Security-aware Replay** 使用 200 条内存，其中 100 条良性、100 条恶意，并实施确定性历史阶段 quota、coverage-key 优先与哈希平局规则。评分只在阶段更新前对 TRAIN 数据冻结：

\[
r(x,y)=
\begin{{cases}}
1-p_\text{{attack}}(x), & y=1 \\ 
p_\text{{attack}}(x), & y=0.
\end{{cases}}
\]

训练 loss 与 Reservoir 相同，为 $(L_\text{{current}}+L_\text{{replay}})/2$。评估数据、外部诊断集和未来阶段从不参与选择。

**M4 Generic Loss-aware Replay** 是匹配控制。检测器、M=200、100/100 标签平衡、阶段 quota、coverage 规则、replay loss、暴露率和确定性平局结构全部相同，唯一变化是 pre-update 真实标签交叉熵：

\[
d(x,y)=\operatorname{{CE}}(f_{{\theta,\mathrm{{pre}}}}(x),y)=-\log(1-r(x,y)).
\]

在共同二分类模型状态和固定标签下，该关系严格单调。实际差异先来自 FP16/FP32 精度与平局，再因保留样本和模型轨迹分化而累积。因此 M4 检验的是操作性 selector 行为，并阻止把结果归因于独特的安全语义评分。

### 4.6 指标与准入规则

主要指标包括 acquisition attack TPR、benign FPR、macro-F1、accuracy、有符号 BWT、peak forgetting 和 FPR increase；attack TPR 还按 head/middle/tail 位置报告。NotInject 与 120 条 B-short 仅作外部诊断，从未用于阈值、epoch、数据选择、replay 设计或 R0/R1/R2 判定。

## 5. 实验设置

| 项目 | 冻结值 |
|---|---|
| 检测器 | `microsoft/deberta-v3-base` |
| Revision | `8ccc9b6f36199bec6961081d44eb72fb3f7353f3` |
| 种子 | 17、42、2026 |
| 阶段顺序 | AF2 → AF3 → AF4 → AF5 |
| Epoch | 每阶段 1 |
| Batch size | 4 |
| 学习率 | $10^{{-5}}$ |
| 优化器 | AdamW；每阶段重置 |
| 最大长度 | 1280 |
| 决策阈值 | 二分类 argmax；$p_\text{{attack}}=0.5$ 边界 |
| Replay 内存 | M=200 历史样本 |
| Replay batch | 4；内存非空时每个 current minibatch 配一个 replay batch |
| 精度/设备 | FP16、CUDA 12.6、NVIDIA GeForce RTX 2060 |
| 软件 | Python 3.14.5、PyTorch 2.14.0+cu126、Transformers 4.57.6 |

最终语料包含 45 个 method-arm-seed run 和 180 个阶段末 checkpoint：No Replay 为 9/36，Reservoir、DER++、Security-aware 合计 27/108，Loss-aware 为 9/36。全部 artifact 与 fairness validation 通过，没有失败或 resumed run。五种方法均通过 12/12 acquisition gate。结果为三个固定种子的描述性均值 ± sample SD，不报告 p 值或显著性。

## 6. 结果

### 6.1 基准资格

初始 AF1–AF4 流没有形成持续 replay 压力。H2-mini 成功 acquisition，两个臂的 attack-TPR BWT 都为 0，保持 R0。Candidate B 的 No Replay 则在 B-SHORT AF3 出现三种子同向 recall 损失，并在 B-HARD 出现独立的良性 FPR 漂移；B-SHORT AF3 的方向还在 B-DOC 中匹配复现，因此达到 R2 并为 replay 提供前瞻性依据。

### 6.2 五方法结果

| 方法 | B-SHORT AF3 最终 TPR loss (pp) | B-HARD FPR increase (pp) | B-HARD D0-1 最终 FPR change (pp) | Acquisition gate |
|---|---:|---:|---:|---:|
| No Replay | 35.000 ± 38.115 | 20.370 ± 18.736 | 62.222 ± 54.297 | 12/12 |
| Reservoir ER | -0.556 ± 2.546 | 1.481 ± 3.060 | 5.000 ± 8.660 | 12/12 |
| DER++ | 0.000 ± 1.667 | 0.370 ± 1.156 | 1.667 ± 2.887 | 12/12 |
| Security-aware | 1.111 ± 0.962 | -0.556 ± 0.962 | 0.000 ± 0.000 | 12/12 |
| Generic Loss-aware | -0.556 ± 0.962 | 0.185 ± 0.321 | 1.111 ± 1.925 | 12/12 |

负的 “loss” 表示正向 backward transfer；负的 FPR increase 表示最终 FPR 低于 acquisition。完整主表还包含 BWT、peak forgetting、外部诊断、运行时间、GPU 内存与辅助存储。

### 6.3 恶意保留

决定性恶意结果是 B-SHORT AF3。No Replay 从 acquisition 到 AF5 最终 checkpoint 损失 **35.000 ± 38.115 pp**，三个种子分别为 **20.000、6.667、78.333 pp**。幅度高度不稳定，但方向全部有害。Reservoir、DER++、Security-aware 和 Loss-aware 分别把均值压至 **-0.556 ± 2.546、0.000 ± 1.667、1.111 ± 0.962、-0.556 ± 0.962 pp**。Security-aware 在该端点并非最佳，通用 replay 与其相当或描述性略优。

![B-SHORT AF3 retention](../results/final_figures/b_short_af3_retention.png)

*图 1：B-SHORT AF3 在 acquisition 及后续阶段的 attack TPR。细线为种子，粗线为均值。No Replay 出现大幅且种子敏感的下降，全部 replay 方法基本保留 AF3；图中不暗示唯一最佳 selector。*

### 6.4 良性保留与过度防御

B-HARD FPR increase 是主要良性结果。No Replay 为 **20.370 ± 18.736 pp**；Reservoir、DER++、Security-aware、Loss-aware 分别降至 **1.481 ± 3.060、0.370 ± 1.156、-0.556 ± 0.962、0.185 ± 0.321 pp**。共享 D0-1 更尖锐：No Replay 为 0/100/86.667 pp；Security-aware 为 0/0/0；Loss-aware 为 3.333/0/0；Reservoir 为 0/15/0；DER++ 为 0/0/5 pp。Security-aware 因而取得观察到的最佳困难良性边界保留，但证据仅为三个种子的描述性结果。

![B-HARD benign retention](../results/final_figures/b_hard_benign_retention.png)

*图 2：B-HARD 中 D0-1、B0-2、BH-1 自 acquisition 后的良性 FPR 轨迹。Replay 基本阻止 No Replay 的大幅 D0-1 漂移。*

![恶意与良性权衡](../results/final_figures/malicious_benign_tradeoff.png)

*图 3：B-SHORT AF3 attack-TPR loss 与 B-HARD benign-FPR increase 的均值及 sample-SD；两轴均越低越好。Replay 方法都接近原点，但具有不同权衡。*

### 6.5 Security-aware 与通用 replay

Security-aware 相对 No Replay 在两个主要方向上都得到强支持；相对 Reservoir 与 DER++ 则结论混合。Reservoir 的 AF3 attack loss 更低，而 Security-aware 的 B-HARD FPR drift 更低；DER++ 同样在 AF3 略优，而 Security-aware 在总体和 D0-1 良性漂移上更优。这些小差异既不证明等价，也不证明普遍优越，只说明 generic replay 已是很强的基线。

Reservoir 与 DER++ 不强制精确 class/stage quota，而 Security-aware 与 Loss-aware 强制。Security-aware/Loss-aware buffer 的 Jaccard overlap 从 AF2/AF3 约 0.93 降到 AF4 约 0.69、AF5 约 0.55，表明即使理想分数单调相关，实际选择仍会分化。

![内存选择比较](../results/final_figures/memory_selection_comparison.png)

*图 4：Replay buffer 的最终标签、历史阶段构成及 Security-aware/Loss-aware 选择重合度，不暴露任何原始 prompt。*

### 6.6 Generic Loss-aware 控制：Case C

最终控制属于 **Case C**，表现为稀疏权衡而非一方支配。Security-aware 减 Loss-aware 的 B-SHORT AF3 attack loss 为 **+1.667 ± 1.667 pp**，偏向 Loss-aware；逐种子为 +1.667、0、+3.333 pp。Security-aware 减 Loss-aware 的 B-HARD benign-FPR increase 约为 **-0.741 ± 1.283 pp**，偏向 Security-aware；非零差异只出现在一个种子。D0-1 同样稀疏：Security-aware 为 0/0/0，Loss-aware 为 3.333/0/0 pp。

精确分数恒等式限制了因果解释。共同模型状态下，security risk 与 true-label CE 的排序在精确算术中一致；实际 all-example Spearman 相关在 AF2/AF3 接近 1，随后因数值平局和已分化轨迹在 AF4/AF5 下降。因此证据支持结构化困难样本 replay，而非独特的安全特定排序原则。

![Security 与 loss 分数关系](../results/final_figures/security_loss_score_relationship.png)

*图 5：左图为精确二分类关系 (\mathrm{{CE}}=-\log(1-r))，右图为实际 FP16/FP32 实现中的相关和 buffer overlap。后期操作性分化不否定解析单调关系。*

### 6.7 外部诊断

B-DOC 在 AF5 后仍严重过度防御。五种方法的最终 NotInject FPR 从 No Replay 的 **69.715 ± 15.933 pp** 到 Reservoir 的 **92.527 ± 4.686 pp**；B-short FPR 从 **37.222 ± 7.561** 到 **63.333 ± 4.640 pp**。B-SHORT 与 B-HARD 明显更低：所有方法最终 B-short FPR 都为 0；replay 方法的 NotInject 均值在 B-SHORT 为 0.098–1.672 pp，在 B-HARD 为 0–1.278 pp。外部 bank 的来源和语言不同，跨 bank 差异不作因果解释；它们从未参与选择或调参。

![外部过度防御诊断](../results/final_figures/external_overdefense.png)

*图 6：各臂与方法的最终 NotInject/B-short FPR。B-DOC 仍严重过度防御，B-SHORT/B-HARD 接近 0。该图仅为诊断，不是方法选择证据。*

## 7. 讨论

结果说明持续安全评估必须保留两个错误方向。记住攻击却越来越多地阻断良性输入并不稳定；良性 FPR 很低也可能伴随早期攻击遗忘。Candidate B 在同一匹配框架中暴露两类失效。全部方法的当前阶段 acquisition 均保持强势，因此主要差异来自 retention，而非后期阶段学不会。

基准发展路径也具有方法学意义。家族分离流与 H2-mini 基本达到天花板；如果在这些流上立即运行 replay，容易围绕细小或短暂差异得出结论。R0/R1/R2 前瞻策略推迟 replay，直到 Candidate B 产生可解释失效。匹配锚点进一步定位两种压力：文档转短文本与 AF3 干扰相关，困难良性暴露与良性边界移动相关。由于格式、来源、语言、长度和语义未单独隔离，这些仍是匹配关联而非纯因果效应。

Replay 在该设置中高度有效。四种 replay 都维持 acquisition 并消除大部分 No Replay 偏移。Reservoir 的强表现说明复杂机制未必必要；DER++ 增加历史 logits，却未在固定预算下取得决定性优势。Security-aware 最强证据是 B-HARD 良性保护。Loss-aware 的强恶意保留以及解析分数关系表明，class/stage 平衡、coverage 与困难样本保留的重要性可能不低于具体评分公式。

对未来 RAG guardrail 的含义是方法学建议，而非部署就绪结论：连续更新后要重复评估，分别维护恶意和良性 retention matrix，审计 replay 数据来源，并用强通用控制比较结构化 selector。外部诊断应与训练和选择隔离。B-DOC 的严重 NotInject/B-short FPR 还说明，如果演化训练分布从未代表某种良性上下文，replay 也无法自动修复该边界。

## 8. 局限

本研究仅使用一个检测器架构/revision、三个种子、一个阶段顺序、每阶段一轮训练、一个 replay 暴露率与一个 M=200 预算，因此不能建立跨检测器泛化、memory-performance 曲线或总体统计不确定性。每阶段测试只有 60 条恶意与 60 条良性记录，类条件指标分辨率为 1.667 pp。全部比较均为描述性，不主张统计显著性。

Candidate B 是构造基准。AF5 为组合式而非 family-disjoint。三臂含来源、格式、长度、语言和语义混杂。B-short 含 42 条英语与 78 条多语言记录，不能与所有臂语言匹配。OR-Bench 不提供官方 rewrite-parent ID，项目语义分组只是近似。Chen 数据是确定性重构而非精确复现。

后期分数存在饱和，尤其是 AF5 恶意记录，因此数值精度与 tie 规则影响操作选择。Security-aware 与 Loss-aware 的单调等价只在共同模型状态下成立；早期选择不同会导致后期模型不同。NotInject 与 B-short 是有限且不匹配的诊断 bank，不是因果触发词测试或部署评估。本项目不主张 deployment-scale、live-RAG、跨检测器、预算或暴露率泛化。

## 9. 结论

持续 IPI 评估能够揭示静态或天花板基准遗漏的保留失效。Candidate B 在匹配顺序条件下同时暴露恶意遗忘与良性过度防御。固定 200 条内存时，Reservoir ER、DER++、Security-aware 与 Generic Loss-aware 都大幅降低主要失效，同时不阻碍新阶段 acquisition。Security-aware 取得观察到的最佳 B-HARD 良性保留，但并非普遍更优；Loss-aware 控制排除了安全特定分数具有独特语义优势的主张。最稳健的结论是：结构化 replay 改善持续安全稳定性，buffer 设计会改变恶意保留与良性接受之间的权衡。未来工作应覆盖更多检测器、预算、暴露率、阶段顺序与更真实的动态攻击流。

{REFERENCES_ZH}

## 附录指引

完整五方法表、复现表和逐种子附录 A–I 位于 `results/final_tables/`。人工复核协议、校准、编码恢复、泄漏审计、来源、memory composition、外部 slices 与 artifact validation 的建议位置见 `docs/final_appendix_plan.md`。
"""


CITATION_AUDIT = r"""# Final citation audit

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
"""


APPENDIX_PLAN = r"""# Final appendix plan

The main report should preserve the scientific argument: motivation, prospective benchmark qualification, Candidate B, five-method comparison, primary malicious and benign results, selector-control interpretation, and limitations. The following material can move to appendices.

| Appendix | Recommended contents | Existing source |
|---|---|---|
| A. Human-review protocol | Reviewer eligibility rubric, independent-review roles, calibration gate, disagreement categories, taxonomy simplification, and workload summary | `h2_mini_review_protocol.md`, `h2_mini_calibration_results.md`, `h2_mini_production_review_results.md`, `h2_mini_taxonomy_amendment.md` |
| B. Review integrity and recovery | UTF-8/GB18030 incidents, immutable-text hashes, human reconfirmation, evidence-preserving recovery, and timing caveats | H2-mini B0 expansion/replenishment reports and review manifests |
| C. Benchmark provenance | Chen, OASST1, OR-Bench, NotInject, B-short revisions/licenses; reconstruction caveats; Candidate B manifest and validation hashes | `source_choices.md`, `hard_benign_data_audit.md`, `stronger_pilot_materialization_report.md` |
| D. Leakage and grouping audit | Exact/normalized/model-view duplicate checks, source/tree grouping, semantic/template approximations, split/stage isolation, AF5 component-overlap policy | Candidate B assignment/materialization validation artifacts |
| E. Full per-seed results | Acquisition, BWT, B-SHORT AF3 trajectories, and B-HARD benign trajectories | `appendix_A` through `appendix_D` CSVs |
| F. External diagnostics | Per-seed NotInject and B-short overall/language-slice results; diagnostic-only policy | `appendix_E_external_diagnostics.csv` and external figure |
| G. Replay memory analysis | Label/stage/source/position composition, DER++ auxiliary logits, Security/Loss score correlations, top-k overlap, and selected-memory Jaccard | `appendix_F_memory_composition.csv`, score-relationship JSON, Figures 4–5 |
| H. Runtime and resources | Per-run time, buffer-scoring time, peak GPU memory, hardware/software environment | `appendix_G_runtime_gpu.csv`, `reproducibility_table.csv` |
| I. Fairness and artifacts | Capacity/exposure controls, train-only selection, no-leakage checks, run/checkpoint/prediction counts, hashes, failures/resumes | `appendix_H_fairness_checks.csv`, `appendix_I_artifact_validation.csv` |
| J. Extended figures | External diagnostics, position-specific TPR, full arm/task trajectories, and any publication-size versions omitted from main text | `results/final_figures/` and frozen aggregate JSON |

If space is constrained, retain Figures 1–3 and the five-method primary table in the main report. Move Figures 4–6, the detailed human-review paragraph, environment table, and all per-seed values to the appendices. Do not move the score-equivalence limitation or Candidate B confounds out of the main text.
"""


REDUCTION_PLAN = r"""# Final report reduction plan

No authoritative course word or page limit was found in the project. The complete English draft is therefore the source version. If a limit is later supplied, reduce in this order:

1. Move the full Experimental Setup table to the appendix, retaining detector, seeds, M=200, epochs, and learning rate in one sentence.
2. Compress the human-review section to calibration, independent review, taxonomy simplification, and final B0/BH eligibility; move encoding recovery and counts to Appendix A/B.
3. Move Figures 4–6 and their detailed text to the appendix; keep Figures 1–3.
4. Reduce Related Work to one paragraph each for security guardrails and continual replay while preserving the direct-PI/IPI and static/continual distinctions.
5. Replace detailed method-by-method numeric prose with the primary five-row table, but retain the Security/Loss paired differences and score-equivalence equation.
6. Shorten Discussion by removing repeated numerical values; retain the two-direction security argument, benchmark qualification, generic-baseline strength, and restricted Security-aware claim.

Never remove the limitations that AF5 is compositional, Candidate B is constructed, external banks are diagnostic-only, n=3 is descriptive, or the two selector scores are monotone-related under a common model state.
"""


def word_count(text: str) -> int:
    return len(re.findall(r"\b[A-Za-z0-9][A-Za-z0-9'’\-]*\b", text))


def validate_numbers() -> None:
    checks = {
        "no_replay": ("35.000 ± 38.115", "20.370 ± 18.736", "62.222 ± 54.297"),
        "reservoir": ("-0.556 ± 2.546", "1.481 ± 3.060", "5.000 ± 8.660"),
        "derpp": ("0.000 ± 1.667", "0.370 ± 1.156", "1.667 ± 2.887"),
        "security_aware": ("1.111 ± 0.962", "-0.556 ± 0.962", "0.000 ± 0.000"),
        "loss_aware": ("-0.556 ± 0.962", "0.185 ± 0.321", "1.111 ± 1.925"),
    }
    for method, expected in checks.items():
        row = MASTER[method]
        actual = (
            f"{float(row['b_short_af3_attack_tpr_loss_mean_pp']):.3f} ± {float(row['b_short_af3_attack_tpr_loss_sample_sd_pp']):.3f}",
            f"{float(row['b_hard_benign_fpr_increase_mean_pp']):.3f} ± {float(row['b_hard_benign_fpr_increase_sample_sd_pp']):.3f}",
            f"{float(row['b_hard_d0_1_final_fpr_change_mean_pp']):.3f} ± {float(row['b_hard_d0_1_final_fpr_change_sample_sd_pp']):.3f}",
        )
        if actual != expected:
            raise AssertionError((method, actual, expected))
        for value in expected:
            if value not in EN or value not in ZH:
                raise AssertionError(f"missing frozen value {value} in report")


def main() -> None:
    if Path.cwd().resolve() != ROOT.resolve():
        raise SystemExit(f"run from canonical root {ROOT}")
    validate_numbers()
    outputs = {
        "final_report_draft_en.md": EN,
        "final_report_draft_zh.md": ZH,
        "final_citation_audit.md": CITATION_AUDIT,
        "final_appendix_plan.md": APPENDIX_PLAN,
        "final_report_reduction_plan.md": REDUCTION_PLAN,
    }
    for name, content in outputs.items():
        (DOCS / name).write_text(content.rstrip() + "\n", encoding="utf-8")
    en_wc = word_count(EN)
    citation_rows = len(re.findall(r"^\| \d+ \|", CITATION_AUDIT, flags=re.MULTILINE))
    if citation_rows != 15:
        raise AssertionError(citation_rows)
    for n in range(1, 9):
        marker = f"[{n}]"
        if marker not in EN or marker not in ZH:
            raise AssertionError(f"citation marker missing: {marker}")
    metadata = {
        "status": "PASS",
        "english_word_count_approx": en_wc,
        "citation_claims_checked": citation_rows,
        "report_sections": 10,
        "figures_embedded": 6,
        "primary_table_rows": 5,
        "frozen_runs": 45,
        "frozen_stage_end_checkpoints": 180,
        "new_experiments": 0,
    }
    (TABLES / "final_report_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
