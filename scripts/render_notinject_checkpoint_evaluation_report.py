"""Render the completed H2-D1 result report from the validated aggregate."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = json.loads((ROOT / 'results/notinject_checkpoint_evaluation_aggregate.json').read_text(encoding='utf-8'))
V = json.loads((ROOT / 'results/notinject_checkpoint_evaluation_artifact_validation.json').read_text(encoding='utf-8'))


def pct(value):
    return f'{100*value:.3f}%'


def pp(value):
    return f'{value:+.3f} pp'


def ms(item, scale=100, suffix='%'):
    return f"{scale*item['mean']:.3f} ± {scale*item['sample_sd']:.3f}{suffix}"

lines = [
    '# NotInject checkpoint evaluation', '',
    '## Scope and validated protocol', '',
    'H2-D1 is a frozen, evaluation-only diagnostic. All 339 records from official NotInject revision `847ae76cf8fea5ed325429e569ae8cfef022d2e0` were evaluated at every stage-end checkpoint from both completed no-replay streams. No training, optimizer loading, threshold change, model selection, H2-F construction, replay, or selective adaptation occurred.', '',
    'The classifier received prompt text only. Group, category, record identity, and other NotInject metadata were joined only after inference. Prediction used the checkpoint’s original binary argmax rule, equivalent to a fixed `p_attack = 0.5` boundary.', '',
    f"Validation passed for **24 checkpoints × 339 records = {A['prediction_count']:,} predictions**. Every checkpoint and `state.pt` file had identical pre/post bytes and SHA-256. A repeated 16-record FP16 sample produced identical labels and maximum probability difference `{V['reproducibility']['max_abs_p_attack_difference']:.3g}` under the recorded `1e-5` numerical tolerance. Aggregate statistics recomputed exactly from raw predictions.", '',
    '## Main result', '',
    'Severe over-defense is already present after AF1: mean NotInject FPR is **96.067% ± 1.802 pp** across seeds, while the same checkpoints retain 100% attack-family TPR and 100% macro-F1 on the existing AF1 test. This demonstrates a large hard-benign boundary failure that the ordinary benign test sets did not expose.', '',
    'The failure does **not** worsen consistently through sequential learning. Primary final FPR is 84.464% ± 8.123 pp, a mean change of -11.603 ± 9.339 pp from AF1, with decreases in all three seeds. Secondary final FPR is 95.772% ± 4.506 pp, a mean change of -0.295 ± 5.628 pp; two seeds increase and one decreases substantially. AF4 immediately after AF1 lowers FPR in every seed (-27.729 ± 28.871 pp), followed by a rebound after AF3. The trajectory is order- and seed-sensitive rather than a reproducible monotonic drift.', '',
    'High NotInject FPR is evidence of over-defense. It is not catastrophic forgetting, and this evaluation-only bank does not establish continual-retention pressure.', '',
    '## Per-checkpoint overall and confidence diagnostics', '',
    '| Order | Seed | Pos. | Stage | FPR | Benign accuracy | Mean p | Median | p90 | p95 | p99 |',
    '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|',
]
for r in A['per_checkpoint']:
    c=r['confidence']
    lines.append(f"| {r['order']} | {r['seed']} | {r['stage_index']} | {r['stage']} | {pct(r['overall_fpr'])} | {pct(r['benign_accuracy'])} | {c['mean_p_attack']:.6f} | {c['median_p_attack']:.6f} | {c['p90_p_attack']:.6f} | {c['p95_p_attack']:.6f} | {c['p99_p_attack']:.6f} |")

lines += ['', '## Per-checkpoint group and category FPR', '',
    '| Order | Seed | Pos. | Stage | HB1 | HB2 | HB3 | Common | Technique | Virtual Creation | Multilingual |',
    '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
for r in A['per_checkpoint']:
    g=r['group_fpr']; c=r['category_fpr']
    lines.append(f"| {r['order']} | {r['seed']} | {r['stage_index']} | {r['stage']} | {pct(g['HB1']['fpr'])} | {pct(g['HB2']['fpr'])} | {pct(g['HB3']['fpr'])} | {pct(c['Common']['fpr'])} | {pct(c['Technique']['fpr'])} | {pct(c['Virtual Creation']['fpr'])} | {pct(c['Multilingual']['fpr'])} |")

lines += ['', '## Cross-seed trajectories', '',
    '| Order | Stage | Overall FPR mean ± SD | Change from AF1 mean ± SD | Harmful seeds | Mean p_attack mean ± SD |',
    '|---|---|---:|---:|---:|---:|']
for order, stages in A['cross_seed_mean_sample_sd'].items():
    for stage,row in stages.items():
        lines.append(f"| {order} | {stage} | {ms(row['overall_fpr'])} | {row['overall_fpr_change_from_AF1_pp']['mean']:+.3f} ± {row['overall_fpr_change_from_AF1_pp']['sample_sd']:.3f} pp | {row['harmful_fpr_change_seed_count']}/3 | {row['mean_p_attack']['mean']:.6f} ± {row['mean_p_attack']['sample_sd']:.6f} |")

lines += ['', '| Order | Seed | FPR path | Final Δ from AF1 | Peak increase | Pattern |', '|---|---:|---|---:|---:|---|']
for t in A['trajectories']:
    path=' → '.join(f"{s['stage']} {pct(s['overall_fpr'])}" for s in t['stages'])
    lines.append(f"| {t['order']} | {t['seed']} | {path} | {pp(t['final_change_pp'])} | {t['peak_increase_pp']:.3f} pp | {t['increase_pattern']} |")

lines += ['', '## Cross-seed group FPR', '',
    '| Order | Stage | HB1 | HB2 | HB3 |', '|---|---|---:|---:|---:|']
for order, stages in A['cross_seed_mean_sample_sd'].items():
    for stage,row in stages.items():
        lines.append(f"| {order} | {stage} | {ms(row['HB1_fpr'])} | {ms(row['HB2_fpr'])} | {ms(row['HB3_fpr'])} |")

lines += ['', '## Cross-seed category FPR', '',
    '| Order | Stage | Common | Technique | Virtual Creation | Multilingual |', '|---|---|---:|---:|---:|---:|']
for order, stages in A['cross_seed_mean_sample_sd'].items():
    for stage,row in stages.items():
        lines.append(f"| {order} | {stage} | {ms(row['category_Common_fpr'])} | {ms(row['category_Technique_fpr'])} | {ms(row['category_Virtual Creation_fpr'])} | {ms(row['category_Multilingual_fpr'])} |")

lines += ['', '## Primary versus secondary order', '',
    'AF4 immediately after AF1 does not increase over-defense more than AF2. It lowers FPR much more in every seed:', '',
    '| Seed | AF2 after AF1 Δ | AF4 after AF1 Δ | AF4 minus AF2 |', '|---:|---:|---:|---:|']
for r in A['order_comparison']['immediate_after_AF1']:
    lines.append(f"| {r['seed']} | {pp(r['AF2_after_AF1_change_pp'])} | {pp(r['AF4_after_AF1_change_pp'])} | {pp(r['AF4_minus_AF2_immediate_change_pp'])} |")

lines += ['', 'Across all 24 correlated checkpoint states, descriptive mean FPR is HB1 92.810%, HB3 91.445%, and HB2 90.118%; two- and three-trigger records are therefore not uniformly more vulnerable than HB1. Virtual Creation is most frequently misclassified (97.520%), followed by Multilingual (91.270%), Common (91.038%), and Technique (89.320%). These are descriptive comparisons because group/category composition is confounded and the 24 states are not independent samples.', '',
    '## Attack-retention versus hard-benign trade-off', '',
    '`retained attack-family TPR` and `macro-F1` are means over all attack-family tests seen at that checkpoint. They remain separate from NotInject metrics.', '',
    '| Checkpoint | Attack TPR | Macro-F1 | NotInject FPR | Mean p_attack |', '|---|---:|---:|---:|---:|']
for r in A['security_tradeoff']:
    lines.append(f"| {r['checkpoint']} | {pct(r['retained_attack_family_tpr'])} | {pct(r['macro_f1'])} | {pct(r['notinject_fpr'])} | {r['notinject_mean_p_attack']:.6f} |")

lines += ['', 'The paired view exposes the central limitation: attack-family TPR and macro-F1 are generally 97–100%, while hard-benign FPR is often 90–100%. AF4 sometimes improves the benign boundary sharply without a comparably large attack-metric loss, but the magnitude is seed- and order-dependent and later stages can reverse it.', '',
    '## Scientific answers and replay status', '',
    '1. **Order affects hard-benign FPR strongly.** The largest contrast is AF4 immediately after AF1, especially seed 42 (97.640% → 36.578%), followed by rebound after AF3.',
    '2. **Over-defense exists and is severe from AF1.** It is present in all three seeds and all groups/categories.',
    '3. **Sequential worsening is not established.** No final primary seed worsens; the secondary final direction is mixed, and the mean is slightly lower than AF1.',
    '4. **Increases are transient or seed-specific.** Primary AF3 rises in two seeds and then falls after AF4. Secondary AF3/AF2 rebounds, but final worsening is not three-seed consistent.',
    '5. **R0 remains active; R1 and R2 are not entered.** NotInject establishes over-defense but does not meet the continual-retention evidence gates.',
    '6. **H2-F construction is more strongly justified.** A training/evaluation stream with licensed, disjoint hard-benign stages is needed to study whether the boundary can be acquired and retained. This supports benchmark construction, not replay implementation.', '',
    'A further limitation is input format: the approved evaluation feeds short NotInject prompt text directly to a detector trained on Chen-derived document contexts. Trigger bias is the intended diagnostic, but prompt-length/context-domain mismatch may also contribute to the absolute FPR. H2-F should include a matched short-form ordinary-benign control.', '',
    '## Artifacts', '',
    '- `results/notinject_checkpoint_evaluation_aggregate.json`',
    '- `results/notinject_checkpoint_evaluation_predictions.jsonl`',
    '- `results/notinject_checkpoint_evaluation_artifact_validation.json`',
    '- `results/notinject_checkpoint_fpr_matrix.csv`',
    '- `results/notinject_checkpoint_group_fpr.csv`',
    '- `results/notinject_checkpoint_category_fpr.csv`', '',
    f"The 24-checkpoint inference pipeline took {A['evaluation_runtime_seconds']:.3f} seconds excluding the separate full-file pre/post hash validation passes. No H2-F records were constructed and no model was trained.",
]
(ROOT / 'docs/notinject_checkpoint_evaluation.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print('wrote docs/notinject_checkpoint_evaluation.md')
