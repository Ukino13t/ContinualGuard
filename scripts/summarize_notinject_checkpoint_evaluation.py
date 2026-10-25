import json
from pathlib import Path

a=json.loads(Path('results/notinject_checkpoint_evaluation_aggregate.json').read_text(encoding='utf-8'))
for r in a['per_checkpoint']:
    hb=','.join(f"{g}:{100*r['group_fpr'][g]['fpr']:.2f}" for g in ('HB1','HB2','HB3'))
    cat=','.join(f"{k}:{100*v['fpr']:.2f}" for k,v in r['category_fpr'].items())
    print(r['order'],r['seed'],r['stage_index'],r['stage'],f"FPR={100*r['overall_fpr']:.3f}",f"HB={hb}",f"CAT={cat}",f"meanp={r['confidence']['mean_p_attack']:.6f}",f"TPR={100*r['retained_attack_family_tpr']:.3f}",f"MF1={100*r['macro_f1']:.3f}")
print('---CROSS-SEED---')
for order, stages in a['cross_seed_mean_sample_sd'].items():
    for stage,row in stages.items():
        print(order,stage,'FPR',100*row['overall_fpr']['mean'],100*row['overall_fpr']['sample_sd'],'delta_pp',row['overall_fpr_change_from_AF1_pp']['mean'],row['overall_fpr_change_from_AF1_pp']['sample_sd'],'harmful',row['harmful_fpr_change_seed_count'],'meanp',row['mean_p_attack']['mean'],row['mean_p_attack']['sample_sd'])
print('---TRAJECTORIES---')
for t in a['trajectories']:
    print(t['order'],t['seed'],[(x['stage'],round(100*x['overall_fpr'],3),round(x['absolute_change_from_AF1_pp'],3)) for x in t['stages']],t['increase_pattern'])
print('---GROUP OVERALL---',json.dumps(a['order_comparison']['all_checkpoint_group_fpr'],indent=2))
print('---CAT OVERALL---',json.dumps(a['order_comparison']['all_checkpoint_category_fpr'],indent=2))
print('---IMMEDIATE---',json.dumps(a['order_comparison']['immediate_after_AF1'],indent=2))
print('---FINAL---',json.dumps(a['order_comparison']['final_vs_AF1'],indent=2))
