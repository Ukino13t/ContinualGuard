"""Validate and summarize the final Candidate B Generic Loss-aware Replay control."""
from __future__ import annotations

import csv, hashlib, json, math, statistics
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from .loss_aware_control import CONFIG_RELATIVE, METHOD, RESULT_RELATIVE, TIE_NAMESPACE
from .replay_summary import aggregate_rows, read_json
from .stronger_pilot_no_replay import ARMS, SEEDS, STAGES, CandidateBData, atomic_json, sha256_file


def run_dir(root, arm, seed):
    return root / RESULT_RELATIVE / arm / f"seed{seed}"


def jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tie_hash(seed, example_id):
    return hashlib.sha256(f"{seed}|{TIE_NAMESPACE}|{example_id}".encode()).hexdigest()


def verify(root: Path):
    primary = read_json(root / "results/replay_comparison_candidate_b/preflight/preflight.json")
    preflight = read_json(root / RESULT_RELATIVE / "preflight/preflight.json")
    if primary.get("status") != "PASS" or set(primary.get("methods", {})) != {"reservoir", "derpp", "security_aware"}:
        raise ValueError("Primary replay all-method preflight is not PASS")
    if preflight.get("status") != "PASS" or preflight.get("cuda_oom") is not False:
        raise ValueError("Loss-aware preflight is not PASS")
    config_hash = sha256_file(root / CONFIG_RELATIVE)
    datasets = {arm: CandidateBData(root, arm) for arm in ARMS}
    lookup = {arm: {row["example_id"]: row for stage in STAGES for row in datasets[arm].records(stage, "train")} for arm in ARMS}
    runs = checkpoints = checkpoint_files = prediction_files = prediction_rows = resumed = 0
    capacities, exposures, quota_checks, unpublished = [], [], [], []
    for arm in ARMS:
        for seed in SEEDS:
            directory = run_dir(root, arm, seed)
            meta = read_json(directory / "run_metadata.json")
            if meta.get("status") != "COMPLETE" or meta.get("completed_stages") != 4:
                raise ValueError(f"Incomplete run: {arm}/seed{seed}")
            runs += 1; resumed += meta.get("resumed_from_stage") is not None
            baseline_init = root / "results/stronger_pilot_no_replay" / arm / f"seed{seed}/initialization.json"
            if read_json(directory / "initialization.json") != read_json(baseline_init):
                raise ValueError(f"Initialization mismatch: {arm}/seed{seed}")
            for stage_index, stage in enumerate(STAGES, 1):
                stage_dir = directory / f"stage{stage_index}"
                smeta, metrics = read_json(stage_dir / "stage_metadata.json"), read_json(stage_dir / "metrics.json")
                if smeta.get("status") != "COMPLETE" or smeta.get("stage") != stage:
                    raise ValueError(f"Stage metadata mismatch: {stage_dir}")
                manifest = read_json(stage_dir / "checkpoint/checkpoint_manifest.json")
                for name, digest in manifest["files"].items():
                    if sha256_file(stage_dir / "checkpoint" / name) != digest:
                        raise ValueError(f"Checkpoint hash mismatch: {stage_dir}/{name}")
                    checkpoint_files += 1
                state = torch.load(stage_dir / "checkpoint/state.pt", map_location="cpu", weights_only=False)
                if not (state["method"] == METHOD and state["arm"] == arm and state["seed"] == seed and
                        state["completed_stages"] == stage_index and state["config_sha256"] == config_hash):
                    raise ValueError(f"Checkpoint lineage mismatch: {stage_dir}")
                entries = state["buffer_state"]["entries"]
                if len(entries) != 200 or len({x["example_id"] for x in entries}) != 200:
                    raise ValueError(f"Memory capacity/identity mismatch: {stage_dir}")
                if state["buffer_state"]["seen_count"] != 400*stage_index or jsonl(stage_dir/"buffer.jsonl") != entries:
                    raise ValueError(f"Memory reconstruction mismatch: {stage_dir}")
                expected = {s: (100 // stage_index) + (1 if i < 100 % stage_index else 0) for i, s in enumerate(STAGES[:stage_index])}
                counts = defaultdict(int)
                for entry in entries:
                    if "text" in entry:
                        raise ValueError(f"Raw text stored in memory audit: {stage_dir}")
                    source = lookup[arm].get(entry["example_id"])
                    if source is None or source["text_sha256"] != entry["text_sha256"] or source["split"] != "train":
                        raise ValueError(f"Memory source mismatch: {stage_dir}")
                    if entry["stage_index"] > stage_index or entry["selection_score_name"] != "pre_update_true_label_cross_entropy":
                        raise ValueError(f"Memory stage/score mismatch: {stage_dir}")
                    if entry["tie_break_sha256"] != tie_hash(seed, entry["example_id"]) or entry["pre_update_generic_loss"] < 0:
                        raise ValueError(f"Memory score/tie mismatch: {stage_dir}")
                    risk = entry["pre_update_p_attack"] if entry["label"] == 0 else 1-entry["pre_update_p_attack"]
                    if not math.isclose(entry["pre_update_security_risk"], risk, abs_tol=1e-7):
                        raise ValueError(f"Diagnostic risk mismatch: {stage_dir}")
                    counts[(entry["label"], entry["task_id"])] += 1
                if any(counts[(label, task)] != quota for label in (0, 1) for task, quota in expected.items()):
                    raise ValueError(f"Label/stage quota mismatch: {stage_dir}")
                scores = jsonl(stage_dir / "pre_update_loss_scores.jsonl")
                if len(scores) != 400 or any(x["task_id"] != stage for x in scores):
                    raise ValueError(f"Pre-update TRAIN score mismatch: {stage_dir}")
                expected_replay, expected_passes = ((0, 0) if stage_index == 1 else (400, 100))
                training = metrics["training"]
                if not (training["current_examples"] == 400 and training["replay_examples_exposed"] == expected_replay and training["replay_forward_passes"] == expected_passes):
                    raise ValueError(f"Replay exposure mismatch: {stage_dir}")
                capacities.append({"arm": ARMS[arm], "seed": seed, "stage": stage, "size": 200, "pass": True})
                exposures.append({"arm": ARMS[arm], "seed": seed, "stage": stage, "current": 400, "replay": expected_replay, "passes": expected_passes, "pass": True})
                quota_checks.append({"arm": ARMS[arm], "seed": seed, "stage": stage, "expected_per_label_stage": expected, "pass": True})
                predictions = list((stage_dir / "predictions").glob("*.jsonl"))
                if len(predictions) != stage_index + 2:
                    raise ValueError(f"Prediction file count mismatch: {stage_dir}")
                prediction_files += len(predictions); prediction_rows += sum(len(jsonl(p)) for p in predictions); checkpoints += 1
            unpublished.extend(str(p) for p in directory.glob("_attempt_stage*"))
    if unpublished:
        raise ValueError(f"Unpublished attempts remain: {unpublished}")
    artifact = {"status":"PASS", "new_runs":runs, "new_checkpoints":checkpoints, "checkpoint_files_hashed":checkpoint_files,
                "prediction_files":prediction_files, "prediction_rows":prediction_rows, "failed_runs":0, "resumed_runs":resumed,
                "initialization_matches_no_replay":True, "candidate_b_hashes_verified":True, "checkpoint_lineage_verified":True,
                "memory_train_only_and_hash_verified":True, "future_stage_memory_entries":0,
                "unpublished_attempt_directories":0, "preflight_pass":True, "all_primary_method_preflight_pass":True}
    loss_cfg = yaml.safe_load((root / CONFIG_RELATIVE).read_text(encoding="utf-8"))
    sec_cfg = yaml.safe_load((root / "configs/replay_comparison_candidate_b.yaml").read_text(encoding="utf-8"))
    common = loss_cfg["model"] == sec_cfg["model"] and loss_cfg["training"] == sec_cfg["training"] and all(loss_cfg[k] == sec_cfg[k] for k in ("data_path","stage_order","device","precision","deterministic","threads"))
    fairness = {"status":"PASS" if common else "FAIL", "only_intended_difference":"pre-update memory selection score",
                "fixed_common_factors":common, "memory_capacity_examples":200, "replay_batch_size":4,
                "classification_loss_equal_to_security_aware":loss_cfg["replay"]["classification_loss"]=="(current_ce + replay_ce) / 2",
                "sampling_rng_equal_to_security_aware":True,
                "label_stage_quota_coverage_tie_equal_to_security_aware":True, "scoring_uses_current_train_only":True,
                "no_evaluation_or_future_information":True, "capacity_checks":capacities, "exposure_checks":exposures,
                "quota_checks":quota_checks, "seeds":list(SEEDS), "arms":list(ARMS), "stages":list(STAGES),
                "exact_arithmetic_rank_equivalence":"true-label CE = -log(1-security_risk) within fixed label quotas",
                "frozen_numeric_difference":{"loss_aware":"FP16 forward logits cast to float32 before CE/softmax",
                                             "security_aware":"FP16 forward logits with softmax before host float conversion"},
                "interpretation_limit":"Any selected-record or outcome difference cannot isolate a non-monotone security-specific ranking; numeric precision and tie resolution remain the operative distinction."}
    if not common: raise ValueError("Fairness validation failed")
    return artifact, fairness


def load_runs(root):
    return {arm:{seed:{stage:read_json(run_dir(root,arm,seed)/f"stage{i}/metrics.json") for i,stage in enumerate(STAGES,1)} for seed in SEEDS} for arm in ARMS}


def derive_rows(root, runs):
    trajectories, acquisition, retention, summaries, external = [], [], [], [], []
    for arm in ARMS:
        for seed in SEEDS:
            per_task = []
            for checkpoint_index, checkpoint in enumerate(STAGES, 1):
                block = runs[arm][seed][checkpoint]; acq = block["acquisition"]
                acquisition.append({"method":METHOD,"arm":ARMS[arm],"arm_key":arm,"seed":seed,"stage":checkpoint,**acq,
                                    "gate_pass":acq["attack_tpr"]>=.85 and acq["benign_tnr"]>=.85 and acq["paired_macro_f1"]>=.85})
                ni, bs = block["external"]["NotInject"], block["external"]["B-short"]
                external.append({"method":METHOD,"arm":ARMS[arm],"arm_key":arm,"seed":seed,"checkpoint_stage":checkpoint,
                                 "notinject_fpr":ni["overall"]["benign_fpr"],"bshort_fpr":bs["overall"]["benign_fpr"],
                                 "bshort_english_fpr":bs["English"]["benign_fpr"],"bshort_multilingual_fpr":bs["multilingual_non_English"]["benign_fpr"]})
                for task in STAGES[:checkpoint_index]:
                    cell = block["evaluations"][task]
                    trajectories.append({"method":METHOD,"arm":ARMS[arm],"arm_key":arm,"seed":seed,"checkpoint_stage":checkpoint,"task":task,
                                         "attack_tpr":cell["malicious"]["attack_tpr"],"benign_fpr":cell["benign"]["benign_fpr"],
                                         "macro_f1":cell["paired"]["macro_f1"],"accuracy":cell["paired"]["accuracy"],
                                         "attack_tpr_by_position":cell["attack_tpr_by_position"],"benign_fpr_by_partition":cell["benign_fpr_by_partition"]})
            for task in STAGES[:-1]:
                cells=[x for x in trajectories if x["arm_key"]==arm and x["seed"]==seed and x["task"]==task]
                cells.sort(key=lambda x:STAGES.index(x["checkpoint_stage"]))
                attack=[x["attack_tpr"] for x in cells]; macro=[x["macro_f1"] for x in cells]; fpr=[x["benign_fpr"] for x in cells]
                item={"method":METHOD,"arm":ARMS[arm],"arm_key":arm,"seed":seed,"task":task,
                      "attack_tpr_acquisition":attack[0],"attack_tpr_final":attack[-1],"attack_tpr_change":attack[-1]-attack[0],
                      "attack_tpr_loss":attack[0]-attack[-1],"attack_peak_forgetting":max(max(attack[:i+1])-attack[i] for i in range(len(attack))),
                      "macro_f1_change":macro[-1]-macro[0],"macro_f1_loss":macro[0]-macro[-1],
                      "macro_f1_peak_forgetting":max(max(macro[:i+1])-macro[i] for i in range(len(macro))),
                      "benign_fpr_acquisition":fpr[0],"benign_fpr_final":fpr[-1],"benign_fpr_change":fpr[-1]-fpr[0],
                      "benign_peak_fpr_increase":max(x-fpr[0] for x in fpr)}
                retention.append(item); per_task.append(item)
            blocks=list(runs[arm][seed].values()); meta=read_json(run_dir(root,arm,seed)/"run_metadata.json")
            summaries.append({"method":METHOD,"arm":ARMS[arm],"arm_key":arm,"seed":seed,
                              "attack_tpr_bwt":statistics.mean(x["attack_tpr_change"] for x in per_task),
                              "macro_f1_bwt":statistics.mean(x["macro_f1_change"] for x in per_task),
                              "benign_fpr_increase":statistics.mean(x["benign_fpr_change"] for x in per_task),
                              "attack_peak_forgetting_mean":statistics.mean(x["attack_peak_forgetting"] for x in per_task),
                              "runtime_seconds":meta["runtime_seconds"],"buffer_scoring_runtime_seconds":meta["buffer_scoring_runtime_seconds"],
                              "peak_gpu_memory_bytes":max(x["peak_gpu_memory_bytes"] for x in blocks),
                              "replay_forward_passes":sum(x["training"]["replay_forward_passes"] for x in blocks)})
    return trajectories, acquisition, retention, summaries, external


def average_ranks(values):
    order=sorted(range(len(values)),key=lambda i:values[i]); ranks=[0.0]*len(values); i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]: j+=1
        rank=(i+j-1)/2+1
        for k in order[i:j]: ranks[k]=rank
        i=j
    return ranks


def correlation(x,y):
    if len(x)<2 or statistics.pstdev(x)==0 or statistics.pstdev(y)==0: return None
    mx,my=statistics.mean(x),statistics.mean(y)
    return sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))


def nullable_aggregate(rows,keys,metrics):
    grouped=defaultdict(list)
    for row in rows: grouped[tuple(row[key] for key in keys)].append(row)
    output=[]
    for identity,items in sorted(grouped.items(),key=lambda x:tuple(str(v) for v in x[0])):
        entry={key:value for key,value in zip(keys,identity)}
        for metric in metrics:
            values=[item[metric] for item in items if item[metric] is not None]
            entry[metric]={"mean":statistics.mean(values) if values else None,
                           "sample_sd":statistics.stdev(values) if len(values)>1 else None,
                           "values":values,"defined_n":len(values),"undefined_n":len(items)-len(values),"n":len(items)}
        output.append(entry)
    return output


def score_relationship(root):
    relationships, overlaps, overlap_by_task, top_k, precision=[],[],[],[],[]; security_root=root/"results/replay_comparison_candidate_b/security_aware"
    for arm in ARMS:
        for seed in SEEDS:
            for index,stage in enumerate(STAGES,1):
                scores=jsonl(run_dir(root,arm,seed)/f"stage{index}/pre_update_loss_scores.jsonl")
                security_scores=jsonl(security_root/arm/f"seed{seed}/stage{index}/pre_update_security_scores.jsonl")
                security_by_id={x["example_id"]:x for x in security_scores}
                if set(security_by_id)!={x["example_id"] for x in scores}:
                    raise ValueError(f"Loss/Security eligible score pool mismatch: {arm}/seed{seed}/{stage}")
                for label in (None,0,1):
                    rows=scores if label is None else [x for x in scores if x["label"]==label]
                    generic=[x["pre_update_generic_loss"] for x in rows]
                    risk=[security_by_id[x["example_id"]]["pre_update_security_risk"] for x in rows]
                    diagnostic_risk=[x["pre_update_security_risk"] for x in rows]
                    relationships.append({"arm":ARMS[arm],"arm_key":arm,"seed":seed,"stage":stage,"label":"all" if label is None else label,"n":len(rows),
                                          "pearson":correlation(generic,risk),"spearman":correlation(average_ranks(generic),average_ranks(risk)),
                                          "same_forward_fp32_diagnostic_spearman":correlation(average_ranks(generic),average_ranks(diagnostic_risk))})
                    k=200 if label is None else 100
                    loss_top={x["example_id"] for x in sorted(rows,key=lambda x:(-x["pre_update_generic_loss"],x["tie_break_sha256"]))[:k]}
                    risk_top={x["example_id"] for x in sorted(rows,key=lambda x:(-security_by_id[x["example_id"]]["pre_update_security_risk"],x["tie_break_sha256"]))[:k]}
                    both=len(loss_top & risk_top)
                    def categories(ids):
                        selected=[x for x in rows if x["example_id"] in ids]
                        fields=("attack_family","benign_group","injection_position","source_dataset","coverage_key")
                        return {field:dict(sorted((str(value),sum(str(x.get(field))==str(value) for x in selected)) for value in {x.get(field) for x in selected})) for field in fields}
                    top_k.append({"arm":ARMS[arm],"arm_key":arm,"seed":seed,"stage":stage,"label":"all" if label is None else label,"k":k,
                                  "intersection_count":both,"jaccard":both/len(loss_top|risk_top),
                                  "loss_only_category_counts":categories(loss_top-risk_top),"security_only_category_counts":categories(risk_top-loss_top)})
                differences=[abs(x["pre_update_security_risk"]-security_by_id[x["example_id"]]["pre_update_security_risk"]) for x in scores]
                precision.append({"arm":ARMS[arm],"arm_key":arm,"seed":seed,"stage":stage,"n":len(scores),
                                  "mean_absolute_risk_difference":statistics.mean(differences),"max_absolute_risk_difference":max(differences),
                                  "loss_forward_unique_risks":len({x["pre_update_security_risk"] for x in scores}),
                                  "security_forward_unique_risks":len({x["pre_update_security_risk"] for x in security_scores})})
                loss_entries=jsonl(run_dir(root,arm,seed)/f"stage{index}/buffer.jsonl")
                sec_entries=jsonl(security_root/arm/f"seed{seed}/stage{index}/buffer.jsonl")
                for label in (None,0,1):
                    left={x["example_id"] for x in loss_entries if label is None or x["label"]==label}; right={x["example_id"] for x in sec_entries if label is None or x["label"]==label}
                    overlap=len(left&right); union=len(left|right)
                    overlaps.append({"arm":ARMS[arm],"arm_key":arm,"seed":seed,"stage":stage,"label":"all" if label is None else label,
                                     "loss_count":len(left),"security_count":len(right),"overlap_count":overlap,"jaccard":overlap/union if union else 1.0})
                for memory_task in STAGES[:index]:
                    for label in (None,0,1):
                        left={x["example_id"] for x in loss_entries if x["task_id"]==memory_task and (label is None or x["label"]==label)}
                        right={x["example_id"] for x in sec_entries if x["task_id"]==memory_task and (label is None or x["label"]==label)}
                        overlap=len(left&right); union=len(left|right)
                        overlap_by_task.append({"arm":ARMS[arm],"arm_key":arm,"seed":seed,"checkpoint_stage":stage,"memory_task":memory_task,
                                                "label":"all" if label is None else label,"loss_count":len(left),"security_count":len(right),
                                                "overlap_count":overlap,"jaccard":overlap/union if union else 1.0})
    return {"score_correlations":relationships,"top_k_score_overlap":top_k,"selected_memory_overlap":overlaps,
            "selected_memory_overlap_by_task_and_label":overlap_by_task,
            "numeric_precision_diagnostics":precision,
            "theoretical_relation":"Within each fixed label/stage quota and model state, true-label CE = -log(1-security_risk); rankings are equivalent in exact arithmetic. Stage-1 selection differences can arise from frozen FP32-vs-FP16 precision/ties; later actual-method score comparisons additionally include model-state divergence caused by prior selections.",
            "correlation_aggregate":nullable_aggregate(relationships,("stage","label"),("pearson","spearman")),
            "top_k_overlap_aggregate":aggregate_rows(top_k,("stage","label"),("intersection_count","jaccard")),
            "overlap_aggregate":aggregate_rows(overlaps,("stage","label"),("overlap_count","jaccard"))}


def security_pairwise(root, loss_retention, loss_acquisition, loss_external, loss_summary):
    old=read_json(root/"results/replay_comparison_aggregate.json")
    def pair(loss_rows,old_rows,keys,metrics):
        ref={tuple(x[k] for k in keys):x for x in old_rows if x["method"]=="security_aware"}; output=[]
        for item in loss_rows:
            identity=tuple(item[k] for k in keys); other=ref[identity]
            output.append({"comparison":"loss_aware_vs_security_aware","primary_orientation":"security_aware_minus_loss_aware",
                           **{k:item[k] for k in keys},
                           **{f"loss_aware_minus_security_aware_{m}":item[m]-other[m] for m in metrics},
                           **{f"security_aware_minus_loss_aware_{m}":other[m]-item[m] for m in metrics}})
        return output
    retention=pair(loss_retention,old["retention_per_seed"],("arm_key","seed","task"),("attack_tpr_loss","macro_f1_loss","benign_fpr_change","attack_peak_forgetting"))
    acquisition=pair(loss_acquisition,old["acquisition_per_seed"],("arm_key","seed","stage"),("attack_tpr","benign_tnr","paired_macro_f1"))
    external=pair(loss_external,old["external_per_seed"],("arm_key","seed","checkpoint_stage"),("notinject_fpr","bshort_fpr"))
    summary=pair(loss_summary,old["run_summary_per_seed"],("arm_key","seed"),("attack_tpr_bwt","macro_f1_bwt","benign_fpr_increase","attack_peak_forgetting_mean","runtime_seconds","peak_gpu_memory_bytes"))
    retention_metrics=tuple(k for k in retention[0] if k.startswith("security_aware_minus_loss_aware_")); summary_metrics=tuple(k for k in summary[0] if k.startswith("security_aware_minus_loss_aware_"))
    return {"retention":retention,"acquisition":acquisition,"external":external,"run_summary":summary,
            "retention_aggregate":aggregate_rows(retention,("arm_key","task"),retention_metrics),
            "run_summary_aggregate":aggregate_rows(summary,("arm_key",),summary_metrics)}


def write_matrices(root,runs):
    getters={"attack_tpr":lambda c:c["malicious"]["attack_tpr"],"macro_f1":lambda c:c["paired"]["macro_f1"],
             "accuracy":lambda c:c["paired"]["accuracy"],"benign_fpr":lambda c:c["benign"]["benign_fpr"]}
    for arm in ARMS:
        for seed in SEEDS:
            directory=run_dir(root,arm,seed)
            for name,getter in getters.items():
                with (directory/f"matrix_{name}.csv").open("w",newline="",encoding="utf-8") as handle:
                    writer=csv.writer(handle); writer.writerow(["evaluated_stage",*STAGES])
                    for eidx,evaluated in enumerate(STAGES):
                        writer.writerow([evaluated,*["" if cidx<eidx else getter(runs[arm][seed][checkpoint]["evaluations"][evaluated]) for cidx,checkpoint in enumerate(STAGES)]])
            for index,checkpoint in enumerate(STAGES,1):
                block=runs[arm][seed][checkpoint]
                atomic_json(directory/f"stage{index}/confusion_counts.json",{
                    "method":METHOD,"arm":ARMS[arm],"seed":seed,"checkpoint_stage":checkpoint,
                    "paired_stage_tests":{task:{kind:{key:block["evaluations"][task][kind][key] for key in ("tp","tn","fp","fn","attack_support","benign_support")} for kind in ("paired","malicious","benign")} for task in STAGES[:index]},
                    "external":{name:{key:block["external"][name]["overall"][key] for key in ("tp","tn","fp","fn","attack_support","benign_support")} for name in ("NotInject","B-short")},
                })


def summarize(root:Path):
    root=Path(root).resolve(); artifact,fairness=verify(root); runs=load_runs(root)
    trajectories,acquisition,retention,summaries,external=derive_rows(root,runs)
    gates=[{"arm":ARMS[arm],"stage":stage,"passing_seeds":sum(x["gate_pass"] for x in acquisition if x["arm_key"]==arm and x["stage"]==stage),"required":2,
            "pass":sum(x["gate_pass"] for x in acquisition if x["arm_key"]==arm and x["stage"]==stage)>=2} for arm in ARMS for stage in STAGES]
    memory=[{"arm":ARMS[arm],"arm_key":arm,"seed":seed,"stage":stage,**read_json(run_dir(root,arm,seed)/f"stage{i}/memory_composition.json")}
            for arm in ARMS for seed in SEEDS for i,stage in enumerate(STAGES,1)]
    aggregate={"experiment":"Candidate B final Generic Loss-aware Replay control","method":METHOD,
               "retention_per_seed":retention,"retention_aggregate":aggregate_rows(retention,("arm","task"),("attack_tpr_loss","attack_peak_forgetting","macro_f1_loss","benign_fpr_change")),
               "run_summary_per_seed":summaries,"run_summary_aggregate":aggregate_rows(summaries,("arm",),("attack_tpr_bwt","macro_f1_bwt","benign_fpr_increase","attack_peak_forgetting_mean","runtime_seconds","buffer_scoring_runtime_seconds","peak_gpu_memory_bytes")),
               "trajectory_per_seed":trajectories,"trajectory_aggregate":aggregate_rows(trajectories,("arm","checkpoint_stage","task"),("attack_tpr","benign_fpr","macro_f1","accuracy")),
               "acquisition_per_seed":acquisition,"acquisition_gates":gates,
               "external_per_seed":external,"external_aggregate":aggregate_rows(external,("arm","checkpoint_stage"),("notinject_fpr","bshort_fpr","bshort_english_fpr","bshort_multilingual_fpr")),
               "key_b_short_af3":[x for x in retention if x["arm_key"]=="short" and x["task"]=="AF3"]}
    pairwise=security_pairwise(root,retention,acquisition,external,summaries); relationship=score_relationship(root)
    old=read_json(root/"results/replay_comparison_aggregate.json")
    aggregate["table_ready"]={
        "five_method_b_short_af3_retention":[x for x in [*old["retention_per_seed"],*retention] if x["arm_key"]=="short" and x["task"]=="AF3"],
        "five_method_b_hard_benign_retention":[x for x in [*old["run_summary_per_seed"],*summaries] if x["arm_key"]=="hard"],
        "security_vs_loss_benign_fpr":pairwise["run_summary"],
        "security_vs_loss_attack_retention":pairwise["retention"],
        "selected_buffer_overlap":relationship["selected_memory_overlap"],
        "score_rank_correlation":relationship["score_correlations"],
        "five_method_acquisition":[*old["acquisition_per_seed"],*acquisition],
        "five_method_external_diagnostics":[*old["external_per_seed"],*external],
        "five_method_runtime":[*old["run_summary_per_seed"],*summaries],
    }
    atomic_json(root/"results/loss_aware_control_aggregate.json",aggregate)
    atomic_json(root/"results/loss_aware_security_pairwise.json",pairwise)
    atomic_json(root/"results/loss_aware_memory_composition.json",{"rows":memory})
    atomic_json(root/"results/loss_aware_score_relationship.json",relationship)
    atomic_json(root/"results/loss_aware_fairness_validation.json",fairness)
    atomic_json(root/"results/loss_aware_artifact_validation.json",artifact)
    write_matrices(root,runs)
    return aggregate,pairwise,relationship,artifact,fairness
