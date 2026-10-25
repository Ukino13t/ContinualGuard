"""Read-only inference and separate security/quality matrices."""
import csv
from pathlib import Path
import torch
from .artifacts import write_json, append_jsonl, read_jsonl
from .metrics import METRICS, classification_metrics, summarize
from .model import batch_inputs
from .reproducibility import isolated_evaluation


def predict(model, tokenizer, records, config):
    output = []
    batch_size = config["training"]["batch_size"]
    with isolated_evaluation(model):
        for offset in range(0, len(records), batch_size):
            batch = records[offset:offset+batch_size]
            inputs = batch_inputs(batch, tokenizer, config["model"]["max_length"], config["device"])
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=config["precision"] == "float16"):
                logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            labels = logits.argmax(dim=-1).cpu().tolist()
            for row, label, probability in zip(batch, labels, probabilities):
                output.append({"example_id": row["example_id"], "label": row["label"], "prediction": label, "attack_probability": probability, "task_id": row["task_id"], "attack_family": row["attack_family"], "injection_position": row["injection_position"], "domain": row["domain"]})
    return output


def evaluate(model, tokenizer, records, config, run_dir, trained_stage, eval_stage, phase):
    predictions = predict(model, tokenizer, records, config)
    checkpoint_id = f"{trained_stage}:{phase}"
    path = Path(run_dir) / "predictions" / f"{phase}_{trained_stage}_{eval_stage}.jsonl"
    for prediction in predictions:
        append_jsonl(path, {**prediction, "checkpoint_id": checkpoint_id})
    metrics = classification_metrics([p["label"] for p in predictions], [p["prediction"] for p in predictions])
    slices = {}
    for field in ("attack_family", "injection_position", "domain"):
        slices[field] = {}
        for value in sorted({p[field] for p in predictions}):
            subset = [p for p in predictions if p[field] == value]
            slices[field][value] = classification_metrics([p["label"] for p in subset], [p["prediction"] for p in subset])
    cell = {"trained_stage": trained_stage, "eval_stage": eval_stage, "phase": phase, **metrics, "slices": slices}
    append_jsonl(Path(run_dir) / "metrics.jsonl", cell)
    append_jsonl(
        Path(run_dir) / "confusion_counts.jsonl",
        {
            "trained_stage": trained_stage,
            "eval_stage": eval_stage,
            "phase": phase,
            **{key: metrics[key] for key in ("tp", "tn", "fp", "fn", "attack_support", "benign_support")},
        },
    )
    return cell


def write_summary(run_dir, cells, order, evidence_type="software smoke test; no research evidence"):
    run_dir = Path(run_dir)
    for metric in METRICS:
        with (run_dir / f"matrix_{metric}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["trained_stage", *order])
            for stage in order:
                writer.writerow([stage, *[next((c[metric] for c in cells if c["phase"] == "post" and c["trained_stage"] == stage and c["eval_stage"] == ev), "") for ev in order]])
    summary = {"stage_summaries": summarize(cells, order), "metric_convention": "Signed BWT/peak forgetting only for higher-is-better; FPR increase separate", "evidence_type": evidence_type}
    # Recompute pooled metrics from saved predictions, never average F1s as pooled F1.
    summary["pooled_by_stage"] = {}
    for stage in order:
        predictions = [p for file in sorted((run_dir / "predictions").glob(f"post_{stage}_*.jsonl")) for p in read_jsonl(file)]
        if predictions:
            summary["pooled_by_stage"][stage] = classification_metrics([p["label"] for p in predictions], [p["prediction"] for p in predictions])
    write_json(run_dir / "summary.json", summary)
    return summary


def write_joint_summary(run_dir, cells, order):
    """Write the offline-reference row without assigning continual-learning meaning."""
    run_dir = Path(run_dir)
    post = {cell["eval_stage"]: cell for cell in cells if cell["phase"] == "post" and cell["trained_stage"] == "JOINT"}
    for metric in METRICS:
        with (run_dir / f"matrix_{metric}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["trained_stage", *order])
            writer.writerow(["JOINT", *[post[stage][metric] for stage in order]])
    predictions = [p for file in sorted((run_dir / "predictions").glob("post_JOINT_*.jsonl")) for p in read_jsonl(file)]
    summary = {
        "per_stage": {stage: {metric: post[stage][metric] for metric in (*METRICS, "tp", "tn", "fp", "fn", "attack_support", "benign_support")} for stage in order},
        "pooled": classification_metrics([p["label"] for p in predictions], [p["prediction"] for p in predictions]),
        "metric_convention": "Offline joint reference only; retention, BWT, and FPR increase are not defined",
        "evidence_type": "Chen-derived deterministic reconstruction; pooled SQuAD/TriviaQA offline reference",
    }
    write_json(run_dir / "summary.json", summary)
    return summary
