"""Explicit signed CL conventions and security metrics."""
from statistics import mean

HIGHER = ("accuracy", "macro_f1", "attack_recall")
METRICS = (*HIGHER, "benign_fpr")


def classification_metrics(labels, predictions):
    if len(labels) != len(predictions) or any(type(x) is not int or x not in (0, 1) for x in [*labels, *predictions]):
        raise ValueError("Expected equally sized binary label/prediction lists")
    tp = sum(y == p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    attack, benign = tp + fn, tn + fp
    def div(a, b):
        return a / b if b else None
    f1a = div(2 * tp, 2 * tp + fp + fn)
    f1b = div(2 * tn, 2 * tn + fp + fn)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "attack_support": attack, "benign_support": benign, "accuracy": div(tp + tn, len(labels)), "macro_f1": ((f1a or 0) + (f1b or 0)) / 2 if attack and benign else None, "attack_recall": div(tp, attack), "benign_fpr": div(fp, benign)}


def summarize(cells, order):
    lookup = {(x["trained_stage"], x["eval_stage"]): x for x in cells if x["phase"] == "post"}
    output = []
    for i, stage in enumerate(order):
        if (stage, stage) not in lookup:
            continue
        entry = {"stage": stage}
        for metric in METRICS:
            current = lookup[(stage, stage)][metric]
            previous = [lookup[(stage, j)][metric] for j in order[:i]]
            entry[metric] = {"acquisition": current, "previous_stage_mean": mean(previous) if previous else None, "seen_mean": mean(previous + [current])}
            if metric in HIGHER:
                bwt = [lookup[(stage, j)][metric] - lookup[(j, j)][metric] for j in order[:i]]
                # Include the current endpoint so an improvement cannot be
                # reported as negative forgetting.
                forgetting = [
                    max(lookup[(k, j)][metric] for k in order[jdx:i + 1])
                    - lookup[(stage, j)][metric]
                    for jdx, j in enumerate(order[:i])
                ]
                entry[metric].update(bwt=mean(bwt) if bwt else None, peak_forgetting=mean(forgetting) if forgetting else None)
            else:
                changes = [lookup[(stage, j)][metric] - lookup[(j, j)][metric] for j in order[:i]]
                entry[metric]["increase_from_acquisition"] = mean(changes) if changes else None
            pre = next((c for c in cells if c["phase"] == "pre" and c["trained_stage"] == stage and c["eval_stage"] == stage), None)
            if pre:
                entry[metric]["adaptation_change"] = current - pre[metric]
        output.append(entry)
    return output
