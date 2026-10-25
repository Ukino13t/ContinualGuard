"""One optimizer step at the largest approved real training-batch shape."""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from continualguard.artifacts import now, write_json
from continualguard.config import load_config
from continualguard.model import make_model, make_tokenizer
from continualguard.reproducibility import seed_everything
from continualguard.train import make_optimizer


def largest_training_rows(count=4):
    candidates = []
    with (ROOT / "data/audits/real_data/token_lengths.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == "train":
                candidates.append((row["token_length"], row["example_id"]))
    selected = sorted(candidates, reverse=True)[:count]
    wanted = {example_id for _, example_id in selected}
    rows = {}
    with (ROOT / "data/processed/chen_domain_shift.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["example_id"] in wanted:
                rows[row["example_id"]] = row
    return [rows[example_id] for _, example_id in selected], selected


def main():
    config = load_config(ROOT / "configs/real_pilot.yaml")
    output = ROOT / "results" / ("real-training-batch-preflight-" + time.strftime("%Y%m%dT%H%M%S", time.gmtime()))
    output.mkdir(parents=True, exist_ok=False)
    result = {"started_at": now(), "status": "running", "batch_size": 4, "precision": "float16"}
    started = time.perf_counter()
    try:
        seed_everything(config["seed"], config["deterministic"], config["threads"])
        tokenizer = make_tokenizer(config["model"], ROOT)
        rows, audited = largest_training_rows()
        encoded = [tokenizer.encode(row["text"], add_special_tokens=True, truncation=False) for row in rows]
        if [len(ids) for ids in encoded] != [length for length, _ in audited]:
            raise RuntimeError("Tokenization differs from approved audit")
        width = max(map(len, encoded))
        inputs = {"input_ids": torch.tensor([ids + [tokenizer.pad_token_id] * (width - len(ids)) for ids in encoded], dtype=torch.long, device="cuda"), "attention_mask": torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids in encoded], dtype=torch.long, device="cuda")}
        labels = torch.tensor([row["label"] for row in rows], dtype=torch.long, device="cuda")
        model = make_model(config["model"], ROOT)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.to("cuda").train()
        optimizer, _ = make_optimizer(model, config)
        scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = model(**inputs, labels=labels).loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            raise RuntimeError(f"Nonfinite values: loss={loss.item()}, gradient_norm={grad_norm.item()}")
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        result.update(status="passed", token_lengths=[len(ids) for ids in encoded], padded_width=width, loss=float(loss.item()), gradient_norm=float(grad_norm.item()), step_seconds=time.perf_counter() - step_started, peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(), peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(), gpu=torch.cuda.get_device_name(0))
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result.update(finished_at=now(), runtime_seconds=time.perf_counter() - started)
        write_json(output / "preflight.json", result)
        print(output)


if __name__ == "__main__":
    main()
