"""One-step 1280-token CUDA preflight for the approved real baseline."""
import ctypes
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


class MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
        ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong), ("available_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    ]


def available_memory():
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.available_physical


def longest_example():
    best = None
    with (ROOT / "data/audits/real_data/token_lengths.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if best is None or row["token_length"] > best["token_length"]:
                best = row
    with (ROOT / "data/processed/chen_domain_shift.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["example_id"] == best["example_id"]:
                return row, best
    raise RuntimeError("Audited longest example is absent from processed data")


def main():
    config = load_config(ROOT / "configs/real_pilot.yaml")
    output = ROOT / "results" / ("real-preflight-" + time.strftime("%Y%m%dT%H%M%S", time.gmtime()))
    output.mkdir(parents=True, exist_ok=False)
    result = {"started_at": now(), "status": "running", "device": config["device"], "precision": config["precision"], "max_length": config["model"]["max_length"], "available_system_memory_before_bytes": available_memory()}
    started = time.perf_counter()
    try:
        seed_everything(config["seed"], config["deterministic"], config["threads"])
        tokenizer = make_tokenizer(config["model"], ROOT)
        row, audit = longest_example()
        ids = tokenizer.encode(row["text"], add_special_tokens=True, truncation=False)
        if len(ids) != audit["token_length"] or len(ids) > config["model"]["max_length"]:
            raise RuntimeError("Tokenization differs from the approved length audit")
        pad = config["model"]["max_length"] - len(ids)
        inputs = {
            "input_ids": torch.tensor([ids + [tokenizer.pad_token_id] * pad], dtype=torch.long, device="cuda"),
            "attention_mask": torch.tensor([[1] * len(ids) + [0] * pad], dtype=torch.long, device="cuda"),
        }
        labels = torch.tensor([row["label"]], dtype=torch.long, device="cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = make_model(config["model"], ROOT)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.to("cuda").train()
        optimizer, _ = make_optimizer(model, config)
        scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = model(**inputs, labels=labels).loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            raise RuntimeError(f"Nonfinite preflight values: loss={loss.item()}, gradient_norm={grad_norm.item()}")
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        result.update(status="passed", audited_example_id=row["example_id"], audited_token_length=len(ids), padded_compute_length=config["model"]["max_length"], loss=float(loss.item()), gradient_norm=float(grad_norm.item()), step_seconds=time.perf_counter() - step_started, peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(), peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(), gpu=torch.cuda.get_device_name(0))
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result.update(finished_at=now(), runtime_seconds=time.perf_counter() - started, available_system_memory_after_bytes=available_memory())
        write_json(output / "preflight.json", result)
        print(output)


if __name__ == "__main__":
    main()
