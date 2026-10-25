"""Strict configuration with no implicit research defaults."""
from pathlib import Path
import math
import re
import yaml


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {"run_type", "data_path", "stage_order", "seed", "model", "training", "device", "precision", "deterministic", "threads"}
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError("Missing or unknown top-level config keys")
    if set(config["model"]) != {"kind", "checkpoint", "revision", "local_artifact", "max_length"}:
        raise ValueError("Missing or unknown model config keys")
    training = config["training"]
    if set(training) != {"epochs", "batch_size", "gradient_accumulation_steps", "gradient_checkpointing", "learning_rate", "betas", "epsilon", "weight_decay", "max_grad_norm", "optimizer_transition", "scheduler", "pre_adaptation"}:
        raise ValueError("Missing or unknown training config keys")
    order = config["stage_order"]
    if not order or len(order) != len(set(order)) or not all(isinstance(x, str) for x in order):
        raise ValueError("Stage order must contain unique string IDs")
    if not all(re.fullmatch(r"[A-Za-z0-9-]+", x) for x in order):
        raise ValueError("Stage IDs must be filesystem-safe alphanumeric/hyphen strings")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if type(config["threads"]) is not int or config["threads"] < 1:
        raise ValueError("threads must be a positive integer")
    if type(config["deterministic"]) is not bool or type(training["pre_adaptation"]) is not bool or type(training["gradient_checkpointing"]) is not bool:
        raise ValueError("deterministic, pre_adaptation, and gradient_checkpointing must be booleans")
    if config["run_type"] not in {"software smoke test", "pilot experiment", "reportable research result"}:
        raise ValueError("Invalid run_type")
    if config["model"]["kind"] not in {"tiny_deberta", "pretrained_deberta"}:
        raise ValueError("Invalid model kind")
    for key in ("learning_rate", "epsilon", "max_grad_norm"):
        value = training[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {key}")
    if not isinstance(training["weight_decay"], (int, float)) or not math.isfinite(training["weight_decay"]) or training["weight_decay"] < 0:
        raise ValueError("Invalid weight_decay")
    if not isinstance(training["betas"], list) or len(training["betas"]) != 2 or not all(isinstance(v, (int, float)) and 0 <= v < 1 for v in training["betas"]):
        raise ValueError("Invalid AdamW betas")
    if training["optimizer_transition"] not in {"reset", "carry"}:
        raise ValueError("optimizer_transition must be reset or carry")
    if training["scheduler"] != "constant" or config["precision"] not in {"float32", "float16"}:
        raise ValueError("Only constant scheduling and explicit float32/float16 precision are supported")
    if config["precision"] == "float16" and config["device"] != "cuda":
        raise ValueError("float16 training requires CUDA")
    for key in ("epochs", "batch_size", "gradient_accumulation_steps"):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f"Invalid {key}")
    if config["model"]["max_length"] is not None and (type(config["model"]["max_length"]) is not int or config["model"]["max_length"] < 2):
        raise ValueError("Invalid max_length")
    if Path(config["data_path"]).is_absolute():
        raise ValueError("data_path must be relative to project root")
    local_artifact = config["model"]["local_artifact"]
    if local_artifact is not None and (not isinstance(local_artifact, str) or Path(local_artifact).is_absolute()):
        raise ValueError("local_artifact must be null or a relative project path")
    return config
