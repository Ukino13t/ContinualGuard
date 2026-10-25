"""Fixed DeBERTa-family classifier; deterministic byte tokenizer is smoke-only."""
import json
import re
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DebertaV2Config, DebertaV2ForSequenceClassification


class ByteTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True, truncation=False):
        if truncation:
            raise ValueError("Truncation is forbidden")
        tokens = [byte + 3 for byte in text.encode("utf-8")]
        return [1, *tokens, 2] if add_special_tokens else tokens

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "smoke_tokenizer.json").write_text(json.dumps({"kind": "utf8-byte", "revision": "local-v1", "vocab_size": 259}), encoding="utf-8")


def _model_source(config, root=None):
    local = config.get("local_artifact")
    if local:
        path = (Path(root or Path.cwd()) / local).resolve()
        if not path.is_dir():
            raise ValueError(f"Pinned local model artifact is missing: {path}")
        return str(path), {"local_files_only": True}
    return config["checkpoint"], {"revision": config["revision"]}


def make_tokenizer(config, root=None):
    if config["kind"] == "tiny_deberta":
        return ByteTokenizer()
    if config["kind"] != "pretrained_deberta" or not re.fullmatch(r"[0-9a-f]{40}", config["revision"] or ""):
        raise ValueError("Pretrained tokenizer needs explicit pinned revision")
    source, location = _model_source(config, root)
    return AutoTokenizer.from_pretrained(source, trust_remote_code=False, use_fast=False, **location)


def make_model(config, root=None):
    if config["kind"] == "tiny_deberta":
        architecture = DebertaV2Config(vocab_size=259, hidden_size=24, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, max_position_embeddings=512, relative_attention=True, position_biased_input=False, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, num_labels=2)
        model = DebertaV2ForSequenceClassification(architecture)
    elif config["kind"] == "pretrained_deberta" and config["revision"]:
        source, location = _model_source(config, root)
        model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=2, trust_remote_code=False, **location)
        if model.config.model_type not in {"deberta", "deberta-v2"}:
            raise ValueError("Only DeBERTa-family classifiers are supported")
    else:
        raise ValueError("Invalid model kind or missing pinned revision")
    model.config.id2label = {0: "benign", 1: "indirect_prompt_injection"}
    model.config.label2id = {value: key for key, value in model.config.id2label.items()}
    model.config.problem_type = "single_label_classification"
    return model


def batch_inputs(records, tokenizer, max_length, device):
    # Only text is encoded: no metadata enters tokens or feature tensors.
    encoded = [tokenizer.encode(r["text"], add_special_tokens=True, truncation=False) for r in records]
    if not encoded or any(len(ids) > max_length for ids in encoded):
        raise ValueError("Empty batch or overlength input; audit/exclude explicitly, never truncate")
    width = max(map(len, encoded))
    return {
        "input_ids": torch.tensor([ids + [tokenizer.pad_token_id] * (width-len(ids)) for ids in encoded], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([[1]*len(ids) + [0]*(width-len(ids)) for ids in encoded], dtype=torch.long, device=device),
    }
