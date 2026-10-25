"""Seed state, environment provenance, and stage-boundary checkpoint IO."""
import importlib.metadata
import hashlib
import os
import platform
import random
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch


def seed_everything(seed, deterministic=True, threads=1):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


@contextmanager
def isolated_evaluation(model):
    state, mode = rng_state(), model.training
    try:
        model.eval()
        with torch.inference_mode():
            yield
    finally:
        model.train(mode)
        restore_rng(state)


def environment(root):
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    commit = git("rev-parse", "HEAD")
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}
    compatibility_warnings = []
    if sys.version_info >= (3, 14):
        compatibility_warnings.append("PyTorch reports that torch.jit.script is unsupported on Python 3.14+; this project uses eager execution, but a research environment should validate its chosen Python/PyTorch pair.")
    return {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(), "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}, "source_file_hashes": sources, "compatibility_warnings": compatibility_warnings, "git_commit": commit, "git_dirty": bool(git("status", "--porcelain")) if commit else None, "git_status_reason": None if commit else "No existing Git repository/commit", "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def save_checkpoint(path, model, tokenizer, optimizer, scheduler, payload):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save({**payload, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": rng_state()}, path / "state.pt")


def load_checkpoint(path):
    from transformers import AutoModelForSequenceClassification
    path = Path(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
    # Only load trusted local checkpoints produced by this experiment.
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    return model, state
