"""Render the human-readable storage audit from history/storage/latest.json."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def size(value):
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    number = float(value)
    for unit in units:
        if number < 1024 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024


def esc(value):
    return value.replace("|", "\\|")


def render():
    data = json.loads((ROOT / "history" / "storage" / "latest.json").read_text(encoding="utf-8"))
    lines = [
        "# Storage audit",
        "",
        f"Generated: {data['generated_at_utc']}. Mode: **read-only; no file was deleted, moved, archived, compressed, or replaced.**",
        "",
        f"The project occupies **{size(data['total_bytes'])}** ({data['total_bytes']:,} bytes) across {data['total_files']:,} files. "
        "Experiment results dominate the footprint. Optimizer/scheduler/RNG states alone are larger than all trained weights and the virtual environment combined.",
        "",
        "## Category breakdown",
        "",
        "| Category | Files | Size | Share |",
        "|---|---:|---:|---:|",
    ]
    for name, item in sorted(data["category_breakdown"].items(), key=lambda x: x[1]["bytes"], reverse=True):
        lines.append(f"| `{name}` | {item['files']:,} | {size(item['bytes'])} | {item['bytes']/data['total_bytes']:.2%} |")
    lines += ["", "The categories cover papers, the virtual environment, the canonical Hugging Face model/tokenizer snapshot, raw and processed data, manifests and audits, trained weights, optimizer state, raw predictions, confusion counts, metric matrices, logs, documentation, history, fixtures, and transient caches. No separate project temporary-file tree was present.",
              "", "## Directory sizes", "", "| Top-level path | Files | Size |", "|---|---:|---:|"]
    for name, item in data["top_level_directory_sizes"].items():
        lines.append(f"| `{esc(name)}` | {item['files']:,} | {size(item['bytes'])} |")
    lines += ["", "### Largest result runs", "", "| Run directory | Files | Size |", "|---|---:|---:|"]
    for name, item in list(data["result_run_directory_sizes"].items())[:20]:
        lines.append(f"| `results/{esc(name)}` | {item['files']:,} | {size(item['bytes'])} |")
    lines += ["", "## Largest 30 files", "", "| Rank | Path | Category | Size |", "|---:|---|---|---:|"]
    for rank, item in enumerate(data["largest_files"], 1):
        lines.append(f"| {rank} | `{esc(item['path'])}` | `{item['category']}` | {size(item['bytes'])} |")
    lines += ["", "## Large exact duplicates", "", f"Files at least {size(data['large_duplicate_threshold_bytes'])} were SHA-256 hashed. The audit found {len(data['large_duplicate_hash_groups'])} duplicate groups.",
              "", "| SHA-256 prefix | Copies | Size each | Reclaimable beyond one | Representative paths |", "|---|---:|---:|---:|---|"]
    for group in data["large_duplicate_hash_groups"]:
        shown = "<br>".join(f"`{esc(p)}`" for p in group["paths"][:4])
        if len(group["paths"]) > 4:
            shown += f"<br>… {len(group['paths'])-4} more in the machine-readable audit"
        lines.append(f"| `{group['sha256'][:16]}…` | {group['copies']} | {size(group['bytes_each'])} | {size(group['reclaimable_bytes_beyond_one'])} | {shown} |")
    lines += ["", "Five duplicate groups are exact trained-weight endpoints: joint seed-42 E1, reverse-order seed-42 first-stage S2, and the three sequential S2 E1 endpoints reproduce previously saved weights byte-for-byte. A 38.34 MiB per-run data manifest is repeated across 14 completed real/control runs. The pinned `spm.model` is copied into 32 checkpoint directories in addition to the canonical Hugging Face snapshot.",
              "", "## Reclaim estimates", "", f"The conservative proposal totals **{size(data['estimated_reclaimable_bytes']['conservative'])}**. It consists of completed-run optimizer/scheduler/RNG states plus transient caches. Removing optimizer state prevents exact resume from those endpoints, so each file still requires explicit approval and validation against the retention policy.",
              "", f"The aggressive proposal totals **{size(data['estimated_reclaimable_bytes']['aggressive'])}**. It includes the conservative set, exact duplicate checkpoints/manifests, checkpoint tokenizer copies after loader changes, smoke weights, and non-selected S2-only E1/E3 weights. This is an estimate of distinct candidate paths, so overlap is not double-counted.",
              "", "The full per-file candidate ledger records path, size, SHA-256, proposed action, reason, risk, recoverability, tier, and `proposed_only_no_action` status in `history/storage/latest.json`. `actions_taken` is empty.",
              "", "## Model and data layout findings", "", "The base model exists once at the pinned Hugging Face path and is identified by model ID, revision, size, and SHA-256 in `data/manifests/real_data/model_weights.json`. Each 703.54 MiB `model.safetensors` under a run is a trained run-specific checkpoint, not an unmodified base-model cache copy. Checkpoint-local tokenizer/config files account for a much smaller but exact duplicated footprint.",
              "", "Raw Chen, SQuAD-derived, TriviaQA-derived, and Alpaca-derived content remains active and local. Canonical processed data are referenced by content hash and construction manifests. Existing result directories duplicate a large ID/hash-only `data_manifest.json`; future runs should use a content-addressed canonical manifest plus a compact per-run reference after readers and verification tools are updated.",
              "", "Compression was not applied. Gzipping JSONL/CSV remains optional and must wait until every reader, verifier, and analysis script supports and tests compressed input.", ""]
    (ROOT / "docs" / "storage_audit.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    render()
