"""Portable artifact IO and paired Markdown/JSONL history."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def history(root, category, record):
    names = {"experiments": "experiment_history", "changes": "change_log", "decisions": "decision_log"}
    append_jsonl(Path(root) / "history" / f"{category}.jsonl", record)
    path = Path(root) / "docs" / f"{names[category]}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n## " + str(record.get("run_id", record.get("id", record.get("date", "Entry")))))
        handle.write("\n\n" + "\n".join(f"- **{key}:** {json.dumps(value, ensure_ascii=False)}" for key, value in record.items()) + "\n")
