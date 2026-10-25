"""Prepare a data-free, reviewable GitHub snapshot of ContinualGuard.

This is an allowlist export. It never copies local benchmark text, reviewer
submissions, model checkpoints, or historical result directories.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import re
import shutil

PROJECT = Path(__file__).resolve().parents[1]
DEST = PROJECT / "deliverables" / "github_release_20261025_v3"
if DEST.exists():
    raise SystemExit(f"Refusing to overwrite existing release: {DEST}")

include: set[Path] = set()


def add(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    relative = path.relative_to(PROJECT)
    if any(part in {"__pycache__", ".pytest_cache", ".venv", "raw", "processed"} for part in relative.parts):
        raise ValueError(f"Forbidden export path: {relative}")
    include.add(relative)


for name in ("pyproject.toml", "requirements-lock.txt"):
    add(PROJECT / name)
for folder, suffixes in (
    ("src", {".py"}),
    ("scripts", {".py"}),
    ("configs", {".yaml", ".yml"}),
    ("tests", {".py"}),
):
    for path in (PROJECT / folder).rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            if b"C:\\Users\\TLS" in path.read_bytes():
                continue  # Local attachment paths are unsuitable for publication.
            add(path)
add(PROJECT / "tests" / "fixtures" / "synthetic.jsonl")

public_docs = (
    "data_schema.md",
    "final_project_summary.md",
    "final_method_description.md",
    "final_descriptive_statistics.md",
    "final_claims_audit.md",
    "final_limitations.md",
    "final_experiment_timeline.md",
    "final_citation_audit.md",
    "final_consistency_audit.md",
    "reproducibility_checklist.md",
    "replay_comparison_protocol.md",
    "replay_comparison_results.md",
    "loss_aware_control_protocol.md",
    "loss_aware_control_results.md",
    "stronger_pilot_candidate_b_protocol.md",
    "stronger_pilot_no_replay_results.md",
    "attack_family_baseline_results.md",
    "h2_mini_no_replay_results.md",
)
for name in public_docs:
    add(PROJECT / "docs" / name)

for path in (PROJECT / "results" / "final_tables").glob("*"):
    if path.is_file() and path.suffix in {".csv", ".json"} and path.name != "reproducibility_table.csv":
        add(path)

paper = PROJECT / "paper" / "ieee_final_report"
for name in ("main.tex", "references.bib", "IEEEtran.cls", "IEEEtran.bst",
             "README.md", "ContinualGuard_IEEE_Final_Report_v2.pdf"):
    add(paper / name)
for folder, suffixes in (("figures", {".pdf"}), ("tables", {".tex"})):
    for path in (paper / folder).rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            add(path)

DEST.mkdir(parents=True)
for relative in sorted(include):
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT / relative, target)

text_suffixes = {".py", ".md", ".yaml", ".yml", ".json", ".csv", ".tex",
                 ".bib", ".cls", ".bst", ".txt", ".toml", ".jsonl"}
for relative in sorted(include):
    target = DEST / relative
    if target.suffix.lower() in text_suffixes:
        payload = target.read_bytes()
        target.write_bytes(payload.replace(b"\r\n", b"\n"))

readme = """# ContinualGuard

ContinualGuard is a research code and report release for continual indirect
prompt-injection detection. It studies both malicious-attack retention and
benign over-defense under sequential updates of a fixed DeBERTa-v3-base binary
detector. The final course report is in
[`paper/ieee_final_report/ContinualGuard_IEEE_Final_Report_v2.pdf`](paper/ieee_final_report/ContinualGuard_IEEE_Final_Report_v2.pdf).

The constructed four-stage Candidate B benchmark combines document, short-form,
and reviewed hard-benign contexts. Across three seeds, no-replay training showed
seed-variable forgetting and false-positive drift. Bounded replay methods reduced
the principal errors, while controlled memory-quota comparisons showed no stable
additional gain over simpler replay. These are descriptive results for one
constructed offline benchmark; no independent malicious stream or deployment
validation is claimed.

## Contents

- `src/continualguard/`: detector, training, evaluation, replay, and audit code.
- `scripts/` and `configs/`: research workflows and frozen experiment settings.
- `tests/`: code tests and a deliberately synthetic software fixture.
- `results/final_tables/`: aggregate and per-seed metrics, without predictions or
  source text.
- `docs/`: selected final methods, results, limitations, and provenance reports.
- `paper/ieee_final_report/`: PDF and self-contained IEEE LaTeX source.

## Data and reproducibility

Raw SQuAD, TriviaQA, Alpaca, NotInject, OR-Bench, OASST1, reviewer submissions,
materialized research JSONL, model weights, checkpoints, and raw predictions are
not distributed. Their source terms and privacy constraints are retained in the
project documentation. Full training requires obtaining the source datasets and
model weights under their own terms and recreating the locally audited benchmark.
The included aggregate tables and paper can be inspected without those inputs.
The synthetic fixture supports software tests only and is not scientific data.

The pinned environment is described in `requirements-lock.txt` (Windows x64,
Python 3.14.5, CUDA 12.6). Install the package locally with `pip install -e .`
and run `pytest tests/test_foundations.py` for a synthetic-data smoke check.
The full test suite and real-data experiment commands require the unavailable
local research inputs. The IEEE source uses `main.tex` with pdfLaTeX and BibTeX.

## Authors

Leshan Tang (experiments and writing) and Haoxiang Liao (design and analysis).

No software license is asserted by this release. Third-party data and model
licenses remain with their respective owners.
"""
(DEST / "README.md").write_text(readme, encoding="utf-8")

ignore = """# Local research inputs and outputs are never published.
data/
review/
history/
results/**
!results/final_tables/
!results/final_tables/*.csv
!results/final_tables/*.json
.venv/
__pycache__/
.pytest_cache/
.pytest_tmp*/
test-output/
*.egg-info/
*.safetensors
*.pt
*.pth
*.ckpt
*.bin
*.onnx
*.env
*.pem
*.key
*.sqlite
*.db
"""
(DEST / ".gitignore").write_text(ignore, encoding="utf-8")
(DEST / ".gitattributes").write_text(
    "* text=auto eol=lf\n*.pdf binary\n", encoding="utf-8"
)

secret = re.compile(
    rb"(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|"
    rb"AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)"
)
manifest = []
for path in sorted(DEST.rglob("*")):
    if not path.is_file():
        continue
    relative = path.relative_to(DEST).as_posix()
    if relative.startswith(("data/", "review/", "history/")):
        raise ValueError(f"Forbidden release item: {relative}")
    payload = path.read_bytes()
    if secret.search(payload):
        raise ValueError(f"Credential-like content in {relative}")
    if len(payload) > 100_000_000:
        raise ValueError(f"Unexpectedly large release file: {relative}")
    manifest.append({"path": relative, "size": len(payload), "sha256": sha256(payload).hexdigest()})

(DEST / "RELEASE_MANIFEST.json").write_text(
    json.dumps({"policy": "curated code/docs/aggregate-only GitHub release",
                "files": manifest}, indent=2) + "\n", encoding="utf-8"
)
print(f"Staged {len(manifest)} files ({sum(x['size'] for x in manifest):,} bytes) at {DEST}")
