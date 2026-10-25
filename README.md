# ContinualGuard

ContinualGuard is a research code and report release for continual indirect
prompt-injection detection. It studies both malicious-attack retention and
benign over-defense under sequential updates of a fixed DeBERTa-v3-base binary
detector. The final course report is in
[`paper/ieee_final_report/ContinualGuard_IEEE_Final_Report_v3.pdf`](paper/ieee_final_report/ContinualGuard_IEEE_Final_Report_v3.pdf).

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
