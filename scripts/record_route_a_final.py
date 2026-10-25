"""Append final validation/package provenance without rewriting prior history."""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

root = Path(r"E:\ContinualGuard")
stamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
zip_path = root / "paper/ContinualGuard_RouteA_Overleaf.zip"
validation = json.loads((root / "results/extended/analysis/validation.json").read_text(encoding="utf-8"))
assert validation["status"] == "PASS" and validation["runs_checked"] == 39
assert sum(item["checkpoint_files_checked"] for item in validation["runs"]) == 903
assert all(item["full_checkpoint_hashes"] for item in validation["runs"])
digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()

history = root / "history/changes.jsonl"
old = history.read_text(encoding="utf-8")
if not any(json.loads(line).get("id") == "C043" for line in old.splitlines() if line.strip()):
    row = {"id": "C043", "date": stamp, "decision_id": "D056",
           "reason": "Finalize Route A P0 validation, coverage identification audit and Overleaf source package",
           "validation": {"new_runs_full_hash_pass": 39, "checkpoint_files_sha256_pass": 903,
                          "focused_tests_pass": 20, "overleaf_static_package_pass": True},
           "overleaf_zip": "paper/ContinualGuard_RouteA_Overleaf.zip",
           "overleaf_zip_sha256": digest,
           "limitations": ["local TeX missing lineno.sty", "coverage rule nonbinding",
                           "P1 training unexecuted"]}
    with history.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

doc = root / "docs/change_log.md"
text = doc.read_text(encoding="utf-8")
marker = "<!-- ROUTE_A_FINAL_VALIDATION_20261017 -->"
if marker not in text:
    text += ("\n" + marker + "\n## 2026-10-17 - Route A final validation and paper package\n\n"
             "All 39 new runs passed SHA-256 checks of 903 checkpoint files; 20 focused tests passed. "
             "The A2/A3 selected-ID overlap and nonbinding coverage-key audit were added to the report and ACL appendix. "
             "The aggregate-only Overleaf ZIP passed static citation/asset/environment and archive integrity checks. "
             "Local PDF compilation still requires the missing TinyTeX `lineno.sty`; no TeX package was installed.\n")
    doc.write_text(text, encoding="utf-8")

print(json.dumps({"status": "recorded", "zip_sha256": digest}))
