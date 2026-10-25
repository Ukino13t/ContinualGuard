"""Build a source-only ACL Overleaf ZIP after full Revision 2 validation.

Only allowlisted aggregate artifacts enter the archive; raw dataset and model
checkpoints are never copied. This script does not invoke a TeX compiler.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(r"E:\ContinualGuard")
PAPER = ROOT / "paper/acl"
ANALYSIS = ROOT / "results/revision2/analysis"
OUT = ROOT / "paper/ContinualGuard_RouteA_Revision2_Overleaf.zip"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_complete() -> None:
    validation = json.loads((ANALYSIS / "validation.json").read_text(encoding="utf-8"))
    if (validation.get("status"), validation.get("full_checkpoint_hashes"), validation.get("runs_checked")) != ("PASS", True, 24):
        raise RuntimeError("All 18 matched and six new current-only runs require full-hash validation")
    if len(validation.get("matched_pair_controls", [])) != 9:
        raise RuntimeError("All nine A1/A2 initialization and order checks must pass")
    overlays = json.loads((ANALYSIS / "protocol_overlays.json").read_text(encoding="utf-8"))
    if overlays.get("status") != "PASS" or overlays.get("count") != 18:
        raise RuntimeError("Actual selector protocol overlays missing")
    for item in overlays["overlays"]:
        if sha(ROOT / item["relative_path"]) != item["sha256"]:
            raise RuntimeError(f"Protocol overlay hash changed: {item['relative_path']}")
    capacity = json.loads((ANALYSIS / "capacity_validation.json").read_text(encoding="utf-8"))
    if (capacity.get("status"), capacity.get("full_checkpoint_hashes"),
            capacity.get("runs_checked")) != ("PASS", True, 4):
        raise RuntimeError("Predeclared P1 capacity subset needs four full-hash-validated runs")
    capacity_controls = json.loads((ANALYSIS / "capacity_pair_controls.json").read_text(encoding="utf-8"))
    if (capacity_controls.get("status"), capacity_controls.get("runs_checked"),
            capacity_controls.get("identical_first_stage_weights"),
            capacity_controls.get("identical_current_order_all_four_stages")) != ("PASS", 5, True, True):
        raise RuntimeError("P1 capacity matched initialization/order controls missing")
    capacity_overlays = json.loads((ANALYSIS / "capacity_protocol_overlays.json").read_text(encoding="utf-8"))
    if capacity_overlays.get("status") != "PASS" or capacity_overlays.get("count") != 4:
        raise RuntimeError("Four P1 actual protocol overlays required")
    for item in capacity_overlays["overlays"]:
        if sha(ROOT / item["relative_path"]) != item["sha256"]:
            raise RuntimeError(f"P1 overlay hash changed: {item['relative_path']}")
    with (ANALYSIS / "availability.json").open(encoding="utf-8") as handle:
        availability = json.load(handle)
    if any(availability.get(method) != 9 for method in ("A1_matched", "A2_matched", "compute_current_only")):
        raise RuntimeError("Incomplete three-seed analysis")


def copy_curated() -> None:
    destination = PAPER / "data/revision2"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("per_seed.csv", "aggregate.csv", "stage_curves.csv", "stage_diagnostics.csv",
                 "buffer_composition.csv", "capacity_pair_controls.json",
                 "paired_per_seed.csv", "paired_aggregate.csv", "availability.json",
                 "validation.json", "capacity_validation.json",
                 "capacity_protocol_overlays.json"):
        shutil.copy2(ANALYSIS / name, destination / name)
    shutil.copy2(ROOT / "results/revision2/selector_audit/summary.json",
                 destination / "selector_summary.json")
    shutil.copy2(ROOT / "results/revision2/selector_audit/actual_candidate_pools.csv",
                 destination / "actual_candidate_pools.csv")
    identities = ROOT / "results/revision2/selector_audit/selected_identities.jsonl"
    with identities.open(encoding="utf-8") as handle:
        for line in handle:
            if set(json.loads(line)) - {"arm", "seed", "selected_ids", "stage", "variant"}:
                raise RuntimeError("Selected-identity audit contains unexpected fields")
    shutil.copy2(identities, destination / "selected_identities.jsonl")
    shutil.copy2(ROOT / "results/revision2/selector_audit/common_model_state_af3_short_seed42.json",
                 destination / "common_state_score_audit.json")
    shutil.copy2(ANALYSIS / "protocol_overlays.json", destination / "protocol_overlays.json")
    protocol_dir = destination / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    for variant in ("a1_matched", "a2_matched"):
        for arm in ("doc", "short", "hard"):
            for seed in (17, 42, 2026):
                source = (ROOT / "results/revision2/matched_ablation_amp256/M200" /
                          variant / arm / f"seed{seed}/actual_selector_protocol.json")
                shutil.copy2(source, protocol_dir / f"{variant}_{arm}_seed{seed}.json")
    for capacity in (50, 400):
        for variant in ("a1_matched", "reservoir_capacity"):
            root = (ROOT / "results/revision2/matched_ablation_amp256" if variant == "a1_matched"
                    else ROOT / "results/revision2/memory_sensitivity_amp256")
            source = root / f"M{capacity}" / variant / "hard/seed42/actual_selector_protocol.json"
            shutil.copy2(source, protocol_dir / f"capacity_{variant}_M{capacity}_hard_seed42.json")
    for name in ("b_short_af3_ablation.pdf", "b_hard_af2_ablation.pdf",
                 "b_short_af3_matched.pdf", "b_hard_af2_matched.pdf"):
        destination_name = (Path(name).stem + "_revision2.pdf"
                            if name.endswith("_ablation.pdf") else name)
        shutil.copy2(ROOT / "figures/revision2" / name,
                     PAPER / "figures" / destination_name)


def source_files() -> list[Path]:
    explicit = ("main.tex", "continualguard_acl.tex", "acl.sty", "acl_natbib.bst",
                "references.bib", "README.md", "build_tables.py")
    files = [PAPER / name for name in explicit]
    for folder, suffixes in (("tables", (".tex", ".json")),
                             ("figures", (".pdf",))):
        files += [p for p in (PAPER / folder).rglob("*") if p.is_file() and p.suffix in suffixes]
    original_data = [f"appendix_{letter}_{stem}.csv" for letter, stem in (
        ("A", "acquisition_per_seed"), ("B", "bwt_per_seed"),
        ("C", "b_short_af3_trajectories"), ("D", "b_hard_benign_trajectories"),
        ("E", "external_diagnostics"), ("F", "memory_composition"),
        ("G", "runtime_gpu"), ("H", "fairness_checks"),
        ("I", "artifact_validation"))]
    original_data += ["five_method_master_table.csv", "reproducibility_table.csv"]
    route_a = ("validation.json", "per_stage_curves.csv", "per_run.csv",
               "paired_per_seed.csv", "paired_aggregate.csv", "memory_composition.csv",
               "joint_reference.csv", "coverage_key_audit.json", "availability.json",
               "aggregate.csv", "a2_a3_memory_overlap.csv")
    revision2 = ("per_seed.csv", "aggregate.csv", "stage_curves.csv", "stage_diagnostics.csv",
                 "buffer_composition.csv", "capacity_pair_controls.json",
                 "paired_per_seed.csv", "paired_aggregate.csv", "availability.json",
                 "validation.json", "selector_summary.json", "common_state_score_audit.json",
                 "protocol_overlays.json", "capacity_validation.json",
                 "capacity_protocol_overlays.json", "actual_candidate_pools.csv",
                 "selected_identities.jsonl")
    files += [PAPER / "data" / name for name in original_data]
    files += [PAPER / "data/route_a" / name for name in route_a]
    files += [PAPER / "data/revision2" / name for name in revision2]
    files += [PAPER / "data/revision2/protocol" / f"{variant}_{arm}_seed{seed}.json"
              for variant in ("a1_matched", "a2_matched")
              for arm in ("doc", "short", "hard") for seed in (17, 42, 2026)]
    files += [PAPER / "data/revision2/protocol" /
              f"capacity_{variant}_M{capacity}_hard_seed42.json"
              for capacity in (50, 400)
              for variant in ("a1_matched", "reservoir_capacity")]
    for p in files:
        if not p.is_file() or not p.resolve().is_relative_to(PAPER.resolve()):
            raise RuntimeError(f"Missing or unsafe package member: {p}")
    return sorted(set(files))


def main() -> None:
    require_complete()
    copy_curated()
    # The paper's static checker must pass on the exact assembled tree.
    from revision2_paper_check import main as static_check
    static_check()
    files = source_files()
    forbidden = ("raw", "checkpoint", "predictions", "reviewer", "_attempt", ".safetensors")
    for file in files:
        relative = file.relative_to(PAPER).as_posix()
        if any(part in relative.lower() for part in forbidden):
            raise RuntimeError(f"Disallowed package path: {relative}")
    temporary = OUT.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                        compresslevel=6) as archive:
        for file in files:
            archive.write(file, file.relative_to(PAPER).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("Duplicate ZIP member")
    os.replace(temporary, OUT)
    manifest = {"package": str(OUT), "sha256": sha(OUT), "bytes": OUT.stat().st_size,
                "members": len(files), "compile_status": "Requires Overleaf pdfLaTeX/BibTeX verification",
                "source_allowlist": [p.relative_to(PAPER).as_posix() for p in files]}
    (ROOT / "reports/revision2/overleaf_package_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("package", "sha256", "bytes", "members")}))


if __name__ == "__main__":
    main()
