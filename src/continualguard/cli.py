"""Command line: validation, length audit, smoke execution, saved summaries."""
import argparse
import json
import uuid
from pathlib import Path
import yaml
from .artifacts import digest, history, now, read_jsonl, write_json
from .config import load_config
from .data import audit_records, length_audit


def audit_run(config, root, config_file):
    from .model import make_tokenizer
    from .reproducibility import environment
    run_id = "audit-" + uuid.uuid4().hex[:16]
    output = root / "results" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    record = {"run_id": run_id, "date": now(), "operation": "data audit; no training", "config_file": config_file, "config_hash": digest(json.dumps(config, sort_keys=True).encode()), "model_checkpoint": config["model"]["checkpoint"], "model_revision": config["model"]["revision"], "seed": config["seed"], "stage_order": config["stage_order"], "hardware_device": "CPU tokenization", "precision": "not applicable; tokenization only", "result_artifact_path": f"results/{run_id}", "data_manifest_hash": None, "exclusions": [], "warnings": [], "run_type": "software smoke test" if config["run_type"] == "software smoke test" else "pilot experiment", "status": "running", "summary_metrics": None}
    try:
        env = environment(root)
        record.update(environment=env, git_commit=env["git_commit"])
        record["warnings"].extend(env["compatibility_warnings"])
        path = (root / config["data_path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Dataset must remain within project root")
        rows = read_jsonl(path)
        record["data_manifest_hash"] = digest(path.read_bytes())
        report = audit_records(rows, config["stage_order"])
        write_json(output / "data_audit.json", report)
        write_json(output / "data_manifest.json", {"source_path": config["data_path"], "sha256": record["data_manifest_hash"], "example_ids": [r["example_id"] for r in rows]})
        coverage = length_audit(rows, make_tokenizer(config["model"]))
        write_json(output / "length_audit.json", coverage)
        for limit in (512, 768, 1024, 1280):
            write_json(output / f"exclusions_{limit}.json", [x for x in coverage["exclusions"] if x["max_length"] == limit])
        record["exclusions"] = coverage["exclusions"]
        record["warnings"] = report["warnings"]
        record["summary_metrics"] = {"coverage": coverage["coverage"], "records": len(rows), "errors": report["errors"]}
        write_json(output / "summary.json", record["summary_metrics"])
        if report["errors"]:
            raise ValueError("Data audit failed; see data_audit.json")
        if coverage["exclusions"] or report["missing_original_content_ids"]:
            record["status"] = "review_required"
            record["warnings"].append("STOP: coverage loss or unverified original provenance requires review before real training")
        else:
            record["status"] = "complete"
        return output
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["finished_at"] = now()
        write_json(output / "metadata.json", record)
        write_json(root / "history" / "runs" / f"{run_id}.json", record)
        history(root, "experiments", record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["validate", "audit", "train", "summary"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "summary":
        if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
            parser.error("summary needs a run ID")
        print((root / "results" / args.run_id / "summary.json").read_text(encoding="utf-8"))
        return
    config = load_config(root / args.config)
    if args.command == "train":
        from .train import run
        print(run(config, root, args.config, resume=args.resume))
    elif args.command == "audit":
        output = audit_run(config, root, args.config)
        print(output)
        print((output / "metadata.json").read_text(encoding="utf-8"))
    else:
        path = (root / config["data_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Dataset must remain within project root")
        report = audit_records(read_jsonl(path), config["stage_order"])
        print(json.dumps(report, indent=2))
        if report["errors"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
