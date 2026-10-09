"""Run each checkpoint in a fresh process, sequentially, for comparable inference cost."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from prepare import sha256, verify_manifest


def run(
    base: Path,
    matrices: list[Path],
    domains: Path,
    output: Path,
    repeats: int = 5,
    pilot: Path | None = None,
    selection: Path | None = None,
) -> dict:
    datasets = sorted(p for p in domains.iterdir() if p.is_dir())
    if pilot:
        datasets.append(pilot)
    if not datasets:
        raise ValueError("evaluation datasets missing")
    manifests = [verify_manifest(path) for path in datasets]
    if any(m.get("split") == "test" for m in manifests):
        if selection is None:
            raise ValueError("reserved test requires a frozen development selection")
        decision = json.loads(selection.read_text())
        if not decision.get("selection_frozen_before_reserved_test"):
            raise ValueError("development selection is not frozen")
        if sha256(Path(decision["evaluation_suite"])) != decision["evaluation_suite_sha256"]:
            raise ValueError("frozen development evidence differs")
    expected_datasets = {p.name: sha256(p / "manifest.json") for p in datasets}
    jobs = [("unchanged", base)]
    for matrix in matrices:
        specification = json.loads(matrix.read_text())
        if specification.get("status", "").startswith("superseded"):
            raise ValueError("cannot evaluate a superseded matrix")
        for row in specification["runs"]:
            path = Path(row["directory"])
            if sha256(path / "training-report.json") != row["training_report_sha256"]:
                raise ValueError("matrix training report checksum mismatch")
            training = json.loads((path / "training-report.json").read_text())
            for filename, digest in training["checkpoint_files"].items():
                if Path(filename).name != filename or sha256(path / filename) != digest:
                    raise ValueError("trained checkpoint checksum mismatch")
            jobs.append((f"{specification['budget']}/{row['arm']}-seed{row['seed']}", path))
    if len({name for name, _ in jobs}) != len(jobs):
        raise ValueError("duplicate model jobs")
    completed = []
    implementations = {
        name: sha256(Path(__file__).with_name(name))
        for name in ("evaluate_encoder.py", "encoder.py", "compare.py", "metrics.py")
    }
    for name, model in jobs:
        destination = output / name
        report_path = destination / "report.json"
        if not report_path.exists():
            if destination.exists():
                raise ValueError(f"incomplete evaluation exists: {destination}")
            command = [
                sys.executable,
                str(Path(__file__).with_name("evaluate_encoder.py")),
                "--model",
                str(model),
                "--domains",
                str(domains),
                "--output",
                str(destination),
                "--benchmark-repeats",
                str(repeats),
            ]
            if pilot:
                command.extend(["--dataset", str(pilot)])
            print(f"Evaluating {name}", flush=True)
            subprocess.run(command, check=True)
        report = json.loads(report_path.read_text())
        if report["implementation_sha256"] != implementations:
            raise ValueError("evaluation implementation differs; use a new run directory")
        if report["model_files"] != {
            p.name: sha256(p) for p in sorted(model.iterdir()) if p.is_file()
        }:
            raise ValueError("evaluation uses a different model/checkpoint")
        observed = {
            key: value["dataset_manifest_sha256"] for key, value in report["results"].items()
        }
        if observed != expected_datasets:
            raise ValueError("evaluation uses different frozen datasets")
        if (report["maximum_tokens"], report["batch_size"], report["cpu_threads"]) != (512, 16, 4):
            raise ValueError("evaluation settings differ")
        runtime = report["runtime"]
        if not runtime or len(runtime["repeat_seconds"]) != repeats:
            raise ValueError("evaluation timing repeats differ")
        for dataset, details in report["results"].items():
            for filename, key in (
                ("rankings.jsonl", "ranking_sha256"),
                ("scores.jsonl", "scores_sha256"),
            ):
                if sha256(destination / dataset / filename) != details[key]:
                    raise ValueError("evaluation ranking/score checksum mismatch")
        for filename, key in (
            ("query-workload.jsonl", "query_workload_sha256"),
            ("document-workload.jsonl", "document_workload_sha256"),
        ):
            if sha256(destination / filename) != runtime[key]:
                raise ValueError("runtime workload checksum mismatch")
        if (
            sha256(destination / "request-workload.jsonl")
            != runtime["single_query"]["workload_sha256"]
        ):
            raise ValueError("single-query workload checksum mismatch")
        completed.append(
            {"name": name, "directory": str(destination), "report_sha256": sha256(report_path)}
        )
    reports = [json.loads((Path(r["directory"]) / "report.json").read_text()) for r in completed]
    workloads = {
        (
            r["runtime"]["query_workload_sha256"],
            r["runtime"]["document_workload_sha256"],
            r["runtime"]["single_query"]["workload_sha256"],
        )
        for r in reports
    }
    if len(workloads) != 1:
        raise ValueError("model inference workloads differ")
    return {
        "runs": completed,
        "datasets": expected_datasets,
        "matrix_files": {str(p): sha256(p) for p in matrices},
        "runtime_repeats": repeats,
        "same_inference_workload_verified": True,
        "process_rule": "one fresh subprocess per model; sequential after training",
        "frozen_selection_sha256": sha256(selection) if selection else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, action="append", required=True)
    parser.add_argument("--domains", type=Path, required=True)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument(
        "--selection", type=Path, help="frozen development decision required for test"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report exists; use a new version")
    result = run(
        args.base,
        args.matrix,
        args.domains,
        args.output,
        pilot=args.pilot,
        selection=args.selection,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
