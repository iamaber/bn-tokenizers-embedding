"""Run the frozen encoder matrix sequentially, verifying completed checkpoints."""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from prepare import sha256


def run(protocol: Path, base: Path, root: Path, budget: str) -> dict:
    specification = json.loads(protocol.read_text())
    alignment = Path(specification["alignment_directory"])
    if sha256(alignment / "manifest.json") != specification["alignment_manifest_sha256"]:
        raise ValueError("alignment snapshot differs from frozen run protocol")
    if base.name != specification["base_revision"]:
        raise ValueError("base revision differs from frozen protocol")
    base_hashes = {p.name: sha256(p) for p in sorted(base.iterdir()) if p.is_file()}
    completed = []
    for seed in specification["training_seeds"]:
        arms = list(specification["arms"])
        random.Random(seed).shuffle(arms)
        for arm in arms:
            output = root / budget / f"{arm}-seed{seed}"
            report_path = output / "training-report.json"
            if not report_path.exists():
                if output.exists():
                    raise ValueError(
                        f"incomplete checkpoint exists: {output}; inspect it before retrying"
                    )
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("train_encoder.py")),
                    "--base",
                    str(base),
                    "--alignment",
                    str(alignment),
                    "--output",
                    str(output),
                    "--seed",
                    str(seed),
                    "--batch-size",
                    str(specification["batch_size"]),
                    "--max-length",
                    str(specification["maximum_training_tokens"]),
                ]
                if arm != "original":
                    command.extend(["--tokenizer", f"models/{arm}.json"])
                if budget == "equal-data":
                    settings = specification["equal_data"]
                    command.extend(
                        [
                            "--adapt-steps",
                            str(settings["representation_steps"]),
                            "--steps",
                            str(settings["contrastive_steps"]),
                        ]
                    )
                else:
                    command.extend(
                        [
                            "--adapt-steps",
                            "100000",
                            "--steps",
                            "100000",
                            "--max-seconds",
                            str(specification["equal_compute"]["total_wall_seconds"]),
                        ]
                    )
                print(f"Training {budget}/{arm}/seed{seed}", flush=True)
                subprocess.run(command, check=True)
            report = json.loads(report_path.read_text())
            saved_inputs = json.loads((output / "encoder.json").read_text())
            expected_custom = arm != "original"
            if bool(saved_inputs.get("custom_tokenizer")) != expected_custom:
                raise ValueError("checkpoint original/custom tokenizer identity differs from arm")
            expected_method = (
                "original-subword mean initialization"
                if expected_custom
                else "retain original tokenizer and input embeddings"
            )
            if report["tokenizer_initialization"]["method"] != expected_method:
                raise ValueError("checkpoint initialization method differs from arm")
            if report["base_model_files"] != base_hashes:
                raise ValueError("completed run uses a different base model snapshot")
            if (
                report["seed"],
                report["budget_kind"],
                report["alignment_manifest_sha256"],
                report["batch_size"],
                report["maximum_tokens"],
            ) != (
                seed,
                budget,
                specification["alignment_manifest_sha256"],
                specification["batch_size"],
                specification["maximum_training_tokens"],
            ):
                raise ValueError(f"completed run does not match frozen protocol: {output}")
            if arm != "original" and report["tokenizer_initialization"].get("tokenizer_sha256") != (
                sha256(Path(f"models/{arm}.json"))
            ):
                raise ValueError("completed run uses a different tokenizer")
            if budget == "equal-data":
                stages = report["stages"]
                settings = specification["equal_data"]
                if (stages["representation"]["steps"], stages["contrastive"]["steps"]) != (
                    settings["representation_steps"],
                    settings["contrastive_steps"],
                ):
                    raise ValueError("completed run did not finish the equal-data exposure")
            elif (
                report["total_training_budget_seconds"]
                != specification["equal_compute"]["total_wall_seconds"]
            ):
                raise ValueError("completed run uses a different compute budget")
            for name, digest in report["checkpoint_files"].items():
                if Path(name).name != name or sha256(output / name) != digest:
                    raise ValueError(f"checkpoint checksum mismatch: {output}/{name}")
            completed.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "directory": str(output),
                    "training_report_sha256": sha256(report_path),
                    "seconds": report["total_training_seconds"],
                }
            )
    if budget == "equal-data":
        for seed in specification["training_seeds"]:
            hashes = {
                json.loads((Path(r["directory"]) / "training-report.json").read_text())[
                    "training_sequence_sha256"
                ]
                for r in completed
                if r["seed"] == seed
            }
            if len(hashes) != 1:
                raise ValueError("equal-data arms consumed different training sequences")
    return {
        "protocol_sha256": sha256(protocol),
        "budget": budget,
        "runs": completed,
        "equal_data_sample_order_verified": budget == "equal-data",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiments/retrieval/run_protocol.json")
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("models/encoders"))
    parser.add_argument("--budget", choices=("equal-data", "equal-compute"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("matrix report exists; use a new report path")
    result = run(args.protocol, args.base, args.root, args.budget)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
