"""Execute and safely resume the five-arm, three-seed extended training matrix."""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from prepare import sha256


def verify(output: Path, protocol: Path, base: Path, arm: str, seed: int, budget: str) -> dict:
    specification = json.loads(protocol.read_text())
    report = json.loads((output / "training-report.json").read_text())
    if (report["protocol_sha256"], report["arm"], report["seed"], report["budget"]) != (
        sha256(protocol), arm, seed, budget
    ):
        raise ValueError("checkpoint does not match frozen protocol and run")
    if report["base_files"] != {p.name: sha256(p) for p in base.iterdir() if p.is_file()}:
        raise ValueError("base model snapshot changed")
    for name, digest in report["implementations"].items():
        if Path(name).name != name or sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError("training implementation changed")
    files = {p.name for p in output.iterdir() if p.is_file()} - {"training-report.json"}
    if files != report["checkpoint_files"].keys():
        raise ValueError("checkpoint file inventory changed")
    for name, digest in report["checkpoint_files"].items():
        if Path(name).name != name or sha256(output / name) != digest:
            raise ValueError("checkpoint checksum mismatch")
    settings = specification["arms"][arm]
    metadata = json.loads((output / "encoder.json").read_text())
    if bool(metadata["custom_tokenizer"]) != bool(settings["tokenizer"]):
        raise ValueError("checkpoint tokenizer identity differs from arm")
    initialization = report["tokenizer_initialization"]
    if settings["tokenizer"]:
        digest = settings["tokenizer_sha256"]
        if (sha256(Path(settings["tokenizer"])), metadata["tokenizer_sha256"],
                initialization["tokenizer_sha256"]) != (digest, digest, digest):
            raise ValueError("custom tokenizer changed")
    if settings["initialization"]:
        digest = settings["initialization_manifest_sha256"]
        if (initialization["artifact_manifest_sha256"],
                sha256(Path(settings["initialization"]) / "manifest.json")) != (digest, digest):
            raise ValueError("context initialization changed")
    else:
        method = "original-subword mean initialization" if settings["tokenizer"] else (
            "retain original tokenizer and input embeddings"
        )
        if initialization["method"] != method:
            raise ValueError("wrong initialization control")
    stages = report["stages"]
    for stage in ("representation", "contrastive"):
        counts = stages[stage]["query_languages"]
        if counts != dict.fromkeys(("english", "banglish"), stages[stage]["steps"] * specification["batch_size"] // 2):
            raise ValueError("language exposure is not balanced")
    expected_hard = stages["contrastive"]["steps"] // 3 * specification["batch_size"]
    if stages["contrastive"]["verified_negative_examples"] != expected_hard:
        raise ValueError("verified-negative exposure differs from protocol")
    if budget == "equal-data":
        if any(stages[stage]["steps"] != specification[f"{stage}_steps"]
               for stage in ("representation", "contrastive")):
            raise ValueError("equal-data run did not finish its exposure")
    elif report["equal_compute_limit_seconds"] != specification["equal_compute_seconds"]:
        raise ValueError("different compute budget")
    return report


def run(protocol: Path, base: Path, root: Path, budget: str) -> dict:
    specification = json.loads(protocol.read_text())
    if base.name != specification["base_revision"]:
        raise ValueError("base revision differs from protocol")
    for field in ("alignment", "hard_negatives"):
        setting = specification[field]
        directory = Path(setting["directory"])
        if sha256(directory / "manifest.json") != setting["manifest_sha256"]:
            raise ValueError("training input manifest changed")
        manifest = json.loads((directory / "manifest.json").read_text())
        if any(Path(name).name != name or sha256(directory / name) != digest
               for name, digest in manifest["files"].items()):
            raise ValueError("training input file changed")
    completed = []
    for seed in specification["training_seeds"]:
        arms = list(specification["arms"])
        random.Random(seed).shuffle(arms)
        for arm in arms:
            output = root / budget / f"{arm}-seed{seed}"
            if not (output / "training-report.json").exists():
                if output.exists():
                    raise ValueError(f"incomplete checkpoint exists; inspect before retrying: {output}")
                print(f"Training {budget}/{arm}/seed{seed}", flush=True)
                subprocess.run([
                    sys.executable, str(Path(__file__).with_name("train_extended.py")),
                    "--protocol", str(protocol), "--base", str(base), "--arm", arm,
                    "--seed", str(seed), "--budget", budget, "--output", str(output),
                ], check=True)
            report = verify(output, protocol, base, arm, seed, budget)
            completed.append({"arm": arm, "seed": seed, "directory": str(output),
                              "training_report_sha256": sha256(output / "training-report.json"),
                              "charged_seconds": report["charged_training_seconds"],
                              "training_sequence_sha256": report["training_sequence_sha256"]})
    if budget == "equal-data":
        for seed in specification["training_seeds"]:
            if len({r["training_sequence_sha256"] for r in completed if r["seed"] == seed}) != 1:
                raise ValueError("equal-data arms consumed different sample sequences")
    return {"protocol_sha256": sha256(protocol), "budget": budget, "runs": completed,
            "equal_data_sample_order_verified": budget == "equal-data"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("models/encoders-round2"))
    parser.add_argument("--budget", choices=("equal-data", "equal-compute"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("matrix report exists; preserve versions")
    result = run(args.protocol, args.base, args.root, args.budget)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
