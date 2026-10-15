"""Run only the original-tokenizer arm of the frozen extended protocol."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from prepare import sha256
from run_extended_matrix import verify


def run(protocol: Path, base: Path, root: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("training inventory exists; preserve versions")
    specification = json.loads(protocol.read_text())
    completed = []
    for budget in ("equal-data", "equal-compute"):
        for seed in specification["training_seeds"]:
            directory = root / budget / f"original-seed{seed}"
            if not (directory / "training-report.json").exists():
                if directory.exists():
                    raise ValueError(f"incomplete checkpoint requires inspection: {directory}")
                subprocess.run([
                    sys.executable, str(Path(__file__).with_name("train_extended.py")),
                    "--protocol", str(protocol), "--base", str(base), "--arm", "original",
                    "--seed", str(seed), "--budget", budget, "--output", str(directory),
                ], check=True)
            report = verify(directory, protocol, base, "original", seed, budget)
            completed.append({"name": f"{budget}/original-seed{seed}", "directory": str(directory),
                              "training_report_sha256": sha256(directory / "training-report.json"),
                              "stages": report["stages"],
                              "charged_seconds": report["charged_training_seconds"]})
    result = {"protocol_sha256": sha256(protocol), "runs": completed,
              "scope": "original tokenizer only; stronger-transfer arms not run",
              "limitations": "joint longer-training, negative and replay intervention; no isolated causal ablation"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.base, args.root, args.output)
