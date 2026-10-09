"""Report reserved-test comparisons without changing the frozen model selection."""

import argparse
import json
from pathlib import Path

from compare import score_rankings
from prepare import read_rows, sha256, verify_manifest
from seed_analysis import DIRECTIONS, paired_seeds


def confirm(suite: Path, selection: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("confirmation report exists; preserve completed test results")
    specification = json.loads(suite.read_text())
    decision = json.loads(selection.read_text())
    if (
        not decision.get("selection_frozen_before_reserved_test")
        or specification.get("frozen_selection_sha256") != sha256(selection)
        or sha256(Path(decision["evaluation_suite"])) != decision["evaluation_suite_sha256"]
    ):
        raise ValueError("test results are not bound to an unchanged frozen development decision")
    runs = {}
    for row in specification["runs"]:
        directory = Path(row["directory"])
        if sha256(directory / "report.json") != row["report_sha256"]:
            raise ValueError("test evaluation report checksum mismatch")
        report = json.loads((directory / "report.json").read_text())
        scored = {}
        for direction, detail in report["results"].items():
            if direction not in DIRECTIONS:
                continue
            dataset = Path(detail["dataset"])
            verify_manifest(dataset)
            if any(q.get("split") != "test" for q in read_rows(dataset / "queries.jsonl")):
                raise ValueError("confirmation requires reserved-test domain queries")
            if sha256(dataset / "manifest.json") != detail["dataset_manifest_sha256"]:
                raise ValueError("test dataset checksum mismatch")
            path = directory / direction / "rankings.jsonl"
            if sha256(path) != detail["ranking_sha256"]:
                raise ValueError("test ranking checksum mismatch")
            scored[direction] = score_rankings(dataset, path)
        if scored.keys() != DIRECTIONS:
            raise ValueError("confirmation requires all nine directions")
        runs[row["name"]] = scored
    seeds = (17, 29, 43)
    comparisons = {}
    for budget in ("equal-data", "equal-compute"):
        for arm in ("original", "unigram-32000", "bpe-32000"):
            candidates = {s: runs[f"{budget}/{arm}-seed{s}"] for s in seeds}
            baselines = {
                "unchanged": {s: runs["unchanged"] for s in seeds},
            }
            if arm != "original":
                baselines["adapted-original"] = {
                    s: runs[f"{budget}/original-seed{s}"] for s in seeds
                }
            comparisons[f"{budget}/{arm}"] = {
                reference: {
                    direction: paired_seeds(
                        {s: baseline[s][direction] for s in seeds},
                        {s: candidates[s][direction] for s in seeds},
                    )
                    for direction in sorted(DIRECTIONS)
                }
                for reference, baseline in baselines.items()
            }
    result = {
        "frozen_selection": decision["selection"],
        "selection_changed_using_test": False,
        "selection_sha256": sha256(selection),
        "test_suite_sha256": sha256(suite),
        "comparisons": comparisons,
        "analysis_implementation_sha256": {
            p.name: sha256(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("seed_analysis.py"),
                Path(__file__).with_name("compare.py"),
            )
        },
        "limits": [
            "18 test intents",
            "three training seeds",
            "single-agent authored/judged proxy",
            "secondary direction intervals are not adjusted for multiple comparisons",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = confirm(args.suite, args.selection, args.output)
    print(
        json.dumps(
            {"frozen_selection": result["frozen_selection"], "selection_changed_using_test": False}
        )
    )
