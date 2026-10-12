"""Rescore fixed first-experiment rankings against the completed expanded pool."""

import argparse
import json
from pathlib import Path

from compare import score_rankings
from expand_pool import EXPECTED_MODELS
from metrics import summarize
from prepare import read_rows, sha256, verify_manifest
from seed_analysis import paired_seeds
from sentence_analysis import coverage, exact_scores


def analyze(dataset: Path, suite: Path, selection: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("expanded analysis exists; preserve versions")
    manifest = verify_manifest(dataset)
    specification = json.loads(suite.read_text())
    if specification["frozen_selection_sha256"] != sha256(selection):
        raise ValueError("suite does not bind the frozen original selection")
    decision = json.loads(selection.read_text())
    if not decision["selection_frozen_before_reserved_test"]:
        raise ValueError("original selection is not frozen")
    previous = next(Path(row["directory"]) for row in specification["runs"]
                    if row["name"] == "unchanged")
    original = json.loads((previous / "report.json").read_text())["results"]["retrieval-reviewed-v1"]
    source = Path(original["dataset"])
    verify_manifest(source)
    if manifest["previous_manifest_sha256"] != sha256(source / "manifest.json"):
        raise ValueError("expanded labels refer to a different cohort")
    if any(sha256(dataset / name) != sha256(source / name)
           for name in ("queries.jsonl", "documents.jsonl")):
        raise ValueError("expanded analysis requires identical query/document inputs")
    query_ids = {r["id"] for r in read_rows(dataset / "queries.jsonl")}
    if query_ids != {r["query_id"] for r in read_rows(dataset / "qrels.jsonl") if r["grade"] == 2}:
        raise ValueError("exact sensitivity needs a grade-two answer for every query")
    rankings = {}
    for run in specification["runs"]:
        name, directory = run["name"], Path(run["directory"])
        if name in rankings or sha256(directory / "report.json") != run["report_sha256"]:
            raise ValueError("duplicate or changed model report")
        detail = json.loads((directory / "report.json").read_text())["results"]["retrieval-reviewed-v1"]
        ranking = directory / "retrieval-reviewed-v1" / "rankings.jsonl"
        if detail["dataset_manifest_sha256"] != sha256(source / "manifest.json") or (
            sha256(ranking) != detail["ranking_sha256"]
        ):
            raise ValueError("ranking or original dataset binding changed")
        rankings[name] = ranking
    if rankings.keys() != EXPECTED_MODELS:
        raise ValueError("require the frozen 19-model cohort")
    pool_coverage = {name: coverage(dataset, path) for name, path in rankings.items()}
    if any(row["unjudged_top10_pairs"] for row in pool_coverage.values()):
        raise ValueError("expanded top-ten judgments are incomplete")
    views = {}
    for view, scorer in (("graded", score_rankings), ("exact_grade2", exact_scores)):
        scores = {name: scorer(dataset, path) for name, path in rankings.items()}
        comparisons = {}
        for budget in ("equal-data", "equal-compute"):
            controls = {s: scores[f"{budget}/original-seed{s}"] for s in (17, 29, 43)}
            comparisons[f"{budget}/original_minus_unchanged"] = paired_seeds(
                {s: scores["unchanged"] for s in controls}, controls
            )
            for arm in ("unigram-32000", "bpe-32000"):
                candidates = {s: scores[f"{budget}/{arm}-seed{s}"] for s in controls}
                comparisons[f"{budget}/{arm}_minus_adapted_original"] = paired_seeds(controls, candidates)
        views[view] = {"summary": {name: summarize(rows) for name, rows in scores.items()},
                       "paired_comparisons": comparisons}
    result = {"dataset": str(dataset), "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
              "suite_sha256": sha256(suite), "rankings_recomputed": False,
              "ranking_sha256": {name: sha256(path) for name, path in rankings.items()},
              "frozen_original_selection": decision["selection"], "selection_sha256": sha256(selection),
              "selection_changed": False, "pool_coverage": pool_coverage, "label_views": views,
              "implementations": {
                  name: sha256(Path(__file__).with_name(name))
                  for name in ("expanded_analysis.py", "expand_pool.py", "prepare.py",
                               "compare.py", "metrics.py", "seed_analysis.py", "sentence_analysis.py")
              },
              "limits": ["agent sentence-correspondence review; no human judgments",
                         "all 19 pooled systems have complete top-ten coverage; outside-pool relevance remains unknown",
                         "larger judged answer sets also change ideal ranking and recall denominators",
                         "secondary rescore cannot change the original frozen domain choice",
                         "bootstrap excludes annotation uncertainty"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.dataset, args.suite, args.selection, args.output)
