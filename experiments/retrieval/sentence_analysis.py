"""Report fixed-label sentence-pilot comparisons without revising domain selection."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from compare import score_rankings
from evaluate import bm25
from metrics import paired_difference, query_metrics, summarize
from prepare import read_rows, sha256, verify_manifest, write_rows
from seed_analysis import paired_seeds


def exact_scores(dataset: Path, ranking: Path) -> list[dict]:
    """Sensitivity check: partial grade-1 correspondences count as zero."""
    scores = score_rankings(dataset, ranking)
    qrels = defaultdict(dict)
    for row in read_rows(dataset / "qrels.jsonl"):
        qrels[row["query_id"]][row["document_id"]] = 2 if row["grade"] == 2 else 0
    rankings = {row["query_id"]: row["documents"] for row in read_rows(ranking)}
    return [
        dict(row, metrics=query_metrics(rankings[row["query_id"]], qrels[row["query_id"]]))
        for row in scores
    ]


def coverage(dataset: Path, ranking: Path) -> dict:
    judged = {(row["query_id"], row["document_id"]) for row in read_rows(dataset / "qrels.jsonl")}
    rows = read_rows(ranking)
    total = sum(len(row["documents"][:10]) for row in rows)
    unknown = sum(
        (row["query_id"], doc) not in judged for row in rows for doc in row["documents"][:10]
    )
    return {
        "top10_pairs": total,
        "unjudged_top10_pairs": unknown,
        "unjudged_top10_fraction": unknown / total if total else 0,
    }


def analyze(suite: Path, selection: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("sentence analysis exists; preserve versions")
    specification = json.loads(suite.read_text())
    if specification["datasets"].keys() != {"retrieval-pilot-v1", "retrieval-reviewed-v1"}:
        raise ValueError("require the original and reviewed sentence cohorts")
    if specification["frozen_selection_sha256"] != sha256(selection):
        raise ValueError("sentence suite does not bind the frozen domain choice")
    decision = json.loads(selection.read_text())
    if not decision.get("selection_frozen_before_reserved_test") or sha256(
        Path(decision["evaluation_suite"])
    ) != decision["evaluation_suite_sha256"]:
        raise ValueError("frozen development evidence differs")
    records = {}
    for run in specification["runs"]:
        if run["name"] in records:
            raise ValueError("duplicate model run")
        directory = Path(run["directory"])
        if sha256(directory / "report.json") != run["report_sha256"]:
            raise ValueError("sentence evaluation report checksum mismatch")
        report = json.loads((directory / "report.json").read_text())
        records[run["name"]] = (directory, report)
    required = {"unchanged"} | {
        f"{budget}/{arm}-seed{seed}"
        for budget in ("equal-data", "equal-compute")
        for arm in ("original", "unigram-32000", "bpe-32000")
        for seed in (17, 29, 43)
    }
    if records.keys() != required:
        raise ValueError("sentence analysis requires exactly the declared 19 models")
    result = {
        "frozen_domain_selection": decision["selection"],
        "selection_changed_using_sentence_results": False,
        "selection_sha256": sha256(selection),
        "suite_sha256": sha256(suite),
        "datasets": {},
        "limits": [
            "Development sentence correspondence, not natural search relevance",
            "Single-agent judgments; no independent human review",
            "Outside original E5/BM25/source pool, relevance remains unknown",
            "Training seeds and group bootstrap exclude label uncertainty",
            "Sentence results are secondary and cannot revise the frozen domain choice",
        ],
        "implementations": {
            name: sha256(Path(__file__).with_name(name))
            for name in ("sentence_analysis.py", "seed_analysis.py", "compare.py", "metrics.py", "evaluate.py")
        },
    }
    # Validate every binding before writing any analysis outputs.
    for name, (_, report) in records.items():
        if report["results"].keys() != specification["datasets"].keys():
            raise ValueError("sentence dataset set differs")
        for key, detail in report["results"].items():
            dataset = Path(detail["dataset"])
            verify_manifest(dataset)
            if sha256(dataset / "manifest.json") != specification["datasets"][key] or (
                detail["dataset_manifest_sha256"] != specification["datasets"][key]
            ):
                raise ValueError("sentence dataset checksum differs")
            if key == "retrieval-reviewed-v1":
                exact_queries = {
                    row["query_id"] for row in read_rows(dataset / "qrels.jsonl")
                    if row["grade"] == 2
                }
                if exact_queries != {row["id"] for row in read_rows(dataset / "queries.jsonl")}:
                    raise ValueError("exact sensitivity requires a grade-2 positive for every reviewed query")
            directory = records[name][0]
            for filename, field in (("rankings.jsonl", "ranking_sha256"), ("scores.jsonl", "scores_sha256")):
                if sha256(directory / key / filename) != detail[field]:
                    raise ValueError("sentence ranking/score checksum differs")
    output.mkdir(parents=True)
    for key in sorted(specification["datasets"]):
        dataset = Path(records["unchanged"][1]["results"][key]["dataset"])
        queries = read_rows(dataset / "queries.jsonl")
        docs = sorted(read_rows(dataset / "documents.jsonl"), key=lambda row: row["id"])
        ranked, runtime = bm25(docs, queries)
        lexical = output / key / "bm25-rankings.jsonl"
        lexical.parent.mkdir()
        write_rows(lexical, [{"query_id": q["id"], "documents": r} for q, r in zip(queries, ranked, strict=True)])
        data = {"manifest_sha256": sha256(dataset / "manifest.json"), "bm25_execution": runtime, "label_views": {}}
        views = [("graded", score_rankings)]
        if key == "retrieval-reviewed-v1":
            views.append(("exact_grade2", exact_scores))
            data["exact_sensitivity_eligibility"] = "Every reviewed query has at least one grade-2 positive; no query omitted."
        else:
            data["exact_sensitivity_unavailable"] = "Provisional source labels are grade 1, not reviewed grade-2 correspondence."
        for view, scorer in views:
            scores = {name: scorer(dataset, directory / key / "rankings.jsonl") for name, (directory, _) in records.items()}
            scores["bm25"] = scorer(dataset, lexical)
            comparisons = {"unchanged_minus_bm25": paired_difference(scores["bm25"], scores["unchanged"])}
            for budget in ("equal-data", "equal-compute"):
                control = {s: scores[f"{budget}/original-seed{s}"] for s in (17, 29, 43)}
                for arm in ("original", "unigram-32000", "bpe-32000"):
                    candidate = {s: scores[f"{budget}/{arm}-seed{s}"] for s in (17, 29, 43)}
                    comparisons[f"{budget}/{arm}_minus_unchanged"] = paired_seeds(
                        {s: scores["unchanged"] for s in candidate}, candidate
                    )
                    if arm != "original":
                        comparisons[f"{budget}/{arm}_minus_adapted_original"] = paired_seeds(control, candidate)
            data["label_views"][view] = {
                "summary": {name: summarize(rows) for name, rows in scores.items()},
                "paired_comparisons": comparisons,
                "misses_no_positive_at_10": {name: sum(row["metrics"]["mrr_at_10"] == 0 for row in rows) for name, rows in scores.items()},
            }
        data["pool_coverage"] = {
            name: coverage(dataset, directory / key / "rankings.jsonl") for name, (directory, _) in records.items()
        }
        data["pool_coverage"]["bm25"] = coverage(dataset, lexical)
        result["datasets"][key] = data
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report exists; preserve versions")
    result = analyze(args.suite, args.selection, args.output)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v["label_views"]["graded"]["misses_no_positive_at_10"]["unchanged"] for k, v in result["datasets"].items()}))
