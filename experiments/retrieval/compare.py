"""Compare rankings on identical frozen labels with paired group uncertainty."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from metrics import paired_difference, query_metrics
from prepare import read_rows, sha256, verify_manifest


def score_rankings(directory: Path, path: Path) -> list[dict]:
    queries = read_rows(directory / "queries.jsonl")
    doc_ids = {r["id"] for r in read_rows(directory / "documents.jsonl")}
    qrels = defaultdict(dict)
    for row in read_rows(directory / "qrels.jsonl"):
        qid, did = row["query_id"], row["document_id"]
        if did in qrels[qid] or did not in doc_ids:
            raise ValueError("duplicate or unknown relevance document")
        qrels[qid][did] = row["grade"]
    rankings = {}
    for row in read_rows(path):
        if row["query_id"] in rankings or set(row["documents"]) - doc_ids:
            raise ValueError("duplicate query or unknown ranking document")
        rankings[row["query_id"]] = row["documents"]
    query_ids = {q["id"] for q in queries}
    if len(query_ids) != len(queries) or rankings.keys() != query_ids or qrels.keys() != query_ids:
        raise ValueError("rankings and labels must cover exactly the dataset queries")
    return [
        {"query_id": q["id"], "group": q["group"], "bucket": q["bucket"],
         "domain": q.get("domain", "unspecified"),
         "metrics": query_metrics(rankings[q["id"]], qrels[q["id"]])}
        for q in queries
    ]


def compare(directory: Path, reference: Path, candidate: Path, seed: int = 17) -> dict:
    manifest = verify_manifest(directory)
    left, right = score_rankings(directory, reference), score_rankings(directory, candidate)
    return {
        "dataset_sha256": sha256(directory / "manifest.json"),
        "label_status": manifest.get("label_status", "unreviewed source counterparts"),
        "ranking_sha256": {"reference": sha256(reference), "candidate": sha256(candidate)},
        "seed": seed, "all": paired_difference(left, right, seed),
        "buckets": {
            bucket: paired_difference([r for r in left if r["bucket"] == bucket],
                                      [r for r in right if r["bucket"] == bucket], seed)
            for bucket in sorted({r["bucket"] for r in left})
        },
        "domains": {
            domain: paired_difference([r for r in left if r["domain"] == domain],
                                      [r for r in right if r["domain"] == domain], seed)
            for domain in sorted({r["domain"] for r in left})
        },
        "uncertainty_scope": "query-group sampling; single run, excludes training-seed variance",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report exists; use a new version")
    result = compare(args.dataset, args.reference, args.candidate)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["all"], indent=2))
