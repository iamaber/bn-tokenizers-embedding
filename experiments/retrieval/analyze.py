"""Prepare a failure audit from rankings; observed features are not causal labels."""

import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from prepare import read_rows, sha256, verify_manifest, write_rows


def failure_audit(directory: Path, ranking: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("failure audit exists; preserve reviewer notes")
    manifest = verify_manifest(directory)
    queries = read_rows(directory / "queries.jsonl")
    documents = {r["id"]: r for r in read_rows(directory / "documents.jsonl")}
    relevant = defaultdict(set)
    for row in read_rows(directory / "qrels.jsonl"):
        if row["grade"] > 0:
            relevant[row["query_id"]].add(row["document_id"])
    rankings = {}
    for row in read_rows(ranking):
        if row["query_id"] in rankings or len(row["documents"]) != len(set(row["documents"])):
            raise ValueError("duplicate ranking/query")
        if set(row["documents"]) - documents.keys():
            raise ValueError("unknown ranked document")
        rankings[row["query_id"]] = row["documents"][:10]
    if rankings.keys() != {q["id"] for q in queries}:
        raise ValueError("rankings must cover exactly the dataset queries")
    failures = []
    bucket_counts = Counter()
    for query in queries:
        qid = query["id"]
        if not relevant[qid]:
            raise ValueError("query has no positive judgments")
        if relevant[qid].intersection(rankings[qid]):
            continue
        bucket_counts[query["bucket"]] += 1
        text = query["text"]
        failures.append({
            "query_id": qid, "group": query["group"], "query": text,
            "bucket": query["bucket"], "words": len(text.split()),
            "has_bengali_script": any("\u0980" <= c <= "\u09ff" for c in text),
            "has_latin_script": any("LATIN" in unicodedata.name(c, "") for c in text),
            "has_digits": any(c.isdecimal() for c in text),
            "reviewed_positives": [documents[d] for d in sorted(relevant[qid])],
            "top_10": [documents[d] for d in rankings[qid]],
            "primary_failure_category": None, "reviewer_id": None,
            "reviewer_origin": None, "notes": None,
        })
    output.mkdir(parents=True)
    write_rows(output / "failure-review.jsonl", failures)
    result = {
        "queries": len(queries), "misses_at_10": len(failures),
        "misses_by_bucket": dict(bucket_counts),
        "label_status": manifest.get("label_status", "unreviewed source counterparts"),
        "causal_categories_completed": False,
        "suggested_categories": ["spelling_variation", "english_insertion", "name_or_number",
                                 "negation", "wrong_source_pair", "domain_gap", "other"],
        "note": "Observed script/digit/length features do not establish a failure cause.",
        "inputs": {str(p): sha256(p) for p in (directory / "manifest.json", ranking)},
        "files": {"failure-review.jsonl": sha256(output / "failure-review.jsonl")},
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(failure_audit(args.pilot, args.ranking, args.output), indent=2))
