"""Summarize blinded agent re-review and prepare an unfilled human review packet."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from prepare import read_rows, sha256, verify_manifest, write_rows


def audit(archive: Path, dataset: Path, reviews: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("audit output exists; preserve versions")
    verify_manifest(dataset)
    manifest = json.loads((archive / "manifest.json").read_text())
    if sha256(archive / "judgments.jsonl") != manifest["judgments_sha256"]:
        raise ValueError("original judgment archive changed")
    originals = read_rows(archive / "judgments.jsonl")
    selected = [r for r in originals if r["grade"] > 0] + random.Random(20261012).sample(
        [r for r in originals if r["grade"] == 0], 120
    )
    random.Random(20261012).shuffle(selected)
    key = lambda row: (row["query_id"], row["document_id"])
    expected = {key(r): r for r in selected}
    judged = {}
    paths = sorted(reviews.glob("audit-*.jsonl"))
    for path in paths:
        for row in read_rows(path):
            if key(row) in judged or key(row) not in expected:
                raise ValueError("duplicate or unexpected audit pair")
            if (type(row["grade"]) is not int or row["grade"] not in (0, 1, 2)
                    or row.get("reviewer_origin") != "agent"
                    or not row.get("reviewer_id") or not row.get("rationale")):
                raise ValueError("incomplete agent audit")
            judged[key(row)] = row
    if judged.keys() != expected.keys():
        raise ValueError("audit must cover all positives and the fixed negative sample")
    queries = {r["id"]: r["text"] for r in read_rows(dataset / "queries.jsonl")}
    documents = {r["id"]: r["text"] for r in read_rows(dataset / "documents.jsonl")}
    disagreements = [
        {"query_id": k[0], "document_id": k[1], "original_grade": expected[k]["grade"],
         "audit_grade": row["grade"], "original_rationale": expected[k]["rationale"],
         "audit_rationale": row["rationale"], "audit_reviewer_id": row["reviewer_id"]}
        for k, row in sorted(judged.items()) if row["grade"] != expected[k]["grade"]
    ]
    packet = [
        {"query_id": r["query_id"], "document_id": r["document_id"],
         "query": queries[r["query_id"]], "document": documents[r["document_id"]],
         "grade": None, "rationale": None, "reviewer_id": None, "reviewer_origin": None}
        for r in selected
    ]
    confusion = Counter(f"{expected[k]['grade']}->{r['grade']}" for k, r in judged.items())
    output.mkdir(parents=True)
    write_rows(output / "human-review-packet.jsonl", packet)
    write_rows(output / "disagreements.jsonl", disagreements)
    result = {
        "original_judgments_sha256": sha256(archive / "judgments.jsonl"),
        "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
        "reviews": {str(p): sha256(p) for p in paths},
        "sampling_seed": 20261012, "reviewed_pairs": len(judged),
        "positive_pairs": sum(r["grade"] > 0 for r in selected), "sampled_negatives": 120,
        "confusion_original_to_audit": dict(sorted(confusion.items())),
        "agreement_fraction": 1 - len(disagreements) / len(judged),
        "disagreements": len(disagreements), "original_labels_modified": False,
        "human_review_completed": False, "independent_human_review_verified": False,
        "human_packet": str(output / "human-review-packet.jsonl"),
        "files": {name: sha256(output / name) for name in (
            "human-review-packet.jsonl", "disagreements.jsonl")},
        "implementation_sha256": sha256(Path(__file__)),
        "limitations": ["second-pass agent review, not human language validation",
                       "sample deliberately includes every positive; overall agreement is not a population estimate",
                       "negative sample is uniform and seeded; unsampled errors remain possible",
                       "disagreements require adjudication, not automatic replacement",
                       "informal transliteration and missing conversation context remain ambiguous"],
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.archive, args.dataset, args.reviews, args.output), indent=2))
