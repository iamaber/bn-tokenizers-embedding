"""Export blinded review material and freeze explicitly attributed judgments."""

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from prepare import read_rows, sha256, verify_manifest, write_rows


def review_pool(directory: Path) -> list[dict]:
    verify_manifest(directory)
    queries = {r["id"]: r for r in read_rows(directory / "queries.jsonl")}
    documents = {r["id"]: r for r in read_rows(directory / "documents.jsonl")}
    rows = read_rows(directory / "review-pool.jsonl")
    seen = set()
    for row in rows:
        key = (row["query_id"], row["document_id"])
        if key in seen or key[0] not in queries or key[1] not in documents:
            raise ValueError("duplicate or unknown review pair")
        query, document = queries[key[0]], documents[key[1]]
        if (row["query"], row["document"], row["group"]) != (
            query["text"], document["text"], query["group"]
        ):
            raise ValueError("review text/group does not match frozen dataset")
        seen.add(key)
    if set(queries) != {r["query_id"] for r in rows}:
        raise ValueError("review pool must cover every query")
    return rows


def export(directory: Path, output: Path, seed: int = 17, calibration: int = 30) -> dict:
    if output.exists():
        raise ValueError("review output exists; preserve prior judgments")
    pool = review_pool(directory)
    groups = sorted({r["group"] for r in pool})
    random.Random(seed).shuffle(groups)
    calibration_groups = set(groups[:calibration])
    blinded = [
        {
            "query_id": r["query_id"], "document_id": r["document_id"],
            "group": r["group"], "query": r["query"], "document": r["document"],
            "grade": None, "rationale": None,
        }
        for r in pool
    ]
    output.mkdir(parents=True)
    for reviewer in ("reviewer-1", "reviewer-2"):
        packet = [dict(row, reviewer_id=None, reviewer_origin=None) for row in blinded]
        write_rows(output / f"{reviewer}.jsonl", packet)
        write_rows(
            output / f"{reviewer}-calibration.jsonl",
            [r for r in packet if r["group"] in calibration_groups],
        )
    shutil.copyfile(directory / "authoring.jsonl", output / "pair-review.jsonl")
    instructions = {
        "rubric": {"2": "same essential meaning", "1": "related but partly different",
                   "0": "unrelated or contradicts; preserve negation, names and numbers"},
        "unresolved": "Leave grade null; null is never a negative.",
        "provenance": "reviewer_origin must be agent or human; never invent human review",
        "independence": "Keep packets separate until grades are recorded independently.",
        "calibration_groups": len(calibration_groups), "seed": seed,
        "pool_sha256": sha256(directory / "review-pool.jsonl"), "pairs": len(pool),
    }
    (output / "instructions.json").write_text(json.dumps(instructions, indent=2) + "\n")
    return instructions


def import_review(
    directory: Path, judgments: Path, pair_review: Path, output: Path,
) -> dict:
    """Import final judgments; do not infer a negative from an unfinished field."""
    if output.exists():
        raise ValueError("reviewed dataset exists; use a new version")
    original = verify_manifest(directory)
    pool = {(r["query_id"], r["document_id"]): r for r in review_pool(directory)}
    expected = set(pool)
    labels = {}
    reviewers = set()
    origins = Counter()
    for row in read_rows(judgments):
        key = (row["query_id"], row["document_id"])
        if key not in expected or key in labels:
            raise ValueError("duplicate or unknown judgment")
        if any(row.get(k) != pool[key][k] for k in ("query", "document", "group")):
            raise ValueError("judgment changed the reviewed text or group")
        if type(row["grade"]) is not int or row["grade"] not in (0, 1, 2):
            raise ValueError("unresolved or invalid judgment; require grade 0, 1 or 2")
        if row.get("reviewer_origin") not in ("human", "agent"):
            raise ValueError("explicit human/agent reviewer origin required")
        if not row.get("reviewer_id") or not str(row.get("rationale") or "").strip():
            raise ValueError("reviewer ID and rationale required")
        labels[key] = row
        reviewers.add(row["reviewer_id"])
        origins[row["reviewer_origin"]] += 1
    if labels.keys() != expected:
        raise ValueError("incomplete review; every pooled pair requires a judgment")
    queries = read_rows(directory / "queries.jsonl")
    documents = read_rows(directory / "documents.jsonl")
    query_ids = {q["id"] for q in queries}
    query_lookup = {q["id"]: q for q in queries}
    doc_lookup = {d["id"]: d for d in documents}
    source_positives = {}
    for row in read_rows(directory / "qrels.jsonl"):
        if row["grade"] > 0:
            source_positives.setdefault(row["query_id"], set()).add(
                doc_lookup[row["document_id"]]["text"]
            )
    source_pairs = {}
    for row in read_rows(directory / "authoring.jsonl"):
        qid = row["query_id"]
        if qid not in query_lookup or qid in source_pairs:
            raise ValueError("duplicate or unknown authoring pair")
        query = query_lookup[qid]
        if (row["group"], row["source_banglish"]) != (query["group"], query["text"]) or (
            row["paired_bengali"] not in source_positives.get(qid, set())
        ):
            raise ValueError("authoring pair does not match frozen source")
        source_pairs[qid] = row
    if source_pairs.keys() != query_ids:
        raise ValueError("authoring pairs must cover every query")
    decisions = {}
    for row in read_rows(pair_review):
        query_id = row["query_id"]
        if query_id not in query_ids or query_id in decisions:
            raise ValueError("duplicate or unknown pair decision")
        if any(row.get(k) != source_pairs[query_id][k] for k in (
            "group", "source_banglish", "paired_bengali"
        )):
            raise ValueError("pair review changed the original pair or group")
        if any(type(row.get(k)) is not bool for k in ("pair_correct", "privacy_review_passed")):
            raise ValueError("source-pair and privacy decisions must be explicit booleans")
        if not row.get("reviewer") or row.get("reviewer_origin") not in ("human", "agent"):
            raise ValueError("pair reviewer and origin required")
        if not row["pair_correct"] or not row["privacy_review_passed"]:
            if not str(row.get("exclusion_reason") or "").strip():
                raise ValueError("excluded query requires a reason")
        decisions[query_id] = row
    if decisions.keys() != query_ids:
        raise ValueError("all source pairs must be reviewed")
    # Privacy exclusions remove the whole associated group, including documents.
    private_groups = {r["group"] for r in decisions.values() if not r["privacy_review_passed"]}
    documents = [d for d in documents if d["group"] not in private_groups]
    document_ids = {d["id"] for d in documents}
    queries = [q for q in queries if decisions[q["id"]]["pair_correct"]
               and q["group"] not in private_groups]
    kept_ids = {q["id"] for q in queries}
    qrels = [
        {"query_id": qid, "document_id": did, "grade": row["grade"],
         "label_origin": f"{row['reviewer_origin']}_reviewed",
         "reviewer_id": row["reviewer_id"], "rationale": row["rationale"]}
        for (qid, did), row in sorted(labels.items())
        if qid in kept_ids and did in document_ids
    ]
    positive_queries = {r["query_id"] for r in qrels if r["grade"] > 0}
    if not queries or positive_queries != kept_ids:
        raise ValueError("every retained query needs a reviewed positive")
    output.mkdir(parents=True)
    for name, rows in (("documents", documents), ("queries", queries), ("qrels", qrels)):
        write_rows(output / f"{name}.jsonl", rows)
    write_rows(output / "judgments.jsonl", [
        row for (qid, did), row in sorted(labels.items()) if qid in kept_ids and did in document_ids
    ])
    write_rows(output / "pair-review.jsonl", [
        row for qid, row in sorted(decisions.items()) if qid in kept_ids
    ])
    manifest = {
        **original, "queries": len(queries), "documents": len(documents),
        "buckets": dict(Counter(q["bucket"] for q in queries)),
        "independent_groups": len({q["group"] for q in queries}),
        "label_status": "reviewed development sentence correspondence; see reviewer provenance",
        "human_review_completed": "agent" not in origins and all(
            r["reviewer_origin"] == "human" for r in decisions.values()
        ),
        "independent_review_verified": False,
        "reviewer_ids": sorted(reviewers), "judgment_origins": dict(origins),
        "exclusions": [
            {"query_id": qid, "reason": r.get("exclusion_reason", "group privacy exclusion")}
            for qid, r in sorted(decisions.items()) if qid not in kept_ids
        ],
        "unjudged_documents": "unknown; pooled judged set is incomplete outside its coverage",
        "review_inputs": {str(p): sha256(p) for p in (
            directory / "manifest.json", directory / "review-pool.jsonl",
            directory / "authoring.jsonl", judgments, pair_review
        )},
        "files": {f"{n}.jsonl": sha256(output / f"{n}.jsonl") for n in (
            "documents", "queries", "qrels", "judgments", "pair-review"
        )},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export", "import"))
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--pair-review", type=Path)
    args = parser.parse_args()
    if args.command == "import":
        if args.judgments is None or args.pair_review is None:
            parser.error("import requires --judgments and --pair-review")
        result = import_review(args.pilot, args.judgments, args.pair_review, args.output)
    else:
        result = export(args.pilot, args.output)
    print(json.dumps(result, indent=2))
