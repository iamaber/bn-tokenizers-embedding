"""Build an attributed agent-authored QA proxy with disjoint intent holdouts."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from prepare import canonical, normalize, read_rows, sha256, write_rows

LANGUAGES = {"bn": "bangla", "bl": "banglish", "en": "english"}
DOMAINS = {"business", "banking", "technology", "science", "conversation", "qa"}


def build(
    material: Path, training: Path, output: Path, seed: int = 17, judgments: Path | None = None
) -> dict:
    if output.exists():
        raise ValueError("benchmark output exists; preserve frozen versions")
    records = read_rows(material)
    ids = {r["id"] for r in records}
    if len(ids) != len(records) or {r["domain"] for r in records} != DOMAINS:
        raise ValueError("require unique intent IDs and all six requested domains")
    reviewed = {}
    if judgments:
        provenance = json.loads(judgments.with_suffix(".manifest.json").read_text())
        if provenance["material_sha256"] != sha256(material) or provenance[
            "intent_judgments_sha256"
        ] != sha256(judgments):
            raise ValueError("domain review source/judgments checksum mismatch")
        for row in read_rows(judgments):
            key = (row["query_intent"], row["document_intent"])
            if (
                key in reviewed
                or not set(key) <= ids
                or type(row["grade"]) is not int
                or row["grade"] not in (0, 1, 2)
                or not row.get("rationale")
                or row.get("reviewer_origin") != "agent"
                or not row.get("reviewer_id")
            ):
                raise ValueError("invalid or duplicate domain intent judgment")
            reviewed[key] = row
        if reviewed.keys() != {(q, d) for q in ids for d in ids}:
            raise ValueError("domain review must cover every query/document intent pair")
        if any(not any(reviewed[q, d]["grade"] > 0 for d in ids) for q in ids):
            raise ValueError("every authored intent needs a reviewed positive")
    surfaces = {}
    for row in records:
        for language in LANGUAGES:
            for field in (language, f"q_{language}"):
                text = row[field]
                if not isinstance(text, str) or not normalize(text):
                    raise ValueError("every intent requires three documents and three queries")
                key = canonical(text)
                if key in surfaces and surfaces[key] != row["id"]:
                    raise ValueError("duplicate text across intent groups")
                surfaces[key] = row["id"]
            if canonical(row[language]) == canonical(row[f"q_{language}"]):
                raise ValueError("queries must not copy their answer passages")
    overlap = []
    with training.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError("overlap audit input must be training records")
            key = canonical(row["text"])
            if key in surfaces:
                overlap.append({"intent": surfaces[key], "training_group": row["group"]})
    if overlap:
        raise ValueError(f"benchmark text overlaps training: {overlap[:5]}")
    assignment = {}
    rng = random.Random(seed)
    for domain in sorted(DOMAINS):
        group_ids = sorted(r["id"] for r in records if r["domain"] == domain)
        if len(group_ids) < 4:
            raise ValueError("each domain requires at least four distinct intent groups")
        rng.shuffle(group_ids)
        test_ids = set(group_ids[: max(1, len(group_ids) * 3 // 10)])
        assignment.update({i: "test" if i in test_ids else "development" for i in group_ids})
    if judgments:
        for query_id in ids:
            if not any(
                reviewed[query_id, document_id]["grade"] > 0
                for document_id in ids
                if assignment[document_id] == assignment[query_id]
            ):
                raise ValueError("authored query has no reviewed positive in its reserved split")
    output.mkdir(parents=True)
    files = {}
    for split in ("development", "test"):
        subset = [r for r in records if assignment[r["id"]] == split]
        for query_language in LANGUAGES:
            for document_language in LANGUAGES:
                direction = f"{query_language}-{document_language}"
                destination = output / split / direction
                destination.mkdir(parents=True)
                documents, queries, qrels = [], [], []
                for row in subset:
                    group = f"agent-domain:{row['id']}"
                    doc_id = f"{group}:{document_language}"
                    query_id = f"{group}:query:{query_language}"
                    provenance = (
                        "agent-authored controlled reference; fictional operational examples"
                    )
                    documents.append(
                        {
                            "id": doc_id,
                            "group": group,
                            "text": normalize(row[document_language]),
                            "source": f"agent-domain:{row['id']}",
                            "domain": row["domain"],
                            "language": LANGUAGES[document_language],
                            "attribution": provenance,
                        }
                    )
                    queries.append(
                        {
                            "id": query_id,
                            "group": group,
                            "text": normalize(row[f"q_{query_language}"]),
                            "source": f"agent-domain:{row['id']}",
                            "domain": row["domain"],
                            "bucket": LANGUAGES[query_language],
                            "target_language": LANGUAGES[document_language],
                            "split": split,
                            "author": "implementation-agent",
                            "author_origin": "agent",
                            "attribution": provenance,
                        }
                    )
                    qrels.append(
                        {
                            "query_id": query_id,
                            "document_id": doc_id,
                            "grade": 2,
                            "reviewer_id": "implementation-agent",
                            "label_origin": "agent_reference_answer",
                            "rationale": (
                                "The reference passage answers this authored information need."
                            ),
                        }
                    )
                if judgments:
                    qrels = []
                    for query in queries:
                        for document in documents:
                            label = reviewed[
                                query["group"].split(":", 1)[1], document["group"].split(":", 1)[1]
                            ]
                            qrels.append(
                                {
                                    "query_id": query["id"],
                                    "document_id": document["id"],
                                    "grade": label["grade"],
                                    "reviewer_id": label["reviewer_id"],
                                    "label_origin": "agent_reviewed_intent_matrix",
                                    "rationale": label["rationale"],
                                }
                            )
                for name, rows in (
                    ("documents", documents),
                    ("queries", queries),
                    ("qrels", qrels),
                ):
                    write_rows(destination / f"{name}.jsonl", rows)
                manifest = {
                    "task": "agent-authored domain QA retrieval proxy",
                    "split": split,
                    "direction": direction,
                    "queries": len(queries),
                    "documents": len(documents),
                    "independent_groups": len(subset),
                    "seed": seed,
                    "domains": dict(Counter(r["domain"] for r in subset)),
                    "label_status": (
                        "agent-authored; all candidate intent pairs explicitly agent-reviewed"
                        if judgments
                        else "agent-authored and agent-reviewed reference-answer labels"
                    ),
                    "human_review_completed": False,
                    "independent_review_verified": False,
                    "training_canonical_overlap": 0,
                    "frozen_project_test_opened": False,
                    "unjudged_documents": (
                        "none within this controlled candidate set"
                        if judgments
                        else "assumed unrelated by authored intent; pool new retrieved alternatives"
                    ),
                    "limitations": [
                        "small controlled corpus",
                        "single-agent author and reference judge",
                        "no naturally collected search traffic",
                        "canonical overlap audit only",
                    ],
                    "inputs": {str(material): sha256(material), str(training): sha256(training)},
                    "files": {
                        f"{n}.jsonl": sha256(destination / f"{n}.jsonl")
                        for n in ("documents", "queries", "qrels")
                    },
                }
                if judgments:
                    manifest["inputs"][str(judgments)] = sha256(judgments)
                    review_manifest = judgments.with_suffix(".manifest.json")
                    manifest["inputs"][str(review_manifest)] = sha256(review_manifest)
                    manifest["judgments"] = len(qrels)
                    manifest["judgment_projection"] = (
                        "shared reviewed intent meanings projected across authored parallel forms; "
                        "single author/judge, no independent language review"
                    )
                (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                files[f"{split}/{direction}/manifest.json"] = sha256(destination / "manifest.json")
    result = {
        "material_sha256": sha256(material),
        "seed": seed,
        "intent_groups": len(records),
        "split_by_intent": assignment,
        "directions": 9,
        "domains": sorted(DOMAINS),
        "training_canonical_overlap": 0,
        "files": files,
        "test_policy": "do not evaluate test rankings until model selection is frozen",
        "provenance": "user-authorized agent-authored and agent-judged proxy, not human data",
        "intent_judgments_sha256": sha256(judgments) if judgments else None,
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--material", type=Path, default=Path("experiments/retrieval/domain_material.jsonl")
    )
    parser.add_argument("--training", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/domain-benchmark-v1"))
    parser.add_argument("--judgments", type=Path, help="complete agent-reviewed intent matrix")
    args = parser.parse_args()
    result = build(args.material, args.training, args.output, judgments=args.judgments)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
