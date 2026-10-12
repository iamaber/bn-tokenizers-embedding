"""Extend reviewed sentence labels with checksum-bound adapted-system candidates."""

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from annotation_archive import GRADE_FIELDS
from prepare import read_rows, sha256, verify_manifest, write_rows

EXPECTED_MODELS = {"unchanged"} | {
    f"{budget}/{arm}-seed{seed}"
    for budget in ("equal-data", "equal-compute")
    for arm in ("original", "unigram-32000", "bpe-32000")
    for seed in (17, 29, 43)
}


def build(source: Path, suite: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("pool already exists; preserve versions")
    verify_manifest(source)
    specification = json.loads(suite.read_text())
    run_names = [r["name"] for r in specification["runs"]]
    if len(run_names) != len(EXPECTED_MODELS) or set(run_names) != EXPECTED_MODELS:
        raise ValueError("pool requires exactly the declared 19 models")
    if specification["datasets"][source.name] != sha256(source / "manifest.json"):
        raise ValueError("suite uses a different reviewed dataset")
    queries = {r["id"]: r for r in read_rows(source / "queries.jsonl")}
    documents = {r["id"]: r for r in read_rows(source / "documents.jsonl")}
    previous = read_rows(source / "qrels.jsonl")
    known = {(r["query_id"], r["document_id"]) for r in previous}
    candidates, ranking_hashes = set(), {}
    names = set()
    for run in specification["runs"]:
        if run["name"] in names:
            raise ValueError("duplicate suite model")
        names.add(run["name"])
        directory = Path(run["directory"])
        report_path = directory / "report.json"
        if sha256(report_path) != run["report_sha256"]:
            raise ValueError("model report checksum mismatch")
        report = json.loads(report_path.read_text())["results"][source.name]
        ranking_path = directory / source.name / "rankings.jsonl"
        digest = sha256(ranking_path)
        if (report["dataset_manifest_sha256"], report["ranking_sha256"]) != (
            sha256(source / "manifest.json"), digest
        ):
            raise ValueError("ranking or dataset checksum mismatch")
        rows = read_rows(ranking_path)
        if len(rows) != len(queries) or {r["query_id"] for r in rows} != queries.keys():
            raise ValueError("rankings must cover each query exactly once")
        for row in rows:
            ids = row["documents"]
            if len(ids) != 10 or len(set(ids)) != 10 or not set(ids) <= documents.keys():
                raise ValueError("require ten distinct known candidates per query")
            candidates.update((row["query_id"], did) for did in ids)
        ranking_hashes[str(ranking_path)] = digest
    packet = [
        {"query_id": qid, "document_id": did, "query": queries[qid]["text"],
         "document": documents[did]["text"], "group": queries[qid]["group"],
         "grade": None, "rationale": None, "reviewer_id": None, "reviewer_origin": None}
        for qid, did in sorted(candidates - known)
    ]
    random.Random(20261009).shuffle(packet)
    manifest = {
        "source": str(source), "source_manifest_sha256": sha256(source / "manifest.json"),
        "suite": str(suite), "suite_sha256": sha256(suite),
        "ranking_files": ranking_hashes, "models": sorted(names),
        "queries": len(queries), "documents": len(documents),
        "previous_judgments": len(previous), "new_pairs": len(packet),
        "top10_union_pairs": len(candidates),
        "rubric": {"2": "same essential meaning", "1": "related but partly different",
                   "0": "unrelated or contradicts; preserve negation, names and numbers"},
        "blindness": "packet hides system names and ranks; implementation agent knows experiment",
    }
    output.mkdir(parents=True)
    write_rows(output / "packet.jsonl", packet)
    manifest["packet_sha256"] = sha256(output / "packet.jsonl")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def import_labels(pool: Path, labels: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("reviewed dataset exists; use a new version")
    manifest = json.loads((pool / "manifest.json").read_text())
    source = Path(manifest["source"])
    original = verify_manifest(source)
    if sha256(source / "manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("source dataset changed")
    if sha256(pool / "packet.jsonl") != manifest["packet_sha256"]:
        raise ValueError("review packet changed")
    if sha256(Path(manifest["suite"])) != manifest["suite_sha256"] or any(
        sha256(Path(path)) != digest for path, digest in manifest["ranking_files"].items()
    ):
        raise ValueError("candidate provenance changed")
    expected = {(r["query_id"], r["document_id"]): r for r in read_rows(pool / "packet.jsonl")}
    judged = {}
    for row in read_rows(labels):
        key = row["query_id"], row["document_id"]
        if key not in expected or key in judged or set(row) != set(GRADE_FIELDS):
            raise ValueError("duplicate, unknown or non-text-free label")
        if type(row["grade"]) is not int or row["grade"] not in (0, 1, 2):
            raise ValueError("unresolved or invalid grade")
        if any(not isinstance(row[field], str) or not row[field].strip()
               for field in ("rationale", "reviewer_id")):
            raise ValueError("rationale and reviewer required")
        if row["reviewer_origin"] not in ("agent", "human"):
            raise ValueError("explicit reviewer origin required")
        judged[key] = row
    if judged.keys() != expected.keys():
        raise ValueError("every new candidate must be judged")
    previous = read_rows(source / "qrels.jsonl")
    additions = [
        {"query_id": row["query_id"], "document_id": row["document_id"],
         "grade": row["grade"], "rationale": row["rationale"],
         "label_origin": row["reviewer_origin"] + "_reviewed",
         "reviewer_id": row["reviewer_id"]}
        for _, row in sorted(judged.items())
    ]
    output.mkdir(parents=True)
    for name in ("queries.jsonl", "documents.jsonl"):
        shutil.copyfile(source / name, output / name)
    write_rows(output / "qrels.jsonl", previous + additions)
    result = {
        **original, "label_status": "agent-reviewed original and adapted-system top10 union",
        "previous_manifest_sha256": manifest["source_manifest_sha256"],
        "pool_manifest_sha256": sha256(pool / "manifest.json"),
        "new_labels_sha256": sha256(labels),
        "new_grade_counts": dict(Counter(r["grade"] for r in additions)),
        "previous_labels_preserved": len(previous),
        "new_labels": len(additions), "pooled_system_top10_coverage": 1.0,
        "reviewer_ids": sorted({r["reviewer_id"] for r in previous + additions if r.get("reviewer_id")}),
        "judgment_origins": dict(Counter(r.get("label_origin", "unknown").split("_", 1)[0]
                                         for r in previous + additions)),
        "human_review_completed": False, "independent_review_verified": False,
        "files": {name: sha256(output / name) for name in (
            "queries.jsonl", "documents.jsonl", "qrels.jsonl"
        )},
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "import"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        if not args.source or not args.suite:
            parser.error("build requires --source and --suite")
        result = build(args.source, args.suite, args.output)
    else:
        if not args.pool or not args.labels:
            parser.error("import requires --pool and --labels")
        result = import_labels(args.pool, args.labels, args.output)
    print(json.dumps(result, indent=2))
