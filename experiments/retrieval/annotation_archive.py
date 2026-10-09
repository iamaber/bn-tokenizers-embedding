"""Archive judgments without corpus text, and restore against the exact frozen packet."""

import argparse
import json
from pathlib import Path

from prepare import read_rows, sha256, verify_manifest, write_rows
from review import export

GRADE_FIELDS = ("query_id", "document_id", "grade", "rationale", "reviewer_id", "reviewer_origin")
PAIR_FIELDS = (
    "query_id",
    "pair_correct",
    "privacy_review_passed",
    "reviewer",
    "reviewer_origin",
    "exclusion_reason",
    "review_notes",
)
SOURCE_FILES = ("manifest.json", "review-pool.jsonl", "authoring.jsonl")


def archive(pilot: Path, judgments: Path, pairs: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("annotation archive exists; preserve reviewed versions")
    verify_manifest(pilot)
    pool = {(r["query_id"], r["document_id"]): r for r in read_rows(pilot / "review-pool.jsonl")}
    original_pairs = {r["query_id"]: r for r in read_rows(pilot / "authoring.jsonl")}
    labels, decisions = [], []
    seen_labels, seen_pairs = set(), set()
    for row in read_rows(judgments):
        if row.get("grade") is None:
            continue
        key = row["query_id"], row["document_id"]
        if key not in pool or key in seen_labels:
            raise ValueError("unknown or duplicate judgment")
        if any(row[k] != pool[key][k] for k in ("group", "query", "document")):
            raise ValueError("judgment changed frozen text/group")
        if (
            type(row["grade"]) is not int
            or row["grade"] not in (0, 1, 2)
            or not row.get("rationale")
            or not row.get("reviewer_id")
            or row.get("reviewer_origin") not in ("agent", "human")
        ):
            raise ValueError("invalid judgment/provenance")
        labels.append({k: row[k] for k in GRADE_FIELDS})
        seen_labels.add(key)
    for row in read_rows(pairs):
        if row.get("pair_correct") is None:
            continue
        key = row["query_id"]
        if key not in original_pairs or key in seen_pairs:
            raise ValueError("unknown or duplicate pair review")
        if any(
            row[k] != original_pairs[key][k] for k in ("group", "source_banglish", "paired_bengali")
        ):
            raise ValueError("pair review changed frozen source")
        if (
            any(type(row.get(k)) is not bool for k in ("pair_correct", "privacy_review_passed"))
            or not row.get("reviewer")
            or row.get("reviewer_origin") not in ("agent", "human")
        ):
            raise ValueError("invalid pair decision/provenance")
        if not all(row[k] for k in ("pair_correct", "privacy_review_passed")) and not row.get(
            "exclusion_reason"
        ):
            raise ValueError("excluded pair requires a reason")
        decisions.append({k: row[k] for k in PAIR_FIELDS if k in row})
        seen_pairs.add(key)
    output.mkdir(parents=True)
    write_rows(output / "judgments.jsonl", labels)
    write_rows(output / "pairs.jsonl", decisions)
    manifest = {
        "sources": {name: sha256(pilot / name) for name in SOURCE_FILES},
        "files": {name: sha256(output / name) for name in ("judgments.jsonl", "pairs.jsonl")},
        "judgments": len(labels),
        "pairs": len(decisions),
        "annotation_coverage_complete": len(labels) == len(pool)
        and len(decisions) == len(original_pairs),
        "scope": "text-free annotation archive; final import still validates labels and exclusions",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def restore(pilot: Path, directory: Path, output: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["sources"] != {name: sha256(pilot / name) for name in SOURCE_FILES}:
        raise ValueError("annotation archive targets a different frozen pilot")
    if manifest["files"] != {
        name: sha256(directory / name) for name in ("judgments.jsonl", "pairs.jsonl")
    }:
        raise ValueError("annotation archive checksum mismatch")
    label_rows = read_rows(directory / "judgments.jsonl")
    pair_rows = read_rows(directory / "pairs.jsonl")
    labels = {(r["query_id"], r["document_id"]): r for r in label_rows}
    decisions = {r["query_id"]: r for r in pair_rows}
    pool_keys = {(r["query_id"], r["document_id"]) for r in read_rows(pilot / "review-pool.jsonl")}
    pair_keys = {r["query_id"] for r in read_rows(pilot / "authoring.jsonl")}
    if len(labels) != len(label_rows) or not labels.keys() <= pool_keys:
        raise ValueError("unknown or duplicate archived judgment")
    if len(decisions) != len(pair_rows) or not decisions.keys() <= pair_keys:
        raise ValueError("unknown or duplicate archived pair review")
    if any(set(r) - set(GRADE_FIELDS) for r in label_rows) or any(
        set(r) - set(PAIR_FIELDS) for r in pair_rows
    ):
        raise ValueError("archive contains fields outside the text-free annotation schema")
    # Reconstruct text only from its original, checksum-bound local corpus packet.
    result = export(pilot, output)
    rows = read_rows(output / "reviewer-1.jsonl")
    for row in rows:
        label = labels.get((row["query_id"], row["document_id"]), {})
        row.update({k: label[k] for k in GRADE_FIELDS if k in label})
    write_rows(output / "reviewer-1.jsonl", rows)
    rows = read_rows(output / "pair-review.jsonl")
    for row in rows:
        decision = decisions.get(row["query_id"], {})
        row.update({k: decision[k] for k in PAIR_FIELDS if k in decision})
    write_rows(output / "pair-review.jsonl", rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("archive", "restore"))
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "archive":
        if not args.judgments or not args.pairs:
            parser.error("archive requires --judgments and --pairs")
        result = archive(args.pilot, args.judgments, args.pairs, args.output)
    else:
        if not args.archive:
            parser.error("restore requires --archive")
        result = restore(args.pilot, args.archive, args.output)
    print(json.dumps(result, indent=2))
