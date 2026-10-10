"""Mine TRAIN-only near neighbours, then freeze explicitly reviewed negative pairs."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
from encoder import Encoder
from prepare import canonical, read_rows, sha256, write_rows

REVIEW_FIELDS = {"group", "negative_group", "pair_correct", "negative_grade",
                "rationale", "reviewer_id", "reviewer_origin"}


def mine(base: Path, alignment: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("mining packet exists; preserve versions")
    manifest = json.loads((alignment / "manifest.json").read_text())
    if sha256(alignment / "pairs.jsonl") != manifest["files"]["pairs.jsonl"]:
        raise ValueError("alignment snapshot changed")
    pairs = read_rows(alignment / "pairs.jsonl")
    if len({row["group"] for row in pairs}) != len(pairs):
        raise ValueError("duplicate alignment groups")
    rng = random.Random(29)
    selected = []
    for language in ("banglish", "english"):
        candidates = sorted([row for row in pairs if row["query_language"] == language], key=lambda r: r["group"])
        if len(candidates) < 64:
            raise ValueError("require 64 training pairs per query language")
        selected.extend(rng.sample(candidates, 64))
    rng.shuffle(selected)
    torch.set_num_threads(4)
    encoder = Encoder(base, max_length=128)
    try:
        documents, _ = encoder.encode(["passage: " + row["document"] for row in pairs])
        queries, _ = encoder.encode(["query: " + row["query"] for row in selected])
        with torch.inference_mode():
            orders = (queries @ documents.T).argsort(descending=True, stable=True).tolist()
        rows = []
        for row, order in zip(selected, orders, strict=True):
            negative = next(pairs[i] for i in order if pairs[i]["group"] != row["group"]
                            and canonical(pairs[i]["document"]) != canonical(row["document"]))
            rows.append({**row, "negative": negative["document"], "negative_group": negative["group"],
                         "pair_correct": None, "negative_grade": None, "rationale": None,
                         "reviewer_id": None, "reviewer_origin": None})
    finally:
        encoder.close()
    output.mkdir(parents=True)
    write_rows(output / "packet.jsonl", rows)
    result = {"alignment": str(alignment), "alignment_manifest_sha256": sha256(alignment / "manifest.json"),
              "packet_sha256": sha256(output / "packet.jsonl"), "queries": len(rows),
              "mining": "pinned unchanged E5 nearest different-group training document",
              "base_files": {p.name: sha256(p) for p in base.iterdir() if p.is_file()},
              "holdout_used": False, "implementation_sha256": sha256(Path(__file__))}
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def freeze(packet: Path, judgments: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("reviewed negative bank exists; preserve versions")
    manifest = json.loads((packet / "manifest.json").read_text())
    if sha256(packet / "packet.jsonl") != manifest["packet_sha256"]:
        raise ValueError("mining packet changed")
    alignment = Path(manifest["alignment"])
    if sha256(alignment / "manifest.json") != manifest["alignment_manifest_sha256"]:
        raise ValueError("alignment snapshot changed")
    rows = read_rows(packet / "packet.jsonl")
    expected = {row["group"]: row for row in rows}
    reviewed = {}
    for row in read_rows(judgments):
        if set(row) != REVIEW_FIELDS:
            raise ValueError("judgments must use the exact text-free review schema")
        group = row["group"]
        if group not in expected or group in reviewed or row["negative_group"] != expected[group]["negative_group"]:
            raise ValueError("duplicate or changed negative pair")
        if type(row["pair_correct"]) is not bool or type(row["negative_grade"]) is not int or row["negative_grade"] not in (0, 1, 2):
            raise ValueError("explicit pair correctness and negative relevance required")
        if row["reviewer_origin"] not in ("human", "agent") or any(
            not isinstance(row[field], str) or not row[field].strip() for field in ("reviewer_id", "rationale")
        ):
            raise ValueError("explicit reviewer and rationale required")
        reviewed[group] = row
    if reviewed.keys() != expected.keys():
        raise ValueError("every mined pair must be reviewed")
    accepted = [
        {**expected[group], **decision} for group, decision in sorted(reviewed.items())
        if decision["pair_correct"] and decision["negative_grade"] == 0
    ]
    counts = Counter(row["query_language"] for row in accepted)
    if any(counts[language] < 16 for language in ("banglish", "english")):
        raise ValueError("require at least 16 verified negatives per language")
    output.mkdir(parents=True)
    write_rows(output / "pairs.jsonl", accepted)
    result = {"mining_manifest_sha256": sha256(packet / "manifest.json"),
              "alignment_manifest_sha256": manifest["alignment_manifest_sha256"],
              "judgments_sha256": sha256(judgments), "accepted": len(accepted),
              "accepted_by_language": dict(counts), "excluded": len(rows) - len(accepted),
              "label_origin": "explicit single-agent review; not independent human judgments",
              "files": {"pairs.jsonl": sha256(output / "pairs.jsonl")}}
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def clean_alignment(alignment: Path, bank: Path, judgments: Path, output: Path) -> dict:
    """Remove positively reviewed pair errors from a new alignment version."""
    if output.exists():
        raise ValueError("clean alignment exists; preserve versions")
    previous = json.loads((alignment / "manifest.json").read_text())
    reviewed = json.loads((bank / "manifest.json").read_text())
    if reviewed["alignment_manifest_sha256"] != sha256(alignment / "manifest.json") or (
        reviewed["judgments_sha256"] != sha256(judgments)
    ):
        raise ValueError("review does not bind this alignment and judgments")
    for directory, manifest in ((alignment, previous), (bank, reviewed)):
        if any(sha256(directory / name) != digest for name, digest in manifest["files"].items()):
            raise ValueError("reviewed training input changed")
    pairs = read_rows(alignment / "pairs.jsonl")
    decisions = read_rows(judgments)
    ids = {row["group"] for row in pairs}
    if any(row["group"] not in ids or type(row["pair_correct"]) is not bool for row in decisions):
        raise ValueError("invalid alignment pair decision")
    excluded = {row["group"] for row in decisions if not row["pair_correct"]}
    kept = [row for row in pairs if row["group"] not in excluded]
    output.mkdir(parents=True)
    write_rows(output / "pairs.jsonl", kept)
    write_rows(output / "adaptation.jsonl", [
        {"text": row[field], "group": row["group"], "source": row["source"]}
        for row in kept for field in ("query", "document")
    ])
    result = {**previous, "pairs": len(kept), "unique_groups": len(kept),
              "parent_alignment_manifest_sha256": sha256(alignment / "manifest.json"),
              "negative_bank_manifest_sha256": sha256(bank / "manifest.json"),
              "reviewed_judgments_sha256": sha256(judgments),
              "known_pair_errors_removed": len(excluded),
              "pairs_by_source": dict(Counter(row["source"] for row in kept)),
              "pairs_by_query_language": dict(Counter(row["query_language"] for row in kept)),
              "review_scope": "128 sampled pairs reviewed; remaining unsampled pair fidelity unverified",
              "files": {n: sha256(output / n) for n in ("pairs.jsonl", "adaptation.jsonl")}}
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("mine", "freeze", "clean-alignment"))
    parser.add_argument("--base", type=Path)
    parser.add_argument("--alignment", type=Path)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "mine":
        if not args.base or not args.alignment:
            parser.error("mine requires --base and --alignment")
        result = mine(args.base, args.alignment, args.output)
    elif args.command == "freeze":
        if not args.packet or not args.judgments:
            parser.error("freeze requires --packet and --judgments")
        result = freeze(args.packet, args.judgments, args.output)
    else:
        if not args.alignment or not args.bank or not args.judgments:
            parser.error("clean-alignment requires --alignment, --bank and --judgments")
        result = clean_alignment(args.alignment, args.bank, args.judgments, args.output)
    print(json.dumps(result, indent=2))
