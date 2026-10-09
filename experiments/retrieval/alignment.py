"""Recover clean bilingual training pairs from checksum-locked source records."""

import argparse
import csv
import io
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

from prepare import canonical, normalize, read_rows, sha256, write_rows


def build(
    data: Path, output: Path, count: int = 6000, seed: int = 17, expanded: Path | None = None
) -> dict:
    if output.exists():
        raise ValueError("alignment output exists; use a new version")
    if count < 2:
        raise ValueError("require at least two training pairs")
    paths = [
        data / "raw/banglatlit/train.csv",
        data / "raw/banglatlit/validation.csv",
        data / "raw/tatoeba/ben-eng.zip",
    ]
    locked = {r["path"]: r for r in json.loads((data / "sources.lock.json").read_text())}
    for path in paths:
        record = locked[str(path.relative_to(data / "raw"))]
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"source checksum mismatch: {path}")
    processed = data / "processed" if expanded is None else expanded / "prepared"
    training = processed / "train.jsonl"
    by_text = {}
    training_groups = set()
    with training.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] != "train":
                raise ValueError("training records must be train-only")
            training_groups.add(row["group"])
            if row["source"] in ("banglatlit", "tatoeba-ben", "vashantor"):
                by_text[normalize(row["text"])] = row
    # Validation includes connected groups reserved for the pilot. No frozen test
    # text is read: saved train membership excludes test groups by construction.
    validation = processed / "validation.jsonl"
    heldout_groups, heldout_keys = set(), set()
    with validation.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            heldout_groups.add(row["group"])
            heldout_keys.add(canonical(row["text"]))
    if training_groups & heldout_groups:
        raise ValueError("saved training and validation groups overlap")
    pools: dict[str, list[dict]] = {"banglatlit": [], "tatoeba-ben": []}
    dropped = Counter()

    def add(query: str, document: str, source: str, raw_id: str, language: str) -> None:
        query, document = normalize(query), normalize(document)
        q, d = by_text.get(query), by_text.get(document)
        if not q or not d or q["group"] != d["group"]:
            dropped["not_same_training_group"] += 1
            return
        allowed_query_buckets = (
            {"banglish", "code-mixed"} if language == "banglish" else {"english"}
        )
        if q["bucket"] not in allowed_query_buckets or d["bucket"] != "bangla":
            dropped["unexpected_language_bucket"] += 1
            return
        if canonical(query) == canonical(document):
            dropped["identity_pair"] += 1
            return
        if len(query.split()) < 3 or len(document.split()) < 3:
            dropped["short_pair"] += 1
            return
        if q["group"] in heldout_groups or {canonical(query), canonical(document)} & heldout_keys:
            raise ValueError("alignment pair overlaps validation")
        pools[source].append(
            {
                "group": q["group"],
                "query": query,
                "document": document,
                "query_language": language,
                "document_language": "bangla",
                "source": source,
                "raw_id": raw_id,
                "attribution": q["attribution"],
                "pair_origin": "original_source_pair; no generated romanization",
            }
        )

    for path in paths[:2]:
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                add(
                    row["text_transliterated"],
                    row["text_bengali"],
                    "banglatlit",
                    row["id"],
                    "banglish",
                )
    with zipfile.ZipFile(paths[2]) as archive:
        with archive.open("ben.txt") as raw:
            for i, line in enumerate(io.TextIOWrapper(raw, encoding="utf-8")):
                english, bengali, _ = line.rstrip("\n").split("\t", 2)
                add(english, bengali, "tatoeba-ben", str(i), "english")
    expansion_inputs = []
    if expanded is not None:
        expansion_manifest = json.loads((expanded / "manifest.json").read_text())
        for name in ("original-pairs.jsonl", "prepared/train.jsonl", "prepared/validation.jsonl"):
            if sha256(expanded / name) != expansion_manifest["files"][name]:
                raise ValueError("expanded source checksum mismatch")
        pools["vashantor"] = []
        original_pairs = expanded / "original-pairs.jsonl"
        expansion_inputs = [expanded / "manifest.json", original_pairs]
        for row in read_rows(original_pairs):
            if row["split"] != "train":
                continue
            forms = row["forms"]
            for query_field, document_field, language in (
                ("banglish", "bangla", "banglish"),
                ("regional_banglish", "regional_bangla", "banglish"),
                ("english", "bangla", "english"),
            ):
                add(
                    forms[query_field],
                    forms[document_field],
                    "vashantor",
                    f"{row['raw_id']}:{query_field}",
                    language,
                )
    rng = random.Random(seed)
    chosen, groups, keys = [], set(), set()
    # Include English alignment explicitly rather than giving all capacity to
    # Banglish. Limited supply is reported; it is never filled with synthetic pairs.
    vashantor_quota = min(1600, count // 5) if expanded is not None else 0
    remaining = count - vashantor_quota
    quotas = {"tatoeba-ben": remaining // 4, "banglatlit": remaining - remaining // 4}
    if expanded is not None:
        quotas["vashantor"] = vashantor_quota
    for source in (["vashantor"] if expanded is not None else []) + ["tatoeba-ben", "banglatlit"]:
        rng.shuffle(pools[source])
        taken = 0
        for row in pools[source]:
            pair_keys = {canonical(row["query"]), canonical(row["document"])}
            if row["group"] in groups or pair_keys & keys:
                continue
            if taken >= quotas[source]:
                break
            chosen.append(row)
            groups.add(row["group"])
            keys.update(pair_keys)
            taken += 1
    if len(chosen) < 2:
        raise ValueError("not enough clean alignment pairs")
    rng.shuffle(chosen)
    output.mkdir(parents=True)
    write_rows(output / "pairs.jsonl", chosen)
    adaptation = [
        {"text": row[field], "group": row["group"], "source": row["source"]}
        for row in chosen
        for field in ("query", "document")
    ]
    write_rows(output / "adaptation.jsonl", adaptation)
    result = {
        "seed": seed,
        "requested_pairs": count,
        "pairs": len(chosen),
        "pairs_by_source": dict(Counter(r["source"] for r in chosen)),
        "pairs_by_query_language": dict(Counter(r["query_language"] for r in chosen)),
        "candidate_pairs_by_source": {s: len(rows) for s, rows in pools.items()},
        "dropped": dict(dropped),
        "unique_groups": len(groups),
        "validation_canonical_or_group_overlap": 0,
        "frozen_test_opened": False,
        "adaptation_scope": "only the exact same clean training groups; full PT corpus not used",
        "limitations": "group/lexical deduplication cannot eliminate all semantic false negatives",
        "inputs": {
            str(p): sha256(p)
            for p in [training, validation, *paths, data / "sources.lock.json", *expansion_inputs]
        },
        "files": {f"{n}.jsonl": sha256(output / f"{n}.jsonl") for n in ("pairs", "adaptation")},
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/alignment-v1"))
    parser.add_argument("--pairs", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--expanded", type=Path, help="audited Vashantor expansion directory")
    args = parser.parse_args()
    print(json.dumps(build(args.data, args.output, args.pairs, args.seed, args.expanded), indent=2))
