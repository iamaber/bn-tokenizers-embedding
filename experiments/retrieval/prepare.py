"""Prepare a development-only paired-sentence retrieval pilot from verified local pairs."""

import argparse
import csv
import hashlib
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def canonical(text: str) -> str:
    return "".join(c.lower() for c in normalize(text) if unicodedata.category(c)[0] in "LMN")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def prepare(root: Path, output: Path, count: int, documents: int, seed: int) -> dict:
    if count < 1 or documents < count:
        raise ValueError("require documents >= queries > 0")
    if output.exists():
        raise ValueError("output already exists; use a new directory to preserve annotations")
    validation = root / "processed/validation.jsonl"
    rows = read_rows(validation)
    if any(row["split"] != "validation" for row in rows):
        raise ValueError("pilot input must contain validation records only")
    by_text = {normalize(row["text"]): row for row in rows if row["source"] == "banglatlit"}
    pairs = []
    raw_paths = [root / f"raw/banglatlit/{split}.csv" for split in ("train", "validation")]
    # Connected corpus groups can contain multiple original pairs. Recover exact
    # pairs from the raw CSV rather than cross-joining texts sharing a group.
    for path in raw_paths:
        with path.open() as stream:
            for raw in csv.DictReader(stream):
                roman = normalize(raw["text_transliterated"])
                bengali = normalize(raw["text_bengali"])
                q, d = by_text.get(roman), by_text.get(bengali)
                if not q or not d or q["group"] != d["group"]:
                    continue
                if q["bucket"] not in ("banglish", "code-mixed") or d["bucket"] != "bangla":
                    continue
                if len(roman.split()) < 3 or len(bengali.split()) < 3:
                    continue
                pairs.append((raw["id"], q, d))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    selected = []
    groups: set[str] = set()
    keys: set[str] = set()
    for pair in pairs:
        _, query, document = pair
        pair_keys = {canonical(query["text"]), canonical(document["text"])}
        if query["group"] in groups or keys & pair_keys:
            continue
        groups.add(query["group"])
        keys.update(pair_keys)
        selected.append(pair)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} eligible distinct groups")
    candidates = {normalize(doc["text"]): doc for _, _, doc in selected}
    doc_keys = {canonical(text) for text in candidates}
    pool = [r for r in rows if r["source"] == "banglatlit" and r["bucket"] == "bangla"]
    rng.shuffle(pool)
    for row in pool:
        if len(candidates) >= documents:
            break
        text = normalize(row["text"])
        key = canonical(text)
        if key and key not in doc_keys:
            candidates[text] = row
            doc_keys.add(key)
    if len(candidates) != documents:
        raise ValueError(f"only {len(candidates)} distinct candidate documents")
    all_keys = keys | doc_keys
    all_groups = groups | {r["group"] for r in candidates.values()}
    training = root / "processed/train.jsonl"
    with training.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] != "train":
                raise ValueError("unexpected training split")
            if row["group"] in all_groups or canonical(row["text"]) in all_keys:
                raise ValueError("pilot overlaps tokenizer training by group or canonical text")
    docs = [
        {
            "id": hashlib.sha256(text.encode()).hexdigest(),
            "text": text,
            "group": row["group"],
            "source": row["source"],
            "attribution": row["attribution"],
        }
        for text, row in candidates.items()
    ]
    docs.sort(key=lambda row: row["id"])
    doc_ids = {d["text"]: d["id"] for d in docs}
    queries, qrels, authoring = [], [], []
    for raw_id, q, d in selected:
        query_id = f"banglatlit:{raw_id}"
        doc_id = doc_ids[normalize(d["text"])]
        queries.append(
            {
                "id": query_id,
                "text": normalize(q["text"]),
                "group": q["group"],
                "bucket": q["bucket"],
                "split": "pilot-development",
                "source": q["source"],
                "attribution": q["attribution"],
            }
        )
        qrels.append(
            {
                "query_id": query_id,
                "document_id": doc_id,
                "grade": 1,
                "label_origin": "source_transliteration_pair_unreviewed",
            }
        )
        authoring.append(
            {
                "query_id": query_id,
                "group": q["group"],
                "source_banglish": normalize(q["text"]),
                "paired_bengali": normalize(d["text"]),
                "natural_bengali_query": None,
                "banglish_paraphrase": None,
                "mixed_query": None,
                "pair_correct": None,
                "privacy_review_passed": None,
                "reviewer": None,
            }
        )
    output.mkdir(parents=True)
    for name, records in [
        ("documents", docs),
        ("queries", queries),
        ("qrels", qrels),
        ("authoring", authoring),
    ]:
        write_rows(output / f"{name}.jsonl", records)
    manifest = {
        "task": "paired-sentence retrieval; development proxy, not human search relevance",
        "seed": seed,
        "queries": len(queries),
        "documents": len(docs),
        "buckets": dict(Counter(q["bucket"] for q in queries)),
        "independent_groups": len(groups),
        "training_canonical_or_group_overlap": 0,
        "frozen_test_opened": False,
        "human_review_completed": False,
        "unjudged_documents": "unknown relevance; treated as zero for provisional metrics",
        "inputs": {str(path): sha256(path) for path in [training, validation, *raw_paths]},
        "files": {
            f"{name}.jsonl": sha256(output / f"{name}.jsonl")
            for name in ("documents", "queries", "qrels")
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/retrieval-pilot-v1"))
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--documents", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.data, args.output, args.queries, args.documents, args.seed), indent=2
        )
    )
