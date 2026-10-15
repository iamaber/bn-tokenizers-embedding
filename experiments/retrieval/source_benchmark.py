"""Freeze source-disjoint multilingual retrieval data with complete reviewed qrels."""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from domains import DOMAINS, LANGUAGES
from prepare import canonical, normalize, read_rows, sha256, write_rows


def material_rows(root: Path) -> list[dict]:
    return [r for domain in sorted(DOMAINS) for r in read_rows(root / f"{domain}.jsonl")]


def surfaces(row: dict) -> list[str]:
    return [row[f] for lang in LANGUAGES for f in (lang, f"q_{lang}")] + row["q_bl_variants"]


def validate(records: list[dict]) -> None:
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate intent IDs")
    if {r["domain"] for r in records} != DOMAINS:
        raise ValueError("require all six domains")
    groups, texts = {}, {}
    for row in records:
        if row["split"] not in ("development", "test"):
            raise ValueError("unknown split")
        group = row["source_group"]
        if group in groups and groups[group] != row["split"]:
            raise ValueError("source group crosses development/test")
        groups[group] = row["split"]
        if row.get("author_origin") != "agent" or not row["q_bl_variants"]:
            raise ValueError("explicit authorship and Banglish variants required")
        for text in surfaces(row):
            if not isinstance(text, str) or not canonical(text):
                raise ValueError("empty text")
            key = canonical(text)
            if key in texts and texts[key] != row["id"]:
                raise ValueError("duplicate text across intents")
            texts[key] = row["id"]
        for field in ("bl", "q_bl"):
            if re.search(r"[\u0980-\u09ff]", row[field]):
                raise ValueError("Banglish field contains Bengali script")
    for domain in DOMAINS:
        if {r["split"] for r in records if r["domain"] == domain} != {"development", "test"}:
            raise ValueError("each domain requires development and test sources")


def shingles(text: str) -> set[tuple[str, ...]]:
    words = [canonical(w) for w in normalize(text).split()]
    return {tuple(words[i:i + 8]) for i in range(len(words) - 7)}


def audit_overlap(records: list[dict], paths: list[Path]) -> dict:
    for path in paths:
        if not path.is_file():
            raise ValueError(f"missing overlap-audit input: {path}")
    exact, inverted, sizes = {}, defaultdict(set), {}
    for row in records:
        for index, text in enumerate(surfaces(row)):
            key = (row["id"], index)
            exact[canonical(text)] = key
            grams = shingles(text)
            sizes[key] = len(grams)
            for gram in grams:
                inverted[gram].add(key)
    hits, counts = [], {}
    for path in paths:
        count = 0
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                for field in ("text", "query", "document", "negative"):
                    if field not in row:
                        continue
                    text = row[field]
                    count += 1
                    if canonical(text) in exact:
                        hits.append({"file": str(path), "row": count, "kind": "canonical",
                                     "intent": exact[canonical(text)][0]})
                    shared = Counter(k for gram in shingles(text) for k in inverted.get(gram, ()))
                    for key, matches in shared.items():
                        if matches >= 3 and matches / sizes[key] >= .8:
                            hits.append({"file": str(path), "row": count,
                                         "kind": "eight-word-shingle-containment", "intent": key[0]})
        counts[str(path)] = {"sha256": sha256(path), "surfaces_checked": count}
    return {"inputs": counts, "matches": hits,
            "method": "canonical equality and >=80% benchmark eight-word shingle containment with >=3 shared shingles",
            "limits": "lexical checks only; cannot rule out semantic overlap or base-model pretraining exposure"}


def build(root: Path, overlap: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("benchmark exists; preserve frozen versions")
    records = material_rows(root)
    validate(records)
    hashes = {f"{d}.jsonl": sha256(root / f"{d}.jsonl") for d in sorted(DOMAINS)}
    audit = json.loads(overlap.read_text())
    if audit["material_sha256"] != hashes or audit["matches"]:
        raise ValueError("overlap audit is stale or contains unresolved matches")
    if any(sha256(Path(p)) != v["sha256"] for p, v in audit["inputs"].items()):
        raise ValueError("audited training/holdout data changed")
    sources = json.loads((root / "sources.json").read_text())
    for source in sources.values():
        if sha256(Path(source["file"])) != source["sha256"]:
            raise ValueError("pinned source changed")
    canonical_groups = {}
    for row in records:
        source = sources[row["source_id"]]
        identity = source["canonical_url"]
        if identity in canonical_groups and canonical_groups[identity] != row["split"]:
            raise ValueError("canonical source crosses splits")
        canonical_groups[identity] = row["split"]
    labels = {}
    for path in sorted(root.glob("review-*.json")):
        review = json.loads(path.read_text())
        if review["material_sha256"] != hashes or review.get("unresolved_issues", True):
            raise ValueError("review is stale or unresolved")
        if review["reviewer_origin"] != "agent":
            raise ValueError("agent review provenance required")
        for qid, detail in review["queries"].items():
            if qid in labels:
                raise ValueError("duplicate query review")
            labels[qid] = dict(detail, reviewer_id=review["reviewer_id"])
    if labels.keys() != {r["id"] for r in records}:
        raise ValueError("review must cover every intent")
    for row in records:
        detail = labels[row["id"]]
        candidates = {r["id"] for r in records if r["split"] == row["split"]}
        if (not detail["rationale"] or not detail["positive_grades"]
                or not detail["positive_grades"].keys() <= candidates
                or any(type(g) is not int or g not in (1, 2) for g in detail["positive_grades"].values())):
            raise ValueError("invalid positive review")
    output.mkdir(parents=True)
    files = {}
    for split in ("development", "test"):
        subset = [r for r in records if r["split"] == split]
        for qlang in LANGUAGES:
            for dlang in LANGUAGES:
                destination = output / split / f"{qlang}-{dlang}"
                destination.mkdir(parents=True)
                documents, queries, qrels = [], [], []
                for row in subset:
                    documents.append({"id": row["id"], "group": row["source_group"],
                                      "text": row[dlang], "domain": row["domain"],
                                      "language": LANGUAGES[dlang], "source": row["source_id"],
                                      "attribution": sources[row["source_id"]]["attribution"]})
                    variants = [row[f"q_{qlang}"]] + (row["q_bl_variants"] if qlang == "bl" else [])
                    for index, text in enumerate(variants):
                        qid = f"{row['id']}:{qlang}:{index}"
                        queries.append({"id": qid, "group": row["domain"],
                                        "source_group": row["source_group"],
                                        "intent": row["id"], "text": text, "domain": row["domain"],
                                        "bucket": LANGUAGES[qlang], "split": split,
                                        "variant": "canonical" if index == 0 else "spelling",
                                        "author_origin": "agent"})
                        detail = labels[row["id"]]
                        for doc in subset:
                            qrels.append({"query_id": qid, "document_id": doc["id"],
                                          "grade": detail["positive_grades"].get(doc["id"], 0),
                                          "reviewer_id": detail["reviewer_id"],
                                          "label_origin": "agent_reviewed_source_intent_matrix",
                                          "rationale": detail["rationale"]})
                for name, rows in (("documents", documents), ("queries", queries), ("qrels", qrels)):
                    write_rows(destination / f"{name}.jsonl", rows)
                manifest = {"task": "source-backed agent-authored multilingual retrieval",
                            "split": split, "direction": destination.name,
                            "queries": len(queries), "documents": len(documents),
                            "source_groups": len({r["source_group"] for r in subset}),
                            "metric_weighting": "equal domain weight; variants share their domain block",
                            "human_review_completed": False, "unjudged_candidates": 0,
                            "files": {f"{n}.jsonl": sha256(destination / f"{n}.jsonl")
                                      for n in ("documents", "queries", "qrels")}}
                (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                files[f"{split}/{destination.name}/manifest.json"] = sha256(destination / "manifest.json")
    result = {"intent_groups": len(records), "split_counts": dict(Counter(r["split"] for r in records)),
              "material_sha256": hashes, "sources_sha256": sha256(root / "sources.json"),
              "reviews": {p.name: sha256(p) for p in sorted(root.glob("review-*.json"))},
              "overlap_audit_sha256": sha256(overlap), "files": files,
              "implementation_sha256": sha256(Path(__file__)),
              "test_policy": "freeze development choice before reserved-test scoring",
              "limits": ["90 agent-authored intents; not the proposed 180-intent benchmark",
                         "source/topic shifts between splits and few independent source groups",
                         "uncertainty resamples six domain blocks and training seeds, not individual variants",
                         "agent-translated documents and synthetic spelling variants; not natural traffic",
                         "short source extracts/summaries; not a long-document benchmark",
                         "agent review only; lexical overlap audit cannot exclude semantic or pretraining overlap"]}
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "build"))
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, action="append", default=[])
    parser.add_argument("--overlap", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; preserve versions")
    if args.command == "audit":
        if not args.corpus:
            parser.error("audit requires corpus inputs")
        rows = material_rows(args.material)
        result = audit_overlap(rows, args.corpus)
        result["material_sha256"] = {f"{d}.jsonl": sha256(args.material / f"{d}.jsonl") for d in sorted(DOMAINS)}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"matches": len(result["matches"])}))
    else:
        if not args.overlap:
            parser.error("build requires overlap audit")
        print(json.dumps(build(args.material, args.overlap, args.output), indent=2))
