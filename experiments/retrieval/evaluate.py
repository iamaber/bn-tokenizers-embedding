"""Run BM25 and optionally a local, pinned multilingual E5 baseline on the pilot."""

import argparse
import json
import math
import platform
import random
import time
import unicodedata
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

from metrics import query_metrics, summarize
from prepare import read_rows, sha256, write_rows


def terms(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).lower()
    return "".join(c if unicodedata.category(c)[0] in "LMN" else " " for c in text).split()


def bm25(documents: list[dict], queries: list[dict]) -> tuple[list[list[str]], dict]:
    started = time.perf_counter()
    counts = [Counter(terms(doc["text"])) for doc in documents]
    lengths = [sum(row.values()) for row in counts]
    average = sum(lengths) / len(lengths)
    if not average:
        raise ValueError("empty document terms")
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i, row in enumerate(counts):
        for term, frequency in row.items():
            postings[term].append((i, frequency))
    index_seconds = time.perf_counter() - started
    started = time.perf_counter()
    rankings = []
    for query in queries:
        scores: dict[int, float] = defaultdict(float)
        for term in sorted(set(terms(query["text"]))):
            entries = postings.get(term, [])
            idf = math.log1p((len(documents) - len(entries) + 0.5) / (len(entries) + 0.5))
            for i, frequency in entries:
                scores[i] += (
                    idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * lengths[i] / average))
                )
        order = sorted(scores, key=lambda i: (-scores[i], documents[i]["id"]))[:10]
        rankings.append([documents[i]["id"] for i in order])
    return rankings, {
        "index_seconds": index_seconds,
        "query_and_search_seconds": time.perf_counter() - started,
        "no_matching_term_queries": sum(not r for r in rankings),
        "k1": 1.5,
        "b": 0.75,
        "query_term_frequency": "binary",
        "normalization": "NFC, lowercase, Unicode letters/marks/numbers",
        "zero_score_documents": "not returned",
    }


def dense(documents: list[dict], queries: list[dict], path: Path) -> tuple[list[list[str]], dict]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(4)
    torch.manual_seed(20260921)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False).eval()
    load_seconds = time.perf_counter() - started

    def encode(rows: list[dict], prefix: str) -> tuple[torch.Tensor, float, int]:
        texts = [prefix + row["text"] for row in rows]
        full = tokenizer(texts, truncation=False, padding=False)["input_ids"]
        truncated = sum(len(ids) > 512 for ids in full)
        outputs = []
        started = time.perf_counter()
        with torch.inference_mode():
            for i in range(0, len(texts), 16):
                batch = tokenizer(
                    texts[i : i + 16],
                    max_length=512,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                hidden = model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                outputs.append(torch.nn.functional.normalize(pooled, p=2, dim=1))
                if i % 128 == 0:
                    print(f"{prefix.strip()} {min(i + 16, len(texts))}/{len(texts)}", flush=True)
        return torch.cat(outputs), time.perf_counter() - started, truncated

    docs, doc_seconds, doc_truncated = encode(documents, "passage: ")
    qs, query_seconds, query_truncated = encode(queries, "query: ")
    started = time.perf_counter()
    scores = qs @ docs.T
    order = scores.argsort(dim=1, descending=True, stable=True)[:, :10].tolist()
    search_seconds = time.perf_counter() - started
    return [[documents[i]["id"] for i in row] for row in order], {
        "model": "intfloat/multilingual-e5-small",
        "revision": path.name,
        "device": "cpu",
        "threads": 4,
        "batch_size": 16,
        "maximum_tokens": 512,
        "pooling": "attention-mask mean, L2 normalization; cosine search",
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
        "load_seconds": load_seconds,
        "document_encode_seconds": doc_seconds,
        "query_encode_seconds": query_seconds,
        "search_seconds": search_seconds,
        "truncated_documents": doc_truncated,
        "truncated_queries": query_truncated,
        "timing_scope": "one local pass, no latency guarantee; excludes length audit",
        "packages": {name: version(name) for name in ("torch", "transformers", "tokenizers")},
        "model_files": {p.name: sha256(p) for p in sorted(path.iterdir()) if p.is_file()},
    }


def evaluate(directory: Path, report: Path, model: Path | None) -> dict:
    if (directory / "review-pool.jsonl").exists():
        raise ValueError("review pool exists; preserve annotations and use a new pilot directory")
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"pilot checksum mismatch: {name}")
    documents = sorted(read_rows(directory / "documents.jsonl"), key=lambda r: r["id"])
    queries = read_rows(directory / "queries.jsonl")
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in read_rows(directory / "qrels.jsonl"):
        if row["document_id"] in qrels[row["query_id"]]:
            raise ValueError("duplicate relevance judgment")
        qrels[row["query_id"]][row["document_id"]] = row["grade"]
    doc_ids = {d["id"] for d in documents}
    query_ids = {q["id"] for q in queries}
    if len(doc_ids) != len(documents) or len(query_ids) != len(queries):
        raise ValueError("duplicate document or query IDs")
    if set(qrels) != query_ids or any(set(rels) - doc_ids for rels in qrels.values()):
        raise ValueError("query/document judgments mismatch")
    runs = {"bm25": bm25(documents, queries)}
    if model:
        runs["multilingual-e5-small"] = dense(documents, queries, model)
    results, rankings_by_model = {}, {}
    for name, (rankings, details) in runs.items():
        scored, saved = [], []
        for q, ranked in zip(queries, rankings, strict=True):
            scored.append(
                {
                    "group": q["group"],
                    "bucket": q["bucket"],
                    "metrics": query_metrics(ranked, qrels[q["id"]]),
                }
            )
            saved.append({"query_id": q["id"], "documents": ranked})
        rankings_by_model[name] = saved
        results[name] = {
            "all": summarize(scored),
            "buckets": {
                b: summarize([r for r in scored if r["bucket"] == b])
                for b in sorted({r["bucket"] for r in scored})
            },
            "execution": details,
        }
        write_rows(directory / f"ranking-{name}.jsonl", saved)
    # Pool candidates without showing systems, ranks, or the assumed positive to reviewers.
    lookup = {d["id"]: d for d in documents}
    review = []
    rng = random.Random(20260921)
    for i, q in enumerate(queries):
        pool = set(qrels[q["id"]])
        for rankings in rankings_by_model.values():
            pool.update(rankings[i]["documents"])
        ordered = sorted(pool)
        rng.shuffle(ordered)
        for doc_id in ordered:
            review.append(
                {
                    "query_id": q["id"],
                    "document_id": doc_id,
                    "group": q["group"],
                    "query": q["text"],
                    "document": lookup[doc_id]["text"],
                    "reviewer_1_grade": None,
                    "reviewer_2_grade": None,
                    "adjudicated_grade": None,
                    "notes": None,
                }
            )
    review_path = directory / "review-pool.jsonl"
    write_rows(review_path, review)
    result = {
        "status": "development proxy; source pairs are unreviewed relevance assumptions",
        "manifest": manifest,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "review_pairs": len(review),
        "implementations": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "results": results,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("data/processed/retrieval-pilot-v1"))
    parser.add_argument("--report", type=Path, default=Path("reports/retrieval-pilot-v1.json"))
    parser.add_argument("--model", type=Path, help="local pinned E5 snapshot; omit for BM25 only")
    args = parser.parse_args()
    result = evaluate(args.pilot, args.report, args.model)
    print(json.dumps({name: row["all"] for name, row in result["results"].items()}, indent=2))
