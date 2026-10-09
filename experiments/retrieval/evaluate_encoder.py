"""Evaluate a local original/adapted encoder without modifying frozen datasets."""

import argparse
import json
import platform
import resource
import statistics
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import torch
from compare import score_rankings
from encoder import Encoder
from prepare import read_rows, sha256, verify_manifest, write_rows


def means(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row["metrics"])
    if not grouped:
        raise ValueError("no scored query groups")
    return {
        "queries": len(rows),
        "groups": len(grouped),
        "weighting": "equal group weight",
        "metrics": {
            metric: statistics.mean(
                statistics.mean(r[metric] for r in group) for group in grouped.values()
            )
            for metric in ("recall_at_10", "ndcg_at_10", "mrr_at_10")
        },
    }


def evaluate(model: Path, datasets: list[Path], output: Path, repeats: int = 5) -> dict:
    if output.exists():
        raise ValueError("evaluation output exists; use a new run directory")
    if repeats != 0 and repeats < 3:
        raise ValueError("timing requires at least three repeated passes")
    if len({p.name for p in datasets}) != len(datasets):
        raise ValueError("dataset directory names must be unique in a suite")
    snapshots = {}
    all_documents, all_queries = set(), set()
    query_buckets = defaultdict(set)
    for path in datasets:
        manifest = verify_manifest(path)
        documents = sorted(read_rows(path / "documents.jsonl"), key=lambda r: r["id"])
        queries = read_rows(path / "queries.jsonl")
        if len({r["id"] for r in documents}) != len(documents):
            raise ValueError("duplicate document IDs")
        snapshots[path.name] = (path, manifest, documents, queries)
        all_documents.update("passage: " + row["text"] for row in documents)
        all_queries.update("query: " + row["text"] for row in queries)
        for row in queries:
            query_buckets[row["bucket"]].add("query: " + row["text"])
    document_texts, query_texts = sorted(all_documents), sorted(all_queries)
    torch.set_num_threads(4)
    started = time.perf_counter()
    encoder = Encoder(model, max_length=512)
    load_seconds = time.perf_counter() - started
    output.mkdir(parents=True)
    try:
        documents_embedded, document_stats = encoder.encode(document_texts)
        queries_embedded, query_stats = encoder.encode(query_texts)
        document_index = {text: i for i, text in enumerate(document_texts)}
        query_index = {text: i for i, text in enumerate(query_texts)}
        results = {}
        for name, (path, manifest, documents, queries) in snapshots.items():
            destination = output / name
            destination.mkdir()
            document_vectors = documents_embedded[
                [document_index["passage: " + row["text"]] for row in documents]
            ]
            query_vectors = queries_embedded[
                [query_index["query: " + row["text"]] for row in queries]
            ]
            with torch.inference_mode():
                # Sorted document IDs provide deterministic ties for every arm.
                indices = (
                    (query_vectors @ document_vectors.T)
                    .argsort(dim=1, descending=True, stable=True)[:, :10]
                    .tolist()
                )
            ranking = [
                {"query_id": query["id"], "documents": [documents[i]["id"] for i in order]}
                for query, order in zip(queries, indices, strict=True)
            ]
            write_rows(destination / "rankings.jsonl", ranking)
            scored = score_rankings(path, destination / "rankings.jsonl")
            write_rows(destination / "scores.jsonl", scored)
            results[name] = {
                "dataset": str(path),
                "dataset_manifest_sha256": sha256(path / "manifest.json"),
                "label_status": manifest.get("label_status", "unreviewed source counterparts"),
                "all": means(scored),
                "buckets": {
                    bucket: means([r for r in scored if r["bucket"] == bucket])
                    for bucket in sorted({r["bucket"] for r in scored})
                },
                "domains": {
                    domain: means([r for r in scored if r["domain"] == domain])
                    for domain in sorted({r["domain"] for r in scored})
                },
                "ranking_sha256": sha256(destination / "rankings.jsonl"),
                "scores_sha256": sha256(destination / "scores.jsonl"),
            }
        timing = None
        if repeats:
            # All unique queries, same fixed batch size and cached document workload.
            encoder.encode(query_texts[:16])
            timings = []
            for _ in range(repeats):
                started = time.perf_counter()
                vectors, _ = encoder.encode(query_texts)
                with torch.inference_mode():
                    (vectors @ documents_embedded.T).argsort(dim=1, descending=True, stable=True)[
                        :, :10
                    ].tolist()
                timings.append(time.perf_counter() - started)
            timing = {
                "repeat_seconds": timings,
                "median_seconds": statistics.median(timings),
                "median_seconds_per_query": statistics.median(timings) / len(query_texts),
                "queries_per_pass": len(query_texts),
                "cached_documents": len(document_texts),
                "scope": (
                    "warm batch throughput: tokenize + encode + exact cached-document search; "
                    "excludes loading/indexing"
                ),
            }
            request_texts = [
                text
                for bucket in sorted(query_buckets)
                for text in sorted(query_buckets[bucket])[:4]
            ]
            write_rows(output / "request-workload.jsonl", [{"text": t} for t in request_texts])
            request_seconds = []
            for _ in range(repeats):
                for text in request_texts:
                    started = time.perf_counter()
                    vector, _ = encoder.encode([text], batch_size=1)
                    with torch.inference_mode():
                        (vector @ documents_embedded.T).argsort(
                            dim=1, descending=True, stable=True
                        )[:, :10].tolist()
                    request_seconds.append(time.perf_counter() - started)
            timing["single_query"] = {
                "repeat_seconds": request_seconds,
                "median_seconds": statistics.median(request_seconds),
                "requests": len(request_seconds),
                "unique_queries": len(request_texts),
                "workload_sha256": sha256(output / "request-workload.jsonl"),
                "scope": "warm batch-1 tokenize + encode + exact cached-document search",
            }
        write_rows(output / "query-workload.jsonl", [{"text": t} for t in query_texts])
        write_rows(output / "document-workload.jsonl", [{"text": t} for t in document_texts])
        if timing:
            timing["query_workload_sha256"] = sha256(output / "query-workload.jsonl")
            timing["document_workload_sha256"] = sha256(output / "document-workload.jsonl")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = peak if platform.system() == "Darwin" else peak * 1024
        result = {
            "model": str(model),
            "model_files": {p.name: sha256(p) for p in sorted(model.iterdir()) if p.is_file()},
            "custom_tokenizer": encoder.custom_path is not None,
            "maximum_tokens": 512,
            "batch_size": 16,
            "cpu_threads": 4,
            "load_seconds": load_seconds,
            "document_encoding": document_stats,
            "query_encoding": query_stats,
            "runtime": timing,
            "peak_process_rss_bytes": peak_bytes,
            "memory_scope": (
                "fresh-process peak including model load, encoding and ranking; not training memory"
            ),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": {name: version(name) for name in ("torch", "transformers", "tokenizers")},
            "implementation_sha256": {
                p.name: sha256(p)
                for p in (
                    Path(__file__),
                    Path(__file__).with_name("encoder.py"),
                    Path(__file__).with_name("compare.py"),
                    Path(__file__).with_name("metrics.py"),
                )
            },
            "results": results,
        }
        (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        encoder.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, action="append", default=[])
    parser.add_argument(
        "--domains", type=Path, help="a development/test directory containing nine directions"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    args = parser.parse_args()
    datasets = args.dataset + (
        sorted(p for p in args.domains.iterdir() if p.is_dir()) if args.domains else []
    )
    if not datasets:
        parser.error("at least one dataset is required")
    result = evaluate(args.model, datasets, args.output, args.benchmark_repeats)
    print(
        json.dumps({name: r["all"]["metrics"] for name, r in result["results"].items()}, indent=2)
    )
