"""Measure real-document exact-search serving, including cold starts and concurrency."""

import argparse
import json
import math
import platform
import random
import resource
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from encoder import Encoder
from prepare import read_rows, sha256, write_rows


def distribution(seconds: list[float]) -> dict:
    if not seconds or any(not math.isfinite(x) or x <= 0 for x in seconds):
        raise ValueError("require finite positive latency samples")
    ordered = sorted(seconds)
    return {
        "samples": len(seconds), "p50_ms": statistics.median(seconds) * 1000,
        "p95_ms": ordered[math.ceil(len(ordered) * .95) - 1] * 1000,
        "p99_ms": ordered[math.ceil(len(ordered) * .99) - 1] * 1000,
        "seconds": seconds,
    }


def peak_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1024 * 1024 if sys.platform == "darwin" else 1024
    )


def workload(alignment: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("workload exists; preserve versions")
    manifest = json.loads((alignment / "manifest.json").read_text())
    if sha256(alignment / "pairs.jsonl") != manifest["files"]["pairs.jsonl"]:
        raise ValueError("alignment checksum mismatch")
    pairs = read_rows(alignment / "pairs.jsonl")
    texts = sorted({r[field] for r in pairs for field in ("query", "document")})
    random.Random(17).shuffle(texts)
    if len(texts) < 10000:
        raise ValueError("require ten thousand distinct real document texts")
    documents = [{"text": "passage: " + text} for text in texts[:10000]]
    # Use training text for performance only; no search-quality inference is made.
    examples = {}
    for row in pairs:
        examples.setdefault(row["query_language"], row["query"])
        examples.setdefault("bangla", row["document"])
    if examples.keys() != {"bangla", "banglish", "english"}:
        raise ValueError("three language workloads required")
    queries = []
    for language, text in sorted(examples.items()):
        words = text.split()
        for length in ("short", "128-words", "512-words"):
            target = 0 if length == "short" else int(length.split("-")[0])
            content = text if not target else " ".join((words * (target // len(words) + 1))[:target])
            queries.append({"text": "query: " + content, "language": language, "length": length})
    output.mkdir(parents=True)
    write_rows(output / "documents.jsonl", documents)
    write_rows(output / "queries.jsonl", queries)
    result = {
        "alignment_manifest_sha256": sha256(alignment / "manifest.json"),
        "documents": len(documents), "queries": len(queries),
        "source": "distinct natural audited training forms; quality not measured",
        "long_inputs": "repeated training sentences, synthetic stress inputs, not natural traffic",
        "files": {n: sha256(output / n) for n in ("documents.jsonl", "queries.jsonl")},
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def load_workload(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text())
    if any(sha256(path / name) != digest for name, digest in manifest["files"].items()):
        raise ValueError("serving workload changed")
    return manifest


def cold(model: Path, workload_path: Path, index: Path) -> dict:
    load_workload(workload_path)
    torch.set_num_threads(4)
    started = time.perf_counter()
    encoder = Encoder(model, max_length=512)
    try:
        vectors = torch.load(index, map_location="cpu", weights_only=True)
        if vectors.shape != (10000, encoder.model.config.hidden_size) or not torch.isfinite(vectors).all():
            raise ValueError("invalid serving index")
        loaded = time.perf_counter() - started
        vector, counts = encoder.encode([read_rows(workload_path / "queries.jsonl")[0]["text"]], 1)
        with torch.inference_mode():
            (vector @ vectors.T).argsort(descending=True, stable=True)[:, :10].tolist()
        return {"load_model_and_index_seconds": loaded,
                "first_response_seconds": time.perf_counter() - started,
                "peak_process_rss_mib": peak_mib(), "query_tokens": counts}
    finally:
        encoder.close()


def benchmark(model: Path, workload_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("benchmark exists; preserve versions")
    manifest = load_workload(workload_path)
    queries = read_rows(workload_path / "queries.jsonl")
    documents = read_rows(workload_path / "documents.jsonl")
    torch.set_num_threads(4)
    encoder = Encoder(model, max_length=512)
    encoder.model.eval()
    lock = threading.Lock()  # HF tokenizers mutate padding/truncation configuration.
    try:
        indexed, counts = encoder.encode([r["text"] for r in documents])
        output.mkdir(parents=True)
        index_path = output / "index.pt"
        torch.save(indexed, index_path)

        def request(row: dict, size: int, submitted: float | None = None) -> tuple[float, list[int], float]:
            started = time.perf_counter()
            with lock:
                inputs, _ = encoder.inputs([row["text"]])
            # Autograd state is thread local, so every worker enters this context.
            with torch.inference_mode():
                vector = encoder.pool(inputs)
                ids = (vector @ indexed[:size].T).argsort(descending=True, stable=True)[:, :10]
            completed = time.perf_counter()
            return completed - started, ids[0].tolist(), completed - (submitted or started)

        reference = {size: [request(row, size)[1] for row in queries] for size in (1000, 10000)}
        settings = []
        for size in (1000, 10000):
            for length in ("short", "128-words", "512-words"):
                requests = [(i, row) for _ in range(12) for i, row in enumerate(queries) if row["length"] == length]
                for concurrency in (1, 4, 8):
                    started = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=concurrency) as executor:
                        futures = [executor.submit(request, row, size, time.perf_counter()) for _, row in requests]
                        samples = [future.result() for future in futures]
                    elapsed = time.perf_counter() - started
                    if any(ids != reference[size][i] for (i, _), (_, ids, _) in zip(requests, samples, strict=True)):
                        raise ValueError("concurrent retrieval differs from serial reference")
                    settings.append({"index_documents": size, "input_length": length,
                                     "concurrency": concurrency, "elapsed_seconds": elapsed,
                                     "requests_per_second": len(samples) / elapsed,
                                     "service_latency": distribution([seconds for seconds, _, _ in samples]),
                                     "burst_response_latency": distribution([response for _, _, response in samples]),
                                     "burst_scope": "36 requests submitted together; includes executor queue",
                                     "latency_scope": "execution plus tokenizer-lock wait; excludes executor queue",
                                     "serial_rankings_verified": True})
        token_counts = []
        for row in queries:
            _, statistics_row = encoder.inputs([row["text"]])
            token_counts.append({"language": row["language"], "length": row["length"], **statistics_row})
        warm_peak = peak_mib()
    finally:
        encoder.close()
    cold_runs = []
    for _ in range(3):
        command = [sys.executable, str(Path(__file__).resolve()), "cold", "--model", str(model),
                   "--workload", str(workload_path), "--index", str(index_path)]
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        cold_runs.append(json.loads(process.stdout))
    result = {
        "model": str(model), "model_files": {p.name: sha256(p) for p in model.iterdir() if p.is_file()},
        "workload_manifest_sha256": sha256(workload_path / "manifest.json"),
        "workload": manifest, "index_sha256": sha256(index_path),
        "index_encoding": counts, "index_tensor_bytes": indexed.numel() * indexed.element_size(),
        "input_tokens": token_counts, "warm_workloads": settings, "cold_starts": cold_runs,
        "warm_peak_process_rss_mib": warm_peak, "torch_intraop_threads": 4,
        "platform": platform.platform(), "torch": torch.__version__,
        "implementation_sha256": {p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name("encoder.py"))},
        "limitations": ["CPU exact search, not ANN or distributed service",
                        "synthetic repeated long inputs; no natural-traffic claim",
                        "36 requests per setting; p99 is sample maximum, not stable tail estimate",
                        "cold time starts after Python/module import; includes model and saved-index load",
                        "service latency excludes queued requests; throughput includes scheduling"],
    }
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("workload", "benchmark", "cold"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--alignment", type=Path)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "workload":
        if not args.alignment or not args.output:
            parser.error("workload requires --alignment and --output")
        result = workload(args.alignment, args.output)
    elif args.command == "cold":
        if not args.model or not args.workload or not args.index:
            parser.error("cold requires --model, --workload and --index")
        result = cold(args.model, args.workload, args.index)
    else:
        if not args.model or not args.workload or not args.output:
            parser.error("benchmark requires --model, --workload and --output")
        result = benchmark(args.model, args.workload, args.output)
    print(json.dumps(result if args.command != "benchmark" else {
        "warm_workloads": len(result["warm_workloads"]), "cold_starts": result["cold_starts"]
    }, indent=2))
