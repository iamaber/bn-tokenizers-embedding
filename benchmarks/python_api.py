"""Measure the installed Python API, including validation and JSON/CGo overhead."""

import argparse
import hashlib
import json
import platform
import statistics
import tempfile
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import TypedDict

from bn_tokenizers_embedding import Tokenizer, normalize

TEXTS = [
    "আমি আজ ভালো আছি। পাসপোর্ট নবায়ন করতে কী কী কাগজপত্র লাগবে?",
    "I am doing well today. Which documents do I need to renew my passport?",
    "ami aaj valo asi. passport renew korte ki ki kagoj lagbe?",
    "ajke meetingটা কখন? filesগুলো update kore dao 🙂",
]


class Measurement(TypedDict):
    sample_seconds: list[float]
    median_seconds: float
    microseconds_per_text: float
    texts_per_second: float
    input_MB_per_second: float


def summarize(durations: list[float], iterations: int, texts: list[str]) -> Measurement:
    seconds = statistics.median(durations)
    total_bytes = sum(len(text.encode("utf-8")) for text in texts) * iterations
    return {
        "sample_seconds": durations,
        "median_seconds": seconds,
        "microseconds_per_text": seconds * 1e6 / (iterations * len(texts)),
        "texts_per_second": iterations * len(texts) / seconds,
        "input_MB_per_second": total_bytes / 1e6 / seconds,
    }


def measure_pair(
    single: Callable[[], object],
    batch: Callable[[], object],
    iterations: int,
    repeats: int,
    texts: list[str],
) -> tuple[Measurement, Measurement]:
    operations = [single, batch]
    samples: list[list[float]] = [[], []]
    for operation in operations:
        operation()  # Warm up outside the measurement.
    for repeat in range(repeats):
        for index in [0, 1] if repeat % 2 == 0 else [1, 0]:
            started = perf_counter()
            for _ in range(iterations):
                operations[index]()
            samples[index].append(perf_counter() - started)
    return summarize(samples[0], iterations, texts), summarize(samples[1], iterations, texts)


def benchmark(model: Path, iterations: int, repeats: int, batch_size: int) -> dict[str, object]:
    started = perf_counter()
    tok = Tokenizer(model)
    load_seconds = perf_counter() - started
    with tok:
        results = {}
        for name, copies in [("short", 1), ("long", 16)]:
            texts = [" ".join([TEXTS[i % len(TEXTS)]] * copies) for i in range(batch_size)]
            expected = [tok.encode(text) for text in texts]
            if tok.encode_batch(texts) != expected:
                raise RuntimeError("single and batch IDs differ")
            if [tok.decode(ids) for ids in expected] != [normalize(text) for text in texts]:
                raise RuntimeError("round-trip check failed")
            single, batch = measure_pair(
                lambda texts=texts: [tok.encode(text) for text in texts],
                lambda texts=texts: tok.encode_batch(texts),
                iterations,
                repeats,
                texts,
            )
            # Compute the ratio from medians rather than claiming a speedup from one run.
            results[name] = {
                "single": single,
                "batch": batch,
                "batch_speedup": single["median_seconds"] / batch["median_seconds"],
                "tokens_per_batch": sum(map(len, expected)),
                "workload_sha256": hashlib.sha256(json.dumps(texts).encode()).hexdigest(),
            }
        return {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "package_version": version("bn-tokenizers-embedding"),
            "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "vocabulary": tok.vocab_size,
            "load_seconds": load_seconds,
            "iterations": iterations,
            "repeats": repeats,
            "batch_size": batch_size,
            "text_fixture_sha256": hashlib.sha256("\n".join(TEXTS).encode()).hexdigest(),
            "round_trip_failures": 0,
            "results": results,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, help="trained JSON model; omit for a CI smoke fixture"
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if min(args.iterations, args.repeats, args.batch_size) < 1:
        parser.error("iterations, repeats and batch-size must be positive")
    with tempfile.TemporaryDirectory() as directory:
        model = args.model
        if model is None:
            model = Path(directory) / "fixture.json"
            model.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "algorithm": "unigram",
                        "normalization": "nfc-whitespace-v1",
                        "pieces": [{"text": "আমি", "score": -1}],
                    }
                ),
                encoding="utf-8",
            )
        result = benchmark(model, args.iterations, args.repeats, args.batch_size)
        result["model_kind"] = "trained" if args.model else "ci_smoke_fixture"
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
