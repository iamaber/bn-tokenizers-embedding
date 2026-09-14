"""Compare installed subword APIs on the full frozen validation split."""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import sentencepiece as spm
from bnlp import SentencepieceTokenizer
from train_comparison import normalized, sha256

from bn_tokenizers_embedding import Tokenizer as GoTokenizer
from tokenizers import Tokenizer as HFTokenizer

STRESS_TEXTS = [
    "আমি বাংলায় গান গাই।",
    "ami banglay gaan gai!",
    "meetingটা ১০টায়, don't be late 🙂",
    "cafe\u0301 ＡＢＣ ①",
    "বাংলা\u200dভাষা বাংলা\u200cভাষা",
    "👩🏽‍💻 🫠 🚀",
    "<unk> <s> </s> <pad> ▁ <0x20>",
    "a\x00b",
    "  বাংলা\tEnglish\nBanglish  ",
    "",
    "\t\n",
    "中文 العربية हिन्दी",
    "https://example.com/a?x=1&y=2",
    "১২৩৪৫ 12345 3.14159",
    "কি? কী! বইগুলোতে",
    "বাংলা" * 200,
]


@dataclass
class Candidate:
    name: str
    path: Path
    vocab_size: int
    encode: Callable[[str], list[int]]
    batch: Callable[[list[str]], list[list[int]]]
    decode: Callable[[list[int]], str]
    unknown_id: int | None
    batch_mode: str


def candidates(go_models: list[GoTokenizer]) -> list[Candidate]:
    result = []
    for algorithm in ["unigram", "bpe"]:
        path = Path(f"models/{algorithm}-32000.json")
        tok = GoTokenizer(path)
        go_models.append(tok)
        result.append(
            Candidate(
                f"ours-{algorithm}-32k",
                path,
                tok.vocab_size,
                tok.encode,
                tok.encode_batch,
                tok.decode,
                1,
                "native batch",
            )
        )
    path = Path("models/comparison/bnlp.model")
    bnlp = SentencepieceTokenizer(model_path=str(path))
    result.append(
        Candidate(
            "bnlp-pretrained",
            path,
            bnlp.model.vocab_size(),
            bnlp.text2id,
            lambda texts: [bnlp.text2id(text) for text in texts],
            bnlp.id2text,
            bnlp.model.unk_id(),
            "Python loop; wrapper has no native batch method",
        )
    )
    for name, path in [
        ("sentencepiece-bnlp-model", Path("models/comparison/bnlp.model")),
        ("sentencepiece-unigram-32k", Path("models/comparison/sentencepiece-32000.model")),
    ]:
        sp = spm.SentencePieceProcessor(model_file=str(path))
        result.append(
            Candidate(
                name,
                path,
                sp.vocab_size(),
                lambda text, sp=sp: sp.encode(text, out_type=int),
                lambda texts, sp=sp: sp.encode(texts, out_type=int, num_threads=1),
                sp.decode,
                sp.unk_id(),
                "native batch, one thread",
            )
        )
    path = Path("models/comparison/hf-bpe-32000.json")
    hf = HFTokenizer.from_file(str(path))
    result.append(
        Candidate(
            "huggingface-byte-bpe-32k",
            path,
            hf.get_vocab_size(),
            lambda text: hf.encode(text, add_special_tokens=False).ids,
            lambda texts: [item.ids for item in hf.encode_batch(texts, add_special_tokens=False)],
            lambda ids: hf.decode(ids, skip_special_tokens=False),
            None,
            "native batch, parallelism off",
        )
    )
    return result


def quality(candidate: Candidate, texts: list[str]) -> dict[str, int | float]:
    encoded = candidate.batch(texts)
    if encoded != [candidate.encode(text) for text in texts]:
        raise ValueError(f"single/batch mismatch: {candidate.name}")
    lengths = sorted(map(len, encoded))
    tokens = sum(lengths)
    words = sum(len(text.split()) for text in texts)
    exact = sum(candidate.decode(ids) == text for ids, text in zip(encoded, texts, strict=True))
    unknowns = sum(ids.count(candidate.unknown_id) for ids in encoded)
    return {
        "texts": len(texts),
        "tokens": tokens,
        "whitespace_words": words,
        "sequence_tokens_per_word": tokens / words if words else 0,
        "exact_normalized_round_trips": exact,
        "round_trip_percent": 100 * exact / len(texts),
        "unknown_tokens": unknowns,
        "unknown_token_percent": 100 * unknowns / tokens if tokens else 0,
        "texts_with_unknowns": sum(candidate.unknown_id in ids for ids in encoded),
        "over_510_token_percent": 100 * sum(length > 510 for length in lengths) / len(texts),
        "sequence_p95": lengths[math.ceil(0.95 * len(lengths)) - 1],
    }


def timings(
    engines: list[Candidate], texts: list[str], repeats: int, batch_size: int
) -> dict[str, dict[str, dict[str, object]]]:
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    samples = {engine.name: {"single": [], "batch": []} for engine in engines}
    operations = [(engine, mode) for engine in engines for mode in ["single", "batch"]]
    for engine in engines:
        engine.batch(texts[:batch_size])
        for text in texts[:batch_size]:
            engine.encode(text)
    rng = random.Random(20260914)
    for _ in range(repeats):
        rng.shuffle(operations)
        for engine, mode in operations:
            started = perf_counter()
            if mode == "single":
                for text in texts:
                    engine.encode(text)
            else:
                for batch in batches:
                    engine.batch(batch)
            samples[engine.name][mode].append(perf_counter() - started)
    byte_count = sum(len(text.encode()) for text in texts)
    return {
        name: {
            mode: {
                "seconds": durations,
                "median_seconds": statistics.median(durations),
                "MB_per_second": byte_count / 1e6 / statistics.median(durations),
                "texts_per_second": len(texts) / statistics.median(durations),
            }
            for mode, durations in modes.items()
        }
        for name, modes in samples.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu-model", default=platform.processor())
    parser.add_argument("--output", type=Path, default=Path("reports/framework-comparison.json"))
    args = parser.parse_args()
    if min(args.repeats, args.batch_size) < 1:
        parser.error("repeats and batch-size must be positive")
    if os.environ.get("TOKENIZERS_PARALLELISM") != "false":
        parser.error("set TOKENIZERS_PARALLELISM=false for the single-thread comparison")
    source = Path("data/processed/validation.jsonl")
    records = [json.loads(line) for line in source.read_text().splitlines()]
    if any(record["split"] != "validation" for record in records):
        raise ValueError("unexpected split in validation input")
    random.Random(20260914).shuffle(records)
    buckets: dict[str, list[str]] = defaultdict(list)
    texts = []
    for record in records:
        text = normalized(record["text"])
        buckets[record["bucket"]].append(text)
        texts.append(text)
    go_models: list[GoTokenizer] = []
    try:
        engines = candidates(go_models)
        speed = timings(engines, texts, args.repeats, args.batch_size)
        scores = {}
        for engine in engines:
            scores[engine.name] = {
                "model_sha256": sha256(engine.path),
                "model_bytes": engine.path.stat().st_size,
                "vocabulary": engine.vocab_size,
                "batch_mode": engine.batch_mode,
                "speed": speed[engine.name],
                "quality": {
                    name: quality(engine, group)
                    for name, group in [("all", texts), *sorted(buckets.items())]
                },
                "stress_cases": [
                    {
                        "input": text,
                        "expected": normalized(text),
                        "decoded": engine.decode(engine.encode(normalized(text))),
                    }
                    for text in STRESS_TEXTS
                ],
            }
        result = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_model": args.cpu_model,
            "logical_cpu_count": os.cpu_count(),
            "packages": {
                name: version(name)
                for name in [
                    "bn-tokenizers-embedding",
                    "bnlp-toolkit",
                    "sentencepiece",
                    "tokenizers",
                ]
            },
            "validation_sha256": sha256(source),
            "ordered_normalized_text_sha256": hashlib.sha256(
                json.dumps(texts).encode()
            ).hexdigest(),
            "records": len(texts),
            "utf8_bytes": sum(len(text.encode()) for text in texts),
            "repeats": args.repeats,
            "batch_size": args.batch_size,
            "seed": 20260914,
            "normalization": "shared NFC/whitespace preprocessing before timing; "
            "each API's internal preprocessing remains timed",
            "sequence_p95_method": "nearest rank: ceil(0.95 * n)",
            "harness_sha256": sha256(Path(__file__)),
            "candidates": scores,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    finally:
        for model in go_models:
            model.close()


if __name__ == "__main__":
    main()
