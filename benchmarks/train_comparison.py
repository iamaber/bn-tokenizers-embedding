"""Train external 32k baselines on the existing training split only."""

import hashlib
import json
import os
import unicodedata
from importlib.metadata import version
from pathlib import Path
from time import perf_counter


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import sentencepiece as spm

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    if os.environ.get("TOKENIZERS_PARALLELISM") != "false":
        raise ValueError("set TOKENIZERS_PARALLELISM=false before training")
    source = Path("data/processed/train.jsonl")
    output = Path("models/comparison")
    output.mkdir(parents=True, exist_ok=True)
    training = output / "train.txt"
    counts: dict[str, int] = {}
    max_bytes = 0
    with source.open() as records, training.open("w", encoding="utf-8") as stream:
        for line in records:
            record = json.loads(line)
            if record["split"] != "train":
                raise ValueError("non-training record in training input")
            text = normalized(record["text"])
            max_bytes = max(max_bytes, len(text.encode()))
            stream.write(text + "\n")
            counts[record["bucket"]] = counts.get(record["bucket"], 0) + 1
    settings = {
        "model_type": "unigram",
        "vocab_size": 32000,
        "character_coverage": 1.0,
        "byte_fallback": True,
        "normalization_rule_name": "identity",
        "num_threads": 1,
        "shuffle_input_sentence": False,
        "input_sentence_size": 0,
        "max_sentence_length": max_bytes + 1,
        "max_sentencepiece_length": 12,
        "hard_vocab_limit": False,
    }
    started = perf_counter()
    spm.SentencePieceTrainer.train(
        input=str(training), model_prefix=str(output / "sentencepiece-32000"), **settings
    )
    sp_seconds = perf_counter() - started
    hf = Tokenizer(models.BPE())
    hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    hf.decoder = decoders.ByteLevel()
    started = perf_counter()
    hf.train(
        [str(training)],
        trainers.BpeTrainer(
            vocab_size=32000,
            min_frequency=2,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    hf_seconds = perf_counter() - started
    hf.save(str(output / "hf-bpe-32000.json"))
    metadata = {
        "training_source_sha256": sha256(source),
        "normalized_training_sha256": sha256(training),
        "records_by_bucket": counts,
        "weighting": "one occurrence per training record; no bucket or word weights",
        "sentencepiece_version": version("sentencepiece"),
        "tokenizers_version": version("tokenizers"),
        "sentencepiece_settings": settings,
        "hf_settings": {
            "algorithm": "ByteLevel BPE",
            "vocab_size": 32000,
            "min_frequency": 2,
            "add_prefix_space": False,
            "alphabet": "all 256 bytes",
            "special_tokens": [],
        },
        "training_seconds": {"sentencepiece": sp_seconds, "hf_bpe": hf_seconds},
        "model_hashes": {
            path.name: sha256(path)
            for path in [output / "sentencepiece-32000.model", output / "hf-bpe-32000.json"]
        },
    }
    Path("reports/comparison-training.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
