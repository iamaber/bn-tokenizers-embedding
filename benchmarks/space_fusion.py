"""Train an experimental space+token layer within a 32k vocabulary budget.

This is a Python research prototype, not a model format supported by our public API.
It expands every new ID back into a space ID and one original ID before decoding.
"""

import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

from train_comparison import normalized, sha256

from bn_tokenizers_embedding import Tokenizer

SPACE = 36


def batches(path: Path, split: str) -> Iterator[list[dict[str, str]]]:
    rows = []
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] != split:
                raise ValueError("unexpected corpus split")
            rows.append(row)
            if len(rows) == 256:
                yield rows
                rows = []
    if rows:
        yield rows


def fuse(ids: list[int], mapping: dict[int, int]) -> list[int]:
    output = []
    i = 0
    while i < len(ids):
        if ids[i] == SPACE and i + 1 < len(ids) and ids[i + 1] in mapping:
            output.append(mapping[ids[i + 1]])
            i += 2
        else:
            output.append(ids[i])
            i += 1
    return output


def expand(ids: list[int], reverse: dict[int, int]) -> list[int]:
    output = []
    for token in ids:
        if token in reverse:
            output.extend((SPACE, reverse[token]))
        else:
            output.append(token)
    return output


def main() -> None:
    training = Path("data/processed/train.jsonl")
    validation = Path("data/processed/validation.jsonl")
    base = Path("models/bpe-16000.json")
    frequency: Counter[int] = Counter()
    with Tokenizer(base) as tokenizer:
        for rows in batches(training, "train"):
            for ids in tokenizer.encode_batch([normalized(row["text"]) for row in rows]):
                frequency.update(
                    right for left, right in zip(ids, ids[1:], strict=False) if left == SPACE
                )
        candidates = sorted(frequency, key=lambda token: (-frequency[token], token))
        mapping = {
            token: tokenizer.vocab_size + rank
            for rank, token in enumerate(candidates[: 32000 - tokenizer.vocab_size])
        }
        reverse = {new: old for old, new in mapping.items()}
        stats: dict[str, Counter[str]] = defaultdict(Counter)
        for rows in batches(validation, "validation"):
            texts = [normalized(row["text"]) for row in rows]
            for row, text, ids in zip(rows, texts, tokenizer.encode_batch(texts), strict=True):
                fused = fuse(ids, mapping)
                restored = expand(fused, reverse)
                if restored != ids or tokenizer.decode(restored) != text:
                    raise ValueError("space fusion changed the base IDs or text")
                for bucket in ("all", row["bucket"]):
                    stats[bucket].update(
                        {
                            "texts": 1,
                            "words": len(text.split()),
                            "base_tokens": len(ids),
                            "fused_tokens": len(fused),
                            "fused_over_510": int(len(fused) > 510),
                        }
                    )
        model = {
            "experimental_format": "prefix-space-fusion-v1",
            "base_model_sha256": sha256(base),
            "base_vocab": tokenizer.vocab_size,
            "vocab_size": tokenizer.vocab_size + len(mapping),
            "prefix_space_id": SPACE,
            "fused_base_ids_in_new_id_order": list(mapping),
            "training_sha256": sha256(training),
            "weighting": "unweighted occurrence counts of space+token in training split",
        }
        artifact = Path("models/comparison/space-fusion.json")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps(model, indent=2) + "\n")
        result = {
            "experiment": {k: v for k, v in model.items() if k != "fused_base_ids_in_new_id_order"},
            "model_sha256": sha256(artifact),
            "validation_sha256": sha256(validation),
            "round_trip_failures": 0,
            "buckets": {
                bucket: {
                    **counts,
                    "tokens_per_word": counts["fused_tokens"] / counts["words"],
                    "reduction_from_base_percent": 100
                    * (1 - counts["fused_tokens"] / counts["base_tokens"]),
                }
                for bucket, counts in sorted(stats.items())
            },
        }
        Path("reports/space-fusion.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
