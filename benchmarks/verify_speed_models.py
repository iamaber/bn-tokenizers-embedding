"""Check full-validation ID equivalence and measure fresh-process model load costs."""

import json
import subprocess
import sys
from pathlib import Path

from compare_frameworks import STRESS_TEXTS
from space_fusion import expand
from train_comparison import normalized, sha256

from bn_tokenizers_embedding import Tokenizer


def main() -> None:
    source = Path("data/processed/validation.jsonl")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    if any(row["split"] != "validation" for row in rows):
        raise ValueError("unexpected split")
    texts = [normalized(row["text"]) for row in rows] + STRESS_TEXTS
    results = {}
    with Tokenizer("models/bpe-16000.json") as base16, Tokenizer("models/bpe-32000.json") as base32:
        expected16 = base16.encode_batch(texts)
        expected32 = base32.encode_batch(texts)
        for path in sorted(Path("models/speed").glob("*.json")):
            model = json.loads(path.read_text())
            fusion = model.get("space_fusion", [])
            reverse = {len(model["pieces"]) + 260 + i: token for i, token in enumerate(fusion)}
            expected = expected16 if fusion else expected32
            with Tokenizer(path) as tok:
                actual = tok.encode_batch(texts)
                for text, ids, wanted in zip(texts, actual, expected, strict=True):
                    if expand(ids, reverse) != wanted or tok.decode(ids) != normalized(text):
                        raise ValueError(f"ID or round-trip mismatch: {path}")
            load = json.loads(
                subprocess.check_output(
                    [
                        sys.executable,
                        "benchmarks/load_cost.py",
                        "--model",
                        str(path),
                        "--repeats",
                        "5",
                    ],
                    text=True,
                )
            )
            results[path.name] = {
                "model_sha256": sha256(path),
                "checked_sequences": len(texts),
                "id_equivalence_failures": 0,
                "round_trip_failures": 0,
                "equivalence": "expanded fused IDs equal 16k base"
                if fusion
                else "IDs equal 32k base",
                "load": load,
            }
    Path("reports/speed-verification.json").write_text(
        json.dumps(
            {
                "validation_sha256": sha256(source),
                "models": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
