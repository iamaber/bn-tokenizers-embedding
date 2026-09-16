"""Fingerprint validation and edge-case token IDs using the installed package."""

import hashlib
import json
from pathlib import Path

from bn_tokenizers_embedding import Tokenizer


def main() -> None:
    source = Path("data/processed/validation.jsonl")
    texts = [json.loads(line)["text"] for line in source.read_text().splitlines()]
    texts += ["", "a\x00b", "<unk> ▁ <0x20>", "👩🏽‍💻", " বাংলা\tEnglish ", "বাংলা" * 200]
    digests = {}
    for algorithm in ["unigram", "bpe"]:
        with Tokenizer(f"models/{algorithm}-32000.json") as tokenizer:
            payload = json.dumps(tokenizer.encode_batch(texts), separators=(",", ":")).encode()
            digests[algorithm] = hashlib.sha256(payload).hexdigest()
    print(json.dumps({"texts": len(texts), "digests": digests}, indent=2))


if __name__ == "__main__":
    main()
