"""Build optional training-word caches and a version-2 fused model for ablation."""

import json
from collections import Counter
from pathlib import Path

from train_comparison import normalized, sha256


def main() -> None:
    training = Path("data/processed/train.jsonl")
    counts: Counter[str] = Counter()
    with training.open() as stream:
        for line in stream:
            record = json.loads(line)
            if record["split"] != "train":
                raise ValueError("cache must use training data only")
            counts.update(w for w in normalized(record["text"]).split() if len(w.encode()) <= 1024)
    words = sorted(counts, key=lambda word: (-counts[word], word))
    output = Path("models/speed")
    output.mkdir(parents=True, exist_ok=True)
    fusion_path = Path("models/comparison/space-fusion.json")
    fusion = json.loads(fusion_path.read_text())
    base16 = Path("models/bpe-16000.json")
    if (
        fusion["experimental_format"] != "prefix-space-fusion-v1"
        or fusion["base_model_sha256"] != sha256(base16)
        or fusion["training_sha256"] != sha256(training)
        or fusion["prefix_space_id"] != 36
    ):
        raise ValueError("fusion provenance does not match the training split and base model")
    artifacts = {}
    for name, source in [("bpe", Path("models/bpe-32000.json")), ("fused", base16)]:
        base = json.loads(source.read_text())
        if name == "fused":
            if len(base["pieces"]) + 260 != fusion["base_vocab"]:
                raise ValueError("fusion base vocabulary mismatch")
            base["version"] = 2
            base["space_fusion"] = fusion["fused_base_ids_in_new_id_order"]
        pieces = {piece["text"] for piece in base["pieces"]}
        candidates = [word for word in words if word not in pieces]
        for size in (0, 16000, 64000, 128000):
            model = {**base, "cache_words": candidates[:size]}
            path = output / f"{name}-cache-{size}.json"
            path.write_text(json.dumps(model, ensure_ascii=False) + "\n")
            artifacts[path.name] = {
                "sha256": sha256(path),
                "base_sha256": sha256(source),
                "cache_words": len(model["cache_words"]),
                "vocabulary": len(model["pieces"]) + 260 + len(model.get("space_fusion", [])),
            }
    metadata = {
        "training_sha256": sha256(training),
        "fusion_sha256": sha256(fusion_path),
        "selection": "training occurrence count descending, lexical tie break; exclude pieces; "
        "maximum 1024 UTF-8 bytes per word",
        "artifacts": artifacts,
    }
    Path("reports/speed-models.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
